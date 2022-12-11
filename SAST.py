import torch
import argparse
from datasets.dataset import dataset
from torch import optim, nn
import time
from utils import AverageMeter, setup_default_logging, plot_confusion_matrix, soft_CrossEntropy
from utils import PData, distribution_alignment, PMovingMeter, one_hot
import os
import numpy as np
from models.create_module import create_model
import random
from itertools import cycle
from torch.cuda.amp import autocast, GradScaler
import contextlib
from scipy.stats import entropy
from datasets.dataset import Resample
from models.center_loss import CenterLoss


def train(data_loader_x, data_loader_u, criterion, criterion_u, model, epoch, optimizer, scheduler, ema, logger,
          criterion_soft, criterion_center, args):
    start = time.time()
    model.train()

    scaler = GradScaler()
    amp_cm = autocast if args.amp else contextlib.nullcontext

    data_loader_x = iter(data_loader_x)
    data_loader_u = iter(data_loader_u)
    pre_acc_meter = AverageMeter()
    train_loss_meter = AverageMeter()
    loss_x_meter = AverageMeter()
    loss_u_meter = AverageMeter()

    for i in range(args.iters):
        img_weak_x, img_strong_x, label_x,  = next(data_loader_x)
        img_weak_u, img_strong_u, label_u_true = next(data_loader_u)
        img_weak_x, landmark_weak_x = img_weak_x[0], img_weak_x[1]
        img_weak_u, landmark_weak_u = img_weak_u[0], img_weak_u[1]
        img_strong_u, landmark_strong_u = img_strong_u[0], img_strong_u[1]
        label_x = label_x.cuda()
        label_u_true = torch.as_tensor(label_u_true, dtype=torch.long).cuda()
        image = torch.cat([img_weak_x, img_weak_u, img_strong_u], dim=0).cuda()
        landmark = torch.cat([landmark_weak_x, landmark_weak_u, landmark_strong_u], dim=0).cuda()
        with amp_cm():
            # logits = model(image)
            logits, d, feature = model(image, landmark)
            logits_weak_x = logits[:args.batch_size]
            logits_weak_u, logits_strong_u = torch.split(logits[args.batch_size:], args.mu * args.batch_size)
            # mask_l = mask_l[args.batch_size: args.batch_size * 2]
            loss_x = criterion(logits_weak_x, label_x)

            with torch.no_grad():
                probs = torch.softmax(logits_weak_u, dim=1)
                scores, label_u_guess = torch.max(probs, dim=1)
                mask_h = (scores.ge(args.thr)).float()
                mask_s = entropy(probs.detach().cpu(), axis=1) > args.thr_e
                mask_s = (torch.tensor(mask_s).cuda() & scores.ge(args.thr_l) & scores.le(args.thr)).float()
                # mask_s = (mask_s & mask_h).float()
                correct = (label_u_guess == label_u_true).sum().cpu().item()
            loss_u = (criterion_u(logits_strong_u, label_u_guess) * mask_h).mean()
            if torch.sum(mask_h) > 1:
                loss_u += (criterion_soft(logits_strong_u, probs) * mask_s).mean()
            loss = loss_x + args.lambda_mu*loss_u - args.alpha*min(d, args.beta*(loss_x+loss_u)) + criterion_center(feature[:args.batch_size], label_x) * args.center
            # loss = loss_x + args.lambda_mu*loss_u - args.alpha*min(d, args.beta*(loss_x+loss_u))
        # if epoch < 10:
        #     optimizer.param_groups[0]['lr'] = (epoch+1) * 1e-4
        cur_lr = optimizer.param_groups[0]['lr']
        optimizer.zero_grad()
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            for param in criterion_center.parameters():
                param.grad.data *= (cur_lr / (args.center * args.lr))
            optimizer.step()
        ema.update_params()

        n = args.mu * args.batch_size
        pre_acc = correct / n

        pre_acc_meter.update(pre_acc)
        train_loss_meter.update(loss)
        loss_x_meter.update(loss_x)
        loss_u_meter.update(loss_u)


    scheduler.step()
    t = time.time() - start
    logger.info('epoch:{} pre_acc:{:.4f} train_loss:{:.4f} loss_x:{:.4f} loss_u:{:.4f} time:{:.4}'.
                format(epoch+1, pre_acc_meter.avg, train_loss_meter.avg, loss_x_meter.avg, loss_u_meter.avg, t))
    ema.update_buffer()

    return pre_acc_meter.avg, train_loss_meter.avg, loss_x_meter.avg, loss_u_meter.avg, cur_lr


def evaluate(dataloader, criterion, model):
    model.eval()
    val_acc_meter = AverageMeter()
    val_loss_meter = AverageMeter()
    y_true, y_pre = [], []

    with torch.no_grad():
        for i, (img, label) in enumerate(dataloader):
            img, landmark = img[0], img[1]
            img, landmark = img.cuda(), landmark.cuda()
            label = torch.as_tensor(label, dtype=torch.long).cuda()

            # logits = model(img)
            logits, _, _ = model(img, landmark)
            loss = criterion(logits, label)
            probs = torch.softmax(logits, dim=1)
            scores, target = torch.max(probs, dim=1)
            correct = (target == label).sum().cpu().item()
            y_true.extend(np.array(label.cpu()))
            y_pre.extend(np.array(target.cpu()))

            acc = correct / img.size(0)
            val_acc_meter.update(acc)
            val_loss_meter.update(loss)

    return val_acc_meter.avg, val_loss_meter.avg, y_true, y_pre


def resample(model, dataloader, resampler, args):
    model.eval()
    infos, labels, scores = [], [], []
    # thrs = torch.zeros(n_class)
    with torch.no_grad():
        for i, (img, info) in enumerate(dataloader):
            img, landmark = img[0], img[1]
            img, landmark = img.cuda(), landmark.cuda()

            logits, _, _ = model(img, landmark)
            prob = torch.softmax(logits, dim=1)
            score, label_guess = torch.max(prob, dim=1)
            infos.extend(info)
            labels.extend(label_guess.cpu().tolist())
            scores.extend(score.cpu().tolist())
    infos = np.array([infos]).transpose([1, 0])
    labels = np.array([labels]).transpose([1, 0])
    scores = np.array([scores]).transpose([1, 0])
    result = np.hstack((infos, labels, scores))
    resampler.resample_image(result)


def set_random_seed(seed=42):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='SAST Pytorch with Affectnet')
    # data
    parser.add_argument('--data', default='FERPlus', type=str)
    parser.add_argument('--pretrain', default='FERPlus', type=str)
    parser.add_argument('--label_num', default=-1, type=float, help='number for labeled images')
    parser.add_argument('--n_class', default=8, type=int)
    parser.add_argument('--mu', default=1, type=int, help='ratio for each batch unlabeled images')  # 6
    parser.add_argument('--batch_size', default=16, type=int)  # 16
    parser.add_argument('--aug_n', default=4, type=float)
    parser.add_argument('--aug_m', default=20, type=float)
    # model
    parser.add_argument('--model', default='new', type=str)
    parser.add_argument('--model_path', default='resnet18_msceleb.pth')
    # parser.add_argument('--model_path', default='output/model/2022_11_05_01_01_41_Pretrain/model')
    parser.add_argument('--model_path1', default='')
    # parser.add_argument('--model_path1', default='output/model/2022_11_15_22_17_33_FERPlus_-1/model')
    parser.add_argument('--ema_alpha', default=0.999, type=float)
    parser.add_argument('--dropout', default=0.5, type=float)
    # solver
    parser.add_argument('--use_cuda', default=True, type=bool)
    parser.add_argument('--amp', default=False, type=bool)
    parser.add_argument('--resample', default=False, type=bool)
    parser.add_argument('--epoch', default=512, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--early_stop', default=512, type=int)
    parser.add_argument('--iters', default=256, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--thr', default=0.95, type=float, help='threshold for mask')
    parser.add_argument('--thr_e', default=0.3, type=float, help='threshold for entropy mask')
    parser.add_argument('--thr_l', default=0.35, type=float, help='threshold for logits low mask')
    parser.add_argument('--lambda_mu', default=2.5, type=float, help='unlabeled weight')
    parser.add_argument('--thr_r', default=0.2, type=float, help='threshold for resample')
    parser.add_argument('--resample_rate', default=100, type=int, help='the number of resample times')
    parser.add_argument('--alpha', default=0.1, type=float, help='weight for distance loss0.1')
    parser.add_argument('--beta', default=0.2, type=float, help='distance0.2')
    parser.add_argument('--lambda_pb', default=0.1, type=float, help='weight of PB feature0.1')
    parser.add_argument('--center', default=1.5, type=float, help='weight for center loss')
    parser.add_argument('--center_feature', default=32, type=int, help='feature dim for center loss')
    parser.add_argument('--lambda_center', default=0.5, type=float, help='weight of center feature')
    parser.add_argument('--output_path', default='./output', type=str)
    args = parser.parse_args()
    set_random_seed()

    # cuda
    if args.use_cuda and torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'

    # logger
    time_name = str(time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time())))+'_'+args.data+'_'+str(args.label_num)
    logger, writer = setup_default_logging(args, time_name)
    logger.info('------ prepare the 1st stage training ------')
    logger.info(dict(args._get_kwargs()))

    # model
    model, ema = create_model(args)

    # loss
    criterion = nn.CrossEntropyLoss().cuda()
    criterion_onehot = soft_CrossEntropy(reduction='none').cuda()
    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda()
    criterion_center = CenterLoss(num_classes=args.n_class, feat_dim=args.center_feature).cuda()

    # optimizer
    parameters = list(model.parameters()) + list(criterion_center.parameters())
    optimizer = optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)

    # data
    data = dataset(args)
    data_loader_x, data_loader_u = data.get_train()
    val_data = data.get_val()
    if args.resample:
        resample_loader = data.get_resample()
        resampler = Resample(args)
    else:
        resample_loader = None

    logger.info(f"  Task = {args.data} with {args.label_num} labeled image")
    logger.info(f"  Number of iterations per epoch = {args.iters}")
    logger.info(f"  Batch size per GPU = {args.batch_size} labeled image, "
                f"{args.batch_size * args.mu} weak unlabeled image and "
                f"{args.batch_size * args.mu} strong unlabeled image")
    logger.info(f"  Total optimization steps = {args.iters * args.epoch}")
    logger.info("  Total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))

    # train
    logger.info('------ start the 1st stage training ------')
    best_acc, best_epoch, stop = 0, 0, 0
    model_name = time_name
    output_dir = os.path.join(args.output_path, 'model', model_name)

    for epoch in range(args.epoch-args.start_epoch):
        epoch = epoch + args.start_epoch
        pre_acc, train_loss, loss_x, loss_u, lr = train(data_loader_x=data_loader_x,
                                                        data_loader_u=data_loader_u,
                                                        criterion=criterion,
                                                        criterion_u=criterion_u,
                                                        model=model,
                                                        epoch=epoch,
                                                        optimizer=optimizer,
                                                        ema=ema,
                                                        logger=logger,
                                                        args=args,
                                                        scheduler=scheduler,
                                                        criterion_soft=criterion_onehot,
                                                        criterion_center=criterion_center,
                                                        )
        val_acc, val_loss, y_true, y_pre = evaluate(dataloader=val_data,
                                                    criterion=criterion,
                                                    model=model)
        stop += 1
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            stop = 0
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            state = model.state_dict()
            torch.save(state, output_dir + '/model')

        logger.info('val_acc:{:.4f} val_loss:{:.4f} Best_acc {:.4f} in epoch {}'.
                    format(val_acc, val_loss, best_acc, best_epoch))
        if stop >= args.early_stop:
            break
        if (epoch + 1) % 5 == 0:
            plot_confusion_matrix(y_true=y_pre, y_pre=y_true, epoch=epoch,
                                  path=args.output_path, time_name=time_name, data=args.data)

        if args.resample and (epoch+1) % (args.epoch//args.resample_rate) == 0:
            logger.info('------ resampling ------')
            resample(model, resample_loader, resampler, args)
            del data_loader_x, data_loader_u
            data_loader_x, data_loader_u = data.get_train()

        writer.add_scalars('train_and_val_loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalar('pre_acc', pre_acc, epoch)
        writer.add_scalar('train_loss_x', loss_x, epoch)
        writer.add_scalar('train_loss_u', loss_u, epoch)
        writer.add_scalar('val_acc', val_acc, epoch)
        writer.add_scalar('lr', lr, epoch)

    writer.close()


if __name__ == '__main__':
    main()
