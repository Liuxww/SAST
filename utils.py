# -*-coding:utf-8-*-
from datetime import datetime
import logging
import os
import sys
import torch
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo
import scipy.stats


def interleave(x, bt):
    s = list(x.shape)
    return torch.reshape(torch.transpose(x.reshape([-1, bt] + s[1:]), 1, 0), [-1] + s[1:])


def de_interleave(x, bt):
    s = list(x.shape)
    return torch.reshape(torch.transpose(x.reshape([bt, -1] + s[1:]), 1, 0), [-1] + s[1:])


def setup_default_logging(args, time_name, default_level=logging.INFO,
                          format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s"):
    output_dir = os.path.join(args.output_path, 'log', time_name)
    os.makedirs(output_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(args.output_path, f'tensorboardX', time_name),
                           comment=f'{args.data}')

    logger = logging.getLogger('train')
    logging.basicConfig(  # unlike the root logger, a custom logger can’t be configured using basicConfig()
        filename=os.path.join(output_dir, args.data + '.log'),
        format=format,
        datefmt="%m/%d/%Y %H:%M:%S",
        level=default_level)

    # print
    # file_handler = logging.FileHandler()
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(default_level)
    console_handler.setFormatter(logging.Formatter(format))
    logger.addHandler(console_handler)

    return logger, writer


def accuracy(output, target, topk=(1,)):
    """Computes the precisio
    n@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, largest=True, sorted=True)  # return value, indices
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """
    Computes and stores the average and current value

    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        # self.avg = self.sum / (self.count + 1e-20)
        self.avg = self.sum / self.count


def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H:%M:%S'

    #     time.strftime(format[, t])
    return datetime.today().strftime(fmt)


class PMovingMeter(object):
    def __init__(self, n_class, buf_size=1024):
        self.ma = torch.ones(buf_size, n_class).cuda()

    def __call__(self):
        v = torch.mean(self.ma, dim=0)
        return v

    def update(self, entry):
        entry = torch.mean(entry, dim=0, keepdim=True)
        self.ma = torch.cat((self.ma[1:], entry), dim=0)


class PData(object):
    def __init__(self, n_class, p_unlabeled=None, p_labeled=None):
        self.name = 'p_data'
        if p_unlabeled is not None:
            self.p_data = p_unlabeled
        elif p_labeled is not None:
            self.p_data = p_labeled
        else:
            self.p_data = torch.ones(n_class).cuda()

    def __call__(self):
        return self.p_data / torch.sum(self.p_data)

    def update(self, entry, p_label, decay=0.999):
        entry = torch.mean(entry, dim=0)/p_label()
        self.p_data = self.p_data * decay + entry * (1 - decay)


def one_hot(labels, n_class=7):
    if labels.dim() == 1:
        one = torch.zeros((len(labels), n_class)).cuda()
        for i in range(len(labels)):
            label = labels[i]
            try:
                one[i][label] = 1
            except:
                print('aa')
    else:
        one = torch.zeros(())

    return one


def plot_confusion_matrix(y_true, y_pre, epoch, path, time_name, data):
    cm = confusion_matrix(np.array(y_pre), np.array(y_true)).astype(np.float32)
    for i in range(cm.shape[0]):
       cm[i] = 100 * cm[i] / cm[i].sum()
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap=plt.cm.Blues)
    indices = range(len(cm))
    if data == 'CK+':
        plt.xticks(indices, ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sadness', 'surprise'],
                   rotation=45)
        plt.yticks(indices, ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sadness', 'surprise'])
    elif data == 'RAF-DB' or data == 'SFEW':
        plt.xticks(indices, ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sadness', 'surprise'], rotation=45)
        plt.yticks(indices, ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sadness', 'surprise'])
    elif data == 'CIFAR10':
        plt.xticks(indices, ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], rotation=45)
        plt.yticks(indices, ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
    elif data == 'FERPlus' or data == 'Affectnet':
        plt.xticks(indices, ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sadness', 'surprise'], rotation=45)
        plt.yticks(indices, ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sadness', 'surprise'])
    plt.colorbar()

    plt.xlabel('predict')
    plt.ylabel('true')
    plt.title('confusion_matrix')

    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False

    for first_index in range(len(cm)):  # 第几行
        for second_index in range(len(cm[first_index])):  # 第几列
            plt.text(second_index - 0.35, first_index, round(cm[first_index][second_index], 2))
    if not os.path.exists(os.path.join(path, 'log', time_name, 'confusion')):
        os.makedirs(os.path.join(path, 'log', time_name, 'confusion'))
    plt.savefig(os.path.join(path, 'log', time_name, 'confusion', '{}.png'.format(epoch + 1)))
    # plt.show()
    plt.clf()
    plt.close()


def kl_divergence(x, y):

    # return torch.sum(x * torch.log((x + 1e-6) / (y + 1e-6)))
    return torch.nn.functional.kl_div(torch.log(y), x, reduction='batchmean')


class soft_CrossEntropy(torch.nn.Module):

    def __init__(self, reduction='mean'):
        super(soft_CrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        x = torch.nn.functional.softmax(x, dim=1)
        loss = y * torch.log(x + 1e-6)
        if self.reduction == 'mean':
            loss = -torch.mean(torch.sum(loss, dim=1), dim=0)
        else:
            loss = -torch.sum(loss, dim=1)
        return loss


class BalanceLoss(nn.Module):
    def __init__(self):
        super(BalanceLoss, self).__init__()
        self.weight = torch.tensor((0.8, 1.5, 1.2, 0.5, 1, 1.2, 0.8), dtype=torch.float, requires_grad=False)

    def to(self, device):
        super().to(device)
        if self.weight is not None:
            self.weight = self.weight.to(device)

    def forward(self, output_logits, target):
        return F.cross_entropy(output_logits, target, weight=self.weight.cuda())


def distribution_alignment(q, t, p_model):
    p_model_item = p_model()
    q = (q * (p_model_item)).cuda()
    q /= torch.sum(q, dim=1, keepdim=True)

    # q_t = q ** t
    # q_target = q_t / torch.sum(q_t, dim=1, keepdim=True)

    return q


def main():
    a = torch.randn((10, 4)).cuda()
    target = torch.tensor((1, 2, 2, 3, 3, 3, 0, 0, 0, 0)).cuda()
    p_model = PMovingMeter(name='model', n_class=4, buf_size=5)
    p_data = PData(n_class=4)
    for i in range(10):
        q = torch.softmax(a[1].unsqueeze(0), dim=1)
        p_data_item = p_data()
        p_model_item = p_model()
        ratio = ((1e-6 + p_data_item) / (1e-6 + p_model_item)).cuda()
        q_norm = q * ratio
        q_norm /= torch.sum(q_norm, dim=1, keepdim=True)

        q_t = q_norm ** 0.5
        q_target = q_t / torch.sum(q_t, dim=1, keepdim=True)
        p_model.update(q)
        p_data.update(one_hot(target[i].unsqueeze(0), n_class=4))


if __name__ == '__main__':
    main()
