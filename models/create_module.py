from torchvision import models
from models.SAST import SAST
from models.ema import EMA
import torch
import timm
import torch.nn as nn


def create_model(args):
    if args.model.lower() == 'resnet18':
        model = models.resnet18(False)
        model.load_state_dict(torch.load(args.model_path)['state_dict'])
        model.fc = torch.nn.Linear(512, args.n_class)
    elif args.model.lower() == 'efficient':
        model = timm.create_model('tf_efficientnet_lite0', True)
        model.classifier = nn.Linear(1280, args.n_class)
    else:
        model = SAST(args)
    if args.model_path1:
        pretrained_dict = torch.load(args.model_path1)
        # new_dict = {}
        # model_dict = model.state_dict()
        #
        # for k, _ in model_dict.items():
        #     if 'fc' not in k:
        #         if k in pretrained_dict:
        #             new_dict[k] = pretrained_dict[k]
        #     # if k in pretrained_dict:
        #     #     new_dict[k] = pretrained_dict[k]
        # model_dict.update(new_dict)
        # model.load_state_dict(model_dict)
        model.load_state_dict(pretrained_dict)
    model.to(args.device)
    ema = EMA(model, args.ema_alpha)

    return model, ema