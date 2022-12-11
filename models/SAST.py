from torch import nn
from torchvision import models
from torch.autograd import Function
import numpy as np
import torch
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
from utils import one_hot


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        o3 = self.layer3(x)
        o4 = self.layer4(o3)

        x = self.avgpool(o4)
        x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x, o3, o4

    def forward(self, x: Tensor):
        return self._forward_impl(x)


class SAST(nn.Module):
    def __init__(self, args):
        super(SAST, self).__init__()
        self.drop_rate = args.dropout
        self.backbone = ResNet(BasicBlock, [2, 2, 2, 2], args.n_class)
        out_dim = list(self.backbone.children())[-1].in_features  # original fc layer's in dimention 512
        dim = out_dim + out_dim//2
        if args.model_path:
            self.pretrain = args.model_path
            self.load_pretrain()
        self.backbone.fc = nn.Linear(out_dim, args.n_class)  # new fc layer 512x7
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.land_fc = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Dropout(self.drop_rate),
            nn.Linear(dim // 2, dim // 2),
            nn.Dropout(self.drop_rate),
            nn.Linear(dim // 2, out_dim),
        )
        self.bn = nn.BatchNorm2d(dim, momentum=0.01)
        self.relu = nn.Hardswish()
        self.fc1 = nn.Linear(out_dim, args.center_feature)
        self.prelu_fc1 = nn.PReLU()
        self.fc2 = nn.Linear(args.center_feature, args.n_class)
        self.lambda_pb = args.lambda_pb
        self.lambda_center = args.lambda_center

    def load_pretrain(self):
        state_dict = torch.load(self.pretrain)['state_dict']
        self.backbone.load_state_dict(state_dict)


    def _upsample_cat(self, x, y):
        _, _, H, W = y.size()
        return torch.cat([F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False), y], dim=1)

    def ambiguity(self, re, le, rm, lm):
        re, le, rm, lm = self.land(re.flatten(1)), self.land(le.flatten(1)), self.land(rm.flatten(1)), self.land(lm.flatten(1))
        with torch.no_grad():
            re, le, rm, lm = torch.softmax(re, dim=1), torch.softmax(le, dim=1), torch.softmax(rm, dim=1), torch.softmax(lm, dim=1)
            _, re = torch.max(re, dim=1)
            _, le = torch.max(le, dim=1)
            _, rm = torch.max(rm, dim=1)
            _, lm = torch.max(lm, dim=1)
            mask_l = ((re == le) & (le == rm)) | ((le == rm) & (rm == lm)) | ((rm == lm) & (lm == re))

        return mask_l

    def forward(self, x, landmark):
        landmark = torch.as_tensor(torch.floor(landmark / 32), dtype=torch.long)
        landmark = landmark[:, :, 1]*7 + landmark[:, :, 0]
        x, o3, o4 = self.backbone._forward_impl(x)
        feature = self._upsample_cat(o4, o3)
        feature = self.avgpool(feature) + self.maxpool(feature)
        feature = self.relu(self.bn(feature))
        feature = feature.view(feature.size(0), feature.size(1), -1)
        re, le, rm, lm = one_hot(landmark[:, 0], feature.size(-1)), one_hot(landmark[:, 1], feature.size(-1)), one_hot(landmark[:, 2], feature.size(-1)), one_hot(landmark[:, 3], feature.size(-1))
        re, le, rm, lm = feature@re.unsqueeze(-1), feature@le.unsqueeze(-1), feature@rm.unsqueeze(-1), feature@lm.unsqueeze(-1)
        feature = torch.cat([re, le, rm, lm], dim=-1)
        feature = torch.mean(feature, dim=-1)
        feature = self.land_fc(feature)
        distance = torch.norm((x-feature), p=2) / x.size(1)
        distance += abs(torch.cosine_similarity(x, feature)).mean()
        center_feature = self.prelu_fc1(self.fc1(x*(1-self.lambda_pb) + feature*self.lambda_pb))
        x = (1-self.lambda_center)*self.backbone.forward(x*(1-self.lambda_pb) + feature*self.lambda_pb) + self.lambda_center*self.fc2(center_feature)

        return x, distance, center_feature