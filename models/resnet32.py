"""
ProCo model zoo for CIFAR experiments.

This keeps the original two-branch ProCo head but adds a backbone factory so
the training script can switch between ResNet-32, ResNet-18 and MobileNetV2,
and can swap the activation from the command line.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_activation(name: str) -> nn.Module:
    """Create a fresh activation module from a CLI-friendly name."""
    key = name.lower()
    if key == 'relu':
        return nn.ReLU(inplace=True)
    if key == 'swish':
        return nn.SiLU(inplace=True)
    if key == 'gelu':
        return nn.GELU()
    raise ValueError(f'Unsupported activation: {name}')


def _init_module_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


class BasicBlock(nn.Module):
    """Standard residual block with configurable activation."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, activation: str = 'relu'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = make_activation(activation)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = make_activation(activation)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.act2(out)


class ResNetBackbone(nn.Module):
    """Small ResNet backbone for CIFAR-style inputs."""

    def __init__(self, blocks_per_stage, stage_channels, activation: str = 'relu'):
        super().__init__()
        self.in_planes = stage_channels[0]
        self.conv1 = nn.Conv2d(3, stage_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stage_channels[0])
        self.stem_act = make_activation(activation)

        layers = []
        for index, (num_blocks, planes) in enumerate(zip(blocks_per_stage, stage_channels)):
            stride = 1 if index == 0 else 2
            layers.append(self._make_layer(planes, num_blocks, stride, activation))
        self.layers = nn.Sequential(*layers)

        self.feat_dim = stage_channels[-1]
        self.apply(_init_module_weights)

    def _make_layer(self, planes: int, num_blocks: int, stride: int, activation: str) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for block_stride in strides:
            blocks.append(BasicBlock(self.in_planes, planes, block_stride, activation=activation))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem_act(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1)
        return out.view(out.size(0), -1)


class ResNet32(ResNetBackbone):
    def __init__(self, activation: str = 'relu'):
        super().__init__([5, 5, 5], [16, 32, 64], activation=activation)


class ResNet18Backbone(ResNetBackbone):
    def __init__(self, activation: str = 'relu'):
        super().__init__([2, 2, 2, 2], [64, 128, 256, 512], activation=activation)


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, groups: int = 1,
                 activation: str = 'relu'):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = make_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class InvertedResidual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, expand_ratio: int,
                 activation: str = 'relu'):
        super().__init__()
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_channels, hidden_dim, kernel_size=1, stride=1, activation=activation))
        layers.append(ConvBNAct(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim,
                                activation=activation))
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


class MobileNetV2Backbone(nn.Module):
    """MobileNetV2 backbone adapted for 32x32 CIFAR inputs."""

    def __init__(self, activation: str = 'relu'):
        super().__init__()
        self.feat_dim = 1280

        input_channel = 32
        self.stem = ConvBNAct(3, input_channel, kernel_size=3, stride=1, activation=activation)

        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        layers = []
        for expand_ratio, out_channels, repeats, stride in inverted_residual_setting:
            layers.append(InvertedResidual(input_channel, out_channels, stride, expand_ratio, activation=activation))
            input_channel = out_channels
            for _ in range(1, repeats):
                layers.append(InvertedResidual(input_channel, out_channels, 1, expand_ratio, activation=activation))
        self.features = nn.Sequential(*layers)

        self.last_conv = nn.Sequential(
            nn.Conv2d(input_channel, self.feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.feat_dim),
            make_activation(activation),
        )

        self.apply(_init_module_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.features(out)
        out = self.last_conv(out)
        out = F.adaptive_avg_pool2d(out, 1)
        return out.view(out.size(0), -1)


def build_backbone(name: str, activation: str) -> nn.Module:
    key = name.lower()
    if key == 'resnet32':
        return ResNet32(activation=activation)
    if key == 'resnet18':
        return ResNet18Backbone(activation=activation)
    if key == 'mobilenetv2':
        return MobileNetV2Backbone(activation=activation)
    raise ValueError(f'Unsupported backbone: {name}')


class ProjectionHead(nn.Module):
    """Two-layer MLP projection head with L2-normalised output."""

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128, activation: str = 'relu'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            make_activation(activation),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )
        self.apply(_init_module_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1, eps=1e-8)


class ProCoModel(nn.Module):
    """Shared backbone + classifier branch + projection branch."""

    def __init__(
        self,
        num_classes: int = 100,
        proj_hidden: int = 512,
        proj_out: int = 128,
        backbone: str = 'resnet32',
        activation: str = 'relu',
    ):
        super().__init__()

        self.backbone_name = backbone.lower()
        self.activation_name = activation.lower()
        self.backbone = build_backbone(self.backbone_name, self.activation_name)
        feat_dim = self.backbone.feat_dim

        self.classifier = nn.Linear(feat_dim, num_classes, bias=True)
        self.proj_head = ProjectionHead(
            in_dim=feat_dim,
            hidden_dim=proj_hidden,
            out_dim=proj_out,
            activation=self.activation_name,
        )

        self.feat_dim = feat_dim
        self.proj_dim = proj_out
        self.apply(_init_module_weights)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        logits = self.classifier(h)
        z_proj = self.proj_head(h)
        return logits, z_proj

    def get_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
