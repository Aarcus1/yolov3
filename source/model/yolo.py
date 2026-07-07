import torch
import torch.nn as nn
from typing import List
from source.model.darknet import Darknet

NUM_ANCHORS_PER_SCALE = 3

class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.layers(x)

class DetectionHead(nn.Module):
    def __init__(self, in_channels: int, num_outputs: int):
        super().__init__()
        self.conv = ConvolutionalBlock(in_channels, in_channels * 2, kernel_size=3)
        self.predict = nn.Conv2d(in_channels * 2, num_outputs, kernel_size=1)

    def forward(self, x):
        return self.predict(self.conv(x))


def _make_convolutional_stack(in_channels: int, out_channels: int) -> nn.Sequential:
    wide = out_channels * 2
    return nn.Sequential(
        ConvolutionalBlock(in_channels, out_channels, kernel_size=1),
        ConvolutionalBlock(out_channels, wide, kernel_size=3),
        ConvolutionalBlock(wide, out_channels, kernel_size=1),
        ConvolutionalBlock(out_channels, wide, kernel_size=3),
        ConvolutionalBlock(wide, out_channels, kernel_size=1),
    )


class YOLOv3(nn.Module):
    def __init__(self, num_classes: int, backbone_config_path: str):
        super().__init__()
        self.backbone = Darknet(backbone_config_path)
        self._is_backbone_frozen = False
        self.num_classes = num_classes

        num_outputs = NUM_ANCHORS_PER_SCALE * (5 + num_classes)

        ch_small, ch_medium, ch_large = self.backbone.channels_before_output  # 256, 512, 1024

        # Large objects
        self.conv_stack_large = _make_convolutional_stack(ch_large, ch_large // 2)  # 1024 -> 512
        self.head_large = DetectionHead(ch_large // 2, num_outputs)

        # Medium objects
        self.upsample_large = nn.Sequential(
            ConvolutionalBlock(ch_large // 2, ch_large // 4, kernel_size=1),  # 512 -> 256
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        concat_medium = ch_large // 4 + ch_medium  # 256 + 512 = 768
        self.conv_stack_medium = _make_convolutional_stack(concat_medium, ch_medium // 2)  # 768 -> 256
        self.head_medium = DetectionHead(ch_medium // 2, num_outputs)

        # Small objects
        self.upsample_medium = nn.Sequential(
            ConvolutionalBlock(ch_medium // 2, ch_medium // 4, kernel_size=1),  # 256 -> 128
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        concat_small = ch_medium // 4 + ch_small  # 128 + 256 = 384
        self.conv_stack_small = _make_convolutional_stack(concat_small, ch_small // 2)  # 384 -> 128
        self.head_small = DetectionHead(ch_small // 2, num_outputs)

    def forward(self, x) -> List[torch.Tensor]:
        feat_s8, feat_s16, feat_s32 = self.backbone(x)  # 256, 512, 1024

        # Large objects
        x_large = self.conv_stack_large(feat_s32)  # 1024 -> 512
        out_large = self.head_large(x_large)

        # Medium objects
        up_large = self.upsample_large(x_large)  # 512 -> 256
        x_medium = torch.cat([up_large, feat_s16], dim=1)  # 256 + 512 = 768
        x_medium = self.conv_stack_medium(x_medium)  # 768 -> 256
        out_medium = self.head_medium(x_medium)

        # Small objects
        up_medium = self.upsample_medium(x_medium)  # 256 -> 128
        x_small = torch.cat([up_medium, feat_s8], dim=1)  # 128 + 256 = 384
        x_small = self.conv_stack_small(x_small)  # 384 -> 128
        out_small = self.head_small(x_small)

        return [self._reshape(out_small), self._reshape(out_medium), self._reshape(out_large)]

    def _reshape(self, pred: torch.Tensor) -> torch.Tensor:
        B, _, H, W = pred.shape
        pred = pred.view(B, NUM_ANCHORS_PER_SCALE, 5 + self.num_classes, H, W)
        return pred.permute(0, 1, 3, 4, 2)

    def is_backbone_frozen(self) -> bool:
        return self._is_backbone_frozen

    def freeze_backbone(self):
        self._is_backbone_frozen = True
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        self._is_backbone_frozen = False
        for param in self.backbone.parameters():
            param.requires_grad = True
