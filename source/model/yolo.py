import torch
import torch.nn as nn
from source.model.darknet import Darknet

class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

class YOLOv3(nn.Module):
    def __init__(self, num_classes, backbone_config_path):
        super().__init__()
        self.backbone = Darknet(backbone_config_path)
        self._is_backbone_frozen = False
        self.num_classes = num_classes
        property_count = (5  # offset x, offset y, width, height, objectness
                          + num_classes
                          )

        self.detect1 = nn.Conv2d(self.backbone.channels_before_output[2], 3 * property_count, 1)
        self.detect2 = nn.Conv2d(self.backbone.channels_before_output[1], 3 * property_count, 1)
        self.detect3 = nn.Conv2d(self.backbone.channels_before_output[0], 3 * property_count, 1)

    def forward(self, x):
        feat1, feat2, feat3 = self.backbone(x)
        y1 = self.detect1(feat3)
        y2 = self.detect2(feat2)
        y3 = self.detect3(feat1)

        def reshape_for_loss(pred):
            B, _, H, W = pred.shape
            pred = pred.view(B, 3, 5 + self.num_classes, H, W)
            pred = pred.permute(0, 1, 3, 4, 2)
            return pred

        return [reshape_for_loss(y) for y in [y3, y2, y1]]

    def is_backbone_frozen(self):
        return self._is_backbone_frozen

    def freeze_backbone(self):
        self._is_backbone_frozen = True
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        self._is_backbone_frozen = False
        for param in self.backbone.parameters():
            param.requires_grad = True

