import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple


def _parse_config_blocks(config_path: str) -> List[Dict]:
    blocks = []
    with open(config_path, "r") as f:
        lines = f.read().splitlines()

   # ignore comments, empty lines
    lines = [line.strip() for line in lines if line and not line.startswith("#")]

    block = {}
    for line in lines:
        # Start of new block
        if line.startswith("["):
            # Save previous block
            if block:
                blocks.append(block)
            block = {"type": line.strip("[]")}
        # Properties of current block
        else:
            k, v = line.split("=")
            block[k.strip()] = v.strip()

    # Save last cached block, if there is any
    if block:
        blocks.append(block)

    return blocks


def _make_convolutional_block(in_channels: int, config: Dict):
    out_channels = int(config["filters"])
    kernel_size = int(config["size"])
    stride = int(config["stride"])

    # pad is a boolean flag, not the actual padding value
    padding_flag = int(config.get("pad", 0))
    padding = kernel_size // 2 if padding_flag else 0
    has_batch_norm = int(config.get("batch_normalize", 0))
    activation = config.get("activation", "linear")

    layers = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=not has_batch_norm,
        )
    ]

    if has_batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    if activation == "leaky":
        layers.append(nn.LeakyReLU(0.1, inplace=True))

    return nn.Sequential(*layers), out_channels


def _build_layers(blocks: List[Dict], in_channel_count: int) -> Tuple[nn.ModuleList, List[int], Dict[int, int]]:
    layers = nn.ModuleList()
    channels_before_output = []
    shortcut_indices: Dict[int, int] = {}
    prev_channel = in_channel_count

    for i, block in enumerate(blocks):
        match block["type"]:
            case "convolutional":
                layer, prev_channel = _make_convolutional_block(prev_channel, block)
                layers.append(layer)
            case "shortcut":
                shortcut_indices[i] = i + int(block["from"])
                layers.append(nn.Identity())
            case "output":
                layers.append(nn.Identity())
                channels_before_output.append(prev_channel)
            case block_type:
                raise ValueError(f"Unsupported block type: {block_type}")

    return layers, channels_before_output, shortcut_indices


class Darknet(nn.Module):
    def __init__(self, config_path: str, in_channel_count: int = 3):
        super().__init__()

        self.blocks = _parse_config_blocks(config_path)
        self.layers, self.channels_before_output, self.shortcut_indices = _build_layers(self.blocks, in_channel_count)

    def forward(self, x):
        outputs = []
        saved_outputs = []

        for i, (block, layer) in enumerate(zip(self.blocks, self.layers)):
            match block["type"]:
                case "convolutional":
                    x = layer(x)
                case "shortcut":
                    x = outputs[i - 1] + outputs[self.shortcut_indices[i]]
                case "output":
                    saved_outputs.append(x)

            outputs.append(x)

        return saved_outputs

    def load_weights(self, weights_path: str):
        with open(weights_path, "rb") as f:
            # header
            np.fromfile(f, dtype=np.int32, count=5)

            weights = np.fromfile(f, dtype=np.float32)

        ptr = 0

        for i, (block, module) in enumerate(zip(self.blocks, self.layers)):
            if block["type"] != "convolutional":
                continue

            conv = module[0]

            has_batch_norm = block.get("batch_normalize", "0") == "1"
            if has_batch_norm:
                batch_norm = module[1]
                num = batch_norm.bias.numel()
                batch_norm.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + num]))
                ptr += num
                batch_norm.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + num]))
                ptr += num
                batch_norm.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr + num]))
                ptr += num
                batch_norm.running_var.data.copy_(torch.from_numpy(weights[ptr:ptr + num]))
                ptr += num
            else:
                num = conv.bias.numel()
                conv.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + num]))
                ptr += num

            num = conv.weight.numel()
            conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + num]).view_as(conv.weight))
            ptr += num

        if ptr != len(weights):
            print(f"Warning: {len(weights) - ptr} unused weights")
        print("Darknet weights loaded successfully")
