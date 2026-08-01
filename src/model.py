import torch
from torch import nn
import torchinfo
from torchinfo import summary

class VGGMini(nn.Module):
  def __init__(self, input_channels:int, hidden_channels:int, output_shape: int):
    """Model architecture copying TinyVGG from:
    https://poloclub.github.io/cnn-explainer/"""
    super().__init__()
    self.block_1 = nn.Sequential(
        nn.Conv2d(in_channels=input_channels,out_channels=hidden_channels,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_channels,out_channels=hidden_channels,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2)
        # 64 x 64 -> 32 x 32
    )
    self.block_2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_channels,out_channels=hidden_channels,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_channels,out_channels=hidden_channels,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2)
        # 32 x 32 -> 16 x 16
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        # colour channels x flattened image (16*16)
        nn.Linear(in_features=hidden_channels*16*16,out_features=output_shape)
    )

  def forward(self,x) -> torch.Tensor:
    # operator fusion
    return self.classifier(self.block_2(self.block_1(x)))