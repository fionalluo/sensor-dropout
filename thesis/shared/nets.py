import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoderResnet(nn.Module):

  def __init__(self, depth, blocks, resize, minres, output_dim=512, in_channels=None, **kw):
    super().__init__()  # This is crucial for PyTorch modules
    self._depth = depth
    self._blocks = blocks
    self._resize = resize
    self._minres = minres
    self._kw = kw
    self.output_dim = output_dim
    self._in_channels = in_channels  # Store for later use

    # Create all layers upfront
    # We'll initialize the conv layer in forward when we know the input shape
    self.initial_conv = None

    # Create all other layers
    self.conv_layers = nn.ModuleDict()
    stages = 4  # Maximum number of stages we'll need
    for i in range(stages):
      # Resize layers
      if resize == 'stride':
        self.conv_layers[f's{i}res'] = nn.Conv2d(depth * (2**i), depth * (2**(i+1)), 4, stride=2, padding=1)
      elif resize == 'stride3':
        s = 2 if i else 3
        k = 5 if i else 4
        self.conv_layers[f's{i}res'] = nn.Conv2d(depth * (2**i), depth * (2**(i+1)), k, stride=s, padding=k//2)
      elif resize in ['mean', 'max', 'bilinear']:
        self.conv_layers[f's{i}res'] = nn.Conv2d(depth * (2**i), depth * (2**(i+1)), 3, stride=1, padding=1)
      
      # Residual blocks
      for j in range(blocks):
        self.conv_layers[f's{i}b{j}conv1'] = nn.Conv2d(depth * (2**(i+1)), depth * (2**(i+1)), 3, padding=1)
        self.conv_layers[f's{i}b{j}conv2'] = nn.Conv2d(depth * (2**(i+1)), depth * (2**(i+1)), 3, padding=1)

    # Initialize all weights
    for layer in self.conv_layers.values():
      nn.init.orthogonal_(layer.weight)
      if layer.bias is not None:
        nn.init.zeros_(layer.bias)

    # Add adaptive pooling and projection
    self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))  # Changed to 2x2 to reduce dimensions
    self.projection = nn.Sequential(
      nn.Linear(depth * (2**stages) * 4, output_dim),  # 4 = 2x2 from adaptive pool
      nn.ReLU()
    )

  def forward(self, x):
    # Initialize initial conv layer if not done yet
    if self.initial_conv is None:
      in_channels = x.shape[1]  # Get channels from input
      self.initial_conv = nn.Conv2d(in_channels, self._depth, 3, stride=1, padding=1).to(x.device)
      nn.init.orthogonal_(self.initial_conv.weight)
      nn.init.zeros_(self.initial_conv.bias)

    stages = int(np.log2(x.shape[-2]) - np.log2(self._minres))
    depth = self._depth
    x = x.float() - 0.5  # Normalize input using PyTorch operations

    # Initial convolution to handle RGB input
    x = self.initial_conv(x)

    for i in range(stages):
      if self._resize == 'stride':
        x = self.conv_layers[f's{i}res'](x)
      elif self._resize == 'stride3':
        x = self.conv_layers[f's{i}res'](x)
      elif self._resize == 'mean':
        x = self.conv_layers[f's{i}res'](x)
        N, C, H, W = x.shape
        x = x.reshape(N, C, H // 2, 2, W // 2, 2).mean(dim=(3, 5))
      elif self._resize == 'max':
        x = self.conv_layers[f's{i}res'](x)
        N, C, H, W = x.shape
        x = x.reshape(N, C, H // 2, 2, W // 2, 2)
        x = x.max(dim=3)[0].max(dim=4)[0]
      elif self._resize == 'bilinear':
        x = self.conv_layers[f's{i}res'](x)
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
      else:
        raise NotImplementedError(self._resize)

      for j in range(self._blocks):
        skip = x
        x = self.conv_layers[f's{i}b{j}conv1'](x)
        x = self.conv_layers[f's{i}b{j}conv2'](x)
        x += skip
        if self._kw.get('preact', False):
          x = F.relu(x)

    if self._blocks and 'act' in self._kw:
      if self._kw['act'] == 'relu':
        x = F.relu(x)
      elif self._kw['act'] == 'tanh':
        x = torch.tanh(x)

    # Adaptive pooling and projection
    x = self.adaptive_pool(x)
    x = x.reshape(x.shape[0], -1)
    x = self.projection(x)
    return x
