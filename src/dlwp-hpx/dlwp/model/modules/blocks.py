from typing import Any, Dict, Optional, Sequence, Union

from hydra.utils import instantiate
from omegaconf import DictConfig
import torch as th
import einops

from dlwp.model.modules.healpix import HEALPixLayer
from dlwp.model.modules.utils import Interpolate


#
# FOLDING/UNFOLDING BLOCKS
#

class FoldFaces(th.nn.Module):
    # perform face folding:
    # [B, F, C, H, W] -> [B*F, C, H, W]

    def __init__(self):
        super().__init__()

    def forward(self, tensor: th.Tensor) -> th.Tensor:

        N, F, C, H, W = tensor.shape
        tensor = th.reshape(tensor, shape=(N*F, C, H, W))
    
        return tensor


class UnfoldFaces(th.nn.Module):
    # perform face unfolding:
    # [B*F, C, H, W] -> [B, F, C, H, W]

    def __init__(self, num_faces=12):
        super().__init__()
        self.num_faces = num_faces

    def forward(self, tensor: th.Tensor) -> th.Tensor:
        
        NF, C, H, W = tensor.shape
        tensor = th.reshape(tensor, shape=(-1, self.num_faces, C, H, W))
    
        return tensor


#
# RECURRENT BLOCKS
#

class ConvNeXtLSTMBlock(th.nn.Module):
    def __init__(
            self,
            geometry_layer: th.nn.Module = HEALPixLayer,
            in_channels: int = 1,
            h_channels: int = 1,
            kernel_size: int = 7,
            dilation: int = 1,
            activation: th.nn.Module = None,
            dropout: float = 0.,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False):
        '''
        :param x_channels: Input channels
        :param h_channels: Latent state channels
        :param kernel_size: Convolution kernel size
        :param activation_fn: Output activation function
        '''
        super().__init__()
        h_channels = in_channels
        conv_channels = in_channels + h_channels
        #spatial mixing
        self.to_latent = th.nn.Sequential(
            geometry_layer(
                layer="torch.nn.Conv2d",
                in_channels=conv_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding="same",
                groups=in_channels,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad
                ),
            geometry_layer(
                layer="torch.nn.GroupNorm",
                num_channels=in_channels,
                num_groups=1,
                affine=True,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad
                ),
            geometry_layer(
                layer="torch.nn.Conv2d",
                in_channels=in_channels,
                out_channels=4*in_channels,
                kernel_size=1,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad
                ),
            geometry_layer(
                layer="torch.nn.GroupNorm",
                num_channels=4*in_channels,
                num_groups=4, 
                affine=True,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad
                )
        )
        #output activation
        self.to_output = th.nn.Sequential(
            geometry_layer(
                layer="torch.nn.Conv2d",
                in_channels=h_channels,
                out_channels=h_channels,
                kernel_size=1,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad
                ),
            geometry_layer(
                layer="torch.nn.GroupNorm",
                num_channels=h_channels,
                num_groups=1, 
                affine=True,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad
                ),
            instantiate(config=activation)
        )
        #dropout
        self.dropout = th.nn.Dropout(dropout)

        # Latent states
        self.h = th.zeros(1)
        self.c = th.zeros(1)

    def forward(self, inputs: Sequence) -> Sequence:
        '''
        LSTM forward pass
        :param inputs: Input
        '''
        if inputs.shape != self.h.shape:
            self.h = th.zeros_like(inputs)
            self.c = th.zeros_like(inputs)

        #Spatial mixing
        z = th.cat((inputs, self.h), dim = 1) if inputs is not None else self.h
        z = self.to_latent(z)
        #LSTM activation
        f, i, g, o = einops.rearrange(z, 'b (gates c) h w -> gates b c h w', gates = 4) #forget gate, input gate, g, output gate
        cell = th.sigmoid(f) * self.c + th.sigmoid(i) * self.dropout(th.tanh(g))
        hidden = th.sigmoid(o) * self.to_output(self.c)
        #hidden = th.sigmoid(o) * self.activation(self.norm3(self.c))
        self.h = hidden
        self.c = cell
        return hidden

    def reset(self):
        self.h = th.zeros_like(self.h)
        self.c = th.zeros_like(self.c)


class ConvGRUBlock(th.nn.Module):
    """
    Code modified from
    https://github.com/happyjin/ConvGRU-pytorch/blob/master/convGRU.py
    """
    def __init__(
            self,
            geometry_layer: th.nn.Module = HEALPixLayer,
            in_channels: int = 3,
            kernel_size: int = 1,
            downscale_factor: int = 4,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False
            ):
        super().__init__()

        self.channels = in_channels
        self.conv_gates = geometry_layer(
            layer="torch.nn.Conv2d",
            in_channels=in_channels + self.channels,
            out_channels=2*self.channels,  # for update_gate,reset_gate respectively
            kernel_size=kernel_size,
            padding="same",
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            )
        self.conv_can = geometry_layer(
            layer="torch.nn.Conv2d",
            in_channels=in_channels+self.channels,
            out_channels=self.channels, # for candidate neural memory
            kernel_size=kernel_size,
            padding="same",
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            )
        self.h = th.zeros(1, 1, 1, 1)

    def forward(self, inputs: Sequence) -> Sequence:
        if inputs.shape != self.h.shape:
            self.h = th.zeros_like(inputs)
        combined = th.cat([inputs, self.h], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = th.split(combined_conv, self.channels, dim=1)
        reset_gate = th.sigmoid(gamma)
        update_gate = th.sigmoid(beta)

        combined = th.cat([inputs, reset_gate*self.h], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = th.tanh(cc_cnm)

        h_next = (1 - update_gate) * self.h + update_gate * cnm
        self.h = h_next

        return inputs + h_next

    def reset(self):
        self.h = th.zeros_like(self.h)


#
# CONV BLOCKS
#

class BasicConvBlock(th.nn.Module):
    """
    Convolution block consisting of n subsequent convolutions and activations
    """
    def __init__(
            self,
            geometry_layer: th.nn.Module = HEALPixLayer,
            in_channels: int = 3,
            out_channels: int = 1,
            kernel_size: int = 3,
            dilation: int = 1,
            n_layers: int = 1,
            latent_channels: int = None,
            activation: th.nn.Module = None,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False
            ):
        super().__init__()
        if latent_channels is None: latent_channels = max(in_channels, out_channels)
        convblock = []
        for n in range(n_layers):
            convblock.append(geometry_layer(
                layer='torch.nn.Conv2d',
                in_channels=in_channels if n == 0 else latent_channels,
                out_channels=out_channels if n == n_layers - 1 else latent_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad
                ))
            if activation is not None: convblock.append(activation)
        self.convblock = th.nn.Sequential(*convblock)

    def forward(self, x):
        return self.convblock(x)


class MobileNetConvBlock(th.nn.Module):
    """
    A convolution block as reported in Figure 4 (d) of https://arxiv.org/pdf/1801.04381.pdf

    Does not seem to improve performance over two simple convolutions
    """
    def __init__(
            self,
            geometry_layer: th.nn.Module = HEALPixLayer,
            in_channels: int = 3,
            out_channels: int = 1,
            kernel_size: int = 3,
            dilation: int = 1,
            activation: th.nn.Module = None,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False
            ):
        super().__init__()
        # Instantiate 1x1 conv to increase/decrease channel depth if necessary
        if in_channels == out_channels:
            self.skip_module = lambda x: x  # Identity-function required in forward pass
        else:
            self.skip_module = geometry_layer(
                layer='torch.nn.Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad
                )
        # Convolution block
        convblock = []
        # Map channels to output depth
        convblock.append(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        if activation is not None: convblock.append(activation)
        # Depthwise convolution
        convblock.append(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=out_channels,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        if activation is not None: convblock.append(activation)
        # Linear postprocessing
        convblock.append(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        self.convblock = th.nn.Sequential(*convblock)

    def forward(self, x):
        return self.skip_module(x) + self.convblock(x)


class ConvNeXtBlock(th.nn.Module):
    """
    A convolution block as reported in Figure 4 of https://arxiv.org/pdf/2201.03545.pdf
    """
    def __init__(
            self,
            geometry_layer: th.nn.Module = HEALPixLayer,
            in_channels: int = 3,
            latent_channels: int = 1,
            out_channels: int = 1,
            kernel_size: int = 3,
            dilation: int = 1,
            upscale_factor: int = 4,
            n_layers: int = 1,
            activation: th.nn.Module = None,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False
            ):
        super().__init__()

        # Instantiate 1x1 conv to increase/decrease channel depth if necessary
        if in_channels == out_channels:
            self.skip_module = lambda x: x  # Identity-function required in forward pass
        else:
            self.skip_module = geometry_layer(
                layer='torch.nn.Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad
                )
        # Convolution block
        convblock = []
        # 7x7 convolution increasing channels
        convblock.append(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=in_channels,
            out_channels=int(latent_channels*upscale_factor),
            kernel_size=kernel_size,
            dilation=dilation,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        # LayerNorm
        #convblock.append(th.nn.LayerNorm([out_channels*upscale_factor, HW, HW]))
        if activation is not None: convblock.append(activation)
        # 1x1 convolution decreasing channels
        convblock.append(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=int(latent_channels*upscale_factor),
            out_channels=int(latent_channels*upscale_factor),
            kernel_size=kernel_size,
            dilation=dilation,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        if activation is not None: convblock.append(activation)
        # Linear postprocessing
        convblock.append(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=int(latent_channels*upscale_factor),
            out_channels=out_channels,
            kernel_size=1,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        self.convblock = th.nn.Sequential(*convblock)

    def forward(self, x):
        return self.skip_module(x) + self.convblock(x)


#
# DOWNSAMPLING BLOCKS
#

class MaxPool(th.nn.Module):
    def __init__(
            self,
            geometry_layer: th.nn.Module = HEALPixLayer,
            pooling: int = 2,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False
            ):
        super().__init__()
        self.maxpool = geometry_layer(
            layer="torch.nn.MaxPool2d",
            kernel_size=pooling,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            )
    def forward(self, x):
        return self.maxpool(x)


class AvgPool(th.nn.Module):
    def __init__(
            self,
            geometry_layer: th.nn.Module = HEALPixLayer,
            pooling: int = 2,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False
            ):
        super().__init__()
        self.avgpool = geometry_layer(
            layer="torch.nn.AvgPool2d",
            kernel_size=pooling,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            )
    def forward(self, x):
        return self.avgpool(x)


class LearnedPool(th.nn.Module):
    def __init__(
            self,
            geometry_layer: th.nn.Module = HEALPixLayer,
            in_channels: int = 1,
            out_channels: int = 1,
            pooling: int = 2,
            activation: th.nn.Module = None,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False
            ):
        super().__init__()
        # "Skip" connection
        self.skip_pool = MaxPool(
            geometry_layer=geometry_layer,
            pooling=pooling,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            )
        # Donwpooling convolution
        downpooler = []
        downpooler.append(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=pooling,
            stride=pooling,
            padding=0,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        if activation is not None:
            downpooler.append(activation)
        self.downpooler = th.nn.Sequential(*downpooler)

    def forward(self, x):
        return self.skip_pool(x) + self.downpooler(x)


#
# UPSAMPLING BLOCKS
#

class InterpolationUpsample(th.nn.Module):
    def __init__(
            self,
            geometry_layer: th.nn.Module = HEALPixLayer,
            in_channels: int = 3,
            out_channels: int = 1,
            kernel_size: int = 3,
            mode: str = "nearest",
            upsampling: int = 2,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False
            ):
        super().__init__()
        self.upsampler = geometry_layer(
            layer=Interpolate,
            scale_factor=upsampling,
            mode=mode,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            )
    def forward(self, x):
        return self.upsampler(x)


class TransposedConvUpsample(th.nn.Module):
    def __init__(
            self,
            geometry_layer: th.nn.Module = HEALPixLayer,
            in_channels: int = 3,
            out_channels: int = 1,
            upsampling: int = 2,
            activation: th.nn.Module = None,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False
            ):
        super().__init__()
        upsampler = []
        # Upsample transpose conv
        upsampler.append(geometry_layer(
            layer='torch.nn.ConvTranspose2d',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=upsampling,
            stride=upsampling,
            padding=0,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        if activation is not None:
            upsampler.append(activation)
        self.upsampler = th.nn.Sequential(*upsampler)

    def forward(self, x):
        return self.upsampler(x)
