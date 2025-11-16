"""Residual network (ResNet) models for 2D operator learning.

This module provides:
- `ResNet`: a HuggingFace-style wrapper that selects the dimensionality-specific
  implementation and handles optional conditioning inputs and reshaping.
- `ResNet2D`: a 2D convolutional residual encoder-decoder constructed from
  configurable residual blocks.

Implementations are adapted from `https://github.com/pdearena/pdearena` and
tailored to this project's configuration, activation, and coordinate-feature
conventions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from utils import activation_func
from .resnet_utils import getblock, make_layer
from utils.grid_utils import twod_meshgrid
from transformers import PreTrainedModel
from .resnet_utils import ResNetConfig

class ResNet(PreTrainedModel):
    """High-level ResNet wrapper for grid-based operator learning.

    - Resolve activation from config and construct `ResNet2D` when `dimension == 2`.
    - Optionally concatenate conditioning inputs along the channel axis.
    - Flatten input sequence and channels before passing to the backbone.

    Adapted from `https://github.com/pdearena/pdearena`.
    """
    
    main_input_name = "input_data"
    conditioning_input_name = "conditioning_input_data"
    config_class = ResNetConfig

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
               
        activation: nn.Module = activation_func.get_activation(config.activation_fn_name)
        if activation is None:
            raise NotImplementedError(f"Activation {config.activation_fn_name} not implemented")
        
        self.resnet = self.build_resnet()(
            config=config,
            activation_fn=activation
        )
           
    def build_resnet(self):
        """Return the dimension-specific ResNet backbone class."""
        if self.config.dimension == 2:
            return ResNet2D
        else:
            raise NotImplementedError(
                "Invalid dimensionality. Only 2D ResNet implemented"
            )
    
    def forward(self, 
                input_data: Tensor,
                **kwargs) -> Tensor: 
        """Run the ResNet model end-to-end.

        Parameters
        ----------
        input_data : Tensor
            Input tensor of shape
            [batch, input_seq, in_channels, grid_x, grid_y].
        **kwargs :
            Optional keyword arguments. Recognized:
            - `conditioning_input_data`: Tensor of shape
              [batch, input_seq, cond_channels, grid_x, grid_y] to be
              concatenated with `input_data` along the channel dimension.

        Returns
        -------
        Tensor
            Output tensor of shape [batch, out_channels, grid_x, grid_y],
            where `out_channels = config.out_size`.
        """

        if "conditioning_input_data" in kwargs:
            #NOTE: Conditioning data can be passed into a conv network before concatination with input_data.
            conditioning_input_data = kwargs["conditioning_input_data"]
            input_data = torch.cat([input_data, conditioning_input_data], dim=2)
        else:
            conditioning_input_data = None        

        batch, input_seq, input_channels, *spatial = input_data.shape
        x=input_data.reshape(batch, input_seq * input_channels, *spatial)
        
        return self.resnet(x, **kwargs)
                                        
class ResNet2D(PreTrainedModel):
    """2D ResNet backbone.

    Architecture:
    - Optional coordinate features concatenation (x, y) if enabled.
    - Two 1x1 conv layers for input lifting with activation.
    - A sequence of residual layers built from `resnet_utils.make_layer`.
    - Two 1x1 conv layers to project to `out_size`.
    - Optional padding before/after the main body to control output size.
    """
    def __init__(self, config, activation_fn: nn.Module = nn.GELU()) -> None:
        super().__init__(config)
        self.activation = activation_fn

        self.conv_in1 = nn.Conv2d(
            config.in_size,
            config.latent_channels,
            kernel_size=1,
            bias=True,
        )
        self.conv_in2 = nn.Conv2d(
            config.latent_channels,
            config.latent_channels,
            kernel_size=1,
            bias=True,
        )
        
        self.block = getblock(config.block)
        
        self.layers = nn.ModuleList(
            [
                make_layer(
                    self.block,
                    config,
                    config.latent_channels,
                    config.latent_channels,
                    config.num_blocks[i],
                    stride = config.stride,
                    dimension = 2,
                    activation_fn = self.activation,
                )
                for i in range(len(config.num_blocks))
            ]
        )
        
        self.conv_out1 = nn.Conv2d(
            config.latent_channels,
            config.latent_channels,
            kernel_size=1,
            bias=True,
        )
        self.conv_out2 = nn.Conv2d(
            config.latent_channels,
            config.out_size,
            kernel_size=1,
            bias=True,
        )           
           

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Encode and decode a 2D grid with residual blocks.

        Parameters
        ----------
        x : Tensor
            Input grid tensor of shape [batch, in_channels, grid_x, grid_y].
            If `config.coord_features` is True, two coordinate feature
            channels are concatenated internally before the lifting convs.

        Returns
        -------
        Tensor
            Output grid tensor of shape [batch, out_channels, grid_x, grid_y].
        """
        if x.dim() != 4:
            raise ValueError(
                "Only 4D tensors [batch, in_channels, grid_x, grid_y] accepted for 2D ResNet"
            )
        
        #add feature map
        if self.config.coord_features: 
            coord_feat = twod_meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)
        
        #encoder    
        x = self.activation(self.conv_in1(x)) 
        x = self.activation(self.conv_in2(x)) 
        if self.config.padding > 0:
            x = F.pad(x, [0, self.config.padding, 0, self.config.padding])
            
        #main part
        for layer in self.layers:
            x = layer(x, **kwargs)
            
        #decoder    
        if self.config.padding > 0:
            x = x[..., : -self.config.padding, : -self.config.padding]

        x = self.activation(self.conv_out1(x))
        x = self.conv_out2(x)
                        
        return x