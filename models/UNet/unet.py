"""U-Net models for grid-based operator learning in 2D.

This module provides:
- `UNet`: a HuggingFace-style wrapper that selects the dimensionality-specific
  implementation and handles optional conditioning inputs and reshaping.
- `UNet2D`: a 2D U-Net with configurable depth, block counts, channel
  multipliers, and optional coordinate features.

Adapted from `https://github.com/pdearena/pdearena` and integrated with this
project's configuration conventions and normalization utilities.
"""
import torch
from torch import Tensor
import torch.nn as nn
from utils import activation_func
import torch.nn.functional as F
from transformers import PreTrainedModel
from .unet_utils import DownBlockND, UpBlockND, MiddleBlockND, DownsampleND, UpsampleND, UNetConfig
from utils.grid_utils import twod_meshgrid
from utils.model_utils import CustomNorm

class UNet(PreTrainedModel): 
    """High-level U-Net wrapper for grid-based operator learning.

    Responsibilities:
    - Resolve activation from config and construct `UNet2D` when `dimension == 2`.
    - Optionally concatenate conditioning inputs along the channel axis.
    - Flatten input sequence and channels before passing to the backbone.

    Shapes
    - Input to `forward`: [batch, input_seq, in_channels, grid_x, grid_y]
    - Output from `forward`: [batch, out_channels, grid_x, grid_y]

    Adapted from `https://github.com/pdearena/pdearena`.
    """
    #TODO: Do this for all models
    main_input_name = "input_data"
    conditioning_input_name = "conditioning_input_data"
    config_class = UNetConfig
    
    def __init__(self, config):
        
        super().__init__(config)

        self.config = config

        activation = activation_func.get_activation(config.activation_fn_name)
        if activation is None:
            raise NotImplementedError(f"Activation {config.activation_fn_name} not implemented")
        
        self.unet = self.build_UNet()(config=config, activation=activation)
       
    def build_UNet(self):
        """Return the dimension-specific UNet backbone class."""
        if self.config.dimension == 2:
            return UNet2D
        else:
            raise NotImplementedError(f"UNet not implemented for dimension {self.config.dimension}")

    
    ### Main Forward function ###
    def forward(self, input_data: Tensor, **kwargs) -> Tensor:
        """Run the U-Net model end-to-end.

        Parameters
        ----------
        input_data : Tensor
            Input tensor of shape
            [batch, input_seq, in_channels, grid_x, grid_y].
        **kwargs :
            Optional keyword arguments. Recognized:
            - `conditioning_input_data`: Tensor of shape
              [batch, input_seq, cond_channels, grid_x, grid_y] concatenated
              along the channel axis before processing.

        Returns
        -------
        Tensor
            Output tensor of shape [batch, out_channels, grid_x, grid_y].
        """
        
        if "conditioning_input_data" in kwargs:
            #NOTE: Conditioning data can be passed into a conv network before concatination with input_data.
            conditioning_input_data = kwargs["conditioning_input_data"]
            input_data = torch.cat([input_data, conditioning_input_data], dim=2)
        else:
            conditioning_input_data = None

        batch, input_seq, input_channels, *spatial = input_data.shape
        x = input_data.reshape(batch, input_seq * input_channels, *spatial)
        return self.unet(x, **kwargs)

#Unet based on the dimension
class UNet2D(PreTrainedModel):
    """2D U-Net backbone.

    Architecture:
    - Initial projection (`image_proj`) to latent channels with either 1x1 or 3x3 conv.
    - Down path: repeated `DownBlockND` blocks with optional `DownsampleND` between resolutions.
    - Middle bottleneck: `MiddleBlockND`.
    - Up path: repeated `UpBlockND` blocks with skip concatenations and `UpsampleND` between resolutions.
    - Final norm, activation, and output projection to `out_size` channels.

    Notes:
    - If needed, inputs are padded so H and W are divisible by the overall downsampling factor.
      Outputs are cropped back to original size.
    - Optional coordinate features (x, y) can be concatenated when `config.coord_features` is True.
    """
    def __init__(self, config, activation: nn.Module):
        super().__init__(config)

        self.activation = activation
        # Number of resolutions (depth of unet)
        unet_depth = len(config.channel_multiplier)
        
        # Project image into feature map
        if config.use1x1: #false by default
            self.image_proj = nn.Conv2d(config.in_size, config.latent_channels, kernel_size=1)
        else:
            self.image_proj = nn.Conv2d(config.in_size, config.latent_channels, kernel_size=(3, 3), padding=(1, 1))

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels_down = in_channels_down = config.latent_channels
        # For each resolution
        for i in range(unet_depth):
            # Number of output channels at this resolution
            out_channels_down = in_channels_down * config.channel_multiplier[i]
            # Add `n_blocks`
            for _ in range(config.n_blocks):
                down.append(
                    DownBlockND(
                        config=config,
                        in_channels=in_channels_down,
                        out_channels=out_channels_down,
                        dim=2,
                        activation=config.activation_fn_name,
                    )
                )
                in_channels_down = out_channels_down
            # Down sample at all resolutions except the last
            if i < unet_depth - 1:
                down.append(DownsampleND(n_channels=in_channels_down, dim=2))

        # Combine the set of modules
        self.down = nn.ModuleList(down)
        
        out_channels_mid = out_channels_down
        self.middle = MiddleBlockND(config=config,
                                    n_channels=out_channels_mid, 
                                    dim=2, 
                                    activation=config.activation_fn_name, 
                                    )

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels_up = out_channels_mid
        # For each resolution
        for i in reversed(range(unet_depth)):
            # `n_blocks` at the same resolution
            out_channels_up = in_channels_up
            for _ in range(config.n_blocks):
                up.append(
                    UpBlockND(
                        config=config,
                        in_channels=in_channels_up,
                        out_channels=out_channels_up,
                        dim=2,
                        activation=config.activation_fn_name,
                    )
                )
            # Final block to reduce the number of channels
            out_channels_up = in_channels_up // config.channel_multiplier[i]
            up.append(UpBlockND(config=config,
                                in_channels=in_channels_up, 
                                out_channels=out_channels_up, 
                                dim=2, 
                                activation=config.activation_fn_name, 
                                ))
            in_channels_up = out_channels_up
            # Up sample at all resolutions except last
            if i > 0:
                up.append(UpsampleND(n_channels=in_channels_up, dim=2))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        self.norm = CustomNorm(config=config, num_channels=config.latent_channels, array_length=4, channel_at_last_position=False)
        if config.use1x1:
            self.final = nn.Conv2d(in_channels_up, config.out_size, kernel_size=1)
        else:
            self.final = nn.Conv2d(in_channels_up, config.out_size, kernel_size=(3, 3), padding=(1, 1))

        # Overall downsampling factor for padding logic
        self._ds_factor = 2 ** (unet_depth - 1)

    def forward(self, x: torch.Tensor, **kwargs):
        """Encode and decode a 2D grid with skip connections.

        Parameters
        ----------
        x : Tensor
            Input grid tensor of shape [batch, in_channels, grid_x, grid_y].
            The number of channels typically equals `in_size * input_seq`
            plus 2 if `coord_features=True`.

        Returns
        -------
        Tensor
            Output grid tensor of shape [batch, out_channels, grid_x, grid_y].
        """
        if x.dim() != 4:
            raise ValueError(
                "Only 4D tensors [batch, in_channels, grid_x, grid_y] accepted for 2D UNet"
            )

        # ---- Pad so H and W divisible by self._ds_factor ----
        H_orig, W_orig = x.shape[-2], x.shape[-1]
        pad_H = (-H_orig) % self._ds_factor
        pad_W = (-W_orig) % self._ds_factor
        top = pad_H // 2
        bottom = pad_H - top
        left = pad_W // 2
        right = pad_W - left
        if pad_H or pad_W:
            x = F.pad(x, (left, right, top, bottom))

        # add coord features after padding
        if self.config.coord_features:
            coord_feat = twod_meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        x = self.image_proj(x)

        h = [x]
        for i,m in enumerate(self.down):
            x = m(x, **kwargs)
            h.append(x)

        x = self.middle(x, **kwargs)

        for i, m in enumerate(self.up):
            if isinstance(m, UpsampleND):
                x = m(x)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, **kwargs)

        x = self.norm(x, **kwargs)
        x = self.activation(x)
        x = self.final(x)

        # Crop back to original size if padded
        if pad_H or pad_W:
            x = x[..., top:top + H_orig, left:left + W_orig]

        return x