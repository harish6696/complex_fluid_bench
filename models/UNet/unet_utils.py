
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from utils.model_utils import PretrainedConfig
from utils import activation_func
from utils.model_utils import CustomNorm

class UNetConfig(PretrainedConfig):
    """
    Args:
        activation_fn_name (str): Name of the activation function to use. Default is "gelu".
        channel_multiplier (Union[Tuple[int, ...], List[int]]): Multipliers for the number of channels at each stage of the UNet. Default is (1, 2, 2, 4).
        is_attn (Union[Tuple[bool, ...], List[bool]]): Flags indicating whether attention is applied at each stage. Default is (False, False, False, False).
        mid_attn (bool): Whether to apply attention in the middle block. Default is False.
        n_blocks (int): Number of blocks in each stage of the UNet. Default is 2.
        norm: "group" or "layer" or "batch" or "none" (default: "layer")
        norm_layer_eps: Used to avoid zero-div error. float (default: 1e-5)
        use1x1 (bool): Whether to use 1x1 convolutions. Default is False.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    model_type = "UNet"
    
    def __init__(
        self,
        activation_fn_name: str = "gelu",
        channel_multiplier: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
        n_blocks: int = 2,
        use1x1: bool = False,
        norm: str = "layer",
        norm_layer_eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.activation_fn_name = activation_fn_name
        self.channel_multiplier = channel_multiplier
        self.n_blocks = n_blocks
        self.use1x1 = use1x1
        self.norm = norm
        self.norm_layer_eps = norm_layer_eps

class ResidualBlockND(nn.Module):
    """Residual Block for 2D data.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        dim (int): Dimensionality of the data. Must be 2.
        activation_fn_name (str): Name of the activation function.
        norm_layer: "group" or "layer" or "batch" (default: "layer")
    """

    def __init__(
        self,
        config,
        in_channels: int,
        out_channels: int,
        dim: int,
        activation_fn_name: str = "gelu",
    ):
        super().__init__()

        # Get activation function
        self.activation = activation_func.get_activation(activation_fn_name)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation_fn_name} not implemented")

        # Select appropriate convolution class
        if dim != 2:
            raise ValueError(f"Unsupported dimension: {dim}. Must be 2.")
        
        Conv = nn.Conv2d
        kernel_size = (3, 3)
        padding = (1, 1)
        shortcut_kernel = (1, 1)

        # Convolution layers
        self.conv1 = Conv(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = Conv(out_channels, out_channels, kernel_size=kernel_size, padding=padding)

        # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = Conv(in_channels, out_channels, kernel_size=shortcut_kernel)
        else:
            self.shortcut = nn.Identity()

        self.norm1 = CustomNorm(config=config, num_channels=in_channels, array_length=dim+2, channel_at_last_position=False)
        self.norm2 = CustomNorm(config=config, num_channels=out_channels, array_length=dim+2, channel_at_last_position=False)

    def forward(self, x: torch.Tensor, **kwargs):
        h = self.norm1(x, **kwargs)
        h = self.activation(h)
        h = self.conv1(h)

        h = self.norm2(h, **kwargs)
        h = self.activation(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class DownBlockND(nn.Module):
    """Generalized Down Block combining ResidualBlock for 2D inputs.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        dim (int): Dimensionality. Must be 2.
        activation (str): Name of activation function.
        config (UNetConfig): Configuration for the UNet.
    """

    def __init__(
        self,
        config,
        in_channels: int,
        out_channels: int,
        dim: int,
        activation: str = "gelu",
    ):
        super().__init__()
        self.res = ResidualBlockND(
            config=config,
            in_channels=in_channels,
            out_channels=out_channels,
            dim=dim,
            activation_fn_name=activation,
        )

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.res(x, **kwargs)
        return x

class UpBlockND(nn.Module):
    """Generalized Up Block combining ResidualBlock for 2D inputs.

    These are used in the second half of U-Net at each resolution.

    Args:
        config (UNetConfig): Configuration for the UNet.
        in_channels (int): Number of input channels (from skip connection).
        out_channels (int): Number of output channels.
        dim (int): Dimensionality. Must be 2.
        activation (str): Activation function.
    """

    def __init__(
        self,
        config,
        in_channels: int,
        out_channels: int,
        dim: int,
        activation: str = "gelu",
    ):
        super().__init__()
        self.res = ResidualBlockND(
            config=config,
            in_channels=in_channels + out_channels,
            out_channels=out_channels,
            dim=dim,
            activation_fn_name=activation,
        )

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.res(x, **kwargs)
        return x

class MiddleBlockND(nn.Module):
    """Middle block

    Combines a `ResidualBlockND`, and another `ResidualBlockND`.
    Used at the lowest resolution of the U-Net.

    Args:
        config (UNetConfig): Configuration for the UNet.
        n_channels (int): Number of channels in the input and output.
        dim (int): Dimensionality. Must be 2.
        activation (str): Activation function to use. Defaults to "gelu".
    """

    def __init__(self, 
                 config,
                 n_channels: int, 
                 dim: int, 
                 activation: str = "gelu"
                 ):
        super().__init__()
        self.res1 = ResidualBlockND(config=config, dim=dim, in_channels=n_channels, out_channels=n_channels, activation_fn_name=activation) 
        self.res2 = ResidualBlockND(config=config, dim=dim, in_channels=n_channels, out_channels=n_channels, activation_fn_name=activation) 

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.res1(x, **kwargs)
        x = self.res2(x, **kwargs)
        return x

class UpsampleND(nn.Module):
    r"""Scale up the feature map by $2 \times$ along all spatial dimensions.

    Uses kernel_size=(4,4), stride=(2,2), padding=(1,1) for 2D transpose convolutions.

    Args:
        n_channels (int): Number of channels in the input and output.
        dim (int): Dimensionality of the data. Must be 2.
    """

    def __init__(self, n_channels: int, dim: int):
        super().__init__()

        if dim != 2:
            raise ValueError(f"Unsupported dimension: {dim}. Must be 2.")
        
        self.conv = nn.ConvTranspose2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding=(1, 1)
        )

    def forward(self, x: torch.Tensor):
        return self.conv(x)

class DownsampleND(nn.Module):
    r"""Scale down the feature map by $\frac{1}{2} \times$ along all spatial dimensions.

    Uses kernel_size=(3,3), stride=(2,2), padding=(1,1) for 2D convolutions.

    Args:
        n_channels (int): Number of channels in the input and output.
        dim (int): Dimensionality of the data. Must be 2.
    """

    def __init__(self, n_channels: int, dim: int):
        super().__init__()

        if dim != 2:
            raise ValueError(f"Unsupported dimension: {dim}. Must be 2.")
        
        self.conv = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1)
        )

    def forward(self, x: torch.Tensor, **kwargs):
        return self.conv(x)
