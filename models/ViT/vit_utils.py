"""Utilities for Vision Transformer (ViT) components tailored to this project.

Includes:
- `ViTConfig`: config extending project `PretrainedConfig` with ViT-specific fields.
- `ViTEmbeddings` and `ViTPatchEmbeddings`: patch tokenization and positional embeddings.
- `ViTLayer`: a single Transformer block using project `CustomNorm`.
- `ViTEncoder`: a stack of `ViTLayer`s producing `BaseModelOutput`.

All components are adapted to use `latent_channels` as the hidden size and
integrate with normalization and configuration utilities used throughout the
project.
"""

from utils.model_utils import PretrainedConfig, CustomNorm
import torch
from torch import nn
from typing import Optional, Tuple, Union
from transformers.models.vit.modeling_vit import ViTAttention, ViTIntermediate, ViTOutput, BaseModelOutput

class ViTConfig(PretrainedConfig):
    """Configuration for Vision Transformer components.

    Parameters
    ----------
    num_hidden_layers : int, default=12
        Number of transformer blocks.
    num_attention_heads : int, default=12
        Number of attention heads per block.
    intermediate_size : int, default=3072
        MLP hidden size inside each transformer block.
    hidden_act : str, default="gelu"
        Activation function used in intermediate layers.
    hidden_dropout_prob : float, default=0.0
        Dropout probability applied to embeddings and MLPs.
    attention_probs_dropout_prob : float, default=0.0
        Dropout probability applied to attention probabilities.
    initializer_range : float, default=0.02
        Std for weight initialization.
    patch_size : int, default=16
        Square patch size for patch embeddings.
    qkv_bias : bool, default=True
        Whether to add bias for QKV projections.
    interpolate_pos_encoding : bool, default=False
        If True, interpolate positional encodings for resized inputs.
    **kwargs :
        Additional fields consumed by `PretrainedConfig`, including:
        - `grid_resolution`: Tuple[int, int], required for patch embeddings.
        - `in_size`: int, input channels for patch embedding conv.
        - `latent_channels`: int, embedding dimension (becomes `hidden_size`).

    """

    model_type = "vit"

    def __init__(
        self,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        patch_size=16,
        qkv_bias=True,
        interpolate_pos_encoding=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.patch_size = patch_size
        self.qkv_bias = qkv_bias

        self.interpolate_pos_encoding = interpolate_pos_encoding

        self.hidden_size = self.latent_channels


class ViTEmbeddings(nn.Module):
    """Patch and position embeddings (optionally with mask token)."""

    def __init__(self, config: ViTConfig, use_mask_token: bool = False) -> None:
        super().__init__()

        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.latent_channels)) if use_mask_token else None
        self.patch_embeddings = ViTPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches, config.latent_channels))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.patch_size = config.patch_size
        self.config = config


    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        """Embed input images into patch tokens and add positional encodings.

        Parameters
        ----------
        pixel_values : Tensor
            Input tensor of shape [batch, in_channels, height, width].
        bool_masked_pos : BoolTensor, optional
            Mask of shape [batch, num_patches] indicating which patches are masked.
        interpolate_pos_encoding : bool, default=False
            If True, interpolate position encodings to match current H/W.

        Returns
        -------
        Tensor
            Patch embeddings of shape [batch, num_patches, latent_channels].
        """
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings
    
class ViTPatchEmbeddings(nn.Module):
    """Turn images into patch embeddings for transformer

    Inputs of shape [batch, in_channels, height, width] are projected into
    patch embeddings of shape [batch, seq_length, latent_channels] using a
    strided convolution with kernel size equal to patch size.
    """

    def __init__(self, config):
        super().__init__()
        grid_resolution, patch_size = config.grid_resolution, config.patch_size
        num_channels, latent_channels = config.in_size, config.latent_channels

        patch_size =  (patch_size, patch_size)
        num_patches = (grid_resolution[1] // patch_size[1]) * (grid_resolution[0] // patch_size[0])
        self.grid_resolution = grid_resolution
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, latent_channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        """Project input into patch tokens.

        Parameters
        ----------
        pixel_values : Tensor
            [batch, in_channels, height, width]
        interpolate_pos_encoding : bool, default=False
            If False, strictly enforce that (height, width) equals `grid_resolution`.

        Returns
        -------
        Tensor
            Patch embeddings of shape [batch, num_patches, latent_channels].

        Raises
        ------
        ValueError
            If input channels mismatch or sizes do not match when interpolation is disabled.
        """
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        if not interpolate_pos_encoding:
            if height != self.grid_resolution[0] or width != self.grid_resolution[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.grid_resolution[0]}*{self.grid_resolution[1]})."
                )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings
    

class ViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ViTAttention(config)
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)

        self.layernorm_before = CustomNorm(config=config, num_channels=config.hidden_size, array_length=3, channel_at_last_position=True)
        self.layernorm_after = CustomNorm(config=config, num_channels=config.hidden_size, array_length=3, channel_at_last_position=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        """Apply attention and MLP with pre/post normalization and residuals.

        Parameters
        ----------
        hidden_states : Tensor
            [batch, seq_length, hidden_size]
        head_mask : Optional[Tensor]
            Attention head mask.
        output_attentions : bool, default=False
            Whether to return attention weights.

        Returns
        -------
        Tuple[Tensor, ...]
            layer_output and optionally attention weights.
        """
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states, **kwargs),  # in ViT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states, **kwargs)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs
    

class ViTEncoder(nn.Module):
    """Stack of `ViTLayer` blocks forming the transformer encoder."""
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([ViTLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs
    ) -> Union[tuple, BaseModelOutput]:
        """Run the transformer encoder.

        Parameters
        ----------
        hidden_states : Tensor
            Input of shape [batch, seq_length, hidden_size].
        head_mask : Optional[Tensor]
            Head mask broadcastable to [num_layers, batch, num_heads, seq_len, seq_len].
        output_attentions : bool, default=False
            Whether to return attention maps.
        output_hidden_states : bool, default=False
            Whether to return hidden states from all layers.
        return_dict : bool, default=True
            Whether to return a `BaseModelOutput`.

        Returns
        -------
        BaseModelOutput or Tuple
            `last_hidden_state`, and optionally `hidden_states`, `attentions`.
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                    **kwargs
                )
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions, **kwargs)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )