import math
import torch
import torch.nn as nn
import numpy as np

from torch import Tensor
from torch.nn import functional as F
from typing import Optional, Tuple, Any, Type


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        """
        Lightly adapted from
        https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
        """
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


class SpatialAttentionModule(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """Apply channel and spatial attention on input for feature recalibration."""
        return (
            x
            * self.act(
                self.cv1(
                    torch.cat(
                        [
                            torch.mean(x, 1, keepdim=True),
                            torch.max(x, 1, keepdim=True)[0],
                        ],
                        1,
                    )
                )
            )
            + x
        )


class ChannelAttentionBlock(nn.Module):
    def __init__(self):
        super(ChannelAttentionBlock, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        # self.proj_query = nn.Linear(512, 64)
        # self.proj_key = nn.Linear(512, 64)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.size()
        query = x.view(B, C, -1)
        key = x.view(B, C, -1).permute(0, 2, 1)
        affinity = torch.matmul(query, key)
        # Subtract the max value for numerical stability
        affinity = (
            torch.max(affinity, -1, keepdim=True)[0].expand_as(affinity) - affinity
        )
        affinity = self.softmax(affinity)
        value = x.view(B, C, -1)
        att_out = torch.matmul(affinity, value)
        att_out = att_out.view(B, C, H, W)
        out = self.gamma * att_out + x
        return out


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        """
        From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
        Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
        """
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class DualMixAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        attention_num_heads: int,
        attention_downsample_rate: int = 2,
    ) -> None:
        super().__init__()
        self.channel_attention = ChannelAttentionBlock()
        self.channel_token_mix_attention = Attention(
            embedding_dim,
            attention_num_heads,
            downsample_rate=attention_downsample_rate,
        )

        self.spatial_attention = SpatialAttentionModule()
        self.spatial_token_mix_attention = Attention(
            embedding_dim,
            attention_num_heads,
            downsample_rate=attention_downsample_rate,
        )

        self.token_layernorm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        mask_tokens: Tensor,
        feature_embeddings: Tensor,
        feature_potional_embeddings: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        spatial_feature_embeddings = feature_embeddings + feature_potional_embeddings
        channel_feature_embeddings = feature_embeddings + feature_potional_embeddings
        # Spatial attention branch
        spatial_attention = self.spatial_attention(spatial_feature_embeddings)
        spatial_attention = spatial_attention.flatten(2).permute(0, 2, 1)
        q = mask_tokens
        k = v = spatial_attention
        spatial_token_mix_attention = self.spatial_token_mix_attention(q, k, v)

        # Channel attention branch
        channel_attention = self.channel_attention(channel_feature_embeddings)
        channel_attention = channel_attention.flatten(2).permute(0, 2, 1)
        q = mask_tokens
        k = v = channel_attention
        channel_token_mix_attention = self.channel_token_mix_attention(q, k, v)

        attention_tokens = (
            spatial_token_mix_attention + channel_token_mix_attention + mask_tokens
        )
        attention_tokens = self.token_layernorm(attention_tokens)
        attention_features = spatial_attention + channel_attention

        return attention_tokens, attention_features


class DMABlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        attention_num_heads: int,
        mlp_dim: int = 1024,
        attention_downsample_rate: int = 2,
    ) -> None:
        super().__init__()
        self.dm_attention = DualMixAttention(
            embedding_dim,
            attention_num_heads,
            attention_downsample_rate,
        )

        self.token_mlp = MLPBlock(embedding_dim, mlp_dim, act=nn.GELU)
        self.token_layernorm = nn.LayerNorm(embedding_dim)
        self.feature_mix = Attention(
            embedding_dim,
            attention_num_heads,
            downsample_rate=attention_downsample_rate,
        )

    def forward(
        self,
        mask_tokens: Tensor,
        feature_embeddings: Tensor,
        feature_potional_embeddings: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        b, c, h, w = feature_embeddings.shape

        attention_tokens, attention_features = self.dm_attention(
            mask_tokens,
            feature_embeddings,
            feature_potional_embeddings,
        )

        # Token MLP
        mask_tokens = (
            self.token_layernorm(self.token_mlp(attention_tokens)) + mask_tokens
        )

        feature_potional_embeddings = feature_potional_embeddings.flatten(2).permute(
            0, 2, 1
        )

        # Mix token to feature embeddings
        q = attention_features + feature_potional_embeddings
        k = v = attention_tokens
        feature_mix_embeddings = self.feature_mix(q, k, v)

        # Reshape feature embeddings
        feature_embeddings = (
            feature_mix_embeddings.permute(0, 2, 1).reshape(b, c, h, w)
            + feature_embeddings
        )

        return mask_tokens, feature_embeddings


class DualMixAttentionBlock(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        attention_downsample_rate: int = 2,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for _ in range(depth):
            self.layers.append(
                DMABlock(
                    embedding_dim=embedding_dim,
                    attention_num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    attention_downsample_rate=attention_downsample_rate,
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        mask_token: Tensor,
        image_embedding: Tensor,
        image_pe: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        b, c, h, w = image_embedding.shape
        # Prepare queries
        queries = mask_token
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(queries, keys, image_pe)

        keys = keys.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)
        # Apply the final attention layer from the points to the image
        q = queries + mask_token
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        keys = keys.permute(0, 2, 1).reshape(b, c, h, w)

        return queries, keys
