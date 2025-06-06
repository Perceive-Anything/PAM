
import torch
import torch.nn.functional as F
from torch import nn, Tensor, einsum
from einops import rearrange, repeat
from typing import Tuple, Type, List
from sam2.modeling.sam2_utils import MLP
from sam2.modeling.position_encoding import PositionEmbeddingRandom
from .modules import Attention

# Function to compute positional embeddings
def compute_image_pe(pe_layer, pe_size, dtype, device):
    """
    Compute image positional embedding.
    """
    image_pe = pe_layer(pe_size).unsqueeze(0)  # Generate positional embeddings
    image_pe = image_pe.flatten(2).permute(0, 2, 1)  # Adjust dimensions
    return image_pe.to(dtype=dtype, device=device)


def pixel_shuffle(x, scale_factor=0.5):
    b, c, h, w = x.size()
    # B, C, H, W--> B, W, H * scale, C // scale
    x = x.view(b, w, int(h * scale_factor), int(c / scale_factor))
    # B, W, H * scale, C // scale --> B, H * scale, W, C // scale
    x = x.permute(0, 2, 1, 3).contiguous()
    # B, H * scale, W, C // scale --> B, , C // (scale ** 2), H * scale, W * scale
    x = x.view(b, int(h * scale_factor), int(w * scale_factor),
                int(c / (scale_factor * scale_factor)))

    x = x.permute(0, 3, 1, 2).contiguous()
    return x

class Semantic_Perceiver(nn.Module):
    def __init__(
        self,
        model_args, 
        num_sem_tokens: int = 16,
        depth: int = 2,
        embedding_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        channel_downsample_rate: int = 2,
    ) -> None:
        """
        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()

        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()
        self.pe_layer = PositionEmbeddingRandom(embedding_dim // 2)
        self.sem_tokens = nn.Embedding(num_sem_tokens, embedding_dim)
        self.proj = nn.Linear(embedding_dim*16, embedding_dim*4)

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    channel_downsample_rate=channel_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, channel_downsample_rate=channel_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        point_embedding: Tensor,
        scale_factor: float,
        *args, 
        **kwargs
    ) -> Tuple[Tensor, Tensor]:

        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)

        # Prepare queries
        bs, mp_t, _ = point_embedding.shape
        sem_tokens = self.sem_tokens.weight.unsqueeze(0).expand(
            bs, -1, -1
        )
        queries = torch.cat((point_embedding, sem_tokens), dim=1)
        keys = image_embedding

        tokens_pe = queries

        # Apply transformer blocks and final layernorm
        for idx, layer in enumerate(self.layers):
            # Generate image positional embedding
            image_pe = compute_image_pe(self.pe_layer, (h, w), keys.dtype, keys.device)

            # Apply transformer block
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=tokens_pe,
                key_pe=image_pe,
                hw=(h, w),
            )
            
        # Apply the final attention layer from the points to the image
        q = queries + tokens_pe
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        keys = keys.reshape(bs, -1, h, w)
        if scale_factor == 1/2:
            # (bs, 256, 64, 64) -> (bs, 1024, 32, 32)
            keys = pixel_shuffle(keys, scale_factor=1/2)
            # (bs, 1024, 32, 32) -> (bs, 1024, 1024)
            keys = keys.flatten(2).permute(0, 2, 1)
        else:
            # (bs, 256, 64, 64) -> (bs, 4096, 16, 16)
            keys = pixel_shuffle(keys, scale_factor=1/4)
            # (bs, 4096, 16, 16) -> (bs, 256, 4096)
            keys = keys.flatten(2).permute(0, 2, 1)
            # (bs, 256, 4096) -> (bs, 256, 1024)
            keys = self.proj(keys)
        
        return queries[:,mp_t:,:], keys
    

    @property
    def config(self):
        return {
            "mm_resampler_type": "semantic_perceiver",
        }

    @property
    def hidden_size(self):
        return self.embedding_dim


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        channel_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:

        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads, token_downsample_rate=1,)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, channel_downsample_rate=channel_downsample_rate, token_downsample_rate=1,
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLP(
            embedding_dim, mlp_dim, embedding_dim, num_layers=2, activation=activation
        )
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, channel_downsample_rate=channel_downsample_rate, token_downsample_rate=1,
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor, hw: Tuple[int, int]
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries, hw=hw)
        # b,_,c = keys.size()
        # keys = F.interpolate(keys.reshape(b, c, hw[0], hw[1]), size=(hw[0]//2, hw[1]//2), mode='bicubic', align_corners=False).flatten(2).permute(0, 2, 1)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys




