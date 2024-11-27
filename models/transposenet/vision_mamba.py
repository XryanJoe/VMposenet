import copy
from typing import Optional
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops.layers.torch import Reduce
from zeta.nn import SSM

# Pair
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class Vim(nn.Module):
    default_config={
        "dim": 256,
        "dt_rank": 32,
        "dim_inner": 256,
        "d_state": 256,
        "depth": 12,  # Number of encoder layers
        "pose_token_dim": 256,  # Pose token dimension
        "image_size": 224,
        "patch_size": 16,
        # "channels": 3,
        "dropout": 0.1
    }

    def __init__(self,config={}):
        config =  {**self.default_config, **config}
        super().__init__()

        self.dim = config.get("dim")
        self.dt_rank = config.get("dt_rank")
        self.dim_inner = config.get("dim_inner")
        self.d_state = config.get("d_state")
        self.depth = config.get("depth")
        self.pose_token_dim = config.get("pose_token_dim")
        self.dropout=config.get("dropout")
        image_height, image_width = pair(config.get("image_size"))
        patch_height, patch_width = pair(config.get("patch_size"))
        channels=256
        patch_dim = channels * patch_height * patch_width

        # Patch embedding for image
        # print(config.get("patch_size"))
        # print(patch_height,patch_width)
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.Linear(patch_dim, self.dim),
        )

        # Encoder layers (VisionMambaBlock)
        self.layers = nn.ModuleList([
            VisionEncoderMambaBlock(dim=self.dim, dt_rank=self.dt_rank, dim_inner=self.dim_inner, d_state=self.d_state)
            for _ in range(self.depth)
        ])

        # Dropout layer
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor, pos_embed: Tensor, pose_token_embed: Tensor):
        b, c, h, w = x.shape

        # Step 1: Patch embedding
     
        x = self.to_patch_embedding(x)  # Shape: (b, num_patches, dim)

       # Step 2: Split pos_embed into pose_pos_embed and activation_pos_embed
        # pose_pos_embed, activation_pos_embed = pos_embed
        # activation_pos_embed = activation_pos_embed.flatten(2).permute(0,2,1)
        # pose_pos_embed = pose_pos_embed.unsqueeze(2).permute(0,2,1)
        # # print(activation_pos_embed.shape)
        # # print(pose_pos_embed.shape)
        # pos_embed = torch.cat([pose_pos_embed, activation_pos_embed],dim=1)

        # # Pose token will be broadcast across all patches
        # cls_tokens = repeat(pose_token_embed, "() n d -> b n d", b=b)
        # print(x.shape)
        # x = torch.cat((cls_tokens, x), dim=1)  # Shape: (b, num_patches+1, dim)
      
        # # print(pos_embed.shape)
        # # Step 3: Apply position embedding (pos_embed)
        # x = torch.cat((x,pos_embed),dim=1)  # Add position embedding (if available)
        print(x.shape)
        # Step 4: Apply VisionMambaBlock layers
        for layer in self.layers:
            x = layer(x)  # Apply VisionMambaBlock for each layer

        # Step 5: Return the final encoded features
        return x  # Shape: (b, num_patches+1, dim)


class VisionEncoderMambaBlock(nn.Module):
    def __init__(self, dim: int, dt_rank: int, dim_inner: int, d_state: int):
        super().__init__()
        self.dim = dim
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state

        self.forward_conv1d = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.backward_conv1d = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)
        self.silu = nn.SiLU()
        self.ssm = SSM(dim, dt_rank, dim_inner, d_state)

        # Linear layer for z and x
        self.proj = nn.Linear(dim, dim)

        # Softplus
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor):
        b, s, d = x.shape

        # Skip connection
        skip = x

        # Normalization
        x = self.norm(x)

        # Split x into x1 and x2 with linears
        z1 = self.proj(x)
        x = self.proj(x)

        # forward conv1d
        x1 = self.process_direction(x, self.forward_conv1d, self.ssm)

        # backward conv1d
        x2 = self.process_direction(x, self.backward_conv1d, self.ssm)

        # Activation
        z = self.silu(z1)

        # Matmul
        x1 *= z
        x2 *= z

        # Residual connection
        return x1 + x2 + skip

    def process_direction(self, x: Tensor, conv1d: nn.Conv1d, ssm: SSM):
        x = rearrange(x, "b s d -> b d s")
        x = self.softplus(conv1d(x))
        x = rearrange(x, "b d s -> b s d")
        x = ssm(x)
        return x


def build_vim(config):
    return Vim(config)
