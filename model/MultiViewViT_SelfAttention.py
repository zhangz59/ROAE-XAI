import torch
import torch.nn as nn
import torch.nn.functional as F
from model.vit import VisionTransformer


class SelfAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super(SelfAttentionFusion, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        # x shape: (batch_size, num_views, embed_dim)
        attn_output, _ = self.attention(x, x, x)
        # Aggregate outputs (mean)
        fused_output = attn_output.mean(dim=1)
        return fused_output


class MultiViewViT_SelfAttention(nn.Module):
    def __init__(self, image_sizes, patch_sizes, num_channals, vit_args, mlp_dims):
        super().__init__()

        # ViT branches
        self.vit_1 = VisionTransformer(image_size=image_sizes[0], num_channals=num_channals[0],
                                       patch_size=patch_sizes[0], **vit_args)
        self.vit_2 = VisionTransformer(image_size=image_sizes[1], num_channals=num_channals[1],
                                       patch_size=patch_sizes[1], **vit_args)
        self.vit_3 = VisionTransformer(image_size=image_sizes[2], num_channals=num_channals[2],
                                       patch_size=patch_sizes[2], **vit_args)

        embed_dim = vit_args['emb_dim']
        self.self_attention_fusion = SelfAttentionFusion(embed_dim=embed_dim, num_heads=4)

        # Final MLP layers
        # Corrected MLP construction
        mlp_layers = []
        for i in range(len(mlp_dims) - 1):
            mlp_layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            if i < len(mlp_dims) - 2:
                mlp_layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        x1 = x.permute(0, 3, 1, 2)
        x2 = x.permute(0, 2, 1, 3)
        x3 = x

        out1 = self.vit_1(x1)
        out2 = self.vit_2(x2)
        out3 = self.vit_3(x3)

        # Combine ViT outputs for Self-Attention Fusion
        combined_out = torch.stack([out1, out2, out3], dim=1)  # shape: (batch, views, emb_dim)

        # Apply self-attention fusion
        fused_out = self.self_attention_fusion(combined_out)

        prediction = self.mlp(fused_out)

        return prediction
