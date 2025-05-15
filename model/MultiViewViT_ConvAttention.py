import torch
import torch.nn as nn
import torch.nn.functional as F
from model.vit import VisionTransformer

# Convolutional Attention Module (CBAM simplified version)
class ConvAttentionFusion(nn.Module):
    def __init__(self, embed_dim, reduction_ratio=16):
        super(ConvAttentionFusion, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(embed_dim, embed_dim // reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv1d(embed_dim // reduction_ratio, embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch_size, num_views, embed_dim)
        x_perm = x.permute(0, 2, 1)  # (batch_size, embed_dim, num_views)
        attention = self.channel_attention(x_perm)
        out = x_perm * attention
        fused_output = out.mean(dim=2)
        return fused_output

class MultiViewViT_ConvAttention(nn.Module):
    def __init__(self, image_sizes, patch_sizes, num_channals, vit_args, mlp_dims):
        super().__init__()

        # ViT branches
        self.vit_1 = VisionTransformer(image_size=image_sizes[0], num_channals=num_channals[0], patch_size=patch_sizes[0], **vit_args)
        self.vit_2 = VisionTransformer(image_size=image_sizes[1], num_channals=num_channals[1], patch_size=patch_sizes[1], **vit_args)
        self.vit_3 = VisionTransformer(image_size=image_sizes[2], num_channals=num_channals[2], patch_size=patch_sizes[2], **vit_args)

        embed_dim = vit_args['emb_dim']
        self.conv_attention_fusion = ConvAttentionFusion(embed_dim=embed_dim, reduction_ratio=16)

        # Final MLP layers
        mlp_layers = []
        mlp_dims = [embed_dim, 512, 256, 1]
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

        # Combine ViT outputs for Convolutional Attention Fusion
        combined_out = torch.stack([out1, out2, out3], dim=1)  # shape: (batch, views, emb_dim)

        # Apply convolutional attention fusion
        fused_out = self.conv_attention_fusion(combined_out)

        prediction = self.mlp(fused_out)

        return prediction
