import torch.nn as nn

import torch
from model.vit import VisionTransformer



class MultiViewViT(nn.Module):
    def __init__(self, image_sizes, patch_sizes, num_channals, vit_args, mlp_dims):
        """
        image_sizes: List of sizes for each of the 3 views e.g. [(91, 109), (91, 91), (109, 91)]
        patch_sizes: List of patch sizes for each of the 3 views e.g. [(16, 16), (16, 16), (16, 16)]
        vit_args: Dictionary containing other arguments for the ViT (e.g. emb_dim, mlp_dim, num_heads, etc.)
        mlp_dims: List of dimensions for the MLP layers e.g. [768*3, 512, 256, 1]
        """
        super(MultiViewViT, self).__init__()

        # Creating 3 ViT models for each view
        self.vit_1 = VisionTransformer(image_size=image_sizes[0],num_channals=num_channals[0], patch_size=patch_sizes[0], **vit_args)
        self.vit_2 = VisionTransformer(image_size=image_sizes[1], num_channals=num_channals[1],patch_size=patch_sizes[1], **vit_args)
        self.vit_3 = VisionTransformer(image_size=image_sizes[2], num_channals=num_channals[2],patch_size=patch_sizes[2], **vit_args)

        # MLP for final prediction after concatenating outputs of three ViTs
        layers = []
        for i in range(len(mlp_dims) - 1):
            layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            if i < len(mlp_dims) - 2:  # No activation after last layer
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)



    def forward(self, x, return_attention_weights=False):
        """
        x1, x2, x3 are the 3 views (slices) of the 3D data.
        """
        x1=x.permute(0, 3, 1, 2)
        x2=x.permute(0, 2, 1, 3)
        x3=x
        if return_attention_weights:
            out1, attn1 = self.vit_1(x1, return_attention_weights=False)
            out2, attn2 = self.vit_2(x2, return_attention_weights=False)
            out3, attn3 = self.vit_3(x3, return_attention_weights=False)
        else:
            out1 = self.vit_1(x1)
            out2 = self.vit_2(x2)
            out3 = self.vit_3(x3)

        # Concatenate the outputs
        combined_out = torch.cat([out1, out2, out3], dim=1)

        # Pass through MLP
        prediction = self.mlp(combined_out)




        if return_attention_weights:
            return prediction, (attn1, attn2, attn3)
        else:
            return prediction



# Example usage
# model = MultiViewViT(
#     image_sizes=[(91, 109), (91, 91), (109, 91)],
#     patch_sizes=[(16, 16), (16, 16), (16, 16)],
#     vit_args={'emb_dim': 768, 'mlp_dim': 3072, 'num_heads': 12, 'num_layers': 12, 'num_classes': 768,
#               'dropout_rate': 0.1, 'attn_dropout_rate': 0.0},
#     mlp_dims=[768 * 3, 512, 256, 1]
# )
