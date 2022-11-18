import torch
import numpy as np

def positional_emb_sin_cos(num_patches, embed_dim):
    # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
    pos_embedding = torch.zeros(num_patches, embed_dim)
    position = torch.arange(0, num_patches, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
    pos_embedding[:, 0::2] = torch.sin(position * div_term)
    pos_embedding[:, 1::2] = torch.cos(position * div_term)
    pos_embedding = pos_embedding.unsqueeze(0)
    return pos_embedding

def img_to_patch(x: torch.Tensor, patch_size):
    """
    Inputs:
        x - torch.Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                        as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5) # [B, H', W', C, p_H, p_W]
    x = x.flatten(1,2)              # [B, H'*W', C, p_H, p_W]
    x = x.flatten(2,4)              # [B, H'*W', C*p_H*p_W]
    return x
    
def patch_to_img(x: torch.Tensor, patch_size, flatten_channels=True):
    """
    Inputs:
        x - torch.Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                        as a feature vector instead of a image grid.
    """
    # TODO: idk when to use it tho?
    pass