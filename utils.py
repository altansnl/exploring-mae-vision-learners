import torch
import numpy as np

def positional_emb_sin_cos(num_patches, embed_dim):
    # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
    """
    print((num_patches,num_patches))
    pos_embedding = torch.zeros(num_patches,num_patches, embed_dim)
    position = torch.arange(0, (num_patches,num_patches), dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
    pos_embedding[:, 0::2] = torch.sin(position * div_term)
    pos_embedding[:, 1::2] = torch.cos(position * div_term)
    pos_embedding = pos_embedding.unsqueeze(0)
    """

    # using their implementation of positional embedding
    grid_h = np.arange(num_patches, dtype=np.float32)
    grid_w = np.arange(num_patches, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)    

    grid = grid.reshape([2, 1, num_patches, num_patches])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    print(type(pos_embed), pos_embed.shape)
    return torch.from_numpy(pos_embed)

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)

    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

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