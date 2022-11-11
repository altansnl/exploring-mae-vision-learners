import torch
import torch.nn as nn

class MaskedAutoencoderViT(nn.Module):
    def __init__(self):
        pass

    def _init_weights(self):
        pass
    
    # divide an image into regular non-overlapping patches
    def patchify(self, imgs):
        pass

    # sample a subset of patches and mask (remove) the remaining ones
    # random without replacement
    def mask(self):
        pass
    
    # apply VIT on visible/unmasked patches
    def encoder_forward(self, x):
        # encode patches with linear projection
        # with added position embeddings

        # apply set of transformer blocks
        pass
    
    # input: full set of tokens consisting of
    # (i) encoded visible patches, (ii) mask tokens
    def decoder_forward(self, x):
        # add positional embeddings to tokens

        # apply series of transformer blocks
        pass
    
    def forward(self, imgs):
        pass

    # mean squared error in the pixel space
    # calculated only on masked patches
    def loss(self):
        pass
