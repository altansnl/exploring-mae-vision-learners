import torch
import torch.nn as nn
import numpy as np

from vanilla_vit import img_to_patch, AttentionBlock, PositionalEncoding
from timm.models.vision_transformer import PatchEmbed, Block
from utils import *


class MAEBackboneViT(nn.Module):
    """
    Simplified version of ViT structure + random patchify
    """
    def __init__(self,
                 embed_dim,
                 img_dim,
                 hidden_dim_ratio,
                 num_channels,
                 num_heads,
                 num_layers,
                 patch_size,
                 mask_ratio=0.75,
                 layer_norm=nn.LayerNorm):
        
        super().__init__()
        
        self.patch_size = patch_size

        # Layers/Networks
        self.input_layer = PatchEmbed(img_dim, patch_size, num_channels, embed_dim)
        self.backbone = nn.Sequential(*[Block(embed_dim, num_heads, hidden_dim_ratio,
                                              qkv_bias=True, qk_scale=None, norm_layer=layer_norm)
                                        for _ in range(num_layers)])
        
        self.norm = layer_norm(embed_dim)
        self.mask_ratio = mask_ratio
        
        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1 ,embed_dim))
        
        # Facebook does a resize of this encodings
        # TODO: see if we need to change it
        self.num_patches = self.patch_embed.num_patches
        pos_embedding = positional_emb_sin_cos(self.num_patches, embed_dim)
        self.register_buffer('pos_embedding', pos_embedding, persistent=False)

        self.initialize_weights()
    
    def _init_weights(self):
        # this one is an struggle to code
        # Facebook does a lot of things that seem random.
        # Maybe they come from DEiT or BEiT
        # TODO: see exactly what of the Facebook code is in original/DEiT/BEiT paper
        pass
    
    def mask_rand(self, x):
        """Random mask patches

        Args:
            x (torch.Tensor): [Batch, Num Tokens, Embed Dim]
            ratio (float, optional): _description_. Defaults to 0.75.

        Returns:
            _type_: _description_
        """
        B, T, E = x.shape
        random_values = torch.rand(B, T, device=x.device)
        
        num_keep = int(T * (1 - self.mask_ratio))

        token_perm = torch.rand(x.shape[:2], device=x.device).argsort(1)
        undo_token_perm = torch.argsort(token_perm, dim=1) # get back indices
        token_perm = token_perm[:, :num_keep]
        token_perm.unsqueeze_(-1)
        token_perm = token_perm.repeat(1, 1, E)  # reformat this for the gather operation

        x_masked = x.gather(1, token_perm)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones(x.shape[:2], device=x.device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, dim=1, index=undo_token_perm)
        
        return x_masked, mask, undo_token_perm
    
    def forward(self, x):
        # Patchify and embed
        x = self.input_layer(x)
        
        # add positional emb (skiping cls)
        x = x + self.pos_embedding[:, 1:]
        
        # random mask
        x, mask, undo_token_perm = self.mask_rand(x)
        
        # Add cls token + positional
        cls_token = self.cls_token + self.pos_embedding[:, 0]
        cls_token = cls_token.repeat(x.shape[0], 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        
        x = self.backbone(x)
        x = self.norm(x)
        
        return x, mask, undo_token_perm
        

class MAEDecoderViT(nn.Module):
    def __init__(self,
                 embed_dim,
                 hidden_dim_ratio,
                 num_channels,
                 num_heads,
                 num_layers,
                 patch_size,
                 enc_embed_dim,
                 num_patches,
                 layer_norm=nn.LayerNorm
                 ):
        super().__init__()

        # Layers/Networks
        self.input_layer = nn.Linear(enc_embed_dim, embed_dim)
        self.neck = nn.Sequential(*[Block(embed_dim, num_heads, hidden_dim_ratio,
                                            qkv_bias=True, qk_scale=None, norm_layer=layer_norm)
                                    for _ in range(num_layers)])
        
        self.norm = layer_norm(embed_dim)
        self.num_patches = num_patches
        # Parameters/Embeddings
        self.mask_token = nn.Parameter(torch.zeros(1, 1 ,embed_dim))
        
        pos_embedding = positional_emb_sin_cos(num_patches, embed_dim)
        self.register_buffer('pos_embedding', pos_embedding, persistent=False)
        
        # Since we have lower dimentional token we need to recover original dimensionality
        self.pred_proj = nn.Linear(embed_dim, patch_size**2 * num_channels)
        
        self.initialize_weights()
    
    def _init_weights(self):
        # this one is an struggle to code
        # Facebook does a lot of things that seem random.
        # Maybe they come from DEiT or BEiT
        # TODO: see exactly what of the Facebook code is in original/DEiT/BEiT paper
        pass
    
    def forward(self, x, undo_token_perm):
        # Patchify and embed
        x = self.input_layer(x)
        
        # Add mask token + positional
        mask_token = self.mask_toke.repeat(x.shape[0],
                                           self.num_patches-(x.shape[1]-1),
                                           1)
        # concat all excluding 
        x_mask = torch.cat([x[:, 1:], mask_token], dim=1)
        
        # recover order
        # apply same permutation to all elements in the same token
        undo_token_perm = undo_token_perm.unsqueeze(-1).repeat(1, 1, x.shape[2])
        x_mask = x_mask.gather(1,undo_token_perm)
        
        # need to put back cls in order to use Block
        x = torch.cat([x[:, 0], x_mask], dim=1)
        
        # add positional emb 
        x = x + self.pos_embedding
        
        x = self.neck(x)
        x = self.norm(x)
        
        # Recover dimensionality
        x = self.pred_proj(x)
        # To test recovery we dont need cls token
        x = x[:, 1:]
        
        return x

class MAEPretainViT(nn.Module):
    def __init__(self,
                 img_dim,
                 num_channels,
                 enc_embed_dim,
                 enc_hidden_dim_ratio,
                 enc_num_heads,
                 enc_num_layers,
                 patch_size,
                 dec_embed_dim,
                 dec_hidden_dim_ratio,
                 dec_num_heads,
                 dec_num_layers,
                 mask_ratio=0.75,
                 norm1=nn.LayerNorm,
                 norm2=nn.LayerNorm):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            img_dim - Suppose that image is square 
            hidden_dim - Dimensionality ratio of the hidden layer in the feed-forward networks
                         within the Transformer wrt to the input tokens
            num_channels - Number of channels of the input (3 for RGB)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            layer_norm - Type of layer norm (Maybe remove param?????)
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()
        
        self.patch_size = patch_size
        
        self.encoder = MAEBackboneViT(embed_dim=enc_embed_dim,
                                    img_dim=img_dim,
                                    hidden_dim_ratio=enc_hidden_dim_ratio,
                                    num_channels=num_channels,
                                    num_heads=enc_num_heads,
                                    num_layers=enc_num_layers,
                                    patch_size=patch_size,
                                    mask_ratio=mask_ratio,
                                    layer_norm=norm1)
        
        self.decoder = MAEDecoderViT(embed_dim=dec_embed_dim,
                                    hidden_dim_ratio=dec_hidden_dim_ratio,
                                    num_channels=num_channels,
                                    num_heads=dec_num_heads,
                                    num_layers=dec_num_layers,
                                    patch_size=patch_size,
                                    enc_embed_dim=enc_embed_dim,
                                    num_patches=self.encoder.num_patches,
                                    layer_norm=norm2
                                    )

        self.initialize_weights()
        
        
    def _init_weights(self):
        # this one is an struggle to code
        # Facebook does a lot of things that seem random.
        # Maybe they come from DEiT or BEiT
        # TODO: see exactly what of the Facebook code is in original/DEiT/BEiT paper
        pass
    
    
    def forward(self, x):
        x, mask, undo_token_perm = self.encoder(x)
        x = self.decoder(x, undo_token_perm)
                
        return x, mask


    # mean squared error in the pixel space
    # calculated only on masked patches
    @staticmethod
    def loss(targets, pred, mask, norm_tar=True):
        """
        targets [B, 3, H, W]
        pred [B, T, E]
        mask [B, T], 0 is keep, 1 is remove, 
        """
        patch_size = int(np.sqrt(pred.shape[-1]/3))
        targets = img_to_patch(targets, patch_size)
        
        if norm_tar:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-5)**.5
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  

        loss = (loss * mask).sum() / mask.sum()  
        return loss
        
        
