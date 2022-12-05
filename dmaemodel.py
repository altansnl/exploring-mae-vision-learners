import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import VisionTransformer

from timm.models.vision_transformer import PatchEmbed, Block
from utils import *
from maemodel import *

class DMAEBackboneViT(nn.Module):
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
                 teacher_dim,
                 extract_lay,
                 layer_norm=nn.LayerNorm):
        
        super().__init__()
        
        self.patch_size = patch_size
        self.extract_lay = extract_lay

        # Layers/Networks
        self.input_layer = PatchEmbed(img_dim, patch_size, num_channels, embed_dim)
        self.backbone = nn.ModuleList([Block(embed_dim, num_heads, hidden_dim_ratio,
                                              qkv_bias=True, norm_layer=layer_norm)
                                        for _ in range(num_layers)])
        
        self.norm = layer_norm(embed_dim)
        
        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1 ,embed_dim))
        
        # Facebook does a resize of this encodings
        self.num_patches = self.input_layer.num_patches
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches  + 1, embed_dim), requires_grad=False)
        
        self.proj_st_2_teach = nn.Sequential(
            nn.Linear(embed_dim, teacher_dim),
            nn.GELU(),
            nn.Linear(teacher_dim, teacher_dim))

        self.initialize_weights()
    
    def initialize_weights(self):
        
        pos_embedding = get_2d_sincos_pos_embed(self.pos_embedding.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.pos_embedding.data.copy_(torch.from_numpy(pos_embedding).float().unsqueeze(0))
        
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # nn.Conv2d uses He initialization because it is better for ReLU
        # But enbeding layer doesnt have activation -> So we want symetric dist
        # PatchEmbed has layer norm that should fix it rewarless but this way we converge faster
        # Not able to track!!!
        w = self.input_layer.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # The use of truncated come from DEiT saying that it is hard to train (maybe too much exploding grads)
        # The sdt seems to come from BEiT but not sure (possible to track)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            # guess: we use GELU so maybe it is better to use xavier
            # Not able to track!!!
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    
    def forward(self, x:torch.Tensor, teacher_model):
      
        # Patchify and embed
        x_og = x.detach()
        teacher, mask, token_perm, undo_token_perm = teacher_model(x_og)
        
        x = self.input_layer(x)
        
        # add positional emb (skiping cls)
        # print(x.shape, self.pos_embedding[:, 1:].shape, self.pos_embedding.shape)

        # we don not generate pos embedding for cls token in the first place
        x = x + self.pos_embedding[:, 1:]
        
        # random mask
        token_perm = token_perm.repeat(1, 1, x.shape[-1])
        x = x.gather(1, token_perm)
        
        # Add cls token + positional
        cls_token = self.cls_token + self.pos_embedding[:, 0]
        cls_token = cls_token.expand(x.shape[0], -1, -1)
      
        x = torch.cat([cls_token, x], dim=1)
        
        student = None
        for idx, blk in enumerate(self.backbone):
            x = blk(x)
            
            if idx == self.extract_lay-1:
                student = self.proj_st_2_teach(x)
            
        x = self.norm(x)
        
        print(student.shape)
        print(teacher.shape)
        
        
        return x, mask, undo_token_perm, student, teacher
        

class DMAEPretainViT(nn.Module):
    def __init__(self,
                 img_dim,
                 num_channels,
                 enc_embed_dim,
                 enc_hidden_dim_ratio,
                 enc_num_heads,
                 enc_num_layers,
                 extract_lay,
                 teacher_dim,
                 patch_size,
                 dec_embed_dim,
                 dec_hidden_dim_ratio,
                 dec_num_heads,
                 dec_num_layers,
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
        
        self.encoder = DMAEBackboneViT(embed_dim=enc_embed_dim,
                                    img_dim=img_dim,
                                    hidden_dim_ratio=enc_hidden_dim_ratio,
                                    num_channels=num_channels,
                                    num_heads=enc_num_heads,
                                    num_layers=enc_num_layers,
                                    extract_lay=extract_lay,
                                    teacher_dim=teacher_dim,
                                    patch_size=patch_size,
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
        
    
    
    def forward(self, x, teacher_model):
        x, mask, undo_token_perm, student, teacher = self.encoder(x, teacher_model)
        x = self.decoder(x, undo_token_perm)
                
        return x, mask, student, teacher


    # mean squared error in the pixel space
    # calculated only on masked patches
    @staticmethod
    def loss(targets, pred, mask, teach, stud, alpha=1., norm_tar=True):
        """
        targets [B, 3, H, W]
        pred [B, T, E]
        mask [B, T], 0 is keep, 1 is remove, 
        """
        loss_mae = MAEPretainViT.loss(targets, pred, mask, norm_tar)
        loss_dist = nn.L1Loss()(stud, teach)
        
        return loss_mae + alpha*loss_dist
