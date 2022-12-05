from maemodel import *
from dmaemodel import *
import numpy as np
from functools import partial
from collections import OrderedDict

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
set_seed(42)


#print(model)

teacher_model = MAEBackboneViT(
        embed_dim=1024,
        img_dim=64,
        hidden_dim_ratio=4.,
        num_channels=3,
        num_heads=16,
        num_layers=18,
        patch_size=16,
        mask_ratio=0.75,
        layer_norm=partial(nn.LayerNorm, eps=1e-6),
        post_norm=False
    )


student_model = DMAEPretainViT(
                 img_dim=64,
                 num_channels=3,
                 enc_embed_dim=768,
                 enc_hidden_dim_ratio=4.,
                 enc_num_heads=12,
                 enc_num_layers=12,
                 extract_lay=9,
                 teacher_dim=1024,
                 patch_size=16,
                 dec_embed_dim=512,
                 dec_hidden_dim_ratio=4.,
                 dec_num_heads=16,
                 dec_num_layers=8,
                 norm1=partial(nn.LayerNorm, eps=1e-6),
                 norm2=partial(nn.LayerNorm, eps=1e-6))


x = torch.rand((4, 3, 64, 64))

x_, mask, student, teacher = student_model.forward(x, teacher_model)
print(DMAEPretainViT.loss(x, x_, mask, teacher, student))
