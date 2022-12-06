from dmaemodel import *
import numpy as np
from functools import partial
from collections import OrderedDict
from utils import set_seed


set_seed(42)

#print(model)

teacher_model = MAEBackboneViT_Teach(
        embed_dim=1024,
        img_dim=224,
        hidden_dim_ratio=4.,
        num_channels=3,
        num_heads=16,
        num_layers=18,
        patch_size=16,
        num_patches=64,
        mask_ratio=0.75,
        layer_norm=partial(nn.LayerNorm, eps=1e-6)
        )

checkpoint_model = torch.load(os.path.join("./models/pretrain_large", "mae_pretrain_vit_large.pth"))["model"]


all_keys = list(checkpoint_model.keys())
new_dict = OrderedDict()
for key in all_keys:
    key_new = key.replace("blocks", "backbone")
    new_dict[key_new] = checkpoint_model[key]

checkpoint_model = new_dict

msg = teacher_model.load_state_dict(checkpoint_model, strict=False)
print(msg)

student_model = DMAEPretainViT(
                 img_dim=64,
                 num_channels=3,
                 enc_embed_dim=768,
                 enc_hidden_dim_ratio=4.,
                 enc_num_heads=12,
                 enc_num_layers=12,
                 extract_lay=9,
                 teacher_dim=1024,
                 patch_size=8,
                 dec_embed_dim=512,
                 dec_hidden_dim_ratio=4.,
                 dec_num_heads=16,
                 dec_num_layers=8,
                 norm1=partial(nn.LayerNorm, eps=1e-6),
                 norm2=partial(nn.LayerNorm, eps=1e-6))


x = torch.rand((4, 3, 64, 64))

x_, mask, student, teacher = student_model.forward(x, teacher_model)
print(DMAEPretainViT.loss(x, x_, mask, teacher, student))

from dataloader import get_finetune_dataloaders
DATA_DIR = './tiny-imagenet-200'

train_loader_finetune, val_loader_finetune = get_finetune_dataloaders(DATA_DIR, 1, 64, opt, use_cuda=True)
