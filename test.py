from maemodel import *
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

model = VisionTransformer(
            img_size=64,
            patch_size=16,
            in_chans=3,
            num_classes=10,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
        )

#print(model)

our_model = MAEPretainViT(
        img_dim=64,
        num_channels=3,
        patch_size=16,
        enc_embed_dim=768,
        enc_hidden_dim_ratio=4.,
        enc_num_heads=12,
        enc_num_layers=12,
        dec_embed_dim=512,
        dec_hidden_dim_ratio=4.,
        dec_num_heads=16,
        dec_num_layers=8,
        mask_ratio=.75,
        norm1=partial(nn.LayerNorm, eps=1e-6),
        norm2=partial(nn.LayerNorm, eps=1e-6)
    )


model_translation = {
    "input_layer": "patch_embed",
    "backbone": "blocks",
    "pos_embedding": "pos_embed",
    "norm": "norm" # depending on the timm implementation has a different name
}


checkpoint_model = our_model.state_dict()
all_keys = list(checkpoint_model.keys())
new_dict = OrderedDict()

for key in all_keys:
    if key.startswith('encoder.'):
        key_new = key[8:]
        for mae_name, vit_name in model_translation.items():
            key_new = key_new.replace(mae_name, vit_name)
            
        new_dict[key_new] = checkpoint_model[key]
        
    elif key.startswith('decoder.'):
        print(f"Removing key {key} from pretrained checkpoint")
        del checkpoint_model[key]
    else:
        new_dict[key] = checkpoint_model[key]
        
checkpoint_model = new_dict
print("="*100)

msg = model.load_state_dict(checkpoint_model, strict=False)
print(msg)