from maemodel import *
import numpy as np

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
set_seed(42)


imgs = torch.rand((1, 3, 64, 64))
set_seed(42)
model_facebook = MaskedAutoencoderViT(img_size=(64,64),
    patch_size=16, embed_dim=768, depth=12, num_heads=12,
    decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
    mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))

set_seed(42)
loss_1, pred_1, mask_1 = model_facebook.forward(imgs, mask_ratio=0.75)
print(loss_1)

set_seed(42)
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

set_seed(42)
pred_2, mask_2 = our_model.forward(imgs)
loss_2 = our_model.loss(imgs, pred_2, mask_2, norm_tar=False)
print(loss_2)
print(mask_1)
print(torch.abs(loss_1-loss_2))
