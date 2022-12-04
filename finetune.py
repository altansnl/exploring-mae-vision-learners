import argparse
from dataloader import get_pretrain_dataloaders

from timm.models.vision_transformer import VisionTransformer
from functools import partial

import torch
import torch.nn as nn
import os

DATA_DIR = './tiny-imagenet-200'
MODELS_DIR = "./models/pretrain_test"

if __name__ == "__main__":    
    
    # options for training
    parser = argparse.ArgumentParser()

    # default parameter setting for Vit-B
    parser.add_argument('--img_dim',  type=int, default=64, help='image dimensionality')
    parser.add_argument('--num_channels',  type=int, default=3, help='number of channels for the input image')
    parser.add_argument('--embed_dim',  type=int, default=768, help='encoder embedding dimensionality')
    parser.add_argument('--hidden_dim_ratio',  type=float, default=4., help='encoder hidden dimension ratio')   
    parser.add_argument('--num_heads',  type=int, default=12, help='encoder number of heads')
    parser.add_argument('--num_layers',  type=int, default=12, help='number of transformer layers in the encoder')
    parser.add_argument('--patch_size',  type=int, default=8, help='patch size')
    parser.add_argument('--epoch_count',  type=int, default=350, help='epoch_count')
    parser.add_argument('--exp_name', type=str, default="pretrain_test", help='Name of the experiment, for tracking purposes')
    parser.add_argument('--nb_classes', default=200, type=int, help='number of the classification types')
    parser.add_argument('--batch_size',  type=int, default=128, help='batch size')
    opt = parser.parse_args()

    train_loader_pretrain, val_loader_pretrain = get_pretrain_dataloaders(DATA_DIR, opt.batch_size, imgsz=64, use_cuda=True)

    # load pre-trained model
    mae_pretrained = torch.load(os.path.join(MODELS_DIR, "mae"), map_location='cpu')
    mae = mae_pretrained.encoder

    # stop masking in the forward pass
    mae.mask_ratio = 0
    del mae_pretrained
    print(f'loaded model of type {type(mae)}')

    device = torch.device('cuda')

    # CHECK: do we need to interpolate position embeddings for higher resolution?

    model = VisionTransformer(
            img_size=opt.img_dim,
            patch_size=opt.patch_size,
            in_chans=3,
            num_classes=opt.nb_classes,
            embed_dim=opt.embed_dim,
            depth=opt.num_layers,
            num_heads=opt.num_heads,
            mlp_ratio=opt.hidden_dim_ratio,
        )

    criterion = torch.nn.CrossEntropyLoss()

