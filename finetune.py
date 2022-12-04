import argparse
from dataloader import get_pretrain_dataloaders
from maemodel import MAEPretainViT

from timm.models.vision_transformer import VisionTransformer
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import os
import json

DATA_DIR = './tiny-imagenet-200'
MODELS_DIR = "./models/"

if __name__ == "__main__":    
    
    # options for training
    parser = argparse.ArgumentParser()

    # default parameter setting for Vit-B
    parser.add_argument('--epoch_count',  type=int, default=350, help='epoch_count')
    parser.add_argument('--exp_name', type=str, default="pretrain_test", help='Name of the experiment, for tracking purposes')
    parser.add_argument('--nb_classes', default=200, type=int, help='number of the classification types')
    parser.add_argument('--batch_size',  type=int, default=128, help='batch size')
    opt = parser.parse_args()

    train_loader_pretrain, val_loader_pretrain = get_pretrain_dataloaders(DATA_DIR, opt.batch_size, imgsz=64, use_cuda=True)

    # load pre-trained model
    model_dir = os.path.join(MODELS_DIR, opt.exp_name)
    args_pre = json.load(open(os.path.join(model_dir, "mae_args.json"), "r"))
    checkpoint_model = torch.load(os.path.join(model_dir, "mae.pt"))
    
    model_translation = {
    "input_layer": "patch_embed",
    "backbone": "blocks",
    "pos_embedding": "pos_embed",
    }
    
    all_keys = list(checkpoint_model.keys())
    new_dict = OrderedDict()
        
    for key in all_keys:
        if key.startswith('encoder.'):
            key_new = key[8:]
            
            if key_new.startswith("norm"):
                print(f"Removing key {key} from pretrained checkpoint")
                del checkpoint_model[key]
                
            else:
                for mae_name, vit_name in model_translation.items():
                    key_new = key_new.replace(mae_name, vit_name)
                    
                new_dict[key_new] = checkpoint_model[key]
            
        elif key.startswith('decoder.'):
            print(f"Removing key {key} from pretrained checkpoint")
            del checkpoint_model[key]
            
        else:
            new_dict[key] = checkpoint_model[key]
            
    checkpoint_model = new_dict

    device = torch.device('cuda')

    # CHECK: do we need to interpolate position embeddings for higher resolution?

    model = VisionTransformer(
            img_size=args_pre["img_dim"],
            patch_size=args_pre["patch_size"],
            in_chans=3,
            num_classes=opt.nb_classes,
            embed_dim=args_pre["embed_dim"],
            depth=args_pre["num_layers"],
            num_heads=args_pre["num_heads"],
            mlp_ratio=args_pre["hidden_dim_ratio"],
            global_pool = 'avg'
        )

    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    
    criterion = torch.nn.CrossEntropyLoss()

