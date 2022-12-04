import argparse
from dataloader import get_finetune_dataloaders
from timm.models.vision_transformer import VisionTransformer
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from functools import partial
from typing import Iterable, Optional
from utils import adjust_learning_rate, param_groups_lrd 
import time
import torch
import torch.nn as nn
import os
import math
import sys
from collections import OrderedDict
import json
from timm.models.layers import trunc_normal_


DATA_DIR = './tiny-imagenet-200'
MODELS_DIR = "./models/"

def finetune_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler,
    mixup: Optional[Mixup],
    print_frequency: int,
    args
):
    model.train(True)
     # Sets the gradients of all optimized :class:`torch.Tensor` s to zero
    losses = []
    t0 = time.time()
    for iter, (samples, targets) in enumerate(data_loader):
        
        lr = adjust_learning_rate(optimizer, iter / len(data_loader) + epoch, args)
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup is not None:
            samples, targets = mixup(samples, targets)

    
        optimizer.zero_grad()
        
        # mixed precision for forward & loss
        # not recommended for backwards pass
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)
    

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_value = loss.item()
        losses.append(loss_value)

        if not math.isfinite(loss_value):
            print("Loss is {}, exiting".format(loss_value))
            sys.exit(1)
        if iter % print_frequency == 0:
            print(f'loss value in epoch {epoch}, step {iter}: {round(loss_value, 5)} with learning rate {round(lr, 7)}')   

    t1 = time.time()
    print(f"Epoch {epoch} took {round(t1-t0, 2)} seconds.")
    return losses

if __name__ == "__main__":    
    
    # options for training
    parser = argparse.ArgumentParser()

    # default parameter setting for Vit-B
    parser.add_argument('--epoch_count',  type=int, default=350, help='epoch_count')
    parser.add_argument('--exp_name', type=str, default="pretrain_test", help='Name of the experiment, for tracking purposes')
    parser.add_argument('--nb_classes', default=200, type=int, help='number of the classification types')
    parser.add_argument('--batch_size',  type=int, default=128, help='batch size')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')

    # Mixup
    parser.add_argument('--mixup', type=float, default=0, help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0, help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None, help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0, help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5, help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch', help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--learning_rate', type=float, default=1e-3, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75, help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--min_learning_rate', type=float, default=1e-6, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N', help='epochs to warmup LR')

    # Augmentation
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT', help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME', help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    
    # Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT', help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')
    opt = parser.parse_args()

    # load pre-trained model
    model_dir = os.path.join(MODELS_DIR, opt.exp_name)
    args_pre = json.load(open(os.path.join(model_dir, "mae_args.json"), "r"))
    checkpoint_model = torch.load(os.path.join(model_dir, "mae.pt"))

    train_loader_pretrain, val_loader_pretrain = get_finetune_dataloaders(DATA_DIR, opt.batch_size, args_pre["img_dim"], opt, use_cuda=True)
    
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

    mixup = None
    mixup_active = opt.mixup > 0 or opt.cutmix > 0. or opt.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=opt.mixup, cutmix_alpha=opt.cutmix, cutmix_minmax=opt.cutmix_minmax,
            prob=opt.mixup_prob, switch_prob=opt.mixup_switch_prob, mode=opt.mixup_mode,
            label_smoothing=opt.smoothing, num_classes=opt.nb_classes)

    model = VisionTransformer(
            img_size=args_pre["img_dim"],
            patch_size=args_pre["patch_size"],
            in_chans=3,
            num_classes=opt.nb_classes,
            embed_dim=args_pre["embed_dim"],
            depth=args_pre["num_layers"],
            num_heads=args_pre["num_heads"],
            mlp_ratio=args_pre["hidden_dim_ratio"],
            global_pool = 'avg',
            drop_path_rate=opt.drop_path
        )
    
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    # manually initialize fc layer [Not said in paper]
    trunc_normal_(model.head.weight, std=2e-5)
    
    model.to(device)

    param_groups = param_groups_lrd(model, opt.weight_decay,
        no_weight_decay_list=model.no_weight_decay(),
        layer_decay=opt.layer_decay
    )

    optimizer = torch.optim.AdamW(param_groups, lr=opt.learning_rate)
    loss_scaler = torch.cuda.amp.GradScaler(enabled=True)

    criterion = torch.nn.CrossEntropyLoss()
    if mixup is not None:
        criterion = SoftTargetCrossEntropy()

    for epoch in range(opt.epoch_count):
        tloss = finetune_epoch(
            model=model,
            criterion=criterion,
            data_loader=train_loader_pretrain,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scaler=loss_scaler,
            mixup=mixup,
            print_frequency=100,
            args=opt)

