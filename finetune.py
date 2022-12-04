import argparse
from dataloader import get_pretrain_dataloaders
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

DATA_DIR = './tiny-imagenet-200'
MODELS_DIR = "./models/pretrain_test"

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
    optimizer.zero_grad() # Sets the gradients of all optimized :class:`torch.Tensor` s to zero
    losses = []
    t0 = time.time()
    for iter, (samples, targets) in enumerate(data_loader):

        lr = adjust_learning_rate(optimizer, iter / len(data_loader) + epoch, args)
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # mixed precision for forward & loss
        # not recommended for backwards pass
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)
        
        if mixup is not None:
            samples, targets = mixup(samples, targets)

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

    # Mixup
    parser.add_argument('--mixup', type=float, default=0, help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0, help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None, help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0, help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5, help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch', help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75, help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N', help='epochs to warmup LR')

    # Augmentation
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')

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

    mixup = None
    mixup_active = opt.mixup > 0 or opt.cutmix > 0. or opt.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=opt.mixup, cutmix_alpha=opt.cutmix, cutmix_minmax=opt.cutmix_minmax,
            prob=opt.mixup_prob, switch_prob=opt.mixup_switch_prob, mode=opt.mixup_mode,
            label_smoothing=opt.smoothing, num_classes=opt.nb_classes)

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

    param_groups = param_groups_lrd(model, opt.weight_decay,
        no_weight_decay_list=model.no_weight_decay(),
        layer_decay=opt.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=opt.lr)
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

