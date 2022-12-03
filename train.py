from maemodel import MAEPretainViT
from dataloader import get_pretrain_dataloaders
from utils import adjust_learning_rate
import argparse
import torch
from typing import Iterable
import timm.optim.optim_factory as optim_factory
import math
import sys
import time

DATA_DIR = './tiny-imagenet-200'

def pretrain_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    current_epoch: int,
    print_frequency: int,
    args
):
    model.train(True)
    iter = 0
    t0 = time.time()
    for iteration, (samples, _) in enumerate(data_loader):
        lr = adjust_learning_rate(optimizer, iteration / len(data_loader) + current_epoch, args)

        samples = samples.to(device, non_blocking=True)
        
        # Facebook accumulates gradients of various steps
        # We will try without it for simplicity
        optimizer.zero_grad()
        
        # mixed precision for forward & loss
        # not recommended for backwards pass
        with torch.cuda.amp.autocast():
            x, mask = model.forward(samples)
            loss = model.loss(samples, x, mask)
            
        loss.backward()
        optimizer.step()

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # not sure if needed
        # I think this is just for multi-thread gpu 
        # torch.cuda.synchronize()
        if iter % print_frequency == 0:
            print(f'loss value in epoch {current_epoch}, step {iter}: {round(loss_value, 5)} with learning rate {round(lr, 7)}')
        iter += 1

    t1 = time.time()
    print(f"Epoch {current_epoch} took {round(t1-t0, 2)} seconds.\n")




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
    parser.add_argument('--patch_size',  type=int, default=16, help='patch size')
    parser.add_argument('--decoder_embed_dim',  type=int, default=512, help='decoder embedding dimensionality')
    parser.add_argument('--decoder_hidden_dim_ratio',  type=float, default=4., help='encoder hidden dimension ratio') 
    parser.add_argument('--decoder_num_heads',  type=int, default=16, help='decoder number of heads')
    parser.add_argument('--decoder_num_layers',  type=int, default=8, help='number of layers in the decoder')
    parser.add_argument('--mask_ratio',  type=float, default=.75, help='mask ratio')
    parser.add_argument('--batch_size',  type=int, default=64, help='batch size')
    parser.add_argument('--epoch_count',  type=int, default=64, help='epoch_count')
    parser.add_argument('--warmup_epochs',  type=int, default=6, help='warmup epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--min_learning_rate', type=float, default=0., help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')

    opt = parser.parse_args()

    # initialize the MAE model
    mae = MAEPretainViT(
        img_dim=opt.img_dim,
        num_channels=opt.num_channels,
        enc_embed_dim=opt.embed_dim,
        enc_hidden_dim_ratio=opt.hidden_dim_ratio,
        enc_num_heads=opt.num_heads,
        enc_num_layers=opt.num_layers,
        patch_size=opt.patch_size,
        dec_embed_dim=opt.decoder_embed_dim,
        dec_hidden_dim_ratio=opt.decoder_hidden_dim_ratio,
        dec_num_heads=opt.decoder_num_heads,
        dec_num_layers=opt.decoder_num_layers,
        mask_ratio=opt.mask_ratio
    )

    train_loader_pretrain, val_loader_pretrain = get_pretrain_dataloaders(DATA_DIR, opt.batch_size, imgsz=64, use_cuda=True)
    device = torch.device('cuda')
    mae.to(device)
    param_groups = optim_factory.add_weight_decay(mae, opt.weight_decay)
    epoch_count = opt.epoch_count
    lr = opt.learning_rate * opt.batch_size / 256
    opt.learning_rate = lr
    optimizer = torch.optim.AdamW(param_groups, lr=opt.learning_rate, betas=(0.9, 0.95))

    for epoch in range(epoch_count):
        pretrain_epoch(mae, train_loader_pretrain, optimizer, device, epoch, 200, opt)
