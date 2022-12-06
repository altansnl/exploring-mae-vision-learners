from dmaemodel import *
from dataloader import get_pretrain_dataloaders
from utils import adjust_learning_rate, save_images_tensors, set_seed
from functools import partial
import torch.nn as nn
import argparse
import torch
from typing import Iterable
import timm.optim.optim_factory as optim_factory
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
import os
import time
import json
from collections import OrderedDict

DATA_DIR = './tiny-imagenet-200'
try:
    os.mkdir(DATA_DIR)
except FileExistsError:
    pass
RESULTS_DIR = "./results/pretraining"
try:
    os.mkdir("./results")
    os.mkdir(RESULTS_DIR)
except FileExistsError:
    pass
MODELS_DIR = "./models"
try:
    os.mkdir(MODELS_DIR)
except FileExistsError:
    pass


def pretrain_epoch(
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    current_epoch: int,
    print_frequency: int,
    args
):
    student_model.train(True)
    losses = []
    t0 = time.time()
    for iter, (samples, _) in enumerate(data_loader):
        lr = adjust_learning_rate(optimizer, iter / len(data_loader) + current_epoch, args)
        samples = samples.to(device, non_blocking=True)
        
        # Facebook accumulates gradients of various steps
        # We will try without it for simplicity
        optimizer.zero_grad()
        
        # mixed precision for forward & loss
        # not recommended for backwards pass
        with torch.cuda.amp.autocast():
            x, mask, student, teacher = student_model.forward(samples, teacher_model)
            loss = student_model.loss(samples, x, mask, teacher, student, alpha=args.alpha)


        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_value = loss.item()
        losses.append(loss_value)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # not sure if needed
        # I think this is just for multi-thread gpu 
        # torch.cuda.synchronize()
        if iter % print_frequency == 0:
            print(f'loss value in epoch {current_epoch}, step {iter}: {round(loss_value, 5)} with learning rate {round(lr, 7)}')

    t1 = time.time()
    print(f"Epoch {current_epoch} took {round(t1-t0, 2)} seconds.")
    return losses


def validate_epoch(
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    current_epoch: int,
    args,
    save_imgs = True
):
    student_model.eval()
    losses = []
    for iter, (samples, _) in enumerate(data_loader):
        samples = samples.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            x, mask, student, teacher = student_model.forward(samples, teacher_model)
            loss = student_model.loss(samples, x, mask, teacher, student, alpha=args.alpha)


        loss_value = loss.item()
        losses.append(loss_value)

    if save_imgs:
        result_dir =  os.path.join(RESULTS_DIR, args.exp_name)
        try:
            os.mkdir(result_dir)
        except FileExistsError:
            pass
        save_images_tensors(samples, x, mask, args.patch_size, result_dir, str(current_epoch))
    avg_loss = sum(losses)/len(losses)
    print(f"Avg. validation loss for epoch {current_epoch} was {round(avg_loss, 5)}.")
    return losses, avg_loss



if __name__ == "__main__":

    # options for training
    parser = argparse.ArgumentParser()

    # default parameter setting for Vit-B
    parser.add_argument('--decoder_embed_dim',  type=int, default=512, help='decoder embedding dimensionality')
    parser.add_argument('--decoder_hidden_dim_ratio',  type=float, default=4., help='decoder hidden dimension ratio') 
    parser.add_argument('--decoder_num_heads',  type=int, default=16, help='decoder number of heads')
    parser.add_argument('--decoder_num_layers',  type=int, default=8, help='number of layers in the decoder')
    parser.add_argument('--mask_ratio',  type=float, default=.75, help='mask ratio')
    parser.add_argument('--batch_size',  type=int, default=2, help='batch size')
    parser.add_argument('--epoch_count',  type=int, default=500, help='epoch_count')
    parser.add_argument('--warmup_epochs',  type=int, default=50, help='warmup epochs')
    parser.add_argument('--learning_rate', type=float, default=1.5e-4, help='learning rate')
    parser.add_argument('--min_learning_rate', type=float, default=0., help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--exp_name', type=str, default="pretrain_test", help='Name of the experiment, for tracking purposes')
    parser.add_argument('--alpha', type=float, default=1.0, help='Destilation parameter')

    opt = parser.parse_args()

    # Seeding
    set_seed(42)

    # Fixed ViT-L params
    teacher_model = MAEBackboneViT_Teach(
            embed_dim=1024,
            img_dim=224,
            hidden_dim_ratio=4.,
            num_channels=3,
            num_heads=16,
            num_layers=18,
            patch_size=16,
            num_patches=64,
            mask_ratio=opt.mask_ratio,
            layer_norm=partial(nn.LayerNorm, eps=1e-6)
            )

    checkpoint_model = torch.load(os.path.join("./models/pretrain_large", "mae_pretrain_vit_large.pth"))["model"]

    model_trans = {
        "blocks": "backbone",
        "patch_embed": "input_layer"
    }

    all_keys = list(checkpoint_model.keys())
    new_dict = OrderedDict()
    for key in all_keys:
        key_new = key
        for vit, mae in model_trans.items():
            key_new = key_new.replace(vit, mae)
        new_dict[key_new] = checkpoint_model[key]

    checkpoint_model = new_dict

    msg = teacher_model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    # Fixed ViT-B params
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
                    dec_embed_dim=opt.decoder_embed_dim,
                    dec_hidden_dim_ratio=opt.decoder_hidden_dim_ratio,
                    dec_num_heads=opt.decoder_num_heads,
                    dec_num_layers=opt.decoder_num_layers,
                    norm1=partial(nn.LayerNorm, eps=1e-6),
                    norm2=partial(nn.LayerNorm, eps=1e-6))


    train_loader_pretrain, val_loader_pretrain = get_pretrain_dataloaders(DATA_DIR, opt.batch_size, imgsz=64, use_cuda=True)
    device = torch.device('cuda')
    student_model.to(device)
    teacher_model.to(device)
    
    for param in teacher_model.parameters():
        param.requires_grad = False
        
    teacher_model.eval()
    param_groups = optim_factory.param_groups_weight_decay(student_model, opt.weight_decay)
    epoch_count = opt.epoch_count
    lr = opt.learning_rate * opt.batch_size / 256
    opt.learning_rate = lr
    optimizer = torch.optim.AdamW(param_groups, lr=opt.learning_rate, betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    train_losses = []
    valid_losses = []
    best_val_loss = 1000
    
    for epoch in range(epoch_count):
        tloss = pretrain_epoch(student_model, teacher_model, train_loader_pretrain, optimizer, scaler, device, epoch, 100, opt)
        vloss, avg_loss = validate_epoch(student_model, teacher_model, val_loader_pretrain, device, epoch, opt)  # remove
        # Saving the best model along with its hyperparameters
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            model_dir = os.path.join(MODELS_DIR, opt.exp_name)
            try:
                os.mkdir(model_dir)
            except FileExistsError:
                pass
            
            torch.save(student_model.state_dict(), os.path.join(model_dir, "mae.pt"))
            
            with open(os.path.join(model_dir, 'mae_args.json'), 'w') as f:
                dictionary_args = opt.__dict__
                dictionary_args["current_epoch"] = epoch
                dictionary_args["validation_loss"] = avg_loss
                json.dump(dictionary_args, f, indent=2)
                
        # Saving check point models
        if epoch % 99 == 0:
            model_dir = os.path.join(MODELS_DIR, opt.exp_name)
            try:
                os.mkdir(model_dir)
            except FileExistsError:
                pass
            
            torch.save(student_model.state_dict(), os.path.join(model_dir, f"mae_{epoch}_epochs.pt"))
            
            with open(os.path.join(model_dir, f'mae_args_{epoch}_epochs.json'), 'w') as f:
                dictionary_args = opt.__dict__
                dictionary_args["current_epoch"] = epoch
                dictionary_args["validation_loss"] = avg_loss
                json.dump(dictionary_args, f, indent=2)
        train_losses += tloss
        valid_losses += vloss
        print()
    
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size

    train_losses = np.convolve(np.array(train_losses), kernel, mode='valid')
    valid_losses = np.convolve(np.array(valid_losses), kernel, mode='valid')
    plt.plot(np.linspace(0, opt.epoch_count, num=len(valid_losses)), valid_losses, label="validation loss", alpha=0.5)
    plt.plot(np.linspace(0, opt.epoch_count, num=len(train_losses)), train_losses, label="train loss", alpha=0.5)
    plt.legend()
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.title("Self-supervised pretraining")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "sst_loss.png"))
    plt.close()

