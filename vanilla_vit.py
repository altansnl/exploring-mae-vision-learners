import os

## tqdm for loading bars
from tqdm import tqdm

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

import numpy as np
## Imports for plotting
import matplotlib.pyplot as plt

## Torchvision
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms

DATASET_PATH = "./CIFAR10/"
MODEL_PATH = "./models/"
BATCHSZ = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def get_dataloaders():
    test_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
                                        ])
    # For training, we add some augmentation. Networks are too powerful and would overfit.
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.RandomResizedCrop((32,32),scale=(0.8,1.0),ratio=(0.9,1.1)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
                                        ])
    # Loading the training dataset. We need to split it into a training and validation part
    # We need to do a little trick because the validation set should not use the augmentation.
    train_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=train_transform, download=True)
    val_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=test_transform, download=True)
    train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000])
    _, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000])

    # Loading the test set
    test_set = CIFAR10(root=DATASET_PATH, train=False, transform=test_transform, download=True)

    # We define a set of data loaders that we can use for various purposes later.
    train_loader = data.DataLoader(train_set, batch_size=BATCHSZ, shuffle=True, drop_last=True, pin_memory=True, num_workers=0)
    val_loader = data.DataLoader(val_set, batch_size=BATCHSZ, shuffle=False, drop_last=False, num_workers=0)
    test_loader = data.DataLoader(test_set, batch_size=BATCHSZ, shuffle=False, drop_last=False, num_workers=0)
    return train_loader, val_loader, test_loader


def img_to_patch(x: torch.Tensor, patch_size, flatten_channels=True):
    """
    Inputs:
        x - torch.Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5) # [B, H', W', C, p_H, p_W]
    x = x.flatten(1,2)              # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2,4)          # [B, H'*W', C*p_H*p_W]
    return x


class AttentionBlock(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )


    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class VisionTransformer(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, patch_size, num_patches, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels - Number of channels of the input (3 for RGB)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            num_patches - Maximum number of patches an image can have
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.patch_size = patch_size

        # Layers/Networks
        self.input_layer = nn.Linear(num_channels*(patch_size**2), embed_dim)
        self.transformer = nn.Sequential(*[AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.pos_embedding = PositionalEncoding(d_model=embed_dim)


    def forward(self, x: torch.Tensor):
        # Preprocess input
        x = img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.pos_embedding(x)

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # Perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        return out


def train_one_epoch(train_loader, model, optimizer, epoch, device):
    model.train()
    running_loss = 0.0
    p = tqdm(total=len(train_loader), disable=False)
    losses = []
    for i, batch in enumerate(train_loader):
        images, targets = batch
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        loss = calculate_loss([images, targets], model)
        loss.backward()
        optimizer.step()
        running_loss += loss.detach().item()
        p.update(1)
        if i % 200 == 199:    # print every 2000 mini-batches
            losses.append((i + 1, running_loss / 2000))
            running_loss = 0.0
    p.close()
    for i, l in losses:
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {l:.3f}')


def calculate_loss(batch, model):
    imgs, labels = batch
    preds = model(imgs)
    loss = F.cross_entropy(preds, labels)
    # acc = (preds.argmax(dim=-1) == labels).float().mean()
    return loss
    

train_loader, val_loader, test_loader = get_dataloaders()
model = VisionTransformer(embed_dim=256, hidden_dim=512, num_heads=8,
                          num_layers=6, patch_size=4, num_channels=3, num_patches=64,
                          num_classes=10, dropout=0.2)

optimizer = optim.AdamW(model.parameters(), lr=3e-4)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)

model.to(device)


for e in range(20):
    train_one_epoch(train_loader, model, optimizer, e, device)
    lr_scheduler.step()
    model.eval()
    val_acc = 0.0
    for i, batch in enumerate(val_loader):
        imgs, labels = batch
        imgs = imgs.to(device)
        labels = labels.to(device)
        prediction = model(imgs)
        val_acc += (prediction.argmax(dim=-1) == labels).float().mean()
    print(f"Validation accuracy: {val_acc/(i+1)}")
        
torch.save(model.state_dict(), os.path.join(MODEL_PATH, "ViT_cifar10"))






