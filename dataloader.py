import os
import matplotlib.pyplot as plt
import numpy as np
from random import randint

from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms as T
from torchvision.utils import make_grid
from torchvision import datasets

DATA_DIR = './tiny-imagenet-200' # Original images come in shapes of [3,64,64]


# Functions to display single or a batch of sample images
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
def show_batch(dataloader):
    for batch in dataloader:
        images, labels = batch
        imshow(make_grid(images)) # Using Torchvision.utils make_grid function
        break
    
def show_image(dataloader):
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    random_num = randint(0, len(images)-1)
    imshow(images[random_num])
    label = labels[random_num]
    print(f'Label: {label}, Shape: {images[random_num].shape}')

# Setup function to create dataloaders for image datasets
def generate_dataloader(data, name, batch_size, transform=None, use_cuda=True):
    if data is None: 
        return None
    
    # Read image files to pytorch dataset using ImageFolder, a generic data 
    # loader where images are in format root/label/filename
    # See https://pytorch.org/vision/stable/datasets.html
    if transform is None:
        dataset = datasets.ImageFolder(data, transform=T.ToTensor())
    else:
        dataset = datasets.ImageFolder(data, transform=transform)

    # Set options for device
    if use_cuda:
        kwargs = {"pin_memory": True, "num_workers": 1}
    else:
        kwargs = {}
    
    # Wrap image dataset (defined above) in dataloader 
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                        shuffle=(name=="train"), 
                        **kwargs)
    
    return dataloader


def get_pretrain_transform(imgsz):
    return T.Compose([
                # T.Resize(256), # Resize images to 256 x 256
                T.RandomResizedCrop(size=(imgsz, imgsz)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),  # Converting cropped images to tensors
                T.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
            ])


def get_pretrain_dataloaders(datadir, batch_size, imgsz=64, use_cuda=True):
    # Define training and validation data paths
    train_dir = os.path.join(datadir, 'train') 
    valid_dir = os.path.join(datadir, 'val')
    transform = get_pretrain_transform(imgsz)

    fp = open(os.path.join(valid_dir, 'val_annotations.txt'), 'r')
    data = fp.readlines()

    # Create dictionary to store img filename (word 0) and corresponding
    # label (word 1) for every line in the txt file (as key value pair)
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    fp.close()

    val_img_dir = os.path.join(valid_dir, 'images')

    for img, folder in val_img_dict.items():
        newpath = (os.path.join(val_img_dir, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(val_img_dir, img)):
            os.rename(os.path.join(val_img_dir, img), os.path.join(newpath, img))

    # Create DataLoaders for pre-trained models (normalized based on specific requirements)
    train_loader_pretrain = generate_dataloader(train_dir, "train", batch_size=batch_size,
                                    transform=transform, use_cuda=use_cuda)

    val_loader_pretrain = generate_dataloader(val_img_dir, "val", batch_size=batch_size,
                                    transform=transform, use_cuda=use_cuda)
    
    return train_loader_pretrain, val_loader_pretrain