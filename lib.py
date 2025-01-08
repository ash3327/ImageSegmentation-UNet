import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from torch import nn

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

def imshow(X, range=range(9), labels:list[str]=None, colorbar=True, a=None, b=None):
    if labels == None:
        labels = [None for i in range]

    for i, j in enumerate(range):
        img = X[j]
        #img = img / 2 + 0.5     # unnormalize
        
        plt.subplot(331 + i)
        
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
        elif a is not None:
            img = img[a:b,:,:]
        A = plt.imshow(np.transpose(img, (1, 2, 0)))
        if colorbar:
            plt.colorbar(A)
        else:
            plt.axis('off')
        plt.title(labels[i])
    plt.tight_layout()
    plt.show()

def get_transforms(train_transform_stack, test_transform_stack, to_tensor=True):
    
    if to_tensor:
        train_transform_stack += [transforms.ToTensor()]
        test_transform_stack += [transforms.ToTensor()]
    # Train transformation
    train_transform = transforms.Compose(
        train_transform_stack
    )
    # Test transformation
    test_transform = transforms.Compose(
        test_transform_stack
    )
    return train_transform, test_transform

def one_hot(x, n_classes):
    x = np.array(x)
    #print(x.shape,  torch.tensor(x, dtype=torch.int64))
    return torch.zeros((n_classes,*x.shape), dtype=torch.float).scatter_(0, torch.tensor(x, dtype=torch.int64).unsqueeze(0), value=1)
    #return F.one_hot(torch.tensor(np.array(x), dtype=torch.int64).long(), num_classes=n_classes)

def get_transforms2(tr_stack, te_stack, 
                    train_transform_stack, test_transform_stack,
                    n_classes, to_tensor=True):
    
    if to_tensor:
        tr_stack = [lambda x:one_hot(x,n_classes)]+tr_stack
        train_transform_stack = [transforms.ToTensor()]+train_transform_stack
        te_stack = [lambda x:one_hot(x,n_classes)]+te_stack
        test_transform_stack = [transforms.ToTensor()]+test_transform_stack
    
    
    train_transform = transforms.Compose(train_transform_stack)#+[transforms.Normalize(mean=[0.],std=[1.]),]
    test_transform = transforms.Compose(test_transform_stack)


    
    trtarg = transforms.Compose(tr_stack)
    tetarg = transforms.Compose(te_stack)
    return train_transform, trtarg, test_transform, tetarg

def fetch_dataset(dataset=datasets.CIFAR10, train_transform=None, test_transform=None):
    """
    EXAMPLE: datasets.MNIST
    """
    # download dataset
    train_data = dataset(
        root="data",
        train=True,
        download=True,
        transform=train_transform,
    )

    test_data = dataset(
        root="data",
        train=False,
        download=True,
        transform=test_transform,
    )
    return train_data, test_data

def fetch_dataset2(dataset=datasets.CIFAR10, 
                   train_transform=None, train_transform_targ=None,
                   test_transform=None, test_transform_targ=None):
    """
    EXAMPLE: datasets.MNIST
    """
    # download dataset
    train_data = dataset(
        root="data",
        train=True,
        download=True,
        transform=train_transform,
        target_transform=train_transform_targ
    )

    test_data = dataset(
        root="data",
        train=False,
        download=True,
        transform=test_transform,
        target_transform=test_transform_targ
    )
    return train_data, test_data

def get_dataloader(train_data, test_data, batch_size=256, **kwargs):
    # create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_data, batch_size=batch_size, **kwargs)
    return train_loader, test_loader

global device
device = None

def setup_device(to_print:bool=False):
    # Get cpu or gpu device for training.
    global device    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if to_print:
        print("Using {} device".format(device))

def get_device():
    """You need to setup with setup_device() first."""
    return device

def set_optimizers(model:nn.Module, loss_fn=nn.CrossEntropyLoss, optimizer=torch.optim.SGD, lr=1e-1, decay=None, **kwargs):
    # Loss function
    loss_fn = loss_fn()

    # SGD Optimizer
    optimizer = optimizer(model.parameters(), lr=lr)
    scheduler = decay(optimizer, **kwargs) if decay != None else None
    return loss_fn, optimizer, scheduler

def text_to_file(JOINED_PATH:str, contents, mode:str='w'):
    with open(JOINED_PATH, mode) as f:
        print(contents, file=f)


def save_model(model:nn.Module, PATH:str='models', FILENAME:str=None, extra_info:str=""):
    if FILENAME == None:
        import time
        FILENAME = f'model_{int(time.time())}.h5'

    import os
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    PATH = os.path.join(PATH, FILENAME)
    torch.save(model.state_dict(), PATH)
    
    text_to_file(PATH+'_details.txt', str(model)+'\n'+extra_info)
    
def load_model(model:nn.Module, FILE_PATH:str):
    model.load_state_dict(torch.load(FILE_PATH))