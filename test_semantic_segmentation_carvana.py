
# In[0]:

# importing libraries
# I'm encapsulating the bulky code in lib.py and models.py
from lib_semantic_segmentation import *
from models_semantic_segmentation import *
import custom_datasets
import detectors
import os

# In[1]:

# loading data
dataset = lambda root, train, download, **kwargs: \
    custom_datasets.CarvanaDataset(
        image_dir=os.path.join(root, 'carvana', 'train' if train else 'val'), 
        mask_dir=os.path.join(root, 'carvana', 'train_masks' if train else 'val_masks'), 
        **kwargs
    )
#IMG_HEI, IMG_WID = 160, 240
IMG_HEI, IMG_WID = 320, 480
traintest_transforms = get_transforms(
    [
        transforms.ToTensor(),
        transforms.Resize((IMG_HEI, IMG_WID)), 
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=35),
        transforms.Normalize(
            mean=[0.],
            std=[1.]
        )
    ], [
        transforms.ToTensor(),
        transforms.Resize((IMG_HEI, IMG_WID)),
        transforms.Normalize(
            mean=[0.],
            std=[1.]
        )
    ], to_tensor=False)
train_data, test_data = fetch_dataset(dataset, *traintest_transforms)
#train_loader, test_loader = get_dataloader(train_data, test_data, batch_size=16)
train_loader, test_loader = get_dataloader(train_data, test_data, batch_size=8)

# In[]:
data = next(iter(train_loader))
print(data[0].shape, data[1].shape)

# In[2]:

# setting up devices
setup_device(to_print=True)
N_CLASSES = 10
import timm
model = UNet(in_channels=3, out_channels=1).to(get_device())

# In[]:
loss_fn, opt, scheduler = set_optimizers(model, nn.BCEWithLogitsLoss, torch.optim.Adam, lr=1E-4)#lr = .1
#loss_fn, opt, scheduler = set_optimizers(model, nn.BCEWithLogitsLoss, torch.optim.Adam, lr=1E-4, 
#                                         decay=torch.optim.lr_scheduler.StepLR, step_size=5, gamma=.5)#lr = .1
#loss_fn, opt, scheduler = set_optimizers(model, nn.BCEWithLogitsLoss, torch.optim.Adam, lr=1E-5, 
#                                         decay=torch.optim.lr_scheduler.ReduceLROnPlateau, patience=5)#lr = .1
print(model)
model.to(get_device())

# In[]:
from torchsummary import summary
print(next(iter(train_loader))[0].shape)
summary(model, next(iter(train_loader))[0].shape[1:])

# In[3]:

import lib_semantic_segmentation as lb
# Total training epochs
epochs = 10
#epochs = 100
training_losses = []
training_accuracy = []
testing_accuracy = []

# In[5]:

import importlib
import lib
importlib.reload(lib)

# In[]:

import importlib
import lib_semantic_segmentation as lb
importlib.reload(lb)
# lb.imshow_test(model, test_loader)
        
# In[]:
model2 = model

# In[8]:

# *for MNIST n_channels=1.
#model2 = Conv(n_channels=3).to(get_device())
lib.load_model(model2, 'models/sem_segmentation/1703926149/model_1703930047.h5')

# In[9]:

setup_device()
print(test(test_loader, model2, loss_fn))


# In[]:
# Test Set Tests
lb.imshow_test(model2, test_loader)

# In[11]:

# Train Augmented Tests
import lib
model2.eval()
test_loss, correct = 0, 0

show_original=False
show_reference=True
# Turn off gradient descent
with torch.no_grad():
    for X, y in train_loader:
        X, y = X.to(get_device()), y.to(get_device())
        pred = model2(X)
        pred = torch.sigmoid(pred)
        pred = (pred>.5).float()

        if show_original:
            lib.imshow(X.cpu().numpy(), range=range(3))
        lib.imshow(pred.cpu().numpy(), range=range(3))
        if show_reference:
            lib.imshow(y.cpu().numpy(), range=range(3))
        
        break
        
# In[11]:

# Intermediate activations
import lib
model2.eval()
test_loss, correct = 0, 0

show_original=False
show_reference=True
activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

I = 2
def to_img(name, activation):
    return np.expand_dims(activation[name].cpu().numpy()[I],axis=1)

# Turn off gradient descent
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(get_device()), y.to(get_device())
        model2.bottleneck.register_forward_hook(get_activation('bottleneck'))
        model2.downs[1].register_forward_hook(get_activation('downs1'))
        model2.ups[1].register_forward_hook(get_activation('ups1'))
        pred = model2(X)
        pred = torch.sigmoid(pred)
        pred = (pred>.5).float()

        if show_original:
            lib.imshow(X.cpu().numpy(), range=range(3))
        lib.imshow(pred.cpu().numpy(), range=range(3))
        lib.imshow(to_img('downs1', activation), range=range(3))
        lib.imshow(to_img('bottleneck', activation), range=range(3))
        lib.imshow(to_img('ups1', activation), range=range(3))
        if show_reference:
            lib.imshow(y.cpu().numpy(), range=range(3))
        
        break

# In[11]:

# Show filters
import lib
importlib.reload(lib)
model2.eval()
test_loss, correct = 0, 0

show_original=False
show_reference=True

I = 2
def to_img2(actv):
    return np.expand_dims(actv.cpu().numpy()[I],axis=1)


setup_device()
# Turn off gradient descent
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(get_device()), y.to(get_device())
        pred = model2(X)
        pred = torch.sigmoid(pred)
        pred = (pred>.5).float()

        if show_original:
            lib.imshow(X.cpu().numpy(), range=range(3))
        lib.imshow(pred.cpu().numpy(), range=range(3))
        filt = model2.downs[1].conv[3].weight.data.clone()
        lib.imshow(to_img2(filt), range=range(9), colorbar=False)
        filt = model2.ups[1].conv[3].weight.data.clone()
        lib.imshow(to_img2(filt), range=range(9), colorbar=False)
        print(filt.shape)
        if show_reference:
            lib.imshow(y.cpu().numpy(), range=range(3))
        
        break

# %%
