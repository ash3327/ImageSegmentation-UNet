
# In[0]:

# importing libraries
# I'm encapsulating the bulky code in lib.py and models.py
from lib_semantic_segmentation import *
from models_semantic_segmentation import *
import custom_datasets
import detectors
import os

# In[]
import warnings
warnings.filterwarnings("ignore")

# In[1]:

# loading data
dataset = lambda root, train, download, **kwargs: \
    datasets.Cityscapes(
        root=os.path.join(root, 'cityscapes'), 
        split='train' if train else 'val',
        mode='fine',
        target_type='semantic', 
        **kwargs
    )
IMG_HEI, IMG_WID = 160, 320
N_CLASSES = 34
#IMG_HEI, IMG_WID = 320, 480
G = [
    #transforms.RandomResizedCrop(size=(IMG_HEI, IMG_WID), scale=(.5, 1), interpolation=transforms.InterpolationMode.NEAREST),
    #transforms.RandomHorizontalFlip(p=0.5),
    #transforms.RandomRotation(degrees=35, 
    #                          interpolation=transforms.InterpolationMode.NEAREST),
    transforms.Resize((IMG_HEI, IMG_WID), 
                        interpolation=transforms.InterpolationMode.NEAREST), 
]
G = transforms.Compose(G)
traintest_transforms = get_transforms2(
    [
        G
    ], [
        transforms.Resize((IMG_HEI, IMG_WID), 
                          interpolation=transforms.InterpolationMode.NEAREST),
    ],
    [
        transforms.ColorJitter(brightness=(.5,1.5),contrast=(.8,1.5),saturation=(.5,1.5),hue=(-.1,.1)),
        G
    ], [
        transforms.Resize((IMG_HEI, IMG_WID), 
                          interpolation=transforms.InterpolationMode.NEAREST),
    ]
    , n_classes=N_CLASSES, to_tensor=True)
train_data, test_data = fetch_dataset2(dataset, *traintest_transforms)
#train_loader, test_loader = get_dataloader(train_data, test_data, batch_size=16)
train_loader, test_loader = get_dataloader(train_data, test_data, batch_size=16)

# In[]:
data = next(iter(train_loader))
print(data[0].shape, data[1].shape)

# In[]
a=data[1][0,:,0,0].numpy()
print(a)

# In[2]:

# setting up devices
setup_device(to_print=True)
#N_CLASSES = 10
import timm
#model = UNet(in_channels=3, out_channels=N_CLASSES, block=ResBlock).to(get_device())
model = UNet(in_channels=3, out_channels=N_CLASSES).to(get_device())

# In[]:
loss_fn, opt, scheduler = set_optimizers(model, nn.CrossEntropyLoss, torch.optim.Adam, lr=1E-4,
                                         )#lr = .1
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
import importlib
importlib.reload(lb)
# Total training epochs
epochs = 100
#epochs = 100
training_losses = []
training_accuracy = []
testing_accuracy = []


# In[]:

setup_device(to_print=True)
print(get_device())

# In[]
lb.imshow_test(model, train_loader, show_original=True, show_reference=True, colorbar=False, softmax=True)


# In[6]:
#lb.imshow_test(model, train_loader, show_original=False, show_reference=True, colorbar=False, softmax=True)
lb.imshow_test(model, test_loader, show_original=False, show_reference=True, colorbar=False, softmax=True)

# In[]
test_accuracy = test(test_loader, model, loss_fn, multiclass=True)



# In[]:
import time
THISID = str(int(time.time()))+'_cityscapes'
# In[]:
last_loss = np.inf
for t in range(epochs):
    print('\n', "=" * 15, "Epoch", t + 1, "=" * 15)
    loss, train_accuracy = train(train_loader, model, loss_fn, opt, 
                                 scheduler=scheduler, scaler=torch.cuda.amp.GradScaler(), multiclass=True)
    test_loss, test_accuracy = test(test_loader, model, loss_fn, multiclass=True)
    #lb.imshow_test(model, train_loader, show_original=False, colorbar=False, softmax=True)
    lb.imshow_test(model, test_loader, show_reference=False, show_original=False, colorbar=False, softmax=True)
    training_losses.append(loss)
    training_accuracy.append(train_accuracy.cpu())
    testing_accuracy.append(test_accuracy.cpu())
    if test_loss < last_loss:
        last_loss = test_loss
        save_model(model, PATH=f"models/sem_segmentation/{THISID}", extra_info=f'\nModel for {train_loader.dataset}, \nTest acc: {test_accuracy}, Test loss: {test_loss}')# with test accuracy: {final_test_acc}')

# In[4]:
    
#training_accuracy = train_accuracy
#testing_accuracy = testing_accuracy
plt.plot(training_accuracy, label=f'Train Accuracy\nBest: {max(training_accuracy)}')
plt.plot(testing_accuracy, label=f'Test Accuracy\nBest: {max(testing_accuracy)}')
plt.legend(loc='best')
plt.show()

# In[4]:

final_test_acc = test(test_loader, model, loss_fn)
print(final_test_acc)

# In[5]:

import importlib
import lib
importlib.reload(lib)

# In[6]:
lb.imshow_test(model, test_loader, show_reference=False, softmax=True)
#imgs = next(iter(test_loader))[0].numpy()
#lib.imshow(imgs, range(9))

# In[7]:

save_model(model, PATH='models/sem_segmentation', extra_info=f'\nModel for {train_loader.dataset}')# with test accuracy: {final_test_acc}')



# In[]:

import importlib
import lib_semantic_segmentation as lb
importlib.reload(lb)
lb.imshow_test(model, test_loader, a=24,b=27, show_original=True, colorbar=False)
        
# In[]:
model2 = model

# In[8]:

# *for MNIST n_channels=1.
#model2 = Conv(n_channels=3).to(get_device())
lib.load_model(model2, 'models/sem_segmentation/1703952735_cityscapes/model_1703956999.h5')



# In[9]:

setup_device()
print(test(test_loader, model2, loss_fn))

# In[]:
lb.imshow_test(model2, test_loader)

# In[11]:

import lib
model2.eval()
test_loss, correct = 0, 0


# Turn off gradient descent
with torch.no_grad():
    for X, y in train_loader:
        X, y = X.to(get_device()), y.to(get_device())
        pred = model2(X)

        lb.imshow_test(model, train_loader,  a=3,b=6, show_original=True)
        
        break
        
# In[12]:

model = model2

# %%
