
# In[0]:

# importing libraries
# I'm encapsulating the bulky code in lib.py and models.py
from lib_classification import *
from models_classificiation import *
import detectors

# In[1]:

# loading data
dataset = datasets.CIFAR10
#dataset = lambda train, **kwargs: datasets.STL10(split='train' if train else 'test', **kwargs)
#dataset = lambda **kwargs: datasets.EMNIST(split='balanced', **kwargs)
traintest_transforms = get_transforms(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10)
    ], [])
train_data, test_data = fetch_dataset(dataset, *traintest_transforms)
train_loader, test_loader = get_dataloader(train_data, test_data, batch_size=256)

# In[]:
data = next(iter(train_loader))
print(data[0].shape, data[1].shape)

# In[2]:

# setting up devices
setup_device(to_print=True)
#model = Conv(n_channels=3).to(get_device())
N_CLASSES = 10
import timm
#model = timm.create_model("resnet18_cifar10", pretrained=True)
#model = ResNet34(in_channel=1, num_classes=N_CLASSES, first_lay_kernel_size=5, first_lay_stride=1).to(get_device())
#model = ResNet34(num_classes=100, first_lay_kernel_size=5, first_lay_stride=1).to(get_device())
model = ResNet34(num_classes=10, first_lay_kernel_size=3, first_lay_stride=1, first_lay_padding=1).to(get_device())

# In[]:
loss_fn, opt, scheduler = set_optimizers(model, nn.CrossEntropyLoss, torch.optim.SGD, lr=.01)#lr = .1
print(model)
model.to(get_device())

# In[]:
from torchsummary import summary
print(next(iter(train_loader))[0].shape)
summary(model, next(iter(train_loader))[0].shape[1:])

# In[3]:

# Total training epochs
epochs = 100
training_losses = []
training_accuracy = []
testing_accuracy = []

for t in range(epochs):
    print('\n', "=" * 15, "Epoch", t + 1, "=" * 15)
    loss, train_accuracy = train(train_loader, model, loss_fn, opt, scheduler=scheduler)
    test_accuracy = test(test_loader, model, loss_fn)
    training_losses.append(loss)
    training_accuracy.append(train_accuracy)
    testing_accuracy.append(test_accuracy)

# In[4]:
    
plt.plot(training_accuracy, label=f'Train Accuracy\nBest: {max(training_accuracy)}')
plt.plot(testing_accuracy, label=f'Test Accuracy\nBest: {max(testing_accuracy)}')
plt.legend(loc='best')
plt.show()

# In[4]:

final_test_acc = test(test_loader, model, loss_fn)
print(final_test_acc)

# In[5]:

import importlib
import lib, lib_classification
importlib.reload(lib)

# In[6]:

#imgs = next(iter(test_loader))[0].numpy()
#lib.imshow(imgs, range(9))

# In[7]:

lib.save_model(model, PATH='models/classification_simp', extra_info=f'\nModel for {train_loader.dataset} with test accuracy: {final_test_acc}')

# In[8]:

conf = lib_classification.confusion_eval(test_loader, model, loss_fn, n_classes=N_CLASSES)
plt.imshow(conf)
plt.show()

# In[]:

classes = train_loader.dataset.classes
confAbs = torch.Tensor(conf)
confAbs.fill_diagonal_(0)
confAbs += np.array(confAbs.T)
plt.imshow(confAbs)
plt.show()

# In[]:

confVals = confAbs.flatten()
sortedVals = confVals.sort(descending=True)
sortedIdx = confVals.argsort(descending=True)
max_confused_classes = np.stack(np.unravel_index(sortedIdx, confAbs.shape)).T
max_confused_classes = [print(f'{[classes[c] for c in cs]}: {sortedVals.values[i]}') for i, cs in enumerate(max_confused_classes) if cs[1]>cs[0] and sortedVals.values[i] != 0]

# In[8]:

# *for MNIST n_channels=1.
#model2 = Conv(n_channels=3).to(get_device())
lib.load_model(model, 'models/classification_simp/model_1703689668.h5')
model2 = model

# In[9]:

print(test(test_loader, model2, loss_fn))

# In[11]:

model2.eval()
test_loss, correct = 0, 0

classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# Turn off gradient descent
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(get_device()), y.to(get_device())
        pred = model2(X)

        lib.imshow(X.cpu().numpy(), range=range(9), 
                   labels=[f"Label: {classes[y[i]]}, \nPred: {classes[np.argmax(pred[i]).tolist()]}" for i in range(9)])
        
        break
        
# In[12]:

model = model2

# %%
