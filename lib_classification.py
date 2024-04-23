from lib import *
from tqdm import tqdm

# Training function
def train(dataloader:DataLoader, model:nn.Module, loss_fn, optimizer, scheduler=None):
    size = len(dataloader.dataset)
    
    # Turn on training mode
    model.to(get_device())
    model.train()
    train_loss, correct = 0, 0
    for X, y in tqdm(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        
        # print(X.shape, y.shape)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # record loss
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    train_loss /= len(dataloader)
    correct /= size
    if scheduler != None:
        scheduler.step()
    
    print(f" Train accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f}")
    return train_loss, correct

# Test function
def test(dataloader:DataLoader, model:nn.Module, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    # Turn on evalution mode
    model.eval()
    test_loss, correct = 0, 0
    
    # Turn off gradient descent
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            
            # record loss
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    test_loss /= num_batches
    correct /= size
    
    print(f" Test accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    return correct

# Test function
def confusion_eval(dataloader:DataLoader, model:nn.Module, loss_fn, n_classes=10):
    from sklearn.metrics import confusion_matrix

    device = 'cpu'
    model.to(device)

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    # Turn on evalution mode
    model.eval()
    test_loss, correct = 0, 0
    conf = None

    # Turn off gradient descent
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            
            # record loss
            test_loss += loss_fn(pred, y).item()
            ypred = pred.argmax(1)
            #print(ypred.shape, y.shape)
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            conf2 = confusion_matrix(y, ypred, labels=np.arange(n_classes))
            conf = conf + conf2 if conf is not None else conf2
            
    test_loss /= num_batches
    #correct /= size
    conf = conf / np.sum(conf, axis=1)

    setup_device()
    model.to(get_device())
    
    #print(f" Test accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    return conf