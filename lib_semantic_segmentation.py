from lib import *
from tqdm import tqdm

#

# Training function
def train(dataloader:DataLoader, model:nn.Module, loss_fn, optimizer, scaler=None, 
          scheduler=None, multiclass=False):
    size = len(dataloader.dataset)
    
    # Turn on training mode
    device = get_device()
    model.to(get_device())
    model.train()
    train_loss, correct, pixels = 0, 0, 0
    
    recall = 0
    recall_den, precision_den = 0, 0

    for X, y in tqdm(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        
        with torch.cuda.amp.autocast():
            # print(X.shape, y.shape)
            pred = model(X)
            loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else: 
            loss.backward()
            optimizer.step()
        
        # record loss
        train_loss += loss.item()
        
        if not multiclass:
            pred = torch.sigmoid(pred)
            preds = (pred>.5).float()
            
            # record loss
            correct += (preds == y).sum()
            pixels += torch.numel(preds)

            recall += torch.logical_and(preds == y, y == 1).sum()
            recall_den += (y == 1).sum()
            precision_den += (preds == 1).sum()
        else:
            pred = torch.softmax(pred, dim=1)
            preds = torch.argmax(pred, dim=1)
            Y = torch.argmax(y, dim=1)
            
            # record loss
            correct += (preds == Y).sum()
            pixels += torch.numel(preds)

            preds = preds.cpu()
            Y = Y.cpu()

            recall += np.array([torch.logical_and(preds == Y, Y == i).sum() for i in range(34)])
            recall_den += np.array([(Y == i).sum() for i in range(34)]) # union
            precision_den += np.array([(preds == i).sum() for i in range(34)])
        
    
    train_loss /= len(dataloader)
    correct = correct / pixels
    if scheduler != None:
        scheduler.step(train_loss)
        print(f" Lr: {optimizer.param_groups[0]['lr']}")
    
    dice_score = recall * 2 / (precision_den + recall_den)
    precision = recall / precision_den
    recall = recall / recall_den
    
    print(f" Test accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f}, \nTest recall: {recall}, precision: {precision}, dice_score: {dice_score}")
    return train_loss, correct

# Test function
def test(dataloader:DataLoader, model:nn.Module, loss_fn, multiclass=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    device = get_device()
    model.to(device)
    # Turn on evalution mode
    model.eval()
    test_loss, correct, pixels = 0, 0, 0
    recall = 0
    recall_den, precision_den = 0, 0
    
    # Turn off gradient descent
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)

            pred = model(X)

            test_loss += loss_fn(pred, y).item()

            if not multiclass:
                pred = torch.sigmoid(pred)
                preds = (pred>.5).float()
                
                # record loss
                correct += (preds == y).sum()
                pixels += torch.numel(preds)

                recall += torch.logical_and(preds == y, y == 1).sum()
                recall_den += (y == 1).sum()
                precision_den += (preds == 1).sum()
            else:
                pred = torch.softmax(pred, dim=1)
                preds = torch.argmax(pred, dim=1)
                Y = torch.argmax(y, dim=1)
                
                preds = preds.cpu()
                Y = Y.cpu()

                # record loss
                correct += (preds == Y).sum()
                pixels += torch.numel(preds)

                recall += np.array([torch.logical_and(preds == Y, Y == i).sum() for i in range(34)])
                recall_den += np.array([(Y == i).sum() for i in range(34)]) # union
                precision_den += np.array([(preds == i).sum() for i in range(34)])
            
    test_loss /= num_batches
    correct = correct / pixels
    dice_score = recall * 2 / (precision_den + recall_den)
    precision = recall / precision_den
    recall = recall / recall_den
    
    print(f" Test accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}, \nTest recall: {recall}, precision: {precision}, dice_score: {dice_score}")
    return test_loss, correct

def imshow_test(model, test_loader, show_reference=True, show_original=False, softmax=False, **kwargs):
    import lib
    import importlib
    importlib.reload(lib)
    model.eval()
    test_loss, correct = 0, 0
    setup_device()
    model.to(get_device())

    #classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

    # Turn off gradient descent
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(get_device()), y.to(get_device())
            pred = model(X)

            if not softmax:
                pred = torch.sigmoid(pred)
                pred = (pred>.5).float()
            else:
                pred = nn.functional.softmax(pred, dim=1)
                pred = torch.argmax(pred, dim=1)
                y = nn.functional.softmax(y, dim=1)
                y = torch.argmax(y, dim=1)

            if show_original:
                lib.imshow(X.cpu().numpy(), range=range(3), colorbar=False)
            lib.imshow(pred.cpu().numpy(), range=range(3), **kwargs)
            if show_reference:
                lib.imshow(y.cpu().numpy(), range=range(3), **kwargs)
            
            break

