import torch
import torch.optim as optim
import torch.nn as nn


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss_list = []
    train_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        #print(X, y)
        loss = loss_fn(pred, y)
        train_loss += loss_fn(pred, y).item()
        train_loss_list.append(loss_fn(pred, y).item())
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    train_loss /= num_batches
    print(f" Avg Train loss: {train_loss:>8f} \n")
    return train_loss_list

def test_loop(dataloader, model, loss_fn):
    test_loss_list = []
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss= 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            test_loss_list.append(loss_fn(pred, y).item())

    test_loss /= num_batches
    print(f" Avg Test loss: {test_loss:>8f} \n")
    return test_loss_list

