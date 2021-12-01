import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm


def final_loss(bce_loss, mu, logvar):
    """
    This function will add the reconstruction loss (BCELoss) and the
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    BCE = bce_loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def fit(model, dataloader, optimizer, criterion, train_data, device, batch_size, fig, ax, foo, animate):
    model.train()
    train_loss_arr = []
    running_loss = 0.0
    ims = []
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        data, _ = data
        data = data.to(device)
        # data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data)
        bce_loss = criterion(reconstruction, data)
        loss = final_loss(bce_loss, mu, logvar)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()
        train_loss_arr.append(loss.item()/len(dataloader.dataset))
        if animate:
            with torch.no_grad():
                test_data = torch.unsqueeze(foo,0)
                test_data = test_data.to(device)
                test_recon, _ ,_ = model(test_data)
                # print(test_recon.cpu().numpy()[0][0].shape)
                im = ax.imshow(test_recon.cpu().numpy()[0][0], animated=True)
                ims.extend([[im]])

    train_loss = running_loss/len(dataloader.dataset)
    return train_loss, train_loss_arr, ims


def validate(model, dataloader, optimizer, criterion, val_data, device, batch_size, epoch):
    model.eval()
    running_loss = 0.0
    val_loss_arr = []

    with torch.no_grad():

        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            data, _ = data
            data = data.to(device)
            #data = data.view(data.size(0), -1)
            reconstruction, mu, logvar = model(data)
            bce_loss = criterion(reconstruction, data)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()
            val_loss_arr.append(loss.item()/len(dataloader.dataset))
            # save the last batch input and output of every epoch
            # plt.imshow(reconstruction.cpu().numpy()[0][0])
            # print(data.shape)

            # if i == int(len(val_data)/dataloader.batch_size) - 1:
            #     num_rows = 8
            #     both = torch.cat((data.view(batch_size, 1, 14, 14)[:8],
            #                       reconstruction.view(batch_size, 1, 14, 14)[:8]))
            #     save_image(both.cpu(), f"outputs/output{epoch}.png", nrow=num_rows)

    val_loss = running_loss/len(dataloader.dataset)

    return val_loss, val_loss_arr

