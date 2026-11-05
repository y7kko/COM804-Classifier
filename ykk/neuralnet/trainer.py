import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import copy

class NetTrainer():
    def __init__(self,model,loss_fn,device=None,valid_loss_delta=1E-4):
        if device is None:
            self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        else:
            self.device = device

        self.model = model
        self.loss_fn = loss_fn
        self.valid_loss = []
        self.train_loss = []
        self.accuracy = []

    def train(self,train_data:DataLoader, optimizer):
        size = len(train_data.dataset)
        self.model.train()

        running_loss,seen = 0.0, 0
        for batch, (X, y) in enumerate(train_data):
            X, y = self._dev_encode(X,y)
            # Compute prediction error
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()*len(X)
            seen += len(X)
            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        self.train_loss.append(running_loss/seen)
        return running_loss/seen


    @torch.no_grad()
    def validate(self,test_data:DataLoader):
        size = len(test_data.dataset)
        self.model.eval()
        valid_loss, correct = 0, 0

        for X, y in test_data:
            X, y = self._dev_encode(X,y)
            
            pred = self.model(X)
            valid_loss += self.loss_fn(pred, y).item()
            correct += ((pred.argmax(1) == y)
                            .type(torch.float)
                            .sum()
                            .item())
        valid_loss /= size
        correct /= size

        self.valid_loss.append(valid_loss)
        self.accuracy.append(correct*100)
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {valid_loss:>8f} \n")
        return valid_loss, correct

    def _dev_encode(self,*args):
        return tuple(arg.to(self.device) for arg in args)

    def plot_loss(self):
        n_epoch = len(self.accuracy)
        x_epoch = np.arange(n_epoch)
        fig,axs = plt.subplots(2,1,figsize=(7,5),sharex=True)

        ax = axs[0]
        _valid_loss = np.asarray(self.valid_loss)
        _train_loss = np.asarray(self.train_loss)
        ax.plot(x_epoch,_valid_loss,label='Validation')
        ax.plot(x_epoch,_train_loss,label='Training')
        ax.legend()

        ax = axs[1]
        _accuracy = np.asarray(self.accuracy)
        ax.plot(x_epoch,_accuracy)
        
        ax.set_xlim(0,min(10,n_epoch))

        plt.tight_layout()
