import torch
import torch.nn as nn
from torch.utils.data import DataLoader
class NetTrainer():
    def __init__(self,model,loss_fn,device=None):
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


    @torch.no_grad()
    def validate(self,test_data:DataLoader):
        size = len(test_data.dataset)
        num_batches = len(test_data)
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
        return valid_loss,correct

    def _dev_encode(self,*args):
        return tuple(arg.to(self.device) for arg in args)
