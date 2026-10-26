import torch
import torch.nn as nn

class NetTrainer():
    def __init__(self,model,loss_fn,device=None):
        if device is None:
            self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        else:
            self.device = device

        self.model = model
        self.loss_fn = loss_fn

    def train(self,dataloader, optimizer):
        size = len(dataloader.dataset)
        self.model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = self._dev_encode(X,y)
            # Compute prediction error
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(self,dataloader):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = self._dev_encode(X,y)
                
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def _dev_encode(self,*args):
        return tuple(arg.to(self.device) for arg in args)
