# !git clone <URL> ykk_repo
# !git -C ./ykk_repo/ pull

# import sys
# sys.path.append('./ykk_repo/')
#%%
import os
os.environ['YKK_NOCHDIR'] = '1'

import ykk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from IPython.display import clear_output
ykk.urban8k().download_dataset()
clear_output()

def separate_folds(val_fold):
    """Dado um fold de validação, retorna os folds
    de treino(todos que não são de validação)
    """
    return [f for f in range(1, 11) if f != val_fold]

def get_data(val_fold,fs,return_dataloader=True):
    train_fld = separate_folds(val_fold)
    valid_data = ykk.urban8k(augmentation=False,
                            fs=fs,filter_folds=val_fold)

    train_data = ykk.urban8k(augmentation=True,
                            fs=fs,filter_folds=train_fld)
    if return_dataloader:
        train_dl= DataLoader(train_data,shuffle=True,)
        valid_dl = DataLoader(valid_data)
        return train_dl,valid_dl
    else:
        return train_data,valid_data
#%%
fs = 44100

#%%
# Note que esse é apenas o código exemplo
class NeuralNetwork(nn.Module):
    def __init__(self,dropout=.5):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # Hidden 1
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Hidden 2
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Out
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

#%%
valid_fld = 1
train_dl,valid_dl = get_data(valid_fld,fs=fs,return_dataloader=False)
#%%
import numpy as np
(valid_dl.metadata['fold'].unique())
#%%
model = NeuralNetwork(dropout=.5)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), 
                            lr=1e-3
                            )

#%%
n_epochs = 2
trainer = ykk.NetTrainer(model   = model,
                         loss_fn = loss_fn,
                         )

for epoch in range(n_epochs):
    trainer.train(train_dl,
                  optimizer=optimizer)
    trainer.validate(valid_dl)

