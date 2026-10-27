"""Um esqueleto do que seria um main
"""
import ykk
import os
# os.environ['YKK_NOCHDIR'] = '1'
ykk.utils.configure()
import torch
from torch.utils.data import DataLoader
#%%
fs = 44100
def separate_folds(val_fold):
    """Dado um fold de validação, retorna os folds
    de treino(todos que não são de validação)
    """
    return [f for f in range(1, 11) if f != val_fold]

valid_fld = 1
train_fld = separate_folds(valid_fld)

valid_data = ykk.urban8k(augmentation=False,
                        fs=fs,folds=valid_fld)

train_data = ykk.urban8k(augmentation=True,
                        fs=fs,folds=train_fld)

#%%
train_dl= DataLoader(train_data,shuffle=True,)
valid_dl = DataLoader(valid_data)
#%%
model = ykk.NeuralNetwork()
#%%
n_epochs = 2
trainer = ykk.NetTrainer(model=model,loss_fn=None,)
for epoch in range(n_epochs):
    trainer.train(train_dl)
    trainer.validate(valid_dl)
