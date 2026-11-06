import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import copy
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

class NetTrainer():
    def __init__(self,model,loss_fn,optimizer,valid_loss_delta=1E-4):
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        print(f'device: {self.device}')
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.valid_loss, self.train_loss = [], []
        self.accuracy, self.f1_score = [], []
        self.epoch = 0
        self.early_stop_flag = False
        self.best_accuracy = 0
        self.best_valid_loss = 0
        self.best_epoch = 0
        self.test_stats = {}

    def _dev_encode(self,*args):
        return tuple(arg.to(self.device) for arg in args)

    def save_checkpoint(self,fname):
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'validation_loss': self.valid_loss,
            'training_loss': self.train_loss,
            'accuracy':self.accuracy,
            'f1_score':self.f1_score,
            'early_stop_flag':self.early_stop_flag,
            'best_valid_loss': self.best_valid_loss,
            'best_accuracy': self.best_accuracy,
            'best_epoch': self.best_epoch,
            'test_stats': self.test_stats
            }
        torch.save(checkpoint, fname)

    def load_checkpoint(self, fname,boot_from_best=False):
        if boot_from_best:
            _slice = slice(checkpoint['best_epoch'])
            current_epoch = checkpoint['best_epoch']
        else:
            _slice = None
            current_epoch = checkpoint['epoch']

        checkpoint = torch.load(fname, weights_only=False)
    
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = current_epoch
        self.valid_loss = checkpoint['validation_loss'][_slice]
        self.train_loss = checkpoint['training_loss'][_slice]
        self.accuracy = checkpoint['accuracy'][_slice]
        self.f1_score = checkpoint['f1_score'][_slice]
        self.early_stop_flag = checkpoint['early_stop_flag']
        self.best_valid_loss = checkpoint['best_valid_loss']
        self.best_accuracy = checkpoint['best_accuracy'] 
        self.best_epoch = checkpoint['best_epoch']
        self.test_stats = checkpoint['test_stats']
