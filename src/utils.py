from torch.utils.data import ConcatDataset, DataLoader, sampler, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import numpy as np
import torch
import os, time

def save_checkpoint(model, epoch, train_loss, train_acc, val_loss, val_acc, val_best_auc, fpr, tpr, prefix=''):
    
    model_out_path = "./saves/" + prefix + '.pth'
    state = {"epoch": epoch, 
             "model": model.state_dict(),
             "train_loss": train_loss, 
             "train_acc": train_acc, 
             "val_loss": val_loss,
             "val_acc": val_acc, 
             "best_auc": val_best_auc}
    
    if not os.path.exists("./saves/"):
        os.makedirs("./saves/")

    torch.save(state, model_out_path)
    print("model checkpoint saved @ {}".format(model_out_path))
    
def load_model(model, pretrained, device):
    weights = torch.load(pretrained, map_location=device)
    model.load_state_dict(weights['model'], strict=False)
