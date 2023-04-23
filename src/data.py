from torch.utils.data import ConcatDataset, DataLoader, sampler, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import numpy as np

class Custom_Dataset(Dataset):
    '''
    class: custom_dataset
    does: create a custom dataset
    parameters: df.data, df.label
    '''
    
    def __init__(self, label, data):
        self.label = label
        self.data = data
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        label = self.label[idx]
        data = self.data[idx]
        sample = {"data": data, "target": label}
        
        return sample

def create_dataloader(df, num_train=45):
    val_size = int((len(df) - num_train) * 0.5)
    X_train, X_test, y_train, y_test = train_test_split(df['wave'].to_numpy(), df['classID'].to_numpy(), 
                                                        train_size=num_train, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=val_size, 
                                                    shuffle=True)
    
    print('-'*10)
    print(f'Training set - {np.unique(y_train, return_counts=True)}')
    print(f'Validation set - {np.unique(y_val, return_counts=True)}')
    print(f'Test set - {np.unique(y_test, return_counts=True)}')
    print('-'*10)
    
    train_set = Custom_Dataset(y_train, X_train)
    val_set = Custom_Dataset(y_val, X_val)
    test_set = Custom_Dataset(y_test, X_test)
    
    print('-'*5, 'Custom Dataset', '-'*5)
    print(f'train data = {next(iter(train_set))}')
    
    train_loader = DataLoader(train_set, batch_size = 1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size = 1, shuffle=True)
    test_loader = DataLoader(test_set, batch_size = 1, shuffle=True)
    
    return train_loader, val_loader, test_loader