from torch.utils.data import ConcatDataset, DataLoader, sampler, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import numpy as np
import torch

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

def adjust_datasize(X_data):
    # Determine the size of the largest tensor
    max_size = max([t.shape[0] for t in X_data])

    # Pad each tensor so it has the same shape as the largest tensor
    padded_data = [torch.nn.functional.pad(t, (0, 0, 0, max_size - t.shape[0]), mode='constant', value=0) for t in X_data]
    
    return padded_data
    
def get_wavedata(df):
    wave = []
    path = './trimmed_data/'
    for file in df['filename']:
        file = file.split('.')[0]
        array = np.load(path + file + '.npy').reshape(-1, 1)
        array = torch.from_numpy(array)
        wave.append(array)
        
    wave = adjust_datasize(wave)
    wave = np.stack(wave)
    
    return wave


def create_dataloader(df, num_train=45):
    val_size = int((len(df) - num_train) * 0.5)
    wave = get_wavedata(df)## return wave data in tensor
    
    X_train, X_test, y_train, y_test = train_test_split(wave, df['classID'].to_numpy(), 
                                                        train_size=num_train, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=val_size, 
                                                    shuffle=True)
    
    
    
#     print('-'*10)
#     print(f'Training set - {np.unique(y_train, return_counts=True)}')
#     print(f'Validation set - {np.unique(y_val, return_counts=True)}')
#     print(f'Test set - {np.unique(y_test, return_counts=True)}')
#     print('-'*10)
    
    train_set = Custom_Dataset(y_train, X_train)
    val_set = Custom_Dataset(y_val, X_val)
    test_set = Custom_Dataset(y_test, X_test)
    
#     train_set = adjust_datasize(X_train, y_train)
#     val_set = adjust_datasize(X_val, y_val)
#     test_set = adjust_datasize(X_test, y_test)
    
    print('-'*5, 'Custom Dataset', '-'*5)
    print(f'train data = {next(iter(train_set))}')
    
    train_loader = DataLoader(train_set, batch_size = 1, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size = 1, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size = 1, shuffle=True, drop_last=True)
    
    return train_loader, val_loader, test_loader