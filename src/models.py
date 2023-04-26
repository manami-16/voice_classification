import numpy as np
import matplotlib.pyplot as plt
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, sampler, Dataset
from sklearn.metrics import roc_curve, auc
import os, time
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import pandas as pd
from src.data import *
from src.utils import save_checkpoint
from torchsummary import summary
from ipywidgets import IntProgress

class CNN_MaxPool(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.kernel_size = 2
        self.stride = 2
        self.input_length = 116584
        # data = [batch_size, in_channels, input_length]
        
        ## layer 1
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=self.kernel_size, stride=self.stride))
        
        self.out_length = ((self.input_length - self.kernel_size) / self.stride) + 1
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        ## layer 2
        self.input_length = self.out_length
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        
        self.out_length = ((self.input_length - self.kernel_size) / self.stride) + 1
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        ## layer 3
        self.input_length = self.out_length
        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        
        self.out_length = ((self.input_length - self.kernel_size) / self.stride) + 1
        self.bn3 = nn.BatchNorm1d(out_dim)

        ## layer 4
        self.layer4 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(int(self.out_length), out_dim),
            nn.Sigmoid())
        
        ## dropout
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)

        x = self.dropout(x)
        x = self.layer2(x)
        x = self.bn2(x)

        x = self.dropout(x)
        x = self.layer3(x)
        x = self.bn3(x)

        x = self.layer4(x)
        return x

class CNN_AvgPool(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.kernel_size = 2
        self.stride = 2
        self.input_length = 116584
        # data = [batch_size, in_channels, input_length]
        
        ## layer 1
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=self.kernel_size, stride=self.stride))
        
        self.out_length = ((self.input_length - self.kernel_size) / self.stride) + 1
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        ## layer 2
        self.input_length = self.out_length
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2))
        
        self.out_length = ((self.input_length - self.kernel_size) / self.stride) + 1
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        ## layer 3
        self.input_length = self.out_length
        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2))
        
        self.out_length = ((self.input_length - self.kernel_size) / self.stride) + 1
        self.bn3 = nn.BatchNorm1d(out_dim)

        ## layer 4
        self.layer4 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(int(self.out_length), out_dim),
            nn.Sigmoid())
        
        ## dropout
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)

        x = self.dropout(x)
        x = self.layer2(x)
        x = self.bn2(x)

        x = self.dropout(x)
        x = self.layer3(x)
        x = self.bn3(x)

        x = self.layer4(x)
        return x