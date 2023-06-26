import os
import numpy as np
import pandas as pd

from rich.progress import track

from PIL import Image
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src.utils.paths import get_project_path

class CustomDataset(Dataset):
    
    def __init__(self, path_to_table, path_to_images):
        self.path_to_images = path_to_images
        self.path_to_table = path_to_table
        
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(60),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        self.images, self.images_edges, self.table, self.y_data = self.prepare_data()
        self.images = torch.tensor(self.images, dtype=torch.float32)
        self.images_edges = torch.tensor(self.images_edges, dtype=torch.float32)
        self.table = torch.tensor(self.table, dtype=torch.float32)
        self.y_data = torch.tensor(self.y_data)
        
    def __len__(self):
        return len(self.table)
    
    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        
        image = self.images[idx]
        image_edges = self.images_edges[idx]
        table = self.table[idx]
        y_data = self.y_data[idx]
        
        return image, image_edges, table, y_data
    
    def prepare_data(self):
        table = pd.read_csv(self.path_to_table)
        
        images = []
        images_edges = []
        
        filenames = table[table.columns[0]].values
        
        for file in track(filenames, description="[green]Loading images..."):
            img = cv2.imread(os.path.join(self.path_to_images, (file[:-4]+'.png')))
            img_edges = cv2.imread(os.path.join(self.path_to_images, (file[:-4]+'_edges.png')))
            images.append(img)
            images_edges.append(img_edges)
            
        y_data = table[table.columns[1]].values
        table.drop(table.columns[1], axis=1, inplace=True)
        table.drop(table.columns[0], axis=1, inplace=True)
        
        return np.array(images), np.array(images_edges), table.to_numpy(), np.array(y_data)