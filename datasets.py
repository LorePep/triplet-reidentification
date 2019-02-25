import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from utils import load_img


class TripletNetworkDataset(Dataset):
    """
    Dataset to generate random triplets.
    For each anchor sample randomly returns a positive and negative samples.
    """
    def __init__(self, path, df, transforms=None):
        self.path = path
        self.df = df
        self.labels = self.df["Id"].values
        self.transforms = transforms
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0] for label in self.labels_set}
                
    def _get_triplet_indexes(self, anchor_label, anchor_idx):
        positive_idx = anchor_idx
        while positive_idx == anchor_idx:
                positive_idx = np.random.choice(self.label_to_indices[anchor_label])
                
        negative_label = np.random.choice(list(self.labels_set - set([anchor_label])))
        negative_idx = np.random.choice(self.label_to_indices[negative_label])
        
        return positive_idx, negative_idx


    def __getitem__(self, index):
        anchor_img_path, anchor_label = self.df.iloc[index]["Image"], self.df.iloc[index]["Id"]
        positive_index, negative_index = self._get_triplet_indexes(anchor_label, index)

        positive_img_path = self.df.iloc[positive_index]["Image"]
        negative_img_path = self.df.iloc[negative_index]["Image"]
        
        anchor_img = load_img(os.path.join(self.path, anchor_img_path))
        positive_img = load_img(os.path.join(self.path, positive_img_path))
        negative_img = load_img(os.path.join(self.path, negative_img_path))
        
        if self.transforms:
            anchor_img = self.transforms(anchor_img)
            positive_img = self.transforms(positive_img)
            negative_img = self.transforms(negative_img)
        return (anchor_img, positive_img, negative_img), []
    
    def __len__(self):
        return len(self.df)
    

class SingleImageDataset(Dataset):
    """
    Dataset to generate random single images.
    For each sample returns an image and its label.
    """
    def __init__(self, path, df, transforms=None):
        self.path = path
        self.df = df
        self.transforms = transforms
        
    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = load_img(os.path.join(self.path, row["Image"]))
    
        if self.transforms is not None:
            img = self.transforms(img)
            
        return img, row["Id"]
    
    def __len__(self):
        return len(self.df)
