
import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader

import pandas as pd 
import matplotlib.pyplot as plt
from typing import *

class IMDBWIKI(Dataset):
    def __init__(self, df, data_dir, transforms):
        self.df = df
        self.data_dir = data_dir

        self.x_transform, self.y_transform = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        index = index % len(self.df)
        row = self.df.iloc[index]
        img = Image.open(os.path.join(self.data_dir, row['path'])).convert('RGB')
        label = np.asarray([row['age']]).astype('float32')

        return self.x_transform.forward(img), self.y_transform.forward(label)

def create_imdb_wiki_dataloader(
        transforms : Tuple,
        data_dir : Union[str, os.PathLike],
        batch_size : int = 32) -> Tuple[DataLoader, DataLoader, Dataset, Dataset]:
    """
    Dataloader for the IMDB-WIKI dataset.

    Args:
        - transforms (Tuple) : Tuple of transforms for the x and y values.
        - data_dir (Union[str, os.PathLike]) : Path to the data directory.
        - batch_size (int) : Batch size for the dataloader.

    Returns:
        - train_loader (DataLoader) : Dataloader for the training set.
        - val_loader (DataLoader) : Dataloader for the validation set.
        - train_dataset (IMDBWIKI) : Dataset for the training set.
        - val_dataset (IMDBWIKI) : Dataset for the validation set.
    """
    df = pd.read_csv(os.path.join(data_dir,"imdb_wiki.csv"))
    df_train, df_val = df[df["split"] == "train"], df[df["split"] == "val"]
    train_ds, val_ds = IMDBWIKI(df_train,data_dir, transforms), IMDBWIKI(df_val, data_dir, transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    
    yield train_loader, val_loader, train_ds, val_ds



    