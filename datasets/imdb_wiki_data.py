
import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader

import pandas as pd 
import matplotlib.pyplot as plt
from typing import *

pd.options.mode.chained_assignment = None  # default='warn'


class IMDBWIKI(Dataset):
    def __init__(self, df, data_dir, transforms : dict = None):
        self.df = df
        self.data_dir = data_dir

        if transforms is None:
            self.x_transform = lambda x: x
            self.y_transform = lambda x: x
        else:
            self.x_transform, self.y_transform = transforms.get('x'), transforms.get('y')

        self.age_to_frequency = {}
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        index = index % len(self.df)
        row = self.df.iloc[index]
        img = Image.open(os.path.join(self.data_dir, row['path'])).convert('RGB')
        label = np.asarray([row['age']]).astype('float32')

        return self.x_transform.forward(img), self.y_transform.forward(label)

    def get_category_bins(self, bin_size = 1, min_label = None, max_label = None):
        """
        Divide the continious region into bins with bin_size. Assign continuous value to each bin.
        """
        if min_label is None:
            min_label = self.df['age'].min()
        if max_label is None:
            max_label = self.df['age'].max()
        bins = (max_label - min_label) // bin_size + 1
        self.df.loc[:,'category_bin'] = pd.cut(self.df.loc[:,'age'], bins = bins, labels = np.arange(bins))
        return min_label, max_label
    
    def assign_frequency_label(self):
        """
        Assign frequency label to bin distribution.
        For 'many-shot' > 100 samples, 'medium-shot' 20-100 samples, 'few-shot' 1-10 samples.
        """
        self.df.loc[:,'frequency'] = self.df.groupby('category_bin')['category_bin'].transform('count')
        self.df.loc[:,'frequency_label'] = pd.cut(self.df.loc[:,'frequency'], bins = [0, 5, 25, np.inf], labels = ['few-shot', 'medium-shot', 'many-shot'])
        self.df.loc[:,'frequency_label'] = self.df.loc[:,'frequency_label'].astype('category')

    def get_categories(self):
        """
        Given continious values of y, return the category of the y.
        """
        return self.age_to_frequency
    
    def get_transform(self):
        return self.x_transform, self.y_transform
    
def create_imdb_wiki_dataloader(
        transforms : dict,
        data_dir : Union[str, os.PathLike],
        batch_size : int = 32) -> Tuple[DataLoader, DataLoader, Dataset, Dataset]:
    """
    Dataloader for the IMDB-WIKI dataset.

    Args:
        - transforms (dict) : Dictionary for the transformation.
        - data_dir (Union[str, os.PathLike]) : Path to the data directory.
        - batch_size (int) : Batch size for the dataloader.

    Returns:
        - train_loader (DataLoader) : Dataloader for the training set.
        - val_loader (DataLoader) : Dataloader for the validation set.
        - train_dataset (IMDBWIKI) : Dataset for the training set.
        - val_dataset (IMDBWIKI) : Dataset for the validation set.
    """
    df = pd.read_csv(os.path.join(data_dir,"imdb_wiki.csv"))
    df_train, df_val,df_test = df[df["split"] == "train"], df[df["split"] == "val"], df[df["split"] == "test"]

    # for age values in df_val, find its frequency in df_train and label accordingly
    train_ds = IMDBWIKI(df_train,data_dir, transforms["train"])
    val_ds = IMDBWIKI(df_val, data_dir, transforms["val"])
    test_ds = IMDBWIKI(df_test, data_dir, transforms["test"])

    min_label, max_label = train_ds.get_category_bins(bin_size=1)
    train_ds.assign_frequency_label()

    # iterate over all bins and assign a frequency label per bin
    for label_bin in train_ds.df["category_bin"].unique():
        if train_ds.df[train_ds.df["age"] == label_bin].shape[0] == 0:
            continue
        label = train_ds.df[train_ds.df["age"] == label_bin]['frequency_label'].values[0]
        train_ds.age_to_frequency[label_bin] = label

    val_ds.get_category_bins(bin_size=1, min_label = min_label, max_label = max_label)
    val_ds.age_to_frequency = train_ds.age_to_frequency
    for label_bin in val_ds.df["category_bin"].unique():
        if label_bin not in val_ds.age_to_frequency.keys():
            val_ds.age_to_frequency[label_bin] = 'zero-shot'

    test_ds.get_category_bins(bin_size=1, min_label = min_label, max_label = max_label)
    test_ds.age_to_frequency = train_ds.age_to_frequency
    for label_bin in test_ds.df["category_bin"].unique():
        if label_bin not in test_ds.age_to_frequency.keys():
            test_ds.age_to_frequency[label_bin] = 'zero-shot'
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=False,pin_memory=True
                              )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              drop_last=False, pin_memory=True
                              )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              drop_last=False, pin_memory=True
                              )
    yield train_loader, val_loader, test_loader
