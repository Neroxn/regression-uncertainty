import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import *
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import seaborn as sns
import os
import pandas as pd
 
class XLSParser():
    """
    Parser for the `.csv`or `.xls` tabular data files. The data will be parsed into x and y values.
    """
    def __init__(self, xls_path : Union[str,os.PathLike]) -> None:
        """
        xls_path (Union[str, os.PathLike]) : Path to the xls file.
        """
        self.xls_path = xls_path
        if self.xls_path.endswith(".csv"):
            self.data = pd.read_csv(self.xls_path)
        else:
            self.data = pd.read_excel(self.xls_path)

class XLSDataset(Dataset):
    def __init__(self, df, x_col : Optional[list] = None, y_col : Optional[list] = None, transforms : Optional[dict] = None,**kwargs):
        self.df = df
        if x_col is None:
            x_col = self.df.columns[:-1]
        if y_col is None:
            y_col = self.df.columns[-1]
        self.x, self.y = self.df[x_col].values,self.df[y_col].values
        self.x_col, self.y_col = x_col, y_col

        if transforms is None:
            self.x_transform = lambda x: x
            self.y_transform = lambda x: x
        else:
            self.x_transform, self.y_transform = transforms.get('x'), transforms.get('y')

        self.age_to_frequency = {}

    def __len__(self):
        """
        Return the shape of the dataset. In this case, it is the number of data points.
        """
        return self.x.shape[0]

    def __getitem__(self, idx):
        """
        Transform the data to torch.Tensor
        """

        x_np = np.array(self.x[idx]).astype('float32')
        y_np = np.array(self.y[idx]).astype('float32')

        # transform x_np and y_np into tensors with torch.float32
        x_tensor = torch.from_numpy(x_np).float()
        y_tensor = torch.from_numpy(y_np).float()
        x_transformed = self.x_transform.forward(x_tensor)
        y_transformed = self.y_transform.forward(y_tensor)

        # return tensor
        return x_transformed, y_transformed


    def get_category_bins(self, bin_size = 1, min_label = None, max_label = None):
            """
            Divide the continious region into bins with bin_size. Assign continuous value to each bin.
            """
            if min_label is None:
                min_label = self.df[self.y_col].min()
            if max_label is None:
                max_label = self.df[self.y_col].max()

            bins = int((max_label - min_label) // bin_size + 1)
            self.df.loc[:,'category_bin'] = pd.cut(self.df.loc[:,self.y_col], bins = bins, labels = np.arange(bins))
            return min_label, max_label

    def assign_frequency_label(self):
        """
        Assign frequency label to bin distribution.
        For 'many-shot' > 100 samples, 'medium-shot' 20-100 samples, 'few-shot' 1-10 samples.
        """
        self.df.loc[:,'frequency'] = self.df.groupby('category_bin')['category_bin'].transform('count')
        self.df.loc[:,'frequency_label'] = pd.cut(self.df.loc[:,'frequency'], bins = [0, 10, 100, np.inf], labels = ['few-shot', 'medium-shot', 'many-shot'])
        self.df.loc[:,'frequency_label'] = self.df.loc[:,'frequency_label'].astype('category')

    def get_categories(self):
        """
        Given continious values of y, return the category of the y.
        """
        return self.age_to_frequency


def create_xls_dataloader(
        transforms : Tuple,
        xls_path : Union[str, os.PathLike],
        batch_size : int = 32,
        cv_split_num : int = 1,
        test_ratio : float = 0.2,
        x_col : Optional[list] = None,
        y_col : Optional[list] = None,
        split_test_val : bool = True,
        **kwargs):
    """
    A simple dataloader for the reading XLS/CSV files. The data will be split into a training and validation set.
    
    Args:
        - transforms (Tuple[Transform,Transform]) : List of x and y transform that will be applied when sampled.
            Transforms are composed with `transforms.Transform` class. 
        - xls_path (Union[str, os.PathLike]) : Path to the xls file.
        - batch_size (int) : Batch size for the dataloader. `-1` denotes that the whole dataset will be used.
        - cv_split_num (int) : Number of cross validation splits.
        - test_ratio (float) : Ratio of the test set.
        - x_col (Optional[list]) : Column index of the x values. If None, all columns except the last one will be used.
        - y_col (Optional[list]) : Column index of the y values. If None, the last column will be used.

    Returns:
        - train_loader (DataLoader) : Dataloader for the training set.
        - val_loader (DataLoader) : Dataloader for the validation set.
        - train_dataset (XLSDataset) : Dataset for the training set.
        - val_dataset (XLSDataset) : Dataset for the validation set.
    """

    parser = XLSParser(xls_path)
    df = parser.data
    num_test = int(df.shape[0] * test_ratio) # half of the test set will be used for validation
    
    # create train and test data for cross validation
    for _ in range(cv_split_num):
        # pick random seperate indices for train set, test set and validation set
        train_idx = np.random.choice(df.index, size = df.shape[0] - num_test, replace = False)
        val_idx = np.random.choice(list(set(df.index) - set(train_idx)), size = num_test, replace = False)
        # val_idx = np.random.choice(test_idx, size = num_test//2, replace = False)
        # test_idx = list(set(test_idx) - set(val_idx))

        # split the data
        df_train, df_val,  = df.loc[train_idx], df.loc[val_idx]

        # create datasets
        train_ds = XLSDataset(df_train, x_col, y_col, transforms["train"])
        val_ds = XLSDataset(df_val, x_col, y_col, transforms["val"])
        #test_ds = XLSDataset(df_test, x_col, y_col, transforms["test"])

        min_label, max_label = train_ds.get_category_bins(bin_size=1)
        train_ds.assign_frequency_label()

        # iterate over all bins and assign a frequency label per bin
        for label_bin in train_ds.df["category_bin"].unique():
            if train_ds.df[train_ds.df[train_ds.y_col] == label_bin].shape[0] == 0:
                continue
            label = train_ds.df[train_ds.df[train_ds.y_col] == label_bin]['frequency_label'].values[0]
            train_ds.age_to_frequency[label_bin] = label

        val_ds.get_category_bins(bin_size=1, min_label = min_label, max_label = max_label)
        val_ds.age_to_frequency = train_ds.age_to_frequency
        for label_bin in val_ds.df["category_bin"].unique():
            if label_bin not in val_ds.age_to_frequency.keys():
                val_ds.age_to_frequency[label_bin] = 'zero-shot'

        # test_ds.get_category_bins(bin_size=1, min_label = min_label, max_label = max_label)
        # test_ds.age_to_frequency = train_ds.age_to_frequency
        # for label_bin in test_ds.df["category_bin"].unique():
        #     if label_bin not in test_ds.age_to_frequency.keys():
        #         test_ds.age_to_frequency[label_bin] = 'zero-shot'
        
        if batch_size == -1:
            batch_size = len(train_ds)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                drop_last=False,pin_memory=True
                                )
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                                drop_last=False, pin_memory=True
                                )
        
        print(f"Train set size : {len(train_ds)}")
        print(f"Val set size : {len(val_ds)}")
        # test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
        #                         drop_last=False, pin_memory=True
        #                         )
        yield train_loader, val_loader, None

if __name__ == "__main__":
    parser = XLSParser('regression_datasets/Concrete_Data.xls')
    df = parser.data
    ds = XLSDataset(df)
    # get the mean and std of the all x values
    print(f"x_cols : {ds.x_col}, y_col : {ds.y_col}")
    x_mean = ds.df[ds.x_col].mean()
    x_std = ds.df[ds.x_col].std()
    print([m for m in x_mean])
    print([s for s in x_std])
    print(ds.df[ds.y_col].mean(), ds.df[ds.y_col].std())

