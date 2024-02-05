import torch
from torch.utils.data import Dataset

class ETTh1TimeSeries(Dataset):
    def __init__(self, ds, seq_len, pred_len):
        """
        Args:
            ds: The preprocessed time series dataset (here, ETTh1)
            seq_len: The input (or window) length.
            pred_len: The prediction length.
        """
        self.ds = ds
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.ds) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        """
        Returns a sample from the data, this is later used for the DataLoader calls
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.ds[idx:idx+self.seq_len]
        target = self.ds[idx+self.seq_len:idx+self.seq_len+self.pred_len]
        data = torch.from_numpy(data).float()
        target = torch.from_numpy(target).float()
        return data, target

