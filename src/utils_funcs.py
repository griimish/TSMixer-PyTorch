from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.etth1_dataset import ETTh1TimeSeries

def scale_data(train_df, test_df, val_df):
    """
    Scale data using StandardScaler.
    """
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)
    val_scaled = scaler.transform(val_df)
    return train_scaled, test_scaled, val_scaled, scaler

def create_datasets(train_df, test_df, val_df, seq_len=512, pred_len=96, batch_size=32):
    """
    Create PyTorch datasets and dataloaders.
    """
    train_dataset = ETTh1TimeSeries(train_df, seq_len, pred_len)
    test_dataset = ETTh1TimeSeries(test_df, seq_len, pred_len)
    val_dataset = ETTh1TimeSeries(val_df, seq_len, pred_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, test_loader, val_loader

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, tolerance=5, delta=0.01):
        self.tolerance = tolerance
        self.delta = delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False

