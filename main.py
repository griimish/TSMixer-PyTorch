import torch
import pandas as pd

from src.utils_funcs import create_datasets, scale_data
from src.train import train_model
from src.TSMixer import TSMixer

HOURS_PER_DAY = 24
DAYS_PER_MONTH = 30

def main(file_path, seq_len=512, pred_len = 96, batch_size=32, num_epochs=20, lr=0.00001, n_mixers=6, dropout=0.9, hidden_dim=512, plot = False):
    # Load and preprocess data
    df = pd.read_csv(file_path)

    # We transform the data into a time series 
    df.set_index('date', inplace=True)    
    _, n_features = df.shape

    # We compute the splits according to the paper guidelines (12/4/4)
    train_months = 12
    val_months = 4
    test_months = 4
    train_steps = train_months * DAYS_PER_MONTH * HOURS_PER_DAY
    val_steps = val_months * DAYS_PER_MONTH * HOURS_PER_DAY
    test_steps = test_months * DAYS_PER_MONTH * HOURS_PER_DAY

    # Split the dataset
    train_df = df[:train_steps]
    val_df = df[train_steps - pred_len : train_steps + val_steps]
    test_df = df[-(test_steps + pred_len):]

    # Scale the data
    train_scaled, test_scaled, val_scaled, scaler = scale_data(train_df, test_df, val_df)

    # Create datasets and dataloaders
    train_loader, test_loader, val_loader = create_datasets(train_scaled, test_scaled, val_scaled, seq_len, pred_len, batch_size)

    # Training settings
    # CUDA is recommended as TSMixer can still be slow on CPU 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TSMixer(seq_len, n_features, hidden_dim, pred_len, n_mixers, dropout).to(device)

    # Train the model
    train_loss_list, val_loss_list = train_model(model, train_loader, val_loader, device, num_epochs, lr)
    torch.save(model.state_dict(), 'model.pt')

    # Plot the results
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(train_loss_list, label='Training Loss')
        plt.plot(val_loss_list, label='Validation Loss')
        plt.legend()
        plt.show()

        with torch.no_grad():
            for images_t, labels_t in test_loader:
                images_t = images_t.to(device)
                labels_t = labels_t.to(device)
                outputs_t = model(images_t)

            unscaled_preds = scaler.inverse_transform(outputs_t.cpu()[0, :, :])

            unscaled_labels = scaler.inverse_transform(labels_t.cpu()[0, :, :])
            for feature_index in range(n_features):
                plt.figure(figsize=(15, 5))
                plt.plot(range(labels_t.shape[1]), unscaled_labels[-96:,feature_index], label="Ground Truth")
                plt.plot(range(labels_t.shape[1]), unscaled_preds[:,feature_index], label="Predicted")
                plt.title(f'{df.columns[feature_index]} - Ground Truth vs Predicted')
                plt.xlabel('Time Step')
                plt.ylabel('Feature Value')
                plt.legend()
                plt.show()



if __name__ == "__main__":
    main('data/ETTh1.csv')
