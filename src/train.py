import torch
from torch import nn
from src.utils_funcs import EarlyStopping
from torch.nn.functional import l1_loss  # equivalent to MAE

def train_model(model, train_loader, val_loader, device, num_epochs=20, lr=0.00001):
	"""
	Train and validate the model.

	Args:
		model: The TSMixer model.
		train_loader: DataLoader for the training set.
		val_loader: DataLoader for the test set.
		device: The device (CPU or GPU) to use for training.
		num_epochs: Number of epochs to train for.
		lr: Learning rate for the Adam optimizer.
	"""


	loss = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	tolerance = 5
	delta = 0.01
	ES_instance = EarlyStopping(tolerance, delta)
	train_loss_list = []
	val_loss_list = []


	for epoch in range(num_epochs):
		model.train()
		total_train_loss = 0
		total_train_mae = 0
		for _, (window, labels) in enumerate(train_loader):
			window = window.to(device)
			labels = labels.to(device)
			outputs = model(window)
			l = loss(outputs, labels)
			mae_train = l1_loss(outputs, labels, reduction='mean')
			with torch.no_grad():
				total_train_loss += l.item() ; total_train_mae += mae_train.item()
			optimizer.zero_grad()
			l.backward()
			optimizer.step()


		model.eval()
		total_val_loss = 0
		total_val_mae = 0
		with torch.no_grad():
			avg_train_loss = total_train_loss / len(train_loader)
			avg_train_mae = total_train_mae / len(train_loader)
			train_loss_list.append(avg_train_loss)

			print(f'Epoch {epoch+1}, Training Loss: {avg_train_loss},      Training MAE: {avg_train_mae}')
			for window_val, labels_val in val_loader:
				window_val = window_val.to(device)
				labels_val = labels_val.to(device)
				outputs_t = model(window_val)
				l_val = loss(outputs_t, labels_val)
				mae_t = l1_loss(outputs_t, labels_val, reduction='mean')
				total_val_loss += l_val.item() ; total_val_mae += mae_t.item()

		avg_val_loss = total_val_loss / len(val_loader)
		avg_val_mae = total_val_mae / len(val_loader)
		print(f'Epoch {epoch+1}, Validation loss: {avg_val_loss}      Validation MAE: {avg_val_mae}')
		val_loss_list.append(avg_val_loss)
		if ES_instance(avg_val_loss) == True:
			print(f"Early Stop at Epoch: {epoch}")
			break
	return (train_loss_list, val_loss_list)



