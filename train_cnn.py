import torch
from torch import nn
from torch.utils.data import DataLoader
from Data import MelSpecDataset
from CNNModel import CNN
    
# initialize the model
cnn_model = CNN()
cnn_model.load_state_dict(torch.load('models/cnn_0.pth'))

# train cnn model
num_epochs = 10
learning_rate = 1e-3
batch_size = 256

train_data = MelSpecDataset(val_range=(0, 1))
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

cnn_model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for i, (X, y) in enumerate(train_loader):
        optimizer.zero_grad()
        X = X.unsqueeze(1)
        y_pred = cnn_model(X)
        loss = loss_fn(y_pred, y)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        print(f'    batch {i}, loss: {loss.item()}')
    print(f'Epoch {epoch}, loss: {epoch_loss}')
    
# save cnn model
torch.save(cnn_model.state_dict(), 'models/cnn_1.pth')