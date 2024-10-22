import torch
from torch.utils.data import DataLoader
from torchvision.models import vgg16, VGG16_Weights
from Data import MelSpecDataset

# init new vgg16 model
vgg_model = vgg16(weights=VGG16_Weights.DEFAULT)

# load trained vgg16 model
# vgg_model = vgg16()
# vgg_model.load_state_dict(torch.load('models/vgg_1.pth'))

# train vgg16
num_epochs = 10
lr = 1e-4
batch_size = 256

train_data = MelSpecDataset(dims=(224, 224), val_range=(0, 255))
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
optimizer = torch.optim.Adam(vgg_model.parameters(), lr=lr)
loss_fn = torch.nn.CrossEntropyLoss()

vgg_model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for i, (x, y) in enumerate(train_loader):
        x = x.unsqueeze(1)
        x = x.repeat(1, 3, 1, 1)
        
        optimizer.zero_grad()
        y_pred = vgg_model(x)
        loss = loss_fn(y_pred, y)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        print(f'    batch {i}, loss: {loss.item()}')
    
    print(f'Epoch {epoch + 1}, loss: {epoch_loss}')
    
# save vgg weights
torch.save(vgg_model.state_dict(), 'vgg_2.pth')