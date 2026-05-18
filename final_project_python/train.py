import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from model import CNN

train_batch_size = 64
test_batch_size = 1000

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# MNIST contains 60,000 images
# training size = 50,000 images
# test size = 10,000 images
train_loader = DataLoader(
    datasets.MNIST('.', train=True, download=True, transform=transform),
    batch_size=train_batch_size, shuffle=True
)

test_loader = DataLoader(
    datasets.MNIST('.', train=False, download=True, transform=transform),
    batch_size=test_batch_size
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
opt = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        opt.zero_grad()     # zero out gradients so they don't accumulate
        loss_fn(model(imgs), labels).backward() # does a CNN.forward(imgs), backward() backpropogates dLoss/dWeight for each parameter through the network
        opt.step()

    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            correct += (model(imgs).argmax(1) == labels).sum().item()
            # model(imgs) = (test_batch_size, 10). argmax(1) takes the largest value of the dim=1 and compares to labels
    print(f"Epoch {epoch}: accuracy = {correct/10000:.4f}")

torch.save(model.state_dict(), 'mnist_cnn.pth')

# (env) PS C:\Users\Nathan\Desktop\cs152b\cs152b\finalproject> python train.py
# Epoch 0: accuracy = 0.9261
# Epoch 1: accuracy = 0.9592
# Epoch 2: accuracy = 0.9653
# Epoch 3: accuracy = 0.9694
# Epoch 4: accuracy = 0.9698