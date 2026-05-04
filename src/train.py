import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import MNIST_CNN  

def train_model():
    
    batch_size = 64
    learning_rate = 0.001
    epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) 
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

 
    model = MNIST_CNN().to(device)
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            
            optimizer.zero_grad()
            
            
            output = model(data)
            loss = criterion(output, target)
            
          
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if batch_idx % 100 == 99:
                print(f'Epoch: {epoch+1} | Batch: {batch_idx+1} | Loss: {running_loss/100:.4f}')
                running_loss = 0.0

   
    torch.save(model.state_dict(), "mnist_cnn.pth")
    print("Training completed and model saved")

if __name__ == "__main__":
    train_model()
