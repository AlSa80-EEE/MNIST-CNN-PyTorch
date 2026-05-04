import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.model import MNIST_CNN
from src.train import train_model
from src.utils import visualize_predictions, plot_loss_curve

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Used device: {device}")

    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

   
    print("\n--- Training Starting ---")
    model, train_losses = train_model() 

    
    print("\n--- Analyze and Visualization ---")
    
  
    plot_loss_curve(train_losses)
    
    
    visualize_predictions(model, test_loader, device)

    print("\nProject succesfully completed. Model saved as 'mnist_cnn.pth'")

if __name__ == "__main__":
    main()
