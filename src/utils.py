import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_predictions(model, test_loader, device, num_images=5):

    model.eval() 

    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    plt.figure(figsize=(12, 4))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        
        img = images[i].cpu().numpy().squeeze()
        
        plt.imshow(img, cmap='gray')
        plt.title(f"Pred: {preds[i].item()}\nActual: {labels[i].item()}")
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

def plot_loss_curve(train_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.xlabel('Batch Index (x100)')
    plt.ylabel('Loss Value')
    plt.title('Model Convergence Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

