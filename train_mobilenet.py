import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import ssl

# Fix SSL certificate verification issue
ssl._create_default_https_context = ssl._create_unverified_context

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

class ChestXrayDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]
    
def load_data(train_dir='train', val_dir='val'):
    """Load and prepare the dataset."""
    # Define transforms for training with data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Define transforms for validation (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Load training data
    train_normal_dir = Path(train_dir) / 'NORMAL'
    train_pneumonia_dir = Path(train_dir) / 'PNEUMONIA'

    train_normal_paths = list(train_normal_dir.glob('*.jpeg'))
    train_pneumonia_paths = list(train_pneumonia_dir.glob('*.jpeg'))

    # Balance the dataset by downsampling the larger class
    if len(train_pneumonia_paths) > len(train_normal_paths):
        train_pneumonia_paths = np.random.choice(train_pneumonia_paths, len(train_normal_paths), replace=False).tolist()
    elif len(train_normal_paths) > len(train_pneumonia_paths):
        train_normal_paths = np.random.choice(train_normal_paths, len(train_pneumonia_paths), replace=False).tolist()
    
    train_paths = train_normal_paths + train_pneumonia_paths
    train_labels = [0] * len(train_normal_paths) + [1] * len(train_pneumonia_paths)

    # Load validation data
    val_normal_dir = Path(val_dir) / 'NORMAL'
    val_pneumonia_dir = Path(val_dir) / 'PNEUMONIA'
    
    val_normal_paths = list(val_normal_dir.glob('*.jpeg'))
    val_pneumonia_paths = list(val_pneumonia_dir.glob('*.jpeg'))
    
    # Balance validation set too
    if len(val_pneumonia_paths) > len(val_normal_paths):
        val_pneumonia_paths = np.random.choice(val_pneumonia_paths, len(val_normal_paths), replace=False).tolist()
    elif len(val_normal_paths) > len(val_pneumonia_paths):
        val_normal_paths = np.random.choice(val_normal_paths, len(val_pneumonia_paths), replace=False).tolist()
    
    val_paths = val_normal_paths + val_pneumonia_paths
    val_labels = [0] * len(val_normal_paths) + [1] * len(val_pneumonia_paths)
    
    # Print dataset info
    print(f"Training set (after balancing): {len(train_normal_paths)} normal, {len(train_pneumonia_paths)} pneumonia")
    print(f"Validation set (after balancing): {len(val_normal_paths)} normal, {len(val_pneumonia_paths)} pneumonia")
    
    # Create datasets
    train_dataset = ChestXrayDataset(train_paths, train_labels, train_transform)
    val_dataset = ChestXrayDataset(val_paths, val_labels, val_transform)

    return train_dataset, val_dataset

class MobileNetPneumoniaClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(MobileNetPneumoniaClassifier, self).__init__()
        self.model = models.mobilenet_v3_small(pretrained=True)

        self.model.classifier = nn.Sequential(
            nn.Linear(576, 256),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)
    
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """Train the model and save the best one."""
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_metrics = {}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Collect predictions and labels for metrics
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')
        print('Confusion Matrix:')
        print(cm)
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_metrics = {
                'accuracy': val_acc/100,
                'confusion_matrix': cm
            }
            torch.save(model.state_dict(), 'mobilenet_model.pth')
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')

    # Plot confusion matrix for best model
    plt.figure(figsize=(8, 6))
    sns.heatmap(best_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
               xticklabels=['Normal', 'Pneumonia'],
               yticklabels=['Normal', 'Pneumonia'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix for Best MobileNet Model')
    plt.savefig('mobilenet_confusion_matrix.png')
    plt.close()
    
    # Print final metrics
    print("\nBest Model Metrics:")
    print(f"Accuracy: {best_metrics['accuracy']:.4f}")
    print("Confusion Matrix:")
    print(best_metrics['confusion_matrix'])
    
    return train_losses, val_losses, train_accs, val_accs

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """Plot training history"""
    plt.figure(figsize=(12, 4))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('mobilenet_training_history.png')
    plt.close()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load data
    print('Loading data...')
    train_dataset, val_dataset = load_data('train', 'val')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Initialize model
    model = MobileNetPneumoniaClassifier().to(device)

    # Define loss function
    weights = torch.tensor([1.0, 1.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    print('Starting training...')
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=20, device=device
    )

    # Plot training history
    print('Plotting training history...')
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    print('Training completed! Best model saved as mobilenet_model.pth')
    
    # Additional step: quantize the model for faster inference
    try:
        print("Quantizing model for faster inference...")
        # Load best model
        model = MobileNetPneumoniaClassifier()
        model.load_state_dict(torch.load('mobilenet_model.pth', map_location=torch.device('cpu')))
        model.eval()

        # Quantize the model
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )

        # Save quantized model
        torch.save(quantized_model.state_dict(), 'mobilenet_model_quantized.pth')
        print("Quantized model saved as mobilenet_model_quantized.pth")

    except Exception as e:
        print(f"Quantization failed: {str(e)}")

if __name__ == "__main__":
    main()