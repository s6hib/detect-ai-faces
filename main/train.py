import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class FaceDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def load_data(data_dir):
    """Load image paths and labels from data directory."""
    real_dir = os.path.join(data_dir, 'real')
    ai_dir = os.path.join(data_dir, 'ai')

    real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    ai_images = [os.path.join(ai_dir, f) for f in os.listdir(ai_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Create labels (0 for real, 1 for AI)
    real_labels = [0] * len(real_images)
    ai_labels = [1] * len(ai_images)

    # Combine datasets
    all_images = real_images + ai_images
    all_labels = real_labels + ai_labels

    return train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

def create_model():
    """Create a transfer learning model using ResNet-50."""
    model = models.resnet50(pretrained=True)
    
    # Freeze all layers except the last few
    for param in list(model.parameters())[:-4]:
        param.requires_grad = False
    
    # Modify the final layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2)
    )
    
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    """Train the model and return training history."""
    model = model.to(device)
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        val_accuracy = accuracy_score(all_labels, all_preds)
        
        # Update history
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_accuracy'].append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}')
        print(f'Val Accuracy: {val_accuracy:.4f}')
    
    return history

def plot_training_history(history):
    """Plot training history."""
    plt.figure(figsize=(12, 4))
    
    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Data augmentation and preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and split data
    train_paths, val_paths, train_labels, val_labels = load_data('data')

    # Create datasets
    train_dataset = FaceDataset(train_paths, train_labels, transform=transform)
    val_dataset = FaceDataset(val_paths, val_labels, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Create model, loss function, and optimizer
    model = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=10,
        device=device
    )

    # Plot training history
    plot_training_history(history)

    # Save model
    torch.save(model.state_dict(), 'models/face_classifier.pth')

if __name__ == '__main__':
    main()
