import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from train import FaceDataset, create_model

def load_trained_model(model_path, device):
    """Load a trained model from disk."""
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def predict_image(model, image_path, transform, device):
    """Make prediction for a single image."""
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    
    return predicted.item(), probabilities[0].cpu().numpy()

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def evaluate_model(model_path, data_dir, batch_size=32):
    """Evaluate model performance on test set."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load model
    model = load_trained_model(model_path, device)

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Prepare data
    real_dir = os.path.join(data_dir, 'real')
    ai_dir = os.path.join(data_dir, 'ai')

    all_predictions = []
    all_labels = []
    all_probabilities = []

    # Evaluate on real images
    print("\nEvaluating real images...")
    for img_name in os.listdir(real_dir):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(real_dir, img_name)
            pred, probs = predict_image(model, img_path, transform, device)
            all_predictions.append(pred)
            all_labels.append(0)  # 0 for real
            all_probabilities.append(probs)

    # Evaluate on AI images
    print("Evaluating AI-generated images...")
    for img_name in os.listdir(ai_dir):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(ai_dir, img_name)
            pred, probs = predict_image(model, img_path, transform, device)
            all_predictions.append(pred)
            all_labels.append(1)  # 1 for AI
            all_probabilities.append(probs)

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)

    # Calculate and print metrics
    classes = ['Real', 'AI-Generated']
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=classes))

    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_predictions, classes)

    # Calculate and print additional metrics
    accuracy = np.mean(all_predictions == all_labels)
    print(f"\nOverall Accuracy: {accuracy:.4f}")

    # Print examples of high-confidence mistakes
    confidence_threshold = 0.9
    for i in range(len(all_labels)):
        pred_prob = all_probabilities[i][all_predictions[i]]
        if pred_prob > confidence_threshold and all_predictions[i] != all_labels[i]:
            print(f"\nHigh-confidence mistake found:")
            print(f"True label: {classes[all_labels[i]]}")
            print(f"Predicted: {classes[all_predictions[i]]} with {pred_prob:.4f} confidence")

if __name__ == "__main__":
    model_path = "models/face_classifier.pth"
    data_dir = "data"
    evaluate_model(model_path, data_dir)
