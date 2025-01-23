import os
import json
import torch
import numpy as np
from torchvision import models, transforms
from preprocessing_data import ImageDataset
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix
)

# ========== Reproducibility Setup ==========
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_zero_shot_metrics(test_metrics, model_name):
    """Save evaluation metrics to JSON file"""
    metrics = {
        'test_metrics': {
            'accuracy': float(test_metrics['accuracy']),
            'precision': float(test_metrics['precision']),
            'recall': float(test_metrics['recall']),
            'f1': float(test_metrics['f1']),
            'confusion_matrix': test_metrics['confusion_matrix'],
            'class_names': test_metrics['class_names']
        }
    }
    
    os.makedirs('results', exist_ok=True)
    filename = f'results/zero_shot_{model_name}_metrics.json'
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=2)

def zero_shot_inference(model_name='resnet50', test_dir='data/test'):
    """Run zero-shot inference with pretrained model"""
    
    # ========== Model Setup with Fixed Initialization ==========
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        # Initialize final layer with fixed weights
        torch.manual_seed(42)  # Fixed seed for layer initialization
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, 14)
        
    elif model_name == 'mobilenet':
        model = models.mobilenet_v3_small(pretrained=True)
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        # Initialize final layer with fixed weights
        torch.manual_seed(42)  # Fixed seed for layer initialization
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(num_features, 14)
    
    model = model.to(device)
    model.eval()

    # ========== Data Loading ==========
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = ImageDataset(
    test_dir, 
    transform=test_transform, 
    return_class_names=True  # Critical for zero-shot
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False,  # Maintain deterministic order
        pin_memory=True
    )
    class_names = test_dataset.get_class_list()

    # ========== Inference Loop ==========
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            # Convert class names to indices
            label_indices = [class_names.index(label) for label in labels]
            
            all_preds.extend(preds)
            all_labels.extend(label_indices)

    # ========== Metrics Calculation ==========
    # Force 14x14 matrix even if some classes are missing
    cm = confusion_matrix(
        all_labels, 
        all_preds, 
        labels=np.arange(len(class_names))
    )
    
    test_metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, 
                                   average='weighted', zero_division=0),
        'recall': recall_score(all_labels, all_preds,
                              average='weighted', zero_division=0),
        'f1': f1_score(all_labels, all_preds,
                      average='weighted', zero_division=0),
        'confusion_matrix': cm.tolist(),
        'class_names': class_names
    }

    # ========== Save and Display Results ==========
    save_zero_shot_metrics(test_metrics, model_name)
    
    print(f"\nZero-Shot {model_name.upper()} Results:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1-Score: {test_metrics['f1']:.4f}")

if __name__ == "__main__":
    # Example usage for both models
    zero_shot_inference(model_name='resnet50', test_dir='data/test')
    zero_shot_inference(model_name='mobilenet', test_dir='data/test')