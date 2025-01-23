import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from preprocessing_data import get_data_loaders
import numpy as np
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ========== Configuration ==========
USE_CUTMIX = True        # Set True for advanced augmentation (CutMix)
USE_FEW_SHOT = True      # Set True for few-shot learning
K_SHOT = 5                # Samples per class for few-shot
EPOCHS = 50           
LR = 0.0001
NUM_CLASSES = 14
# ===================================

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_metrics(history, test_metrics):
    metrics = {
        'training_history': history,
        'test_metrics': {
            'accuracy': test_metrics['accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1': test_metrics['f1'],
            'confusion_matrix': test_metrics['confusion_matrix'].tolist(),
            'class_names': test_metrics['class_names']
        }
    }
    
    os.makedirs('results', exist_ok=True)
    filename = f'results/metrics_resnet_{"cutmix" if USE_CUTMIX else "standard"}_{"fewshot" if USE_FEW_SHOT else "full"}.json'
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=2)

def evaluate_model(model, test_loader, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds, labels=np.arange(len(class_names)))
    
    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
        'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'confusion_matrix': cm,
        'class_names': class_names
    }

def build_resnet(num_classes):
    model = models.resnet50(pretrained=True)
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    # Replace final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

def train_model(train_loader, val_loader, num_classes, class_names):
    model = build_resnet(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': []
    }

    print("\n=== Training Setup ===")
    print(f"Mode: {'CutMix' if USE_CUTMIX else 'Standard'} {'+ Few-Shot' if USE_FEW_SHOT else ''}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Learning rate: {LR}\n")

    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            if USE_CUTMIX and labels.dim() > 1:
                labels = labels.argmax(dim=1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        # Store metrics
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(val_acc)

        print(f'Epoch {epoch+1}/{EPOCHS}')
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        print(f'Val Accuracy: {val_acc:.4f}\n')

    torch.save(model.state_dict(), 'results/model_final.pth')
    return model, history

if __name__ == "__main__":
    try:
        # Load data
        train_loader, val_loader, test_loader = get_data_loaders(
            train_dir='data/train',
            val_dir='data/valid',
            test_dir='data/test',
            num_classes=NUM_CLASSES,
            augmentation='advanced' if USE_CUTMIX else 'standard',
            samples_per_class=K_SHOT if USE_FEW_SHOT else None
        )

        # Get class names
        class_names = test_loader.dataset.classes

        # Train model
        model, history = train_model(train_loader, val_loader, NUM_CLASSES, class_names)

        # Final evaluation
        print("\n=== Final Test Evaluation ===")
        test_metrics = evaluate_model(model, test_loader, class_names)
        save_metrics(history, test_metrics)
        
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test Precision: {test_metrics['precision']:.4f}")
        print(f"Test Recall: {test_metrics['recall']:.4f}")
        print(f"Test F1-Score: {test_metrics['f1']:.4f}")
        print(f"\nMetrics saved to results directory")

    except Exception as e:
        print(f"Error: {str(e)}")