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
            'accuracy': float(test_metrics['accuracy']),
            'precision': float(test_metrics['precision']),
            'recall': float(test_metrics['recall']),
            'f1': float(test_metrics['f1']),
            'confusion_matrix': test_metrics['confusion_matrix'],
            'class_names': test_metrics['class_names']
        }
    }
    
    os.makedirs('results', exist_ok=True)
    filename = f'results/metrics_mobilenet_{"cutmix" if USE_CUTMIX else "standard"}_{"fewshot" if USE_FEW_SHOT else "full"}.json'
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=2)

def evaluate_model(model, test_loader, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.numpy())

    cm = confusion_matrix(all_labels, all_preds, labels=np.arange(len(class_names)))
    
    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
        'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'confusion_matrix': cm.tolist(),
        'class_names': class_names
    }

def build_mobilenet_v3(num_classes):
    model = models.mobilenet_v3_small(pretrained=True)
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze classifier
    for param in model.classifier.parameters():
        param.requires_grad = True
    # Replace final layer
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model.to(device)

def train_mobilenet_v3(train_loader, val_loader, num_classes, class_names):
    model = build_mobilenet_v3(num_classes)
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

    best_acc = 0.0

    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            if USE_CUTMIX and isinstance(targets, tuple):
                inputs, labels_a, labels_b, lam = targets
                targets = (labels_a, labels_b, lam)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)

        # Calculate metrics
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        acc = correct / total
        
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(acc)

        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {acc:.4f}")

        if acc > best_acc:
            torch.save(model.state_dict(), 'results/mobilenet_v3_final.pth')
            best_acc = acc

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
        model, history = train_mobilenet_v3(train_loader, val_loader, NUM_CLASSES, class_names)

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