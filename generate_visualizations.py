import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

def load_metrics():
    with open('results/metrics_standard_fewshot.json', 'r') as f:
        return json.load(f)

def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/training_history.png')
    plt.close()

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(20, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('results/confusion_matrix.png')
    plt.close()

def generate_visualizations():
    metrics = load_metrics()
    
    # Plot training history
    plot_training_history(metrics['training_history'])
    
    # Plot confusion matrix
    cm = np.array(metrics['test_metrics']['confusion_matrix'])
    plot_confusion_matrix(cm, metrics['test_metrics']['class_names'])
    
    # Print metrics
    print("\nTest Metrics:")
    print(f"Accuracy: {metrics['test_metrics']['accuracy']:.4f}")
    print(f"Precision: {metrics['test_metrics']['precision']:.4f}")
    print(f"Recall: {metrics['test_metrics']['recall']:.4f}")
    print(f"F1-Score: {metrics['test_metrics']['f1']:.4f}")

if __name__ == "__main__":
    generate_visualizations()
