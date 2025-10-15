# train_cnn.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from models.cnn_model import EmotionCNN, count_parameters
from utils.data_loader import SpectrogramDataset
import json
from pathlib import Path

class CNNTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load dataset info
        with open('data/processed/dataset_info.json', 'r') as f:
            self.dataset_info = json.load(f)
        
        # Initialize model
        self.model = EmotionCNN(num_classes=4, config=config)
        self.model.to(self.device)
        
        print(f"Model parameters: {count_parameters(self.model):,}")
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)
        
    def load_data(self):
        """Load train, validation, and test datasets"""
        train_dataset = SpectrogramDataset('data/processed/train_metadata.csv')
        val_dataset = SpectrogramDataset('data/processed/val_metadata.csv')
        test_dataset = SpectrogramDataset('data/processed/test_metadata.csv')
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        print(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}, Test batches: {len(self.test_loader)}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, loader):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        val_loss = running_loss / len(loader)
        val_acc = accuracy_score(all_targets, all_preds)
        val_f1 = f1_score(all_targets, all_preds, average='macro')
        
        return val_loss, val_acc, val_f1, all_preds, all_targets
    
    def train(self):
        """Full training loop"""
        self.load_data()
        
        best_val_f1 = 0.0
        train_losses = []
        val_losses = []
        val_f1_scores = []
        
        print("Starting training...")
        for epoch in range(self.config.NUM_EPOCHS):
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_f1, _, _ = self.validate(self.val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_f1_scores.append(val_f1)
            
            print(f'Epoch {epoch+1:03d}/{self.config.NUM_EPOCHS}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
            
            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(self.model.state_dict(), 'best_cnn_model.pth')
                print(f'  → New best model saved with Val F1: {val_f1:.4f}')
        
        # Load best model for final evaluation
        self.model.load_state_dict(torch.load('best_cnn_model.pth'))
        
        return train_losses, val_losses, val_f1_scores
    
    def evaluate(self):
        """Comprehensive evaluation on test set"""
        test_loss, test_acc, test_f1, test_preds, test_targets = self.validate(self.test_loader)
        
        # Confusion matrix
        cm = confusion_matrix(test_targets, test_preds)
        emotion_labels = list(self.dataset_info['idx_to_label'].values())
        
        # Classification report
        report = classification_report(test_targets, test_preds, target_names=emotion_labels)
        
        print("\n" + "="*60)
        print("FINAL TEST RESULTS")
        print("="*60)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Macro F1: {test_f1:.4f}")
        print(f"\nClassification Report:\n{report}")
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=emotion_labels, 
                    yticklabels=emotion_labels)
        plt.title('CNN Model - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('cnn_confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved as 'cnn_confusion_matrix.png'")
        
        return test_acc, test_f1, cm

class Config:
    # Training
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50
    
    # For multiple runs
    SEEDS = [42, 123, 456]

def run_cnn_experiment(seed=42):
    """Run CNN experiment with a specific seed"""
    print(f"\n{'='*50}")
    print(f"RUNNING CNN EXPERIMENT WITH SEED: {seed}")
    print(f"{'='*50}")
    
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    config = Config()
    trainer = CNNTrainer(config)
    
    # Train the model
    train_losses, val_losses, val_f1_scores = trainer.train()
    
    # Evaluate on test set
    test_acc, test_f1, cm = trainer.evaluate()
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_f1_scores, label='Val F1 Score', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('Validation F1 Score')
    
    plt.tight_layout()
    plt.savefig(f'cnn_training_history_seed_{seed}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training history saved as 'cnn_training_history_seed_{seed}.png'")
    
    return test_acc, test_f1

def main():
    """Run CNN experiments with multiple seeds"""
    config = Config()
    test_accuracies = []
    test_f1_scores = []
    
    for seed in config.SEEDS:
        test_acc, test_f1 = run_cnn_experiment(seed)
        test_accuracies.append(test_acc)
        test_f1_scores.append(test_f1)
    
    # Report final results
    print("\n" + "="*60)
    print("CNN EXPERIMENT FINAL RESULTS")
    print("="*60)
    print(f"Seeds: {config.SEEDS}")
    print(f"Test Accuracy: {np.mean(test_accuracies):.4f} ± {np.std(test_accuracies):.4f}")
    print(f"Test Macro F1: {np.mean(test_f1_scores):.4f} ± {np.std(test_f1_scores):.4f}")
    
    # Save results
    results = {
        'seeds': config.SEEDS,
        'test_accuracies': test_accuracies,
        'test_f1_scores': test_f1_scores,
        'mean_accuracy': np.mean(test_accuracies),
        'std_accuracy': np.std(test_accuracies),
        'mean_f1': np.mean(test_f1_scores),
        'std_f1': np.std(test_f1_scores)
    }
    
    with open('cnn_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to 'cnn_results.json'")

if __name__ == "__main__":
    main()