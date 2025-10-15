# train_pretrained_enhanced.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import torchaudio
from collections import Counter


class EnhancedTorchAudioPretrainedClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(EnhancedTorchAudioPretrainedClassifier, self).__init__()
        
        print("Loading enhanced torchaudio pretrained model...")
        
        # Load pretrained wav2vec2 from torchaudio
        try:
            self.bundle = torchaudio.pipelines.WAV2VEC2_BASE
            self.model = self.bundle.get_model()
            self.feature_dim = self.bundle._params['encoder_embed_dim']
            print(f"Successfully loaded WAV2VEC2_BASE with feature dim: {self.feature_dim}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Freeze the encoder (transfer learning)
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Enhanced classifier head with batch normalization and better architecture
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, num_classes)
        )
        
    def forward(self, audio_waveforms):
        # audio_waveforms: (batch_size, sequence_length)
        
        # Forward pass through pretrained model (frozen)
        with torch.no_grad():
            features, _ = self.model(audio_waveforms)
            
        # Global average pooling over time
        features = features.mean(dim=1)
        
        # Enhanced classifier
        logits = self.classifier(features)
        
        return logits

# Data loader
class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_file, target_sample_rate=16000, target_duration=3.0):
        self.metadata = pd.read_csv(metadata_file)
        self.target_sample_rate = target_sample_rate
        self.target_duration = target_duration
        self.target_length = int(target_duration * target_sample_rate)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        audio_path = self.metadata.iloc[idx]['file_path']
        label = self.metadata.iloc[idx]['emotion_idx']
        
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample if necessary
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            waveform = resampler(waveform)
        
        # Ensure consistent duration (pad or trim)
        if waveform.shape[1] < self.target_length:
            pad_length = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        elif waveform.shape[1] > self.target_length:
            waveform = waveform[:, :self.target_length]
            
        return waveform.squeeze(0), label  # Remove channel dimension

class EnhancedPretrainedTrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Load dataset info
        with open('data/processed/dataset_info.json', 'r') as f:
            self.dataset_info = json.load(f)
        
        # Initialize enhanced model
        self.model = EnhancedTorchAudioPretrainedClassifier(num_classes=4)
        self.model.to(self.device)
        
        # Count parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Total parameters: {total_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")
        
        # Compute class weights for imbalance handling
        self.class_weights = self._compute_class_weights()
        print(f"Class weights: {self.class_weights}")
        
        # Enhanced loss function with class weighting
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))
        
        # Enhanced optimizer with weight decay
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        
        # Enhanced learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=30, eta_min=1e-6)
        
    def _compute_class_weights(self):
        """Compute class weights to handle imbalance"""
        train_df = pd.read_csv('data/processed/train_metadata.csv')
        class_counts = train_df['emotion_idx'].value_counts().sort_index()
        
        # Inverse frequency weighting
        total_samples = len(train_df)
        num_classes = len(class_counts)
        weights = total_samples / (num_classes * class_counts)
        
        # Normalize weights
        weights = weights / weights.sum() * num_classes
        
        return torch.tensor(weights.values, dtype=torch.float32)
    
    def load_data(self):
        """Load train, validation, and test datasets with optional class balancing"""
        train_dataset = AudioDataset('data/processed/train_metadata.csv')
        val_dataset = AudioDataset('data/processed/val_metadata.csv')
        test_dataset = AudioDataset('data/processed/test_metadata.csv')
        
        batch_size = 8
        
        # Use weighted sampler for training to handle class imbalance
        train_df = pd.read_csv('data/processed/train_metadata.csv')
        class_counts = train_df['emotion_idx'].value_counts().sort_index()
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[idx] for idx in train_df['emotion_idx']]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}, Test batches: {len(self.test_loader)}")
        
        # Print class distribution
        print("Training class distribution:", dict(class_counts))
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            if batch_idx % 10 == 0:
                batch_acc = (predicted == target).float().mean().item()
                print(f'  Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}, Acc: {batch_acc:.4f}')
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        epoch_f1 = f1_score(all_targets, all_preds, average='macro')
        
        return epoch_loss, epoch_acc, epoch_f1
    
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
        
        # Per-class F1
        per_class_f1 = f1_score(all_targets, all_preds, average=None)
        
        return val_loss, val_acc, val_f1, per_class_f1, all_preds, all_targets
    
    def train(self, num_epochs=35):
        """Enhanced training loop with early stopping and better monitoring"""
        self.load_data()
        
        best_val_f1 = 0.0
        best_val_acc = 0.0
        patience = 8
        patience_counter = 0
        train_losses = []
        val_losses = []
        val_f1_scores = []
        train_f1_scores = []
        
        print("Starting enhanced training...")
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc, train_f1 = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_f1, per_class_f1, _, _ = self.validate(self.val_loader)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_f1_scores.append(val_f1)
            train_f1_scores.append(train_f1)
            
            print(f'Epoch {epoch+1:03d}/{num_epochs}:')
            print(f'  Train => Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, F1: {train_f1:.4f}')
            print(f'  Val   => Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}')
            print(f'  Per-class F1: {per_class_f1}')
            print(f'  LR: {current_lr:.2e}')
            
            # Save best model based on validation F1
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_acc = val_acc
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_f1': best_val_f1,
                    'val_acc': best_val_acc,
                    'epoch': epoch
                }, 'best_enhanced_pretrained_model.pth')
                print(f'  → New best model saved with Val F1: {val_f1:.4f}')
                patience_counter = 0
            else:
                patience_counter += 1
                print(f'  No improvement for {patience_counter} epochs')
            
            # Early stopping
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        # Load best model for final evaluation
        checkpoint = torch.load('best_enhanced_pretrained_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Best validation - F1: {checkpoint['val_f1']:.4f}, Acc: {checkpoint['val_acc']:.4f}")
        
        return train_losses, val_losses, val_f1_scores, train_f1_scores
    
    def evaluate(self):
        """Comprehensive evaluation on test set"""
        test_loss, test_acc, test_f1, per_class_f1, test_preds, test_targets = self.validate(self.test_loader)
        
        # Confusion matrix
        cm = confusion_matrix(test_targets, test_preds)
        emotion_labels = list(self.dataset_info['idx_to_label'].values())
        
        # Classification report
        report = classification_report(test_targets, test_preds, target_names=emotion_labels)
        
        print("\n" + "="*60)
        print("ENHANCED PRETRAINED MODEL - FINAL TEST RESULTS")
        print("="*60)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Macro F1: {test_f1:.4f}")
        print(f"Per-class F1: {dict(zip(emotion_labels, per_class_f1))}")
        print(f"\nClassification Report:\n{report}")
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=emotion_labels, 
                    yticklabels=emotion_labels)
        plt.title('Enhanced Pretrained Model - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('enhanced_pretrained_confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved as 'enhanced_pretrained_confusion_matrix.png'")
        
        return test_acc, test_f1, per_class_f1

def plot_enhanced_training_history(train_losses, val_losses, train_f1_scores, val_f1_scores):
    """Plot enhanced training history"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.plot(val_losses, label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Training and Validation Loss')
    ax1.grid(True, alpha=0.3)
    
    # F1 score plot
    ax2.plot(train_f1_scores, label='Train F1', linewidth=2)
    ax2.plot(val_f1_scores, label='Val F1', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.legend()
    ax2.set_title('Training and Validation F1 Score')
    ax2.grid(True, alpha=0.3)
    
    # Combined loss and F1
    ax3.plot(train_losses, label='Train Loss', color='blue', alpha=0.7)
    ax3.plot(val_losses, label='Val Loss', color='red', alpha=0.7)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss', color='black')
    ax3.tick_params(axis='y', labelcolor='black')
    ax3.legend(loc='upper left')
    
    ax4 = ax3.twinx()
    ax4.plot(train_f1_scores, label='Train F1', color='green', linestyle='--', alpha=0.7)
    ax4.plot(val_f1_scores, label='Val F1', color='orange', linestyle='--', alpha=0.7)
    ax4.set_ylabel('F1 Score', color='black')
    ax4.tick_params(axis='y', labelcolor='black')
    ax4.legend(loc='upper right')
    ax3.set_title('Combined Loss and F1 Score')
    ax3.grid(True, alpha=0.3)
    
    # Final comparison
    epochs = range(1, len(train_losses) + 1)
    ax4.bar(epochs, [val_f1_scores[-1]], alpha=0.6, color='green', label='Final Val F1')
    ax4.set_ylim(0, 1)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('enhanced_pretrained_training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Enhanced training history saved as 'enhanced_pretrained_training_history.png'")

def main():
    """Run enhanced pretrained model experiment"""
    print("="*60)
    print("ENHANCED PRETRAINED MODEL EXPERIMENT")
    print("="*60)
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    trainer = EnhancedPretrainedTrainer()
    
    # Train the enhanced model
    train_losses, val_losses, val_f1_scores, train_f1_scores = trainer.train(num_epochs=35)
    
    # Evaluate on test set
    test_acc, test_f1, per_class_f1 = trainer.evaluate()
    
    # Plot enhanced training history
    plot_enhanced_training_history(train_losses, val_losses, train_f1_scores, val_f1_scores)
    
    # Save enhanced results
    results = {
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'per_class_f1': dict(zip(list(trainer.dataset_info['idx_to_label'].values()), per_class_f1)),
        'model_type': 'enhanced_pretrained_torchaudio',
        'improvements': [
            'Class weighting for imbalance',
            'Enhanced classifier architecture',
            'Batch normalization',
            'AdamW optimizer with weight decay',
            'Cosine annealing scheduler',
            'Gradient clipping',
            'Early stopping',
            'Weighted random sampling'
        ]
    }
    
    with open('enhanced_pretrained_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEnhanced results saved to 'enhanced_pretrained_results.json'")
    
    # Compare with previous results
    print("\n" + "="*60)
    print("COMPARISON WITH PREVIOUS MODELS")
    print("="*60)
    
    # Load previous results if available
    try:
        with open('pretrained_results.json', 'r') as f:
            original_pretrained = json.load(f)
        print(f"Original Pretrained - Accuracy: {original_pretrained['test_accuracy']:.4f}, F1: {original_pretrained['test_f1']:.4f}")
    except:
        print("Original pretrained results not found")
    
    try:
        with open('cnn_results.json', 'r') as f:
            cnn_results = json.load(f)
        print(f"CNN Baseline        - Accuracy: {cnn_results['mean_accuracy']:.4f}, F1: {cnn_results['mean_f1']:.4f}")
    except:
        print("CNN results not found")
    
    print(f"Enhanced Pretrained - Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
    print(f"Per-class F1 improvement: {per_class_f1}")

if __name__ == "__main__":
    main()