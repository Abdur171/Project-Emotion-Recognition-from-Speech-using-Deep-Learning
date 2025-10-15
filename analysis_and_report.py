# analysis_and_report.py
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import torch

def load_results():
    """Load results from both models"""
    with open('cnn_results.json', 'r') as f:
        cnn_results = json.load(f)
    
    with open('pretrained_results.json', 'r') as f:
        pretrained_results = json.load(f)
    
    return cnn_results, pretrained_results

def create_comparison_plot(cnn_results, pretrained_results):
    """Create comparison plots between models"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy comparison
    models = ['CNN', 'Pretrained (wav2vec2)']
    accuracies = [cnn_results['mean_accuracy'], pretrained_results['test_accuracy']]
    acc_errors = [cnn_results['std_accuracy'], 0]  # Only CNN has multiple seeds
    
    bars = ax1.bar(models, accuracies, yerr=acc_errors, capsize=5, color=['skyblue', 'lightcoral'], alpha=0.7)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Comparison - Accuracy')
    ax1.set_ylim(0, 0.7)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # F1 comparison
    f1_scores = [cnn_results['mean_f1'], pretrained_results['test_f1']]
    f1_errors = [cnn_results['std_f1'], 0]
    
    bars = ax2.bar(models, f1_scores, yerr=f1_errors, capsize=5, color=['skyblue', 'lightcoral'], alpha=0.7)
    ax2.set_ylabel('Macro F1 Score')
    ax2.set_title('Model Comparison - Macro F1')
    ax2.set_ylim(0, 0.7)
    
    # Add value labels on bars
    for bar, f1 in zip(bars, f1_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{f1:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Model comparison plot saved as 'model_comparison.png'")

def analyze_class_performance():
    """Analyze per-class performance from the classification reports"""
    # This would typically come from the actual predictions
    # For now, we'll use the reported values
    
    classes = ['neutral', 'happy', 'sad', 'angry']
    
    # Approximate F1 scores from the classification reports
    cnn_f1 = [0.42, 0.43, 0.67, 0.62]  # From CNN seed 42 report
    pretrained_f1 = [0.11, 0.62, 0.66, 0.79]  # From pretrained report
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(classes))
    width = 0.35
    
    ax.bar(x - width/2, cnn_f1, width, label='CNN', alpha=0.7)
    ax.bar(x + width/2, pretrained_f1, width, label='Pretrained', alpha=0.7)
    
    ax.set_xlabel('Emotion Class')
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Class F1 Score Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.set_ylim(0, 1.0)
    
    # Add value labels
    for i, (cnn, pre) in enumerate(zip(cnn_f1, pretrained_f1)):
        ax.text(i - width/2, cnn + 0.02, f'{cnn:.2f}', ha='center', va='bottom')
        ax.text(i + width/2, pre + 0.02, f'{pre:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('class_performance_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Class performance comparison saved as 'class_performance_comparison.png'")

def generate_final_report(cnn_results, pretrained_results):
    """Generate a comprehensive final report"""
    
    report = f"""
EMOTION RECOGNITION PROJECT - FINAL REPORT
===========================================

DATASET
--------
- RAVDESS dataset, 4 emotions: neutral, happy, sad, angry
- 672 total samples (192 happy, 192 sad, 192 angry, 96 neutral)
- Speaker-independent split: 448 train, 112 validation, 112 test
- 24 actors (16 train, 4 validation, 4 test)

MODELS COMPARED
---------------

1. CNN Baseline (Spectrogram-based)
   - Architecture: 3 conv layers + 2 fully connected
   - Parameters: 1,535,556 (all trainable)
   - Input: Log-mel spectrograms (64 mel bands)
   - Training: 50 epochs, 3 random seeds

2. Pretrained Model (Transfer Learning)
   - Base: wav2vec2-base (torchaudio)
   - Parameters: 94,601,220 total, 230,276 trainable
   - Input: Raw audio waveforms
   - Training: 25 epochs, classifier only (encoder frozen)

RESULTS
-------

Overall Performance:
- CNN Baseline:      Accuracy = {cnn_results['mean_accuracy']:.4f} ± {cnn_results['std_accuracy']:.4f}, 
                     Macro F1 = {cnn_results['mean_f1']:.4f} ± {cnn_results['std_f1']:.4f}
- Pretrained Model:  Accuracy = {pretrained_results['test_accuracy']:.4f}, 
                     Macro F1 = {pretrained_results['test_f1']:.4f}

Key Findings:
1. The pretrained model achieved higher overall accuracy (+6% absolute)
2. The CNN model showed more balanced performance across classes
3. Both models struggled with neutral emotion recognition
4. The pretrained model excelled at recognizing angry emotion (F1=0.79)
5. Transfer learning required significantly fewer trainable parameters

ERROR ANALYSIS
--------------

CNN Model:
- Most balanced performance across emotions
- Neutral class remains challenging (F1=0.42)
- Consistent performance across random seeds

Pretrained Model:
- Excellent for angry emotion recognition
- Very poor performance on neutral class (F1=0.11)
- Potential reasons: 
  * Neutral speech patterns may be too similar to the base model's training data
  * Class imbalance (fewer neutral samples)
  * May be conflating neutral with other emotions

CONCLUSIONS
-----------

1. Both models provide reasonable baselines for emotion recognition
2. The pretrained model shows promise but needs better handling of neutral class
3. The CNN model offers more consistent and balanced performance
4. Transfer learning can provide good performance with minimal training

RECOMMENDATIONS FOR FUTURE WORK
-------------------------------

1. Address class imbalance for neutral emotion
2. Experiment with fine-tuning parts of the pretrained encoder
3. Add data augmentation specifically for emotional speech
4. Try ensemble approaches combining both models
5. Explore attention mechanisms for better temporal modeling
"""
    
    with open('final_report.md', 'w') as f:
        f.write(report)
    
    print("Final report saved as 'final_report.md'")
    print("\n" + "="*60)
    print("FINAL REPORT SUMMARY")
    print("="*60)
    print(report)

def main():
    """Generate comprehensive analysis and report"""
    print("Loading results and generating final analysis...")
    
    # Load results
    cnn_results, pretrained_results = load_results()
    
    # Create comparison plots
    create_comparison_plot(cnn_results, pretrained_results)
    analyze_class_performance()
    
    # Generate final report
    generate_final_report(cnn_results, pretrained_results)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("Generated files:")
    print("- model_comparison.png: Overall performance comparison")
    print("- class_performance_comparison.png: Per-class F1 scores")
    print("- final_report.md: Comprehensive project report")

if __name__ == "__main__":
    main()