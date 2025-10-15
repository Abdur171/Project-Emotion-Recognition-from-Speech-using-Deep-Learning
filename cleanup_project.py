# cleanup_project.py
import os
import shutil
from pathlib import Path

def cleanup_project():
    """Remove unnecessary files and keep only essential project files"""
    
    # Essential directories to keep
    essential_dirs = [
        'data/processed',
        'models',
        'utils'
    ]
    
    # Essential files to keep
    essential_files = [
        'data_preparation.py',
        'train_cnn.py',
        'train_pretrained_final.py',
        'analysis_and_report.py',
        'cleanup_project.py',  # This file itself
        'requirements.txt' if os.path.exists('requirements.txt') else None
    ]
    
    # Results files to keep (final outputs)
    results_files = [
        'cnn_results.json',
        'pretrained_results.json',
        'final_report.md',
        'model_comparison.png',
        'class_performance_comparison.png',
        'cnn_confusion_matrix.png',
        'pretrained_confusion_matrix.png'
    ]
    
    # Files to remove (training artifacts)
    files_to_remove = [
        # Training history files
        'cnn_training_history_seed_42.png',
        'cnn_training_history_seed_123.png', 
        'cnn_training_history_seed_456.png',
        'pretrained_training_history.png',
        
        # Model checkpoints
        'best_cnn_model.pth',
        'best_pretrained_model.pth',
        
        # Temporary files
        'test_installation.py',
        'train_pretrained_safe.py',
        'train_pretrained_torchaudio.py',
        'train_pretrained.py',
        
        # Backup files
        '*.py.bak',
        '*.ipynb'
    ]
    
    print("PROJECT CLEANUP")
    print("=" * 50)
    
    # Remove unnecessary files
    removed_count = 0
    for pattern in files_to_remove:
        for file_path in Path('.').glob(pattern):
            try:
                if file_path.is_file():
                    file_path.unlink()
                    print(f"✓ Removed: {file_path}")
                    removed_count += 1
            except Exception as e:
                print(f"✗ Could not remove {file_path}: {e}")
    
    # Remove any __pycache__ directories
    for pycache_dir in Path('.').rglob('__pycache__'):
        try:
            shutil.rmtree(pycache_dir)
            print(f"✓ Removed: {pycache_dir}")
            removed_count += 1
        except Exception as e:
            print(f"✗ Could not remove {pycache_dir}: {e}")
    
    print(f"\nRemoved {removed_count} files/directories")
    
    # List essential files that should remain
    print("\n" + "=" * 50)
    print("ESSENTIAL PROJECT FILES")
    print("=" * 50)
    
    essential_count = 0
    for file_pattern in essential_files + results_files:
        if file_pattern and Path(file_pattern).exists():
            print(f"✓ Keeping: {file_pattern}")
            essential_count += 1
    
    for dir_pattern in essential_dirs:
        if Path(dir_pattern).exists():
            print(f"✓ Keeping: {dir_pattern}/")
            essential_count += 1
    
    print(f"\nTotal essential files/directories: {essential_count}")

    
    print("\n" + "=" * 50)
    print("CLEANUP COMPLETE!")
    print("=" * 50)
    print("Project is now ready for submission/report writing.")
    print("All essential files for reproducibility are preserved.")

if __name__ == "__main__":
    cleanup_project()