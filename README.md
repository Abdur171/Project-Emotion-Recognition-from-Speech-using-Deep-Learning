
## Quick Start
1. Preprocess data: `python data_preparation.py`
2. Train CNN baseline: `python train_cnn.py` 
3. Train pretrained baseline: `python train_pretrained_final.py`
4. Generate report: `python analysis_and_report.py`

## Results Summary
- **CNN Baseline**: 57.4% accuracy, 55.7% F1
- **Pretrained Model**: 63.4% accuracy, 54.4% F1
- I have removed a file named best_enhanced_pretrainend_model.pth. This because the file size was too big. One can produce this file by running the train_pretrained_final.py
