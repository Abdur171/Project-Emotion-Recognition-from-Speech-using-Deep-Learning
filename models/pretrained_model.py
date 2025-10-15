# models/pretrained_model.py
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torchaudio

class PretrainedEmotionClassifier(nn.Module):
    def __init__(self, num_classes=4, model_name="facebook/wav2vec2-base"):
        super(PretrainedEmotionClassifier, self).__init__()
        
        # Load pretrained model and processor with safer settings
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Trying alternative approach...")
            # Try with local files only
            self.processor = Wav2Vec2Processor.from_pretrained(model_name, local_files_only=True)
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name, local_files_only=True)
        
        # Freeze the encoder (only train classifier)
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
            
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),  # wav2vec2-base hidden size is 768
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, audio_waveforms):
        # audio_waveforms: (batch_size, sequence_length)
        
        # Process audio through wav2vec2 processor
        inputs = self.processor(
            audio_waveforms, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True,
            return_attention_mask=True
        )
        
        # Move inputs to same device as model
        input_values = inputs.input_values.to(self.wav2vec2.device)
        attention_mask = inputs.attention_mask.to(self.wav2vec2.device)
        
        # Forward pass through wav2vec2 (frozen)
        with torch.no_grad():
            outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
            
        # Use mean of hidden states as features (pooling)
        hidden_states = outputs.last_hidden_state
        # Use attention mask for mean pooling
        if attention_mask is not None:
            input_lengths = attention_mask.sum(dim=1)
            # Mask out padded positions
            hidden_states = hidden_states * attention_mask.unsqueeze(-1)
            features = hidden_states.sum(dim=1) / input_lengths.unsqueeze(-1)
        else:
            features = hidden_states.mean(dim=1)
        
        # Classifier
        logits = self.classifier(features)
        
        return logits

def count_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params