# data_preparation.py
import os
import pandas as pd
import numpy as np
import torch
import torchaudio
from pathlib import Path
import librosa
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

class RAVDESSDataPreparer:
    def __init__(self, data_path="data/raw", output_path="data/processed"):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.sample_rate = 16000
        self.n_mels = 64
        self.n_fft = 1024
        self.hop_length = 512
        self.duration = 3.0  # Target duration in seconds
        
        # Emotion mapping based on RAVDESS documentation
        self.emotion_map = {
            1: 'neutral',  # 01 = neutral
            3: 'happy',    # 03 = happy  
            4: 'sad',      # 04 = sad
            5: 'angry'     # 05 = angry
        }
        
        # Label to index mapping
        self.label_to_idx = {emotion: idx for idx, emotion in enumerate(self.emotion_map.values())}
        self.idx_to_label = {idx: emotion for emotion, idx in self.label_to_idx.items()}
        
    def parse_filename(self, filename):
        """
        Parse RAVDESS filename according to the convention:
        03-01-06-01-02-01-12.wav
        Modality (03=audio-only) - Vocal channel (01=speech) - Emotion (06=fearful) - 
        Intensity (01=normal,02=strong) - Statement (01="Kids",02="Dogs") - 
        Repetition (01=1st) - Actor (01-24)
        """
        parts = filename.stem.split('-')
        if len(parts) != 7:
            return None
            
        try:
            modality = int(parts[0])  # 03 = audio-only
            vocal_channel = int(parts[1])  # 01 = speech
            emotion = int(parts[2])  # Emotion code
            intensity = int(parts[3])  # 01=normal, 02=strong
            statement = int(parts[4])  # 01="Kids", 02="Dogs"
            repetition = int(parts[5])  # 01 or 02
            actor = int(parts[6])  # 01-24
            
            return {
                'modality': modality,
                'vocal_channel': vocal_channel,
                'emotion': emotion,
                'intensity': intensity,
                'statement': statement,
                'repetition': repetition,
                'actor': actor,
                'gender': 'female' if actor % 2 == 0 else 'male'
            }
        except ValueError:
            return None
    
    def find_audio_files(self):
        """Find all RAVDESS audio files and parse their metadata"""
        audio_files = list(self.data_path.rglob("*.wav"))
        data_records = []
        
        print(f"Found {len(audio_files)} audio files")
        
        for file_path in tqdm(audio_files, desc="Parsing audio files"):
            metadata = self.parse_filename(file_path)
            if metadata is None:
                continue
                
            # Filter for our target emotions and audio-only modality
            if (metadata['modality'] == 3 and  # audio-only
                metadata['vocal_channel'] == 1 and  # speech
                metadata['emotion'] in self.emotion_map):
                
                record = {
                    'file_path': str(file_path),
                    'filename': file_path.name,
                    'emotion_id': metadata['emotion'],
                    'emotion': self.emotion_map[metadata['emotion']],
                    'emotion_idx': self.label_to_idx[self.emotion_map[metadata['emotion']]],
                    'intensity': metadata['intensity'],
                    'statement': metadata['statement'],
                    'repetition': metadata['repetition'],
                    'actor': metadata['actor'],
                    'gender': metadata['gender']
                }
                data_records.append(record)
        
        return pd.DataFrame(data_records)
    
    def analyze_dataset(self, df):
        """Analyze the dataset distribution"""
        print("\n" + "="*50)
        print("DATASET ANALYSIS")
        print("="*50)
        
        print(f"Total samples: {len(df)}")
        print(f"Number of actors: {df['actor'].nunique()}")
        print(f"Male actors: {df[df['gender'] == 'male']['actor'].nunique()}")
        print(f"Female actors: {df[df['gender'] == 'female']['actor'].nunique()}")
        
        print("\nEmotion distribution:")
        emotion_counts = df['emotion'].value_counts()
        for emotion, count in emotion_counts.items():
            print(f"  {emotion}: {count} samples")
        
        print("\nSamples per actor:")
        actor_counts = df['actor'].value_counts().sort_index()
        for actor, count in actor_counts.items():
            print(f"  Actor {actor:02d}: {count} samples")
        
        return emotion_counts, actor_counts
    
    def preprocess_audio(self, audio_path):
        """Load and preprocess audio to log-mel spectrogram"""
        try:
            # Load audio
            waveform, sr = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Ensure consistent duration (pad or trim to 3 seconds)
            target_length = int(self.duration * self.sample_rate)
            if waveform.shape[1] < target_length:
                # Pad with zeros
                pad_length = target_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_length))
            elif waveform.shape[1] > target_length:
                # Trim to target length
                waveform = waveform[:, :target_length]
            
            # Compute log-mel spectrogram
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels
            )
            
            spectrogram = mel_transform(waveform)
            log_spectrogram = torch.log(spectrogram + 1e-9)  # Add small value to avoid log(0)
            
            return log_spectrogram.squeeze(0)  # Remove channel dimension
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def save_spectrograms(self, df):
        """Precompute and save spectrograms for all audio files"""
        spectrogram_dir = self.output_path / "spectrograms"
        spectrogram_dir.mkdir(exist_ok=True)
        
        spectrogram_paths = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing spectrograms"):
            # Create unique filename for spectrogram
            spec_filename = f"spec_{Path(row['filename']).stem}.pt"
            spec_path = spectrogram_dir / spec_filename
            
            # Process and save spectrogram
            spectrogram = self.preprocess_audio(row['file_path'])
            if spectrogram is not None:
                torch.save(spectrogram, spec_path)
                spectrogram_paths.append(str(spec_path))
            else:
                spectrogram_paths.append(None)
        
        # Add spectrogram paths to dataframe
        df['spectrogram_path'] = spectrogram_paths
        
        # Remove any rows where spectrogram processing failed
        df = df.dropna(subset=['spectrogram_path'])
        
        return df
    
    def create_splits(self, df, split_type='speaker_independent', test_actors=[21, 22, 23, 24]):
        """
        Create train/validation/test splits
        Options:
        - 'speaker_independent': Hold out specific actors for test
        - 'leave_one_speaker_out': For cross-validation (we'll implement later)
        """
        if split_type == 'speaker_independent':
            # Use specific actors for test set
            test_mask = df['actor'].isin(test_actors)
            train_val_mask = ~test_mask
            
            train_val_df = df[train_val_mask].copy()
            test_df = df[test_mask].copy()
            
            # Further split train_val into train and validation (80-20)
            actors_train_val = train_val_df['actor'].unique()
            np.random.shuffle(actors_train_val)
            split_point = int(0.8 * len(actors_train_val))
            
            train_actors = actors_train_val[:split_point]
            val_actors = actors_train_val[split_point:]
            
            train_df = train_val_df[train_val_df['actor'].isin(train_actors)]
            val_df = train_val_df[train_val_df['actor'].isin(val_actors)]
            
            print(f"\nData splits:")
            print(f"Train: {len(train_df)} samples ({len(train_actors)} actors)")
            print(f"Validation: {len(val_df)} samples ({len(val_actors)} actors)")  
            print(f"Test: {len(test_df)} samples ({len(test_actors)} actors)")
            
            return train_df, val_df, test_df
        
        else:
            raise ValueError(f"Unknown split type: {split_type}")
    
    def save_metadata(self, train_df, val_df, test_df):
        """Save metadata and splits to files"""
        # Save full dataset info
        dataset_info = {
            'sample_rate': self.sample_rate,
            'n_mels': self.n_mels,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'duration': self.duration,
            'emotion_map': self.emotion_map,
            'label_to_idx': self.label_to_idx,
            'idx_to_label': self.idx_to_label
        }
        
        with open(self.output_path / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        # Save splits
        train_df.to_csv(self.output_path / 'train_metadata.csv', index=False)
        val_df.to_csv(self.output_path / 'val_metadata.csv', index=False) 
        test_df.to_csv(self.output_path / 'test_metadata.csv', index=False)
        
        # Save full dataset
        full_df = pd.concat([train_df, val_df, test_df])
        full_df.to_csv(self.output_path / 'full_metadata.csv', index=False)
        
        print(f"\nMetadata saved to {self.output_path}")
    
    def visualize_sample_spectrograms(self, df, num_samples=4):
        """Visualize sample spectrograms for each emotion"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.ravel()
        
        for idx, emotion in enumerate(self.emotion_map.values()):
            emotion_samples = df[df['emotion'] == emotion]
            if len(emotion_samples) > 0:
                sample = emotion_samples.iloc[0]
                spectrogram = torch.load(sample['spectrogram_path'])
                
                # Plot spectrogram
                im = axes[idx].imshow(spectrogram.numpy(), aspect='auto', origin='lower')
                axes[idx].set_title(f'{emotion.capitalize()} (Actor {sample["actor"]})')
                axes[idx].set_xlabel('Time frames')
                axes[idx].set_ylabel('Mel bins')
                plt.colorbar(im, ax=axes[idx])
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'sample_spectrograms.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Sample spectrograms saved to {self.output_path / 'sample_spectrograms.png'}")

def main():
    # Initialize data preparer
    preparer = RAVDESSDataPreparer(
        data_path="data/raw/ravdess",  # Update this path to your RAVDESS location
        output_path="data/processed"
    )
    
    # Step 1: Find and parse audio files
    print("Step 1: Finding and parsing audio files...")
    df = preparer.find_audio_files()
    
    if len(df) == 0:
        print("No audio files found! Please check the data path.")
        return
    
    # Step 2: Analyze dataset
    print("\nStep 2: Analyzing dataset...")
    emotion_counts, actor_counts = preparer.analyze_dataset(df)
    
    # Step 3: Precompute and save spectrograms
    print("\nStep 3: Precomputing spectrograms...")
    df = preparer.save_spectrograms(df)
    
    # Step 4: Create data splits
    print("\nStep 4: Creating data splits...")
    train_df, val_df, test_df = preparer.create_splits(
        df, 
        split_type='speaker_independent',
        test_actors=[21, 22, 23, 24]  # Hold out last 4 actors for test
    )
    
    # Step 5: Save metadata
    print("\nStep 5: Saving metadata...")
    preparer.save_metadata(train_df, val_df, test_df)
    
    # Step 6: Visualize samples
    print("\nStep 6: Creating visualizations...")
    preparer.visualize_sample_spectrograms(df)
    
    print("\n" + "="*50)
    print("DATA PREPARATION COMPLETE!")
    print("="*50)
    print(f"Final dataset size: {len(df)} samples")
    print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    print(f"Output directory: {preparer.output_path}")

if __name__ == "__main__":
    main()