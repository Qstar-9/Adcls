import os
import pandas as pd
import numpy as np
import torch
import torchaudio
from tqdm.notebook import tqdm
import random
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, PitchShift, Gain, Resample
from audiomentations.core.transforms_interface import BaseTransform
import traceback

# --- Configuration (Unchanged) ---
SAMPLE_RATE = 44100
TARGET_DURATION = 5
MAX_DURATION_TRUNCATE = 30.0
NUM_AUGMENTATIONS_PER_SAMPLE = 2
OUTPUT_DIR = '/home/renmengxing/audioRec/audio_data/augmented_audio_data'
INFO_CSV_NAME = '/home/renmengxing/audioRec/audio_data/augmented_train.csv'

# --- Corrected PadOrTruncate Class ---
class PadOrTruncate(BaseTransform):
    """
    Pads or truncates audio to a target duration.
    This class inherits from audiomentations.BaseTransform and correctly
    implements the required abstract methods.
    """
    def __init__(self, target_duration_seconds, sample_rate, p=1.0):
        super().__init__(p)
        self.target_duration_seconds = target_duration_seconds
        self.sample_rate = sample_rate
        self.target_samples = int(target_duration_seconds * sample_rate)

    def get_transform_parameters(self, samples, sample_rate):
        """
        This method is required by the BaseTransform class.
        For this specific transform, the parameters are fixed during initialization,
        so we don't need to determine any random parameters for each call.
        We can return an empty dictionary.
        """
        return {}

    def to_dict_private(self):
        """
        This method is required by the BaseTransform class for serialization.
        It should return a dictionary of the transform's initialization parameters.
        """
        return {
            "target_duration_seconds": self.target_duration_seconds,
            "sample_rate": self.sample_rate,
            "p": self.p,
        }

    def apply(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Applies the padding or truncation to the input audio samples.
        """
        current_samples = samples.shape[0]

        if current_samples == self.target_samples:
            return samples  # No change needed

        if current_samples < self.target_samples:
            # Pad the audio to the target length
            pad_needed = self.target_samples - current_samples
            pad_left = pad_needed // 2
            pad_right = pad_needed - pad_left
            padded_samples = np.pad(samples, (pad_left, pad_right), 'constant', constant_values=0)
            return padded_samples
        else:  # current_samples > self.target_samples
            # Truncate the audio from a random start point
            start_index = np.random.randint(0, current_samples - self.target_samples + 1)
            truncated_samples = samples[start_index : start_index + self.target_samples]
            return truncated_samples


# --- Logic: Separating Preprocessing and Augmentation (Unchanged) ---

# 1. Define a transform for preprocessing (ensuring all clips are TARGET_DURATION long)
preprocess_transform = PadOrTruncate(
    target_duration_seconds=TARGET_DURATION,
    sample_rate=SAMPLE_RATE,
    p=1.0
)

# 2. Define a pipeline for "augmentation" to be applied to the preprocessed audio
augmentation_pipeline = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.3),
    Gain(min_gain_db=-6.0, max_gain_db=6.0, p=0.3),
    Resample(min_sample_rate=int(SAMPLE_RATE * 0.9), max_sample_rate=int(SAMPLE_RATE * 1.1), p=0.2),
])

# --- Main Function (Modified to call .apply()) ---
def augment_audio_dataset_fixed(input_csv_path, output_dir, target_duration,
                                max_duration_truncate, num_augmentations_per_sample,
                                preprocessor, augmentor):
    df = pd.read_csv(input_csv_path)
    augmented_data = []
    # df = df.head(5) # Keeping this for quick testing as per your original code

    os.makedirs(output_dir, exist_ok=True)
    for label in df['label'].unique():
        os.makedirs(os.path.join(output_dir, str(label)), exist_ok=True)

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting audio"):
        fname = row['fname']
        label = row['label']
        original_duration = row['duration']
        audio_path = os.path.join('/home/renmengxing/audioRec/audio_data/train/', fname)

        if not os.path.exists(audio_path):
            print(f"Warning: File not found {audio_path}, skipping.")
            continue

        try:
            samples, sr = sf.read(audio_path)
            if samples.ndim > 1:
                samples = samples.mean(axis=1)

            if sr != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
                samples = resampler(torch.from_numpy(samples).float().unsqueeze(0)).squeeze(0).numpy()
                sr = SAMPLE_RATE

            if original_duration > max_duration_truncate:
                num_samples_to_keep = int(max_duration_truncate * sr)
                start_idx = random.randint(0, len(samples) - num_samples_to_keep)
                samples = samples[start_idx : start_idx + num_samples_to_keep]

            # Step 1: Preprocess the original audio to a uniform length
            # Corrected: Call the .apply() method for custom BaseTransform instances
            processed_samples = preprocessor.apply(samples=samples, sample_rate=sr)

            # Save the preprocessed "original" version
            original_output_fname = f"original_{os.path.splitext(fname)[0]}.wav"
            original_output_path = os.path.join(output_dir, str(label), original_output_fname)
            sf.write(original_output_path, processed_samples, sr)
            augmented_data.append({
                'fname': os.path.join(str(label), original_output_fname),
                'label': label,
                'duration': target_duration
            })

            # Step 2: Apply the augmentation pipeline to the processed audio
            # This is correct as 'augmentor' is a Compose object and is callable
            for i in range(num_augmentations_per_sample):
                augmented_samples = augmentor(samples=processed_samples, sample_rate=sr)
                augmented_output_fname = f"aug_{i+1}_{os.path.splitext(fname)[0]}.wav"
                augmented_output_path = os.path.join(output_dir, str(label), augmented_output_fname)
                sf.write(augmented_output_path, augmented_samples, sr)
                augmented_data.append({
                    'fname': os.path.join(str(label), augmented_output_fname),
                    'label': label,
                    'duration': target_duration
                })

        except Exception as e:
            print(f"Error processing file {audio_path}: {e}, skipping.")
            traceback.print_exc()
            continue

    augmented_df = pd.DataFrame(augmented_data)
    augmented_df.to_csv(INFO_CSV_NAME, index=False)
    print(f"\nData augmentation completed! Augmented data info saved to {INFO_CSV_NAME}")
    print(f"Generated {len(augmented_df)} audio files in total.")


# --- Execute the corrected data augmentation ---
if __name__ == "__main__":
    input_csv_path = '/home/renmengxing/audioRec/audio_data/train_with_duration.csv'
    
    augment_audio_dataset_fixed(input_csv_path=input_csv_path,
                                output_dir=OUTPUT_DIR,
                                target_duration=TARGET_DURATION,
                                max_duration_truncate=MAX_DURATION_TRUNCATE,
                                num_augmentations_per_sample=NUM_AUGMENTATIONS_PER_SAMPLE,
                                preprocessor=preprocess_transform,
                                augmentor=augmentation_pipeline)
