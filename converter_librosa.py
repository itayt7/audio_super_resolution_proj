import os
import pickle
import numpy as np
import librosa


def convert_mp3_to_mus(mp3_path, mus_path):
    # Load the MP3 file using librosa
    audio, sr = librosa.load(mp3_path, sr=None)

    # Convert the audio to a floating-point numpy array
    audio = audio.astype(np.float32)

    # You can process `audio` here if needed

    # Save as .mus file using pickle
    with open(mus_path, 'wb') as mus_file:
        pickle.dump(audio, mus_file)


# Example usage
mp3_directory = '/home/morm/Audio-Super-Resolution-ViT/resources/fma_small/fma_small/030'
# mus_directory = '/home/morm/Audio-Super-Resolution-ViT/resources/train'
# mus_directory = '/home/morm/Audio-Super-Resolution-ViT/resources/validation'
mus_directory = '/home/morm/Audio-Super-Resolution-ViT/resources/test'

if not os.path.exists(mus_directory):
    os.makedirs(mus_directory)

for mp3_file in os.listdir(mp3_directory):
    try:
        if mp3_file.endswith('.mp3'):
            mp3_path = os.path.join(mp3_directory, mp3_file)
            mus_path = os.path.join(mus_directory, os.path.splitext(mp3_file)[0] + '.mus')
            convert_mp3_to_mus(mp3_path, mus_path)
            print(f"Converted {mp3_path} to {mus_path}")
    except Exception as e:
        print(f"Error loading {mp3_file}: {e}")

