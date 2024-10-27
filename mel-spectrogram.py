import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import cv2
import matplotlib
from PIL import Image


def preprocessing_wav(wav_file_path, fs_down=8000):
    # Read and normalize the audio file
    data, sample_rate = librosa.load(wav_file_path, sr=fs_down)
    return data, sample_rate

def create_mel_spectrogram_image(wav_file_path, img_size=(128, 128), fs_down=8000):
    # 1. Process the audio file (preprocessing and downsampling)
    data, sample_rate = preprocessing_wav(wav_file_path, fs_down)
    
    # 2. Generate Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=data, sr=sample_rate, n_mels=128, fmax=4000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 3. Normalize (to range 0-255)
    mel_spec_normalized = 255 * (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
    mel_spec_normalized = mel_spec_normalized.astype(np.uint8)
    
    # 4. Apply Viridis color map
    colormap = matplotlib.colormaps['viridis']
    colored_image = colormap(mel_spec_normalized)
    
    # 5. Convert the colored image to range 0-255 and save as uint8 format
    rgb_image = (colored_image[:, :, :3] * 255).astype(np.uint8)
    
    # 6. Resize the image to 128x128 pixels
    rgb_image_resized = cv2.resize(rgb_image, img_size, interpolation=cv2.INTER_LINEAR)
    
    # 7. Convert to PIL image and return
    rgb_image_pil = Image.fromarray(rgb_image_resized)

    return rgb_image_pil


# Set the base directory (the main path to the IEMOCAP dataset folder)
folder_path = os.getcwd()
base_directory = os.path.join(folder_path, 'IEMOCAP_full_release')
images = []

# List to store paths of each .wav file
wav_file_paths = []

# Loop over 5 Sessions
for session in range(1, 6):  # From Session1 to Session5
    session_folder = os.path.join(base_directory, f'Session{session}', 'sentences', 'wav')
    images = []
    
    # Find all .wav files in subfolders of `sentences/wav`
    for sub_folder in sorted(os.listdir(session_folder)):
        sub_folder_path = os.path.join(session_folder, sub_folder)
        for file in sorted(os.listdir(sub_folder_path)):
            if file.endswith('.wav'):
                # Construct the full path of the .wav file
                file_path = os.path.join(sub_folder_path, file)
                
                # Create Mel-spectrogram image and add to list
                image = create_mel_spectrogram_image(file_path)
                output_path = os.path.join(folder_path, f"output_images/{file}.png")
                image.save(output_path)
                images.append(image)
                print(f'{file} successfully processed')
                
                # Append file path to wav_file_paths list
                wav_file_paths.append(file_path)
    
    print(f'Session {session} has been completed')
    images = np.array(images)
    np.save(f'C:\\Users\\Keaton\\Desktop\\EDU\\AffectiveComputing\\Research\\Session{session}_images.npy', images)

# Display the total number of .wav files found
print(f"Total .wav files processed: {len(wav_file_paths)}")
