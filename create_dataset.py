# %%
import os
import pickle

# %%
import librosa
import numpy as np
import torch
from scipy import signal

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# %%
class LHB_Dataset(torch.utils.data.Dataset):

    def __init__(self, path, ext):
        self.path = path
        self.ext = ext
        self.len = len(os.listdir(self.path))
        self.items_in_dir = os.listdir(self.path)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        name = self.path + '/' + self.items_in_dir[idx]  # self.path + '/' + str(idx) + "." + self.ext

        with open(name, 'rb') as fd:
            image = pickle.load(fd)

        return image


# %%
train_path = '/resources/train'

train_ds = LHB_Dataset(train_path, 'mus')

print(train_ds[0].shape)
print(len(train_ds))

# train
train_generator = torch.Generator(device='cpu')
train_generator.manual_seed(13)
trainloader = torch.utils.data.DataLoader(
    dataset=train_ds,
    batch_size=1,
    shuffle=True,
    generator=train_generator
)


# %%
def create_dataset(trainloader):
    index = 0

    NUM_COLS = 1024

    for data in trainloader:
        print("WORKING ON SPECTROGRAM NUMBER ", index)
        data = data.squeeze(dim=0)  # one song

        # Compute spectrograms
        train_stft = librosa.stft(np.asarray(data), n_fft=4096, win_length=4096, window=signal.windows.hamming(4096))
        train_spectrogram = torch.tensor(librosa.amplitude_to_db(abs(train_stft)))

        rows = train_spectrogram.shape[0]
        cols = train_spectrogram.shape[1]

        if cols < NUM_COLS:
            continue

        if cols % NUM_COLS > 0:
            cols_to_add = NUM_COLS - cols % NUM_COLS
            new_data = torch.zeros(size=(rows, cols + cols_to_add))
            new_data[:, : cols] = train_spectrogram
            new_data[:, cols: cols + cols_to_add] = train_spectrogram[:, -cols_to_add:]
            train_spectrogram = new_data
            cols = cols + cols_to_add

        train_spectrogram = train_spectrogram.reshape(1, 1, rows, cols)

        PTS = cols // NUM_COLS

        for j in range(PTS):
            with open('./Spectrograms/spec' + str(index) + '.ds', 'wb') as fd:
                pickle.dump({'data': train_spectrogram[:, :, 1: 1025, j * NUM_COLS: (j + 1) * NUM_COLS].float()}, fd)

            index = index + 1

# %%
# !rm - r. / Spectrograms /
# %%
# !mkdir. / Spectrograms /
# %%
create_dataset(trainloader)
# %%
