# %%
import os

# Per avere una traccia più precisa dell'errore
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import math
import numpy as np

from scipy import signal
from scipy.fft import fft, fftshift

import torch
import torch.nn as nn
from torch.nn import functional as F

import cv2
import pickle
from PIL import Image
import matplotlib.pyplot as plt

import IPython
from tqdm import tqdm
# %%
# !pip
# install
# transformers
# !pip
# install
# librosa

from transformers import ViTModel, ViTConfig, AdamW

import librosa
import librosa.display as display
# %%
# !nvidia - smi
# %%
# CUDA_VISIBLE_DEVICES = 7
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# %%
# device = torch.device('cuda:7')
# %%
# device
# %%
# PYTORCH_CUDA_ALLOC_CONF = 1.1


# %%
class ResidualBlock(nn.Module):

    def __init__(self, device='cpu'):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(3, 3), padding=(2, 1), stride=(2, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(3, 3), padding=1, stride=1)
        ).to(device)

        self.ext_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(3, 3), padding=(2, 1), stride=(2, 1))
        ).to(device)

    def forward(self, inputs):
        extended_input = self.ext_block(inputs)
        convolved_input = self.block(inputs)
        return convolved_input + extended_input


# %%
class ConvResidualBlock(nn.Module):

    def __init__(self, device='cpu'):
        super(ConvResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(1, 1, 3, stride=1, padding="same"),
            nn.Conv2d(1, 1, 3, stride=1, padding="same")
        ).to(device)

        self.conv_1d = nn.Conv2d(1, 1, 3, stride=1, padding="same").to(device)

    def forward(self, inputs):
        convolved_input = self.block(inputs)
        skip_con = self.conv_1d(inputs)
        return convolved_input + skip_con


# %%
class GenerativeNetwork(nn.Module):
    """
    Input Shape: (b, 1, 800, 800)
    Output ViT: (b, num_pathces, hidden_size)
    After Reshape: (b, 1, x, x) dove x è una dimensione generica che puoi decidere
    Ouput Shape: (b, 1, 1025, 800)
    """

    def __init__(self, device='cpu'):
        super(GenerativeNetwork, self).__init__()
        self.device = device
        self.hidden_size = 1312
        self.patch_size = 32
        configuration = ViTConfig(num_attention_heads=8,
                                  num_hidden_layers=8,
                                  hidden_size=self.hidden_size,
                                  patch_size=self.patch_size,
                                  num_channels=1,
                                  image_size=800)
        self.vit = ViTModel(configuration).to(self.device)
        self.refine_model = nn.Sequential(*[ConvResidualBlock() for _ in range(3)]).to(device)

    def patch_to_img(self, x):
        row_patch_size = 41
        col_patch_size = 32
        B, NumPatches, HiddenSize = x.shape
        x = x.reshape(B, NumPatches, 1, HiddenSize)
        x = x.reshape(B, NumPatches, 1, row_patch_size, col_patch_size)
        x = x.permute(0, 1, 3, 4, 2)
        x = x.reshape(B, int(math.sqrt(NumPatches)), int(math.sqrt(NumPatches)), row_patch_size, col_patch_size, 1)
        x = x.permute(0, 1, 3, 2, 4, 5)
        new_h = x.shape[1] * x.shape[2]
        new_w = x.shape[3] * x.shape[4]
        x = x.reshape(B, new_h, new_w, 1)  # ultima posizione = num_channels che è sempre 1
        x = x.swapaxes(3, 1)
        x = x.swapaxes(3, 2)
        return x

    def forward(self, inputs):
        if inputs.device == 'cpu':
            inputs = inputs.to(self.device)
        vit_res = self.vit(pixel_values=inputs)
        inputs = vit_res.last_hidden_state[:, 1:, :]
        # patch_size_after_vit = int(math.sqrt(inputs.shape[2]))
        inputs = self.patch_to_img(inputs)  # , patch_size_after_vit)
        return self.refine_model(inputs)


# %%
class DiscriminativeNetwork(nn.Module):

    def __init__(self, device='cpu'):
        super(DiscriminativeNetwork, self).__init__()
        self.device = device
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(2),
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(4),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(8),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(256),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(in_features=1536, out_features=128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid()

        ).to(self.device)

    def forward(self, inputs):
        if inputs.device == 'cpu':
            inputs = inputs.to(self.device)
        return self.classifier(inputs)


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
            song = pickle.load(fd)

        return song  # [:1321967]


# %%
test_path = "/home/morm/Audio-Super-Resolution-ViT/resources/test"

test_ds = LHB_Dataset(test_path, 'mus')

print(test_ds[0].shape)
print(len(test_ds))
# %%
# test
test_generator = torch.Generator(device='cpu')
test_generator.manual_seed(13)
testloader = torch.utils.data.DataLoader(
    dataset=test_ds,
    batch_size=1,
    shuffle=False,
    generator=test_generator
)
# %%
# Models
generator = GenerativeNetwork(device).to(device)
# %%
checkpoint = torch.load('/home/morm/Audio-Super-Resolution-ViT/ViT-SR/chceckpoints/GEN_BestVal_reshapeAfterVit_V1_100', map_location='cpu')
checkpoint
# %%
generator.load_state_dict(checkpoint['model_state_dict'])


# %%

# %%

# %%
def compute_signal_to_noise(truth, reco):
    den = np.sqrt(np.sum((truth - reco) ** 2)) + 1e-6
    return 10. * np.log10(1e-6 + np.sqrt(np.sum(truth ** 2)) / den)


def compute_signal_to_noise_pytorch(truth, reco):
    truth = truth.view(-1, truth.shape[2], truth.shape[3])
    reco = reco.view(-1, reco.shape[2], reco.shape[3])
    den = torch.sqrt(torch.sum(torch.pow((truth - reco), 2)))
    return torch.mean(10.0 * torch.log10(1e-6 + torch.sqrt(torch.sum(torch.pow(truth, 2))) / den))


def compute_lsd(truth, reco):
    true_X = np.log10(np.abs(truth) ** 2 + 1e-6)
    reco_X = np.log10(np.abs(reco) ** 2 + 1e-6)
    reco_X_diff_squared = (true_X - reco_X) ** 2
    return np.mean(np.sqrt(np.mean(reco_X_diff_squared, axis=0)))


def compute_lsd_pytorch(truth, reco):
    truth = truth.view(-1, truth.shape[2], truth.shape[3])
    reco = reco.view(-1, reco.shape[2], reco.shape[3])
    true_X = torch.log10(torch.pow(torch.abs(truth), 2) + 1e-6)
    reco_X = torch.log10(torch.pow(torch.abs(reco), 2) + 1e-6)
    diff = true_X - reco_X
    reco_X_diff_squared = torch.pow(diff, 2)
    return torch.mean(torch.mean(torch.sqrt(torch.mean(reco_X_diff_squared, dim=1)), dim=1))


def get_metric_comparison(testloader, metric, generator=None, device='cpu'):
    generator.eval()

    NUM_COLS = 800
    TOT_ROWS = 1825
    HF_ROWS = 1025
    LF_ROWS = 800

    total_value = 0.0
    count = 0

    for test_data in testloader:

        data = np.asarray(test_data).squeeze(axis=0)
        # Compute spectrograms
        stft = librosa.stft(np.asarray(data), n_fft=4096, win_length=4096, window=signal.windows.hamming(4096))
        spectrogram = librosa.amplitude_to_db(abs(stft))

        rows = spectrogram.shape[0]
        real_cols = spectrogram.shape[1]

        if real_cols % NUM_COLS > 0:
            cols_to_add = NUM_COLS - real_cols % NUM_COLS
            new_data = np.zeros(shape=(rows, real_cols + cols_to_add))
            new_data[:, : real_cols] = spectrogram
            new_data[:, real_cols:] = spectrogram[:, -cols_to_add:]
            spectrogram = new_data
            cols = real_cols + cols_to_add
        else:
            cols = real_cols

        PTS = cols // NUM_COLS

        temp_data = np.zeros(shape=(PTS, HF_ROWS + LF_ROWS, NUM_COLS))
        for i in range(PTS):
            temp_data[i, :, :] = spectrogram[: HF_ROWS + LF_ROWS, i * NUM_COLS: (i + 1) * NUM_COLS]

        temp_data = torch.from_numpy(temp_data).view(PTS, 1, -1, NUM_COLS).float()
        ds_lf = temp_data[:, :, : LF_ROWS, :]
        ds_hf = temp_data[:, :, LF_ROWS: LF_ROWS + HF_ROWS, :]

        ds_lf = ds_lf.to(device)

        generated_hf = np.asarray(generator(ds_lf).detach().cpu())
        ds_hf = np.asarray(ds_hf.detach().cpu())
        ds_lf = np.asarray(ds_lf.detach().cpu())

        tmp_real = np.zeros(shape=(PTS, 1, HF_ROWS + LF_ROWS, NUM_COLS))
        tmp_real[:, :, : LF_ROWS, :] = ds_lf
        tmp_real[:, :, LF_ROWS: LF_ROWS + HF_ROWS, :] = ds_hf

        tmp_pred = np.zeros(shape=(PTS, 1, HF_ROWS + LF_ROWS, NUM_COLS))
        tmp_pred[:, :, : LF_ROWS, :] = ds_lf
        tmp_pred[:, :, LF_ROWS: LF_ROWS + HF_ROWS, :] = generated_hf

        real_spec = np.zeros(shape=(TOT_ROWS, NUM_COLS * PTS))
        pred_spec = np.zeros(shape=(TOT_ROWS, NUM_COLS * PTS))

        for j in range(PTS):
            real_spec[: TOT_ROWS, j * NUM_COLS: (j + 1) * NUM_COLS] = tmp_real[j, :, :, :]

            pred_spec[: TOT_ROWS, j * NUM_COLS: (j + 1) * NUM_COLS] = tmp_pred[j, :, :, :]

        real_spec = real_spec[:, :real_cols]
        pred_spec = pred_spec[:, :real_cols]

        total_value = total_value + metric(real_spec, pred_spec)
        count = count + 1

    return total_value / count


def get_metric_comparison_onlyLB(testloader, metric, device='cpu'):
    NUM_COLS = 800
    TOT_ROWS = 1825
    HF_ROWS = 1025
    LF_ROWS = 800

    total_value = 0.0
    count = 0

    for test_data in testloader:

        data = np.asarray(test_data).squeeze(axis=0)
        # Compute spectrograms
        stft = librosa.stft(np.asarray(data), n_fft=4096, win_length=4096, window=signal.windows.hamming(4096))
        spectrogram = librosa.amplitude_to_db(abs(stft))

        rows = spectrogram.shape[0]
        real_cols = spectrogram.shape[1]

        if real_cols % NUM_COLS > 0:
            cols_to_add = NUM_COLS - real_cols % NUM_COLS
            new_data = np.zeros(shape=(rows, real_cols + cols_to_add))
            new_data[:, : real_cols] = spectrogram
            new_data[:, real_cols:] = spectrogram[:, -cols_to_add:]
            spectrogram = new_data
            cols = real_cols + cols_to_add
        else:
            cols = real_cols

        PTS = cols // NUM_COLS

        temp_data = np.zeros(shape=(PTS, HF_ROWS + LF_ROWS, NUM_COLS))
        for i in range(PTS):
            temp_data[i, :, :] = spectrogram[: HF_ROWS + LF_ROWS, i * NUM_COLS: (i + 1) * NUM_COLS]

        temp_data = torch.from_numpy(temp_data).view(PTS, 1, -1, NUM_COLS).float()

        ds_lf = temp_data[:, :, : LF_ROWS, :]
        ds_hf = temp_data[:, :, LF_ROWS: LF_ROWS + HF_ROWS, :]

        min_value = torch.min(ds_hf)
        generated_hf = torch.ones_like(ds_hf) * min_value
        ds_hf = np.asarray(ds_hf.detach().cpu())
        ds_lf = np.asarray(ds_lf.detach().cpu())

        tmp_real = np.zeros(shape=(PTS, 1, HF_ROWS + LF_ROWS, NUM_COLS))
        tmp_real[:, :, : LF_ROWS, :] = ds_lf
        tmp_real[:, :, LF_ROWS: LF_ROWS + HF_ROWS, :] = ds_hf

        tmp_pred = np.zeros(shape=(PTS, 1, HF_ROWS + LF_ROWS, NUM_COLS))
        tmp_pred[:, :, : LF_ROWS, :] = ds_lf
        tmp_pred[:, :, LF_ROWS: LF_ROWS + HF_ROWS, :] = generated_hf

        real_spec = np.zeros(shape=(TOT_ROWS, NUM_COLS * PTS))
        pred_spec = np.zeros(shape=(TOT_ROWS, NUM_COLS * PTS))

        for j in range(PTS):
            real_spec[: TOT_ROWS, j * NUM_COLS: (j + 1) * NUM_COLS] = tmp_real[j, :, :, :]
            pred_spec[: TOT_ROWS, j * NUM_COLS: (j + 1) * NUM_COLS] = tmp_pred[j, :, :, :]

        real_spec = real_spec[:, :real_cols]
        pred_spec = pred_spec[:, :real_cols]

        total_value = total_value + metric(real_spec, pred_spec)
        count = count + 1

    return total_value / count


def get_metric_comparison_on_interpolation(testloader, metric):
    NUM_COLS = 800
    TOT_ROWS = 1825
    HF_ROWS = 1025
    LF_ROWS = 800

    total_value = 0.0
    count = 0

    for test_data in testloader:

        data = np.asarray(test_data).squeeze(axis=0)
        # Compute spectrograms
        stft = librosa.stft(np.asarray(data), n_fft=4096, win_length=4096, window=signal.windows.hamming(4096))
        spectrogram = librosa.amplitude_to_db(abs(stft))

        rows = spectrogram.shape[0]
        real_cols = spectrogram.shape[1]

        if real_cols % NUM_COLS > 0:
            cols_to_add = NUM_COLS - real_cols % NUM_COLS
            new_data = np.zeros(shape=(rows, real_cols + cols_to_add))
            new_data[:, : real_cols] = spectrogram
            new_data[:, real_cols:] = spectrogram[:, -cols_to_add:]
            spectrogram = new_data
            cols = real_cols + cols_to_add
        else:
            cols = real_cols

        PTS = cols // NUM_COLS

        temp_data = np.zeros(shape=(PTS, HF_ROWS + LF_ROWS, NUM_COLS))
        for i in range(PTS):
            temp_data[i, :, :] = spectrogram[: HF_ROWS + LF_ROWS, i * NUM_COLS: (i + 1) * NUM_COLS]

        temp_data = temp_data.reshape(PTS, 1, LF_ROWS + HF_ROWS, NUM_COLS)
        ds_lf = temp_data[:, :, : LF_ROWS, :]
        ds_hf = temp_data[:, :, LF_ROWS: LF_ROWS + HF_ROWS, :]

        dim = (NUM_COLS, HF_ROWS + LF_ROWS)
        tmp_hf = ds_hf.copy().reshape(ds_hf.shape[0], ds_hf.shape[2], ds_hf.shape[3])
        tmp_pred = np.zeros(shape=(PTS, HF_ROWS + LF_ROWS, NUM_COLS))
        for i in range(PTS):
            tmp_pred[i] = cv2.resize(tmp_hf[i], dim, interpolation=cv2.INTER_CUBIC)

        tmp_real = np.zeros(shape=(PTS, 1, HF_ROWS + LF_ROWS, NUM_COLS))
        tmp_real[:, :, :LF_ROWS, :] = ds_lf
        tmp_real[:, :, LF_ROWS:LF_ROWS + HF_ROWS, :] = ds_hf

        real_spec = np.zeros(shape=(TOT_ROWS, NUM_COLS * PTS))
        pred_spec = np.zeros(shape=(TOT_ROWS, NUM_COLS * PTS))

        for j in range(PTS):
            real_spec[: LF_ROWS + HF_ROWS, j * NUM_COLS: (j + 1) * NUM_COLS] = tmp_real[j, :, :, :]
            pred_spec[: LF_ROWS + HF_ROWS, j * NUM_COLS: (j + 1) * NUM_COLS] = tmp_pred[j, :, :]

        real_spec = real_spec[:, :real_cols]
        pred_spec = pred_spec[:, :real_cols]

        total_value = total_value + metric(real_spec, pred_spec)
        count = count + 1

    return total_value / count


# %%
# TEST GENERATOR
print('LSD: ', get_metric_comparison(testloader=testloader, metric=compute_lsd, generator=generator, device=device))
print('SNR: ',
      get_metric_comparison(testloader=testloader, metric=compute_signal_to_noise, generator=generator, device=device))
# %%
# TEST ONLY LB
print('LSD: ', get_metric_comparison_onlyLB(testloader=testloader, metric=compute_lsd, device=device))
print('SNR: ', get_metric_comparison_onlyLB(testloader=testloader, metric=compute_signal_to_noise, device=device))
# %%
# TEST INTERPOLATION
print('LSD: ', get_metric_comparison_on_interpolation(testloader=testloader, metric=compute_lsd))
print('SNR: ', get_metric_comparison_on_interpolation(testloader=testloader, metric=compute_signal_to_noise))
# %%
