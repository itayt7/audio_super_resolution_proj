# %%
import os

# Per avere una traccia più precisa dell'errore
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import math
import numpy as np

from scipy import signal

import torch
import torch.nn as nn

import pickle

from transformers import ViTModel, ViTConfig, AdamW

import librosa

# %%
# CUDA_VISIBLE_DEVICES = 7
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
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
train_path = "/home/morm/Audio-Super-Resolution-ViT/resources/train"

train_ds = LHB_Dataset(train_path, 'mus')

print(train_ds[0].shape)
print(len(train_ds))
# %%
validation_path = '/home/morm/Audio-Super-Resolution-ViT/resources/validation'

validation_ds = LHB_Dataset(validation_path, 'mus')

print(validation_ds[0].shape)
print(len(validation_ds))
# %%
# train
train_generator = torch.Generator(device='cpu')
train_generator.manual_seed(13)
trainloader = torch.utils.data.DataLoader(
    dataset=train_ds,
    batch_size=1,
    shuffle=True,
    generator=train_generator
)

# validation
validation_generator = torch.Generator(device='cpu')
validation_generator.manual_seed(13)
validloader = torch.utils.data.DataLoader(
    dataset=validation_ds,
    batch_size=1,
    shuffle=False,
    generator=validation_generator
)
# %%
# Models
generator = GenerativeNetwork(device).to(device)
discriminator = DiscriminativeNetwork(device).to(device)

# Optimizers
optimizer_gen = AdamW(generator.parameters(), lr=1e-4, weight_decay=1e-4)
optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=1e-5)

# Loss
loss_gen = nn.MSELoss()
loss_dis = nn.BCELoss()


# %%
def train_with_val(trainloader, validloader, generator, discriminator, optimizer_gen, optimizer_dis, loss_gen, loss_dis,
                   name_to_save, epoches=1, start=0, beta=1.0, best_val=float('inf'), device='cpu'):
    from tqdm import tqdm

    TRAIN_CHECKPOINTS_DIR_PATH = '/home/morm/Audio-Super-Resolution-ViT/ViT-SR/chceckpoints'
    filename = 'afterIsmis_test1204_100.txt'

    alpha = 1.5

    best_val_loss = best_val
    old_valid_discriminator_loss = float('inf')

    NUM_COLS = 800
    NUM_ROWS_LB = 800

    GRAD_ACC = 8

    # TrainSteps
    for epoch in tqdm(range(epoches), desc='Epoch '):
        generator.train()
        discriminator.train()
        num_samples_seen = 0
        total_gen_loss = 0
        total_dis_loss = 0

        # Iter on batches
        num_songs = 0
        for data_batch in trainloader:
            try:
                print(f'Song n.{num_songs} on {len(trainloader)}')
                batch_lf = []
                batch_hf = []

                for data in data_batch:

                    data = data.squeeze(dim=0)  # one song

                    # Compute spectrograms
                    train_stft = librosa.stft(np.asarray(data), n_fft=4096, win_length=4096,
                                              window=signal.windows.hamming(4096))
                    train_spectrogram = torch.tensor(librosa.amplitude_to_db(abs(train_stft)))

                    rows = train_spectrogram.shape[0]
                    cols = train_spectrogram.shape[1]

                    if cols % NUM_COLS > 0:
                        cols_to_add = NUM_COLS - cols % NUM_COLS
                        new_data = torch.zeros(size=(rows, cols + cols_to_add))
                        new_data[:, : cols] = train_spectrogram
                        new_data[:, cols: cols + cols_to_add] = train_spectrogram[:, -cols_to_add:]
                        train_spectrogram = new_data
                        cols = cols + cols_to_add

                    train_spectrogram = train_spectrogram.reshape(1, rows, cols).float()

                    PTS = cols // NUM_COLS

                    for i in range(PTS):
                        batch_lf.append(train_spectrogram[:, : 800, i * NUM_COLS: (i + 1) * NUM_COLS])
                        batch_hf.append(train_spectrogram[:, 800: 1825, i * NUM_COLS: (i + 1) * NUM_COLS])
            except RuntimeError:
                continue

            batch_lf = torch.stack(batch_lf).to(device)
            batch_hf = torch.stack(batch_hf).to(device)

            num_samples_seen += batch_lf.shape[0]

            shuffled_indexes = np.random.permutation(batch_lf.shape[0])  # shuffle
            batch_lf = batch_lf[shuffled_indexes]
            batch_hf = batch_hf[shuffled_indexes]

            # Train the discriminator on the true/generated data
            generated_data = generator(batch_lf)
            combined_data = torch.cat((batch_hf.to(device), generated_data), dim=0)
            labels = torch.cat((torch.ones(batch_hf.shape[0]), torch.zeros(generated_data.shape[0])), dim=0)

            shuffled_indexes = np.random.permutation(combined_data.shape[0])  # shuffle
            combined_data = combined_data[shuffled_indexes]
            labels = labels[shuffled_indexes].to(device)

            discriminator_out = discriminator(combined_data).reshape(-1)
            # print(f'DIS: {discriminator_out.shape}, LABELS: {labels.shape}')
            optimizer_dis.zero_grad()
            discriminator_loss = loss_dis(discriminator_out, labels)
            discriminator_loss.backward()
            optimizer_dis.step()

            # Train the generator
            optimizer_gen.zero_grad()
            generator_out = generator(batch_lf)
            generator_loss = loss_gen(batch_hf, generator_out)

            discriminator_out_gen = discriminator(generator_out.detach()).reshape(-1)
            discriminator_loss_gen = loss_dis(discriminator_out_gen.to('cpu'),
                                              torch.ones(size=(discriminator_out_gen.shape[0],)))  # bce

            total_dis_loss = total_dis_loss + discriminator_loss_gen.detach()
            total_gen_loss = total_gen_loss + generator_loss.detach()

            loss = alpha * generator_loss + beta * discriminator_loss_gen

            loss.backward()
            optimizer_gen.step()

            num_songs += 1

        # if num_songs % GRAD_ACC != 0:
        """
        optimizer_gen.step()
        optimizer_gen.zero_grad()
        optimizer_dis.step()
        optimizer_dis.zero_grad()
        """

        # End Trainloader Loop

        print('START VALIDATION')
        # Validation
        with torch.no_grad():
            generator.eval()
            discriminator.eval()
            total_val_gen = 0
            total_val_dis = 0

            valid_samples_seen = 0

            for valid_data in validloader:
                valid_data = valid_data.squeeze()
                valid_stft = librosa.stft(np.asarray(valid_data), n_fft=4096, win_length=4096,
                                          window=signal.windows.hamming(4096))
                valid_spectrogram = torch.tensor(librosa.amplitude_to_db(abs(valid_stft)))

                valid_rows = valid_spectrogram.shape[0]
                valid_cols = valid_spectrogram.shape[1]

                if valid_cols % NUM_COLS > 0:
                    cols_to_add = NUM_COLS - valid_cols % NUM_COLS
                    new_data = torch.zeros(size=(valid_rows, valid_cols + cols_to_add))
                    new_data[:, : valid_cols] = valid_spectrogram
                    new_data[:, valid_cols: valid_cols + cols_to_add] = valid_spectrogram[:, -cols_to_add:]
                    valid_spectrogram = new_data
                    valid_cols = valid_cols + cols_to_add

                valid_spectrogram = valid_spectrogram.reshape(1, 1, valid_rows, valid_cols)

                VALID_PTS = valid_cols // NUM_COLS

                valid_samples_seen = valid_samples_seen + VALID_PTS

                valid_lf = torch.zeros(size=(VALID_PTS, 1, 800, NUM_COLS))
                valid_hf = torch.zeros(size=(VALID_PTS, 1, 1025, NUM_COLS)).to(device)

                for j in range(VALID_PTS):
                    valid_lf[i, :, :, :] = valid_spectrogram[:, :, : 800, i * NUM_COLS: (i + 1) * NUM_COLS]
                    valid_hf[i, :, :, :] = valid_spectrogram[:, :, 800: 1825, i * NUM_COLS: (i + 1) * NUM_COLS]

                valid_hf_generated = generator(valid_lf.to(device))

                valid_gen_loss = loss_gen(valid_hf.to('cpu'), valid_hf_generated.to('cpu'))

                valid_dis_out = discriminator(valid_hf_generated.detach()).reshape(-1)
                valid_dis_loss = loss_dis(valid_dis_out.to('cpu'), torch.ones(size=(valid_dis_out.shape[0],)))

                total_val_gen = total_val_gen + valid_gen_loss
                total_val_dis = total_val_dis + valid_dis_loss

            mean_val_loss_gen = total_val_gen / valid_samples_seen
            mean_val_loss_dis = total_val_dis / valid_samples_seen

            mean_val_loss = alpha * mean_val_loss_gen + beta * mean_val_loss_dis

            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                print('UPDATE TO ', best_val_loss)
                gen_name = os.path.join(TRAIN_CHECKPOINTS_DIR_PATH, 'GEN_BestVal_' + name_to_save)
                torch.save({
                    'epoch': epoch + start,
                    'model_state_dict': generator.state_dict(),
                    'optimizer_state_dict': optimizer_gen.state_dict(),
                    'generator_loss': loss,
                    'discriminator_loss': discriminator_loss,
                    'val_loss': mean_val_loss,
                    'beta': beta
                }, gen_name)

                # Compute alpha e beta
        mean_gen_loss = total_gen_loss / num_samples_seen
        mean_dis_loss = total_dis_loss / num_samples_seen

        gen_order = torch.floor(torch.log10(mean_gen_loss))
        dis_order = 0 if mean_dis_loss == 0 else torch.floor(torch.log10(mean_dis_loss))
        b_pow = gen_order - dis_order
        if b_pow > 0:
            b_pow = b_pow
        beta = pow(10.0, b_pow)

        # Ad ogni epoca stampiamo la loss
        file = open(filename, 'a')  # w+ is to append new lines to the file
        file.write(
            'EPOCH ' + str(epoch + 1) +
            '\n\t -> Discriminative Loss during D Training = ' + str(
                mean_dis_loss.item()) + ', during G Training = ' + str(discriminator_loss_gen.item()) +
            '\n\t -> Generative Loss = ' + str(loss.item()) + ' ---> alpha * ' + str(
                mean_gen_loss.item()) + ' beta * ' + str(mean_dis_loss.item()) +
            '\n\t -> Validation Loss = ' + str(mean_val_loss.item()) + '\n\n')
        file.flush()
        file.close()

        print('EPOCH ' + str(epoch + 1) +
              '\n\t -> Discriminative Loss during D Training = ' + str(
            mean_dis_loss.item()) + ', during G Training = ' + str(discriminator_loss_gen.item()) +
              '\n\t -> Generative Loss = ' + str(loss.item()) + ' ---> alpha * ' + str(
            mean_gen_loss.item()) + ' beta * ' + str(mean_dis_loss.item()) +
              '\n\t -> Validation Loss = ' + str(mean_val_loss.item()) + '\n\n')

        dis_name = os.path.join(TRAIN_CHECKPOINTS_DIR_PATH, 'DIS_' + name_to_save)
        torch.save({
            'epoch': epoch + start,
            'model_state_dict': discriminator.state_dict(),
            'optimizer_state_dict': optimizer_dis.state_dict(),
            'generator_loss': loss,
            'discriminator_loss': discriminator_loss,
            'val_loss': 0,
            'beta': beta
        }, dis_name)

        gen_name = os.path.join(TRAIN_CHECKPOINTS_DIR_PATH, 'GEN_' + name_to_save)
        torch.save({
            'epoch': epoch + start,
            'model_state_dict': generator.state_dict(),
            'optimizer_state_dict': optimizer_gen.state_dict(),
            'generator_loss': loss,
            'discriminator_loss': discriminator_loss,
            'val_loss': mean_val_loss,
            'beta': beta
        }, gen_name)


# %%
TRAIN_CHECKPOINTS_DIR_PATH = '/home/morm/Audio-Super-Resolution-ViT/ViT-SR/chceckpoints'
name_to_save = 'reshapeAfterVit_V1_100'
# %%
# train_with_val(trainloader, validloader, generator, discriminator, optimizer_gen, optimizer_dis, loss_gen, loss_dis, name_to_save=name_to_save, epoches=20, start=0, beta=1.0, best_val=float('inf'), device=device)


### Restore and Resume training
# %%

# Restore discriminator
dis_name = os.path.join(TRAIN_CHECKPOINTS_DIR_PATH, 'DIS_' + name_to_save)
dis_checkpoint = torch.load(dis_name)
discriminator.load_state_dict(dis_checkpoint['model_state_dict'])
optimizer_dis.load_state_dict(dis_checkpoint['optimizer_state_dict'])
discriminator_loss = dis_checkpoint['discriminator_loss']
epoch = dis_checkpoint['epoch']

# best val loss
ckp = torch.load(os.path.join(TRAIN_CHECKPOINTS_DIR_PATH, 'GEN_BestVal_' + name_to_save))
best_val_loss = ckp['val_loss']

# Restore generator
gen_name = os.path.join(TRAIN_CHECKPOINTS_DIR_PATH, 'GEN_' + name_to_save)
gen_checkpoint = torch.load(gen_name)
generator.load_state_dict(gen_checkpoint['model_state_dict'])
optimizer_gen.load_state_dict(gen_checkpoint['optimizer_state_dict'])
discriminator_loss = gen_checkpoint['discriminator_loss']
epoch = gen_checkpoint['epoch']
mean_val_loss = gen_checkpoint['val_loss']
beta = gen_checkpoint['beta']


train_with_val(trainloader, validloader, generator, discriminator, optimizer_gen, optimizer_dis, loss_gen, loss_dis,
               name_to_save=name_to_save, epoches=200, start=epoch, beta=beta, best_val=best_val_loss, device=device)
# %%
