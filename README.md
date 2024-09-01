# Audio Super-Resolution Using GANs and ViT
## Project Overview
This project focuses on enhancing audio resolution using Generative Adversarial Networks (GANs) and Vision Transformer (ViT) architectures. The goal is to develop a deep learning model capable of increasing the quality of low-resolution audio samples, effectively reconstructing high-resolution audio from its low-resolution counterpart.
## Table of Contents
* Usage
* Project Structure
* Training
* Evaluation
* Results
* Contributing
* Installation

### Usage
* Creating Datasets:
  Use the create_dataset.py script to generate the datasets needed for training and testing. This script processes audio files, converts them to low-resolution versions, and prepares them for input into the model.

### Training the Model
* The training script utilizes GANs and ViT for learning the mapping between low-resolution and high-resolution audio.
* Training progress, including loss metrics and validation results, can be monitored via the logs generated.
* The model is trained over 142 epochs with a combination of discriminative and generative loss functions. The training script logs both the discriminative loss during training and the generative loss, along with the validation loss at each epoch. Detailed epoch-by-epoch logs can be found in the train_results.txt file.

### Tetisting the Model
* After training, evaluate the model's performance using the Test_ViT-SR.py script.
* This will output the reconstructed high-resolution audio and save the results in the specified directory.
* The model's performance is evaluated on a separate test dataset, and the results are logged for each epoch. The test script provides metrics that assess the model's ability to reconstruct high-resolution audio.

### [Results:](https://github.com/itayt7/audio_super_resolution_proj/blob/b51bbf7f95f26bb3f8ffa14f6bce5f04a382de15/audio%20super%20resolution.xlsx)

* During training, the model shows varying levels of performance across different epochs, with both training and validation losses tracked closely. The results of this training are critical for understanding how well the model generalizes to unseen data.

  
