# ResNet LSTM Image Captioning

This document provides a comprehensive guide to the ResNet34-LSTM image captioning model, including architecture details, training instructions, and troubleshooting tips.

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Data Preparation](#data-preparation)
4. [Model Architecture](#model-architecture)
   - [Encoder: ResNet34](#encoder-resnet34)
   - [Attention Mechanism](#attention-mechanism)
   - [Decoder: LSTM with Attention](#decoder-lstm-with-attention)
5. [Training the Model](#training-the-model)
   - [Basic Commands](#basic-commands)
   - [Training Parameters](#training-parameters)
   - [Resuming Training](#resuming-training)
   - [Training Metrics](#training-metrics)
   - [Tips for Effective Training](#tips-for-effective-training)
6. [Generating Captions](#generating-captions)
   - [Using test_caption.py](#using-test_captionpy)
   - [Using captions.py](#using-captionspy)
   - [Visualizing Attention](#visualizing-attention)
   - [Beam Search](#beam-search)
7. [Model Outputs](#model-outputs)
   - [Directory Structure](#directory-structure)
   - [Output Files](#output-files)
   - [Visualizing Results](#visualizing-results)
8. [Key Functions](#key-functions)
9. [Troubleshooting](#troubleshooting)
   - [Training Issues](#training-issues)
   - [Graphviz Installation](#graphviz-installation)
   - [Common Errors](#common-errors)
10. [How A2C Works](#how-a2c-works)

## Overview

This image captioning system combines a ResNet34 convolutional neural network (CNN) for image encoding with a Long Short-Term Memory (LSTM) network with attention for text generation. The model follows the encoder-decoder architecture pattern common in sequence generation tasks.

Key features of the model:
- ResNet34 CNN encoder for feature extraction
- LSTM decoder with attention mechanism
- Beam search for caption generation
- Attention visualization
- BLEU score tracking
- Training metrics tracking
- Pretrained weights support
- Training resumption capability

## Installation

### Dependencies

```bash
# Install Python dependencies
pip install torch torchvision numpy matplotlib nltk tqdm scikit-image Pillow
pip install graphviz torchviz
```

### Additional Software
For model architecture visualization, Graphviz is required:
1. Download and install from https://graphviz.org/download/
2. Add the Graphviz bin directory to your system PATH:
   - Windows: Add to system environment variables
   - macOS: `export PATH=$PATH:/usr/local/Cellar/graphviz/X.XX/bin`
   - Linux: `export PATH=$PATH:/usr/bin/graphviz`
3. Restart your terminal/command prompt

### Directory Structure

```
resNet_LSTM/
├── model.py              # Model architecture definition
├── train.py              # Training script
├── test_caption.py       # Script to generate captions with visualization
├── captions.py           # Simpler script to generate captions
├── create_data_n_prep.py # Script to prepare dataset
├── model_architecture.py # Script to display model architecture
├── utils.py              # Utility functions
├── dataset.py            # Dataset loading and processing
├── visualize_metrics.py  # Script to visualize training metrics
├── data_output/          # Directory for dataset files
└── model_outputs/        # Directory for model outputs
```

## Data Preparation

Before training the model, you need to prepare the dataset. This involves downloading the image dataset and corresponding caption annotations, then processing them into the format required by the model.

### Step 1: Download Dataset
Download one of the supported datasets:
- Flickr8k (recommended for beginners): [Dataset](https://forms.illinois.edu/sec/1713398)
- Flickr30k: [Dataset](https://shannon.cs.illinois.edu/DenotationGraph/)
- MSCOCO: [Dataset](https://cocodataset.org/#download)

### Step 2: Download Karpathy's Splits
Download the JSON file containing the training/validation/test splits created by Andrej Karpathy:
- [Karpathy's Splits](https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip)

### Step 3: Prepare the Data
Use the `create_data_n_prep.py` script to process the images and captions:

```bash
python create_data_n_prep.py
```

By default, the script is configured for Flickr8k with the following settings:
```python
create_input_data(
    dataset='flickr8k',
    json_path='data/caption_datasets/dataset_flickr8k.json',
    image_folder='data/flickr8k/Images',
    captions_per_image=5,
    min_word_freq=5,
    output_folder='data_output',
    max_len=50
)
```

Modify these parameters in the script if you're using a different dataset or want to change the configuration:
- `dataset`: Choose from 'flickr8k', 'flickr30k', or 'coco'
- `json_path`: Path to Karpathy's JSON file
- `image_folder`: Directory containing the images
- `captions_per_image`: Number of captions to use per image
- `min_word_freq`: Minimum frequency for a word to be included in vocabulary
- `output_folder`: Directory to save processed data
- `max_len`: Maximum caption length

The script will create the following files in the `data_output` directory:
- Word map: `WORDMAP_flickr8k_5_5.json`
- Encoded captions and lengths: `TRAIN/VAL/TEST_CAPTIONS/CAPLENS_flickr8k_5_5_*.json`
- Image features: `TRAIN/VAL/TEST_IMAGES_flickr8k_5_5.hdf5`

## Model Architecture

The image captioning model consists of three main components: an encoder, an attention mechanism, and a decoder.

### Encoder: ResNet34

The encoder is a ResNet34 convolutional neural network with the following characteristics:

- **Input**: RGB images of size 256x256 pixels
- **Output**: Feature maps of size 14x14 with 512 channels
- **Structure**: ResNet34 architecture (3-4-6-3 layers configuration)
- **Pretrained Option**: Can use pretrained weights from torchvision or train from scratch
- **Fine-tuning**: The encoder can be fine-tuned during training or frozen

The encoder extracts spatial features from the input image, which are then used by the attention mechanism to focus on relevant parts of the image during caption generation.

### Attention Mechanism

The attention mechanism allows the decoder to focus on different parts of the image when generating each word:

- **Inputs**: 
  - Encoder features: (batch_size, 196, 512)
  - Decoder hidden state: (batch_size, 512)
- **Processing**:
  1. Transform encoder features with a linear layer
  2. Transform decoder hidden state with a linear layer
  3. Combine transformed features and apply tanh activation
  4. Score each pixel location with another linear layer
  5. Apply softmax to get attention weights
  6. Compute weighted sum of encoder features using attention weights
- **Output**:
  - Attention weights: (batch_size, 196)
  - Context vector: (batch_size, 512)

### Decoder: LSTM with Attention

The decoder is an LSTM network that generates captions word by word:

- **Inputs**:
  - Encoder features: (batch_size, 196, 512)
  - Captions: (batch_size, caption_length) [during training]
- **Embedding Layer**: Converts word indices to dense vectors of size 512
- **LSTM Cell**: Processes embedded word and context vector to update hidden state
- **Attention Gate**: Controls how much attention information to use
- **Output Layer**: Linear projection from LSTM hidden state to vocabulary size
- **Outputs**:
  - Word predictions: (batch_size, max_length, vocab_size)
  - Attention weights: (batch_size, max_length, 196)

## How This Works

The image captioning system is designed to generate descriptive captions for images by combining a convolutional neural network (CNN) encoder with a Long Short-Term Memory (LSTM) decoder. Here's an overview of how the system works:

### 1. Data Preparation
- The `create_data_n_prep.py` script processes the dataset by:
  - Extracting image features using ResNet34.
  - Tokenizing and encoding captions into numerical format.
  - Creating a word map (vocabulary) and saving it as a JSON file.
  - Splitting the dataset into training, validation, and test sets.

### 2. Model Architecture
- **Encoder**: A ResNet34 CNN extracts spatial features from input images.
- **Attention Mechanism**: Focuses on relevant parts of the image for each word in the caption.
- **Decoder**: An LSTM generates captions word by word, guided by the attention mechanism.

### 3. Training
- The `train.py` script trains the model using the prepared dataset.
- Key steps during training:
  1. Images are passed through the encoder to extract features.
  2. Captions are tokenized and fed into the decoder.
  3. The decoder generates predictions for the next word in the sequence.
  4. Loss is calculated using cross-entropy and attention regularization.
  5. Gradients are computed and used to update model weights.
- Training metrics (loss, accuracy, BLEU scores) are tracked and saved for visualization.

### 4. Caption Generation
- The `captions.py` script generates captions for new images.
- Steps:
  1. The image is passed through the encoder to extract features.
  2. The decoder generates a caption using beam search or greedy decoding.
  3. Optionally, attention weights are visualized to show which parts of the image influenced each word.

### 5. Evaluation
- BLEU-4 scores are used to evaluate the quality of generated captions.
- Validation and test sets are used to measure model performance.

### 6. Outputs
- Model checkpoints, training metrics, and generated captions are saved in the `model_outputs` directory.
- Attention visualizations and captioned images are also saved for analysis.

This system is modular and extensible, allowing you to experiment with different datasets, architectures, and training strategies.

## Training the Model

### Basic Commands

#### Start Training from Scratch

```bash
python train.py
```

This will train the model with default parameters:
- 120 epochs
- Batch size of 64
- Random initialization of ResNet34 weights
- No fine-tuning of the encoder

#### Using Pretrained Weights

```bash
python train.py --pretrained
```

This loads pretrained ResNet34 weights from torchvision to initialize the encoder.

#### Fine-tuning the Encoder

```bash
python train.py --pretrained --fine_tune_encoder
```

This allows the encoder parameters to be updated during training, which can improve performance.

### Training Parameters

You can customize the training with the following parameters:

- `--resume`: Resume training from the latest checkpoint
- `--fine_tune_encoder`: Enable fine-tuning of the encoder
- `--epochs`: Number of training epochs (default: 120)
- `--batch_size`: Batch size for training (default: 64)
- `--checkpoint`: Specific checkpoint to resume from
- `--pretrained`: Use pretrained ResNet34 weights
- `--one_hot`: Use one-hot-encoding on the word embeding of the captions (default: False)

Example with custom parameters:
```bash
python train.py --pretrained --fine_tune_encoder --epochs 50 --batch_size 32 --one_hot
```

### Resuming Training

To resume training from the last saved checkpoint:

```bash
python train.py --resume
```

When using the `--resume` flag, the model will:
1. Look for the checkpoint file at `model_outputs/checkpoint_flickr8k_5_5.pth.tar`
2. Load the model weights, optimizer states, and training progress
3. Continue training from the epoch where it left off
4. Preserve the BLEU score history
5. Load and continue tracking training metrics (loss and accuracy)

You can also resume with modified settings:

```bash
python train.py --resume --fine_tune_encoder --epochs 20
```

**Note:** When resuming training, the `--epochs` parameter specifies the *total* number of epochs to train, not additional epochs. For example, if you've already trained for 5 epochs and set `--epochs 10`, the model will train for 5 more epochs.

### Training Metrics

The training script automatically tracks and saves the following metrics for each epoch:

#### Tracked Metrics

1. **Training Loss**: Average loss across all training batches
2. **Training Top-5 Accuracy**: Percentage of predictions where the correct word is among the top 5 predictions
3. **Validation Loss**: Average loss on validation data
4. **Validation Top-5 Accuracy**: Top-5 accuracy on validation data
5. **BLEU-4 Score**: Evaluation metric for generated captions

#### Visualizing Metrics

All metrics are saved to JSON files in the `model_outputs` directory. To visualize the training progress:

```bash
python visualize_metrics.py
```

This will generate comprehensive plots and print summary statistics about the training.

#### Interpreting BLEU Scores

For image captioning models, BLEU-4 scores typically fall in these ranges:
- 0.00-0.05: Poor performance
- 0.05-0.10: Basic performance (model is learning)
- 0.10-0.20: Reasonable/moderate performance
- 0.20-0.30: Good performance
- >0.30: Excellent (state-of-the-art on some datasets)

### Tips for Effective Training

1. **Start with pretrained weights** (`--pretrained`): This gives the model a better starting point.
2. **Initially train with frozen encoder**: First train without `--fine_tune_encoder` for about 10-15 epochs.
3. **Fine-tune the encoder**: After initial training, enable `--fine_tune_encoder` to improve performance.
4. **Resume training**: Use `--resume` if training is interrupted.
5. **Monitor the metrics**: Watch for plateaus in the validation metrics to determine when to stop training.
6. **Automatic learning rate adjustment**: If BLEU scores don't improve for 10 epochs, the learning rate is automatically reduced.
7. **Early stopping**: After 20 epochs without improvement, training stops automatically.

## Generating Captions

There are two scripts for generating captions: `captions.py`. The first provides more features including attention visualization, while the second is simpler.

### Using captions.py

The `captions.py` script provides a simpler way to generate captions:

```bash
python captions.py --img path/to/your/image.jpg --model model_outputs/BEST_flickr8k_5_5.pth.tar --word_map data_output/WORDMAP_flickr8k_5_5.json
```

Required arguments:
- `--img` or `-i`: Path to the image you want to caption
- `--model` or `-m`: Path to your trained model checkpoint file
- `--word_map` or `-wm`: Path to the word map JSON file (created during data preparation)

Optional arguments:
- `--beam_size` or `-b`: Size of beam search (default is 5)
- `--visualize_attention` or `-v`: Add this flag to visualize attention weights for each word
- `--captions_json` or `-cj`: Path to the original captions JSON file (Karpathy's format) to display original captions alongside generated ones

The script will:
1. Load the specified model and word map
2. Generate a caption using beam search
3. Print the generated caption to the console
4. If original captions are provided, print them as well
5. Display the image with both generated and original captions (if available)
6. Save the captioned image to `model_outputs/caption_result.png`
7. If original captions are available, save a comparison visualization to `model_outputs/caption_comparison.png`
8. If attention visualization is enabled, it will also generate and save attention maps to `model_outputs/attention_visualization.png`

Example for visualizing attention:
```bash
python captions.py --img test_images/dog.jpg --model model_outputs/BEST_flickr8k_5_5.pth.tar --word_map data_output/WORDMAP_flickr8k_5_5.json --visualize_attention
```

Example with original captions:
```bash
python captions.py --img test_images/dog.jpg --model model_outputs/BEST_flickr8k_5_5.pth.tar --word_map data_output/WORDMAP_flickr8k_5_5.json --captions_json data/caption_datasets/dataset_flickr8k.json
```

### Visualizing Attention

To visualize which parts of the image the model focuses on when generating each word:

```bash
python test_caption.py --image path/to/your/image.jpg --visualize_attention
```

This will generate:
- A caption for the image
- A visualization showing attention weights for each word in the caption
- The attention visualization is saved to `model_outputs/attention_visualization.png`

### Beam Search

The model uses beam search to generate captions during inference:

- **Beam Size**: Controls how many caption candidates are maintained during generation (default: 5)
- **Process**:
  - Start with the start token
  - At each step, consider all possible next words for each candidate sequence
  - Keep the top-k (beam size) sequences with highest probability
  - Continue until all sequences reach the end token or maximum length
  - Return the sequence with highest probability

To adjust the beam size:

```bash
python test_caption.py --image path/to/your/image.jpg --beam_size 10
```

## Model Outputs

### Directory Structure

All model outputs are saved to the `model_outputs` directory, which is automatically created by the scripts if it doesn't exist.

### Output Files

| File Pattern | Description |
|--------------|-------------|
| `checkpoint_*.pth.tar` | Regular model checkpoints saved during training |
| `BEST_*.pth.tar` | Best model weights based on validation BLEU scores |
| `bleu_scores_*.json` | JSON file tracking BLEU scores across epochs |
| `bleu_scores_*.png` | Plot of BLEU scores across training epochs |
| `training_metrics_*.json` | JSON file with loss and accuracy values |
| `loss_plot_*.png` | Plot of training and validation loss |
| `accuracy_plot_*.png` | Plot of training and validation accuracy |
| `attention_visualization.png` | Visualization of attention weights for each word in a caption |
| `caption_result.png` | Generated caption overlaid on the input image |
| `model_architecture.png` | Visual diagram of the model architecture |
| `training_visualization_*.png` | Comprehensive visualization of all training metrics |

### Visualizing Results

To visualize the model architecture:

```bash
python model_architecture.py
```

This will:
1. Print detailed information about the model architecture
2. Generate a visual diagram of the model architecture (requires Graphviz)
3. Save the visualization to `model_outputs/model_architecture.png`

## Key Functions

### Training Functions (train.py)

- **main()**: Entry point for training, handles model initialization and training loop
- **train()**: Performs one epoch of training
- **validate()**: Performs validation and calculates BLEU scores

### Utility Functions (utils.py)

- **save_bleu_scores()**: Saves BLEU scores to a file and creates a plot
- **save_training_metrics()**: Saves training metrics to a file and creates plots
- **save_checkpoint()**: Saves model state to a checkpoint file
- **adjust_learning_rate()**: Reduces learning rate to help convergence
- **AverageMeter**: Class for tracking metrics during training
- **clip_gradient()**: Prevents gradient explosion during training
- **accuracy()**: Calculates top-k accuracy for evaluation

### Model Architecture Functions (model.py)

- **EncoderCNN**: ResNet34-based encoder class
- **Attention**: Attention mechanism implementation
- **LSTMDecoderWithAttention**: Decoder with attention implementation
- **ImageCaptioningModel**: Combined model for visualization

### Testing/Inference Functions (test_caption.py)

- **caption_image()**: Generates a caption for a given image
- **visualize_att()**: Visualizes attention weights for each word

## Troubleshooting

### Training Issues

1. **Out of Memory Errors**:
   - Reduce batch size with `--batch_size 32` or even lower
   - Disable fine-tuning of encoder
   - Use a smaller model variant

2. **Slow Training**:
   - Ensure you're using GPU if available
   - Set `workers=0` to avoid h5py pickling issues
   - Optimize data loading

3. **Low BLEU Scores**:
   - Ensure you're using pretrained weights
   - Train for more epochs
   - Enable fine-tuning after initial training
   - Verify dataset preparation is correct

4. **Training Doesn't Progress When Resuming**:
   - Check that the `--epochs` parameter is greater than the current epoch number
   - Verify the checkpoint file exists
   - Make sure there's enough disk space for new checkpoints

### Graphviz Installation

If you encounter errors with model visualization:

1. Verify Graphviz is installed: Run `dot -v` in a terminal
2. If not installed, download from https://graphviz.org/download/
3. Add the Graphviz bin directory to your system PATH
4. On Windows, you may need to restart your computer after installation
5. The model will function without Graphviz, but visualization will not be available

### Common Errors

1. **"TypeError: h5py objects cannot be pickled"**: 
   - Set `num_workers=0` in DataLoader to resolve this

2. **"CUDA out of memory"**: 
   - Reduce batch size
   - Move to CPU if necessary (`device = torch.device('cpu')`)

3. **"checkpoint_*.pth.tar not found"**: 
   - Ensure you have previously trained the model
   - Check the `model_outputs` directory exists

4. **"ImportError: cannot import name 'ResNet34_Weights'"**:
   - Update to a newer version of PyTorch/torchvision
   - The code should fall back to an older method if this fails

5. **"Dimension out of range"** in encoder outputs:
   - This can happen if your model architecture has changed or was trained differently
   - The captions.py script has been updated to handle different tensor dimensions automatically
   - If you still encounter this error, check if your encoder outputs dimensions match what the decoder expects
   - Common tensor shape should be (batch_size, channels, height, width) for encoder outputs

6. **Image loading errors** in captions.py:
   - The script now supports both PIL and OpenCV image loading methods
   - Ensure your image file exists and is a valid image format (jpg, png, etc.)
   - If you're getting errors with PIL, the script will automatically try OpenCV as a fallback

## How A2C Works

The Advantage Actor-Critic (A2C) reinforcement learning approach is implemented to improve the image captioning model by directly optimizing for BLEU-4 scores. Here's an overview of how A2C works in this project:

### 1. Actor-Critic Framework
- **Actor**: The existing image captioning model (encoder-decoder with attention) acts as the policy network.
  - It generates captions by sampling words from its probability distribution.
- **Critic**: A separate neural network predicts the expected reward (value) of the generated sequence at each time step.

### 2. Training Process
- **Sampling**: The actor generates captions by sampling words from its probability distribution.
- **Reward**: BLEU-4 scores are used as rewards for the generated captions.
- **Advantage Calculation**: The advantage is computed as the difference between the actual rewards and the critic's predicted values.
  - Advantage = Reward - Predicted Value
- **Policy Gradient Update**: The actor is updated using the policy gradient method, weighted by the advantage.
- **Critic Update**: The critic is trained to minimize the mean squared error between its predictions and the actual rewards.

### 3. Key Components
- **Entropy Regularization**: Encourages exploration by penalizing low-entropy (overconfident) predictions.
- **Discount Factor (Gamma)**: Rewards are discounted over time to prioritize immediate rewards.
- **Gradient Clipping**: Prevents exploding gradients during training.

### 4. Implementation Details
- The actor uses the existing encoder-decoder model with attention.
- The critic is a simple multi-layer perceptron (MLP) that takes the decoder's hidden states as input and predicts the expected reward.
- BLEU-4 scores are computed for each generated caption and used as the final reward.
- The A2C training loop alternates between updating the actor and the critic.

### 5. Running A2C
To train the model using A2C, run the following command:

```bash
python -m src.A2C --checkpoint model_outputs/BEST_flickr8k_5_5.pth.tar --epochs 20 --batch_size 32
```

### 6. Outputs
- **Actor Loss**: Measures how well the actor improves its policy based on the advantage.
- **Critic Loss**: Measures the accuracy of the critic's value predictions.
- **BLEU-4 Scores**: Tracks the quality of generated captions over time.

This approach allows the model to directly optimize for BLEU-4 scores, leading to more accurate and diverse captions.

## Acknowledgements

This implementation is based on the "Show, Attend and Tell" paper by Xu et al. # Image_captioning
