### 
# Added new critic network that combines CNN image features and decoder hidden state features
# to predict the expected reward for an image-caption pair.
#Change CriticNetwork to HybridCriticNetwork 
###

import time
import torch.backends.cudnn as cudnn
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
from src.model import EncoderCNN, LSTMDecoderWithAttention
from src.dataset import *
from src.utils import *
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from tqdm import tqdm
import argparse
import os
import numpy as np
import sys  # Add sys import
import matplotlib.pyplot as plt  # Add matplotlib for plotting metrics
import json

data_folder = 'data_output'  # files saved by create_input_files.py
data_name = 'flickr8k_5_5'  # base name shared by data files

checkpoint = 'model_outputs/BEST_flickr8k_5_5.pth.tar'  # model checkpoint

# A2C (Advantage Actor-Critic) hyperparameters
critic_lr = 1e-4
actor_lr = 5e-5
entropy_weight = 0.05  # Weight for the entropy regularization term
value_loss_weight = 0.2  # Weight for the value loss term
gamma = 0.99  # Discount factor
max_grad_norm = 5.0  # For gradient clipping


class CriticNetwork(nn.Module):
    """
    Critic network for the A2C model.
    Predicts the value (expected reward) of a state.
    Takes features from the actor's decoder and predicts the expected future reward.
    """
    def __init__(self, input_dim, hidden_dim=256):
        """
        :param input_dim: dimension of input features (should match decoder's output dimension)
        :param hidden_dim: dimension of hidden layer
        """
        super(CriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # Initialize weights
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
    
    def forward(self, features):
        """
        Forward pass through the critic network
        :param features: features from the decoder (batch_size, feature_dim)
        :return: predicted state value (batch_size, 1)
        """
        x = self.relu(self.fc1(features))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        value = self.fc3(x)
        return value

#Hybrid critic network (combines CNN and LSTM features)

class HybridCriticNetwork(nn.Module):
    """
    Critic network that combines CNN image features and decoder hidden state features
    to predict the expected reward for an image-caption pair.
    """
    def __init__(self, img_channels=512, decoder_dim=512, hidden_dim=256):
        super(HybridCriticNetwork, self).__init__()

        # Convolutional branch for image features
        self.conv1 = nn.Conv2d(img_channels, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Output: (batch_size, 128, 1, 1)

        # Fully connected layers
        self.fc_img = nn.Linear(128, hidden_dim // 2)
        self.fc_dec = nn.Linear(decoder_dim, hidden_dim // 2)
        self.fc_combined = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

        # Init weights
        for layer in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0)
        for layer in [self.fc_img, self.fc_dec, self.fc_combined, self.output_layer]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)

    def forward(self, img_feats, decoder_feats):
        """
        :param img_feats: CNN feature map (batch_size, img_channels, H, W)
        :param decoder_feats: decoder hidden state (batch_size, decoder_dim)
        :return: scalar value for each state (batch_size, 1)
        """
        # Image path
        x_img = self.relu(self.conv1(img_feats))
        x_img = self.relu(self.conv2(x_img))
        x_img = self.pool(x_img).squeeze(-1).squeeze(-1)  # -> (batch_size, 128)
        x_img = self.relu(self.fc_img(x_img))

        # Decoder path
        x_dec = self.relu(self.fc_dec(decoder_feats))

        # Combine both
        x = torch.cat([x_img, x_dec], dim=1)
        x = self.relu(self.fc_combined(x))
        x = self.dropout(x)
        value = self.output_layer(x)
        return value


class A2CImageCaptioning:
    def __init__(self, word_map, device, 
                 encoder_lr=1e-5, decoder_lr=1e-4, critic_lr=1e-4, 
                 entropy_weight=0.01, value_loss_weight=0.5, gamma=0.99,
                 checkpoint=None, fine_tune_encoder=False,
                 beam_size=5, temperature=1.0):
        """
        Initialize the A2C Image Captioning model
        :param word_map: dictionary mapping words to indices
        :param device: device to run the model on
        :param encoder_lr: learning rate for the encoder
        :param decoder_lr: learning rate for the decoder (actor)
        :param critic_lr: learning rate for the critic
        :param entropy_weight: weight for the entropy regularization term
        :param value_loss_weight: weight for the value loss term
        :param gamma: discount factor for rewards
        :param checkpoint: path to a checkpoint file to load
        :param fine_tune_encoder: whether to fine-tune the encoder
        :param beam_size: beam size for beam search decoding
        :param temperature: temperature for sampling
        """
        self.device = device
        self.word_map = word_map
        self.rev_word_map = {v: k for k, v in word_map.items()}
        self.vocab_size = len(word_map)
        self.fine_tune_encoder = fine_tune_encoder
        self.gamma = gamma
        self.entropy_weight = entropy_weight
        self.value_loss_weight = value_loss_weight
        self.beam_size = beam_size
        self.temperature = temperature
        
        # Special tokens
        self.start_token = word_map['<start>']
        self.end_token = word_map['<end>']
        self.pad_token = word_map['<pad>']
        
        # Load or initialize the model
        if checkpoint is not None:
            print(f"Loading checkpoint from {checkpoint}")
            
            # Create a temporary module mapping to handle the import mismatch
            if 'model' not in sys.modules:
                sys.modules['model'] = sys.modules['src.model']
            
            # Load checkpoint with the right device mapping
            checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
            
            # Check if this is a regular checkpoint or an A2C checkpoint
            if 'critic' in checkpoint_data:
                print("Loading A2C checkpoint with critic model")
                is_a2c_checkpoint = True
            else:
                print("Loading regular checkpoint (encoder/decoder only)")
                is_a2c_checkpoint = False
            
            # Load encoder and decoder (actor) from checkpoint
            self.encoder = checkpoint_data['encoder'].to(device)
            self.decoder = checkpoint_data['decoder'].to(device)
            
            # Get encoder dimension from loaded encoder
            encoder_dim = self.encoder.enc_dim
            decoder_dim = self.decoder.decoder_dim
            
            # Initialize or load critic
            if is_a2c_checkpoint:
                self.critic = checkpoint_data['critic'].to(device)
            else:
                # Initialize new critic for regular checkpoints
                self.critic = CriticNetwork(decoder_dim).to(device)
        else:
            # Initialize encoder and decoder from scratch (unlikely to be used, but included for completeness)
            print("Initializing new encoder and decoder models")
            is_a2c_checkpoint = False
            encoder_dim = 512  # Default for ResNet34
            
            # Initialize encoder with optional pretrained weights
            self.encoder = EncoderCNN(pretrained_path="torchvision").to(device)
            
            # Initialize decoder (actor)
            attention_dim = 512
            embed_dim = 512
            decoder_dim = 300
            dropout = 0.3
            
            self.decoder = LSTMDecoderWithAttention(
                attention_dim=attention_dim,
                embed_dim=embed_dim,
                decoder_dim=decoder_dim,
                vocab_size=self.vocab_size,
                encoder_dim=encoder_dim,
                dropout=dropout
            ).to(device)
            
            # Initialize critic
            self.critic = CriticNetwork(decoder_dim).to(device)
        
        # Set fine-tuning mode
        self.encoder.fine_tune(fine_tune_encoder)
        
        # Setup optimizers
        # The actor is the decoder (which generates captions)
        self.actor_optimizer = optim.Adam(
            params=filter(lambda p: p.requires_grad, self.decoder.parameters()),
            lr=decoder_lr
        )
        
        # The critic optimizer
        self.critic_optimizer = optim.Adam(
            params=self.critic.parameters(),
            lr=critic_lr
        )
        
        # Optional encoder optimizer if fine-tuning
        if fine_tune_encoder:
            self.encoder_optimizer = optim.Adam(
                params=filter(lambda p: p.requires_grad, self.encoder.parameters()),
                lr=encoder_lr
            )
        else:
            self.encoder_optimizer = None
            
        # Load optimizer states if it's an A2C checkpoint
        if checkpoint is not None and is_a2c_checkpoint:
            if 'actor_optimizer' in checkpoint_data:
                self.actor_optimizer.load_state_dict(checkpoint_data['actor_optimizer'].state_dict())
            if 'critic_optimizer' in checkpoint_data:
                self.critic_optimizer.load_state_dict(checkpoint_data['critic_optimizer'].state_dict())
            if self.encoder_optimizer is not None and 'encoder_optimizer' in checkpoint_data:
                self.encoder_optimizer.load_state_dict(checkpoint_data['encoder_optimizer'].state_dict())
            
        # Loss functions
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token, reduction='none').to(device)
        self.mse_loss = nn.MSELoss().to(device)
    
    def compute_returns(self, final_reward, sequence_length):
        """
        Compute discounted returns for a given episode
        :param final_reward: final reward at the end of the episode (BLEU score)
        :param sequence_length: length of the generated sequence
        :return: tensor of discounted returns for each time step
        """
        returns = torch.zeros(sequence_length, device=self.device)
        # Start with the final reward
        R = final_reward
        # Work backwards computing returns
        for t in reversed(range(sequence_length)):
            returns[t] = R
            R = self.gamma * R
        return returns
    
    def compute_sentence_bleu(self, references, hypothesis):
        """
        Compute BLEU score for a single sentence
        :param references: list of reference sentences (list of lists of tokens)
        :param hypothesis: predicted sentence (list of tokens)
        :return: BLEU score
        """
        smoothing = None
        try:
            from nltk.translate.bleu_score import SmoothingFunction
            smoothing = SmoothingFunction().method1
        except ImportError:
            pass
        
        # Remove <start>, <end>, and <pad> tokens
        clean_hypothesis = [w for w in hypothesis if w not in [self.start_token, self.end_token, self.pad_token]]
        
        # Clean references similarly
        clean_references = []
        for reference in references:
            clean_reference = [w for w in reference if w not in [self.start_token, self.end_token, self.pad_token]]
            if clean_reference:  # Only add non-empty references
                clean_references.append(clean_reference)
        
        if not clean_references:
            return 0.0
        
        # Calculate BLEU score with smoothing for short sentences
        weights = (0.25, 0.25, 0.25, 0.25)  # Default for BLEU-4
        try:
            return sentence_bleu(clean_references, clean_hypothesis, smoothing_function=smoothing, weights=weights)
        except:
            # Fallback if there are issues
            return 0.0
    
    def generate_caption_with_sampling(self, encoder_out, max_length=20):
        """
        Generate a caption using sampling from the model's probability distribution
        :param encoder_out: encoded image features
        :param max_length: maximum caption length
        :return: generated caption, log probabilities, and entropy values
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        
        # Flatten image features
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        
        # Initialize LSTM state
        h_list, c_list = self.decoder.init_hidden_state(encoder_out)
        
        # Start with <start> token
        prev_words = torch.full((batch_size, 1), self.start_token, device=self.device, dtype=torch.long)
        
        # Tensors to store generated sequences, log probabilities, and entropies
        sequences = torch.full((batch_size, max_length), self.pad_token, device=self.device, dtype=torch.long)
        log_probs = torch.zeros(batch_size, max_length, device=self.device)
        entropies = torch.zeros(batch_size, max_length, device=self.device)
        
        # Store hidden states for critic
        hidden_states = []
        
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        # Generate caption word by word
        for step in range(max_length):
            # Get embeddings for current words
            if self.decoder.use_one_hot:
                # Convert to one-hot vectors
                one_hot_words = self.decoder.one_hot_encoder(prev_words[:, -1])
                embeddings = self.decoder.embedding(one_hot_words.float())
            else:
                embeddings = self.decoder.embedding(prev_words[:, -1])
            
            # Run attention LSTM for one step
            # For simplicity, using only the active (not finished) sequences
            active_indices = (~finished).nonzero(as_tuple=True)[0]
            
            if len(active_indices) == 0:
                break  # All sequences finished
                
            # Calculate attention using the last layer's hidden state
            attention_weighted_encoding, alpha = self.decoder.attention(
                encoder_out[active_indices], 
                h_list[-1][active_indices]
            )
            
            gate = self.decoder.sigmoid(self.decoder.f_beta(h_list[-1][active_indices]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            
            h_new = []
            c_new = []
            
            # Process through each LSTM layer
            for i in range(self.decoder.num_layers):
                if i == 0:
                    # First layer receives embeddings and attention
                    lstm_input = torch.cat(
                        [embeddings[active_indices], attention_weighted_encoding], 
                        dim=1
                    )
                else:
                    # Other layers receive previous layer's hidden state
                    lstm_input = h_new[-1]
                
                h, c = self.decoder.lstm_layers[i](
                    lstm_input,
                    (h_list[i][active_indices], c_list[i][active_indices])
                )
                h_new.append(h)
                c_new.append(c)
            
            # Update hidden and cell states for active sequences
            for i in range(self.decoder.num_layers):
                h_list[i][active_indices] = h_new[i]
                c_list[i][active_indices] = c_new[i]
            
            # Save the final layer's hidden state for critic
            hidden_states.append(h_list[-1].clone())
            
            # Get predicted scores for next word
            scores = self.decoder.fc(self.decoder.dropout(h_list[-1]))  # (batch_size, vocab_size)
            
            # Apply temperature for exploration
            scores = scores / self.temperature
            
            # Compute probabilities and log probabilities
            probs = torch.softmax(scores, dim=1)
            log_prob = torch.log_softmax(scores, dim=1)
            
            # Compute entropy: -sum(p * log(p))
            entropy = -torch.sum(probs * log_prob, dim=1)
            
            # Sample from the probability distribution for active sequences only
            next_word_idx_active = torch.multinomial(probs[active_indices], 1).squeeze(1)  # (active_batch_size,)
            
            # Save generated word, its log probability, and entropy for active sequences
            sequences[active_indices, step] = next_word_idx_active
            log_probs[active_indices, step] = log_prob[active_indices].gather(1, next_word_idx_active.unsqueeze(1)).squeeze(1)
            entropies[active_indices, step] = entropy[active_indices]
            
            # Create a full batch tensor for previous words update (needed for next iteration)
            next_word_idx = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            next_word_idx[active_indices] = next_word_idx_active
            
            # Update finished sequences mask
            just_finished = next_word_idx_active == self.end_token
            finished[active_indices] = finished[active_indices] | just_finished
            
            # Update previous words for next iteration
            prev_words = torch.cat([prev_words, next_word_idx.unsqueeze(1)], dim=1)
            
            # Early stopping if all sequences have finished
            if finished.all():
                break
        
        # Stack hidden states for critic
        hidden_states = torch.stack(hidden_states, dim=1)  # (batch_size, seq_len, hidden_dim)
        
        return sequences, log_probs, entropies, hidden_states
    
    def a2c_train_step(self, images, captions, caption_lengths, allcaps=None):
        """
        Perform a single A2C training step
        :param images: batch of images
        :param captions: ground truth captions
        :param caption_lengths: caption lengths
        :param allcaps: all available captions for each image (for BLEU calculation)
        :return: actor loss, critic loss, BLEU score
        """
        batch_size = images.size(0)
        
        # Move data to device
        images = images.to(self.device)
        captions = captions.to(self.device)
        caption_lengths = caption_lengths.to(self.device)
        
        if allcaps is not None:
            allcaps = allcaps.to(self.device)
        
        # 1. Get image features from encoder
        encoder_out = self.encoder(images)
        
        # 2. Generate captions by sampling from the model
        samples, log_probs, entropies, hidden_states = self.generate_caption_with_sampling(encoder_out)
        
        # Ensure we only use valid time steps in hidden states (up to the sequence length)
        seq_length = hidden_states.size(1)
        
        # 3. Compute critic values for each step of each sequence
        values = self.critic(hidden_states.view(-1, hidden_states.size(-1))).view(batch_size, seq_length)
        
        # 4. Compute rewards and advantages
        rewards = torch.zeros_like(values)  # Match dimensions with values tensor
        advantages = torch.zeros_like(values)
        bleu_scores = []
        
        # Initialize smoothing function for BLEU calculation
        from nltk.translate.bleu_score import SmoothingFunction
        smoothing = SmoothingFunction().method1
        
        # Compute sequence lengths (excluding padding)
        sequence_lengths = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        for i in range(batch_size):
            # Find position of first <end> token or use max length
            end_pos = (samples[i] == self.end_token).nonzero(as_tuple=True)[0]
            sequence_lengths[i] = end_pos[0] + 1 if len(end_pos) > 0 else samples.size(1)
        
        # For each sample, compute BLEU score against all reference captions
        all_rewards = []
        for i in range(batch_size):
            sample_tokens = samples[i].tolist()
            sample_length = sequence_lengths[i].item()
            
            # Truncate to actual sequence length
            sample_tokens = sample_tokens[:sample_length]
            
            # Get all reference captions for this image
            references = []
            if allcaps is not None:
                for j in range(allcaps.size(1)):
                    ref_tokens = allcaps[i, j].tolist()
                    # Remove padding and end tokens
                    ref_tokens = [t for t in ref_tokens if t not in [self.pad_token]]
                    if ref_tokens:
                        references.append(ref_tokens)
            else:
                # Use ground truth caption if no alternative captions provided
                ref_tokens = captions[i, :caption_lengths[i]].tolist()
                references.append(ref_tokens)
            
            # Compute BLEU scores with smoothing
            bleu1 = sentence_bleu(references, sample_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
            bleu2 = sentence_bleu(references, sample_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
            bleu4 = sentence_bleu(references, sample_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
            
            # Combined reward (gives credit for partial matches)
            reward = 0.2 * bleu1 + 0.3 * bleu2 + 0.5 * bleu4
            bleu_scores.append(bleu4)
            all_rewards.append(reward)
            
            # Compute returns using the BLEU score as the final reward
            sample_returns = self.compute_returns(reward, min(sample_length, seq_length))
            
            # Apply reward clipping to stabilize training
            sample_returns = torch.clamp(sample_returns, min=-1.0, max=1.0)
            
            # Fill in rewards for this sample
            rewards[i, :min(sample_length, seq_length)] = sample_returns
        
        # Calculate the baseline as the mean reward for this batch
        # This helps reduce variance in updates
        reward_baseline = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        
        # Create a tensor that has the baseline for each sequence
        baseline_tensor = torch.full_like(rewards, reward_baseline)
        
        # 5. Compute advantages (rewards - baseline - value predictions)
        # Use a combination of critic values and mean reward baseline
        advantages = rewards - 0.5 * (values.detach() + baseline_tensor)
        
        # 6. Compute losses
        # Actor loss: -log_prob * advantage - entropy_weight * entropy (entropy is for exploration)
        actor_loss = -(log_probs[:, :seq_length] * advantages.detach()).mean() - self.entropy_weight * entropies[:, :seq_length].mean()
        
        # Critic loss: MSE between predicted values and actual returns
        critic_loss = self.mse_loss(values, rewards.detach())
        
        # Combined loss
        total_loss = actor_loss + self.value_loss_weight * critic_loss
        
        # 7. Backpropagate and update weights
        # Zero gradients
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.zero_grad()
            
        # Backward pass
        total_loss.backward()
        
        # Clip gradients
        if self.encoder_optimizer is not None:
            clip_gradient(self.encoder_optimizer, max_grad_norm)
        clip_gradient(self.actor_optimizer, max_grad_norm)
        clip_gradient(self.critic_optimizer, max_grad_norm)
        
        # Update weights
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.step()
            
        # 8. Return losses and metrics
        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
        return actor_loss.item(), critic_loss.item(), avg_bleu
    
    def train_epoch(self, train_loader, epoch):
        """
        Train for one epoch
        :param train_loader: DataLoader for training data
        :param epoch: epoch number
        :return: average losses and BLEU score for this epoch
        """
        # Set models to training mode
        self.encoder.train()
        self.decoder.train()
        self.critic.train()
        
        # Metrics
        batch_time = AverageMeter()
        actor_losses = AverageMeter()
        critic_losses = AverageMeter()
        bleu_scores = AverageMeter()
        
        start = time.time()
        
        # Iterate over data batches
        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"A2C Training Epoch {epoch}"):
            # Handle both 3-item and 4-item batches
            if len(batch) == 4:
                images, captions, caplens, allcaps = batch
            else:
                images, captions, caplens = batch
                allcaps = None
            
            # Training step
            actor_loss, critic_loss, bleu = self.a2c_train_step(images, captions, caplens, allcaps)
            
            # Update metrics
            actor_losses.update(actor_loss, images.size(0))
            critic_losses.update(critic_loss, images.size(0))
            bleu_scores.update(bleu, images.size(0))
            
            batch_time.update(time.time() - start)
            start = time.time()
            
            # Print status
            if i % 10 == 0:  # Print every 10 batches
                print(f"Epoch: [{epoch}][{i}/{len(train_loader)}] "
                      f"Time {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                      f"Actor Loss {actor_losses.val:.4f} ({actor_losses.avg:.4f}) "
                      f"Critic Loss {critic_losses.val:.4f} ({critic_losses.avg:.4f}) "
                      f"BLEU {bleu_scores.val:.3f} ({bleu_scores.avg:.3f})")
        
        return actor_losses.avg, critic_losses.avg, bleu_scores.avg
    
    def validate(self, val_loader):
        """
        Evaluate the model on the validation set
        :param val_loader: DataLoader for validation data
        :return: BLEU-4 score on validation data
        """
        # Set models to evaluation mode
        self.encoder.eval()
        self.decoder.eval()
        self.critic.eval()
        
        # Lists to store references and hypotheses for BLEU score calculation
        references = list()
        hypotheses = list()
        
        # For timing
        batch_time = AverageMeter()
        start = time.time()
        
        # No gradients needed for validation
        with torch.no_grad():
            # Batches
            for i, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc="A2C Validation"):
                # Handle both 3-item and 4-item batches
                if len(batch) == 4:
                    images, captions, caplens, allcaps = batch
                else:
                    images, captions, caplens = batch
                    allcaps = None
                
                # Move to device
                images = images.to(self.device)
                allcaps = allcaps.to(self.device) if allcaps is not None else None
                
                # Forward pass through encoder
                encoder_out = self.encoder(images)
                
                # Generate captions using the model with sampling
                samples, _, _, _ = self.generate_caption_with_sampling(encoder_out)
                
                # Convert word indices to words for all sampled captions
                for j in range(samples.size(0)):
                    # Find position of first <end> token or use max length
                    end_pos = (samples[j] == self.end_token).nonzero(as_tuple=True)[0]
                    length = end_pos[0].item() + 1 if len(end_pos) > 0 else samples.size(1)
                    
                    # Convert to list of word indices (excluding <start>)
                    hyp_tokens = samples[j, :length].tolist()
                    # Convert indices to words
                    hypothesis = [self.rev_word_map[token] for token in hyp_tokens if token not in [self.start_token, self.pad_token]]
                    # Remove <end> token if present
                    if self.rev_word_map[self.end_token] in hypothesis:
                        hypothesis = hypothesis[:hypothesis.index(self.rev_word_map[self.end_token])]
                    
                    # Add to hypotheses list
                    hypotheses.append(hypothesis)
                    
                    # For each image, get all reference captions
                    references_for_image = []
                    if allcaps is not None:
                        for k in range(allcaps.size(1)):
                            # Get reference caption
                            ref_tokens = allcaps[j, k].tolist()
                            # Convert indices to words
                            reference = [self.rev_word_map[token] for token in ref_tokens if token not in [self.start_token, self.end_token, self.pad_token]]
                            # Add to references list for this image
                            if reference:
                                references_for_image.append(reference)
                    
                    # Add references for this image to the main list
                    references.append(references_for_image)
                
                # Update batch time
                batch_time.update(time.time() - start)
                start = time.time()
                
                # Print status
                if i % 10 == 0:
                    print(f"Validation: [{i}/{len(val_loader)}] "
                          f"Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})")
        
        # Calculate BLEU-4 score
        bleu4 = corpus_bleu(references, hypotheses)
        
        print(f"\n * BLEU-4 - {bleu4:.4f}")
        
        return bleu4

    def save_model(self, data_name, epoch, bleu4, is_best=False):
        """
        Save the model checkpoint
        :param data_name: name of the dataset
        :param epoch: epoch number
        :param bleu4: BLEU-4 score
        :param is_best: whether this checkpoint is the best so far
        """
        state = {
            'epoch': epoch,
            'bleu-4': bleu4,
            'encoder': self.encoder,
            'decoder': self.decoder,
            'critic': self.critic,
            'encoder_optimizer': self.encoder_optimizer,
            'actor_optimizer': self.actor_optimizer,
            'critic_optimizer': self.critic_optimizer
        }
        
        # Create directory if it doesn't exist
        output_dir = 'model_outputs'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the checkpoint
        filename = os.path.join(output_dir, f'a2c_checkpoint_{data_name}.pth.tar')
        torch.save(state, filename)
        
        # If this is the best model, save a copy
        if is_best:
            best_filename = os.path.join(output_dir, f'a2c_BEST_{data_name}.pth.tar')
            torch.save(state, best_filename)


def a2c_train(data_folder, data_name, batch_size=32, epochs=20, checkpoint=None, 
              fine_tune_encoder=False, freeze_encoder=False, workers=0, resume_training=False,
              temperature=1.0, entropy_weight=0.05, entropy_annealing=True, value_loss_weight=0.2):
    """
    Train the image captioning model with A2C
    :param data_folder: folder with data files
    :param data_name: base name shared by data files
    :param batch_size: batch size
    :param epochs: number of epochs
    :param checkpoint: checkpoint to resume from
    :param fine_tune_encoder: whether to fine-tune the encoder
    :param freeze_encoder: whether to freeze the encoder completely (overrides fine_tune_encoder)
    :param workers: number of workers for data loading
    :param resume_training: whether to resume training from the checkpoint (load optimizer states and metrics)
    :param temperature: temperature for exploration (higher values = more exploration)
    :param entropy_weight: initial weight for entropy regularization
    :param entropy_annealing: whether to apply annealing to entropy weight
    :param value_loss_weight: weight for the value loss term
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    
    # Load word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    
    # If freeze_encoder is set, fine_tune_encoder should be False
    if freeze_encoder:
        fine_tune_encoder = False
        print("Freezing encoder (all parameters)")
    
    # Initialize the A2C model
    model = A2CImageCaptioning(
        word_map=word_map,
        device=device,
        checkpoint=checkpoint,
        fine_tune_encoder=fine_tune_encoder,
        entropy_weight=entropy_weight,
        value_loss_weight=value_loss_weight,
        temperature=temperature
    )
    
    # Data loaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_loader = DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True
    )
    
    # Initialize or load metrics
    start_epoch = 0
    best_bleu4 = 0.0
    epochs_since_improvement = 0
    actor_losses = []
    critic_losses = []
    train_bleu_scores = []
    val_bleu_scores = []
    
    # If resuming training from an A2C checkpoint, try to load metrics
    if resume_training and checkpoint is not None:
        try:
            # Load the checkpoint to get the epoch number
            checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
            
            if 'epoch' in checkpoint_data and 'bleu-4' in checkpoint_data:
                start_epoch = checkpoint_data['epoch'] + 1  # Start from the next epoch
                best_bleu4 = checkpoint_data['bleu-4']
                print(f"Resuming from epoch {start_epoch}, best BLEU-4: {best_bleu4:.4f}")
            
                # Try to load metrics data
                metrics_file = os.path.join('model_outputs', f'a2c_metrics_{data_name}.json')
                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r') as f:
                        metrics_data = json.load(f)
                    
                    actor_losses = metrics_data.get('actor_loss', [])
                    critic_losses = metrics_data.get('critic_loss', [])
                    train_bleu_scores = metrics_data.get('train_bleu', [])
                    val_bleu_scores = metrics_data.get('val_bleu', [])
                    
                    print(f"Loaded metrics from {metrics_file}")
                    
                    # Sanity check: number of epochs should match
                    expected_epochs = checkpoint_data['epoch'] + 1
                    if len(actor_losses) != expected_epochs:
                        print(f"Warning: Expected {expected_epochs} epochs in metrics, but found {len(actor_losses)}")
                        # Continue anyway with what we have
            else:
                print("Checkpoint does not contain epoch info, starting from epoch 0")
        except Exception as e:
            print(f"Error loading metrics, starting from epoch 0: {e}")
            start_epoch = 0
    
    # Training loop
    for epoch in range(start_epoch, start_epoch + epochs):
        if epochs_since_improvement == 50:
            print("No improvement for 10 epochs. Stopping training.")
            break
            
        # Adjust learning rates if necessary
        if epochs_since_improvement > 0 and epochs_since_improvement % 10 == 0:
            for optimizer in [model.actor_optimizer, model.critic_optimizer]:
                adjust_learning_rate(optimizer, 0.8)
            if model.encoder_optimizer is not None:
                adjust_learning_rate(model.encoder_optimizer, 0.8)
        
        # Apply entropy annealing if enabled
        if entropy_annealing:
            current_entropy_weight = max(0.001, entropy_weight * (0.95 ** epoch))
            model.entropy_weight = current_entropy_weight
            print(f"Current entropy weight: {current_entropy_weight:.6f}")
                
        # One epoch's training with A2C
        actor_loss, critic_loss, train_bleu = model.train_epoch(train_loader, epoch)
        
        # Store metrics
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)
        train_bleu_scores.append(train_bleu)
        
        # One epoch's validation
        val_bleu = model.validate(val_loader)
        val_bleu_scores.append(val_bleu)
        
        # Check if there was an improvement
        is_best = val_bleu > best_bleu4
        best_bleu4 = max(val_bleu, best_bleu4)
        
        if not is_best:
            epochs_since_improvement += 1
            print(f"Epochs since last improvement: {epochs_since_improvement}")
        else:
            epochs_since_improvement = 0
            print(f"New best BLEU-4 score: {best_bleu4:.4f}")
        
        # Save checkpoint
        model.save_model(data_name, epoch, val_bleu, is_best=is_best)
        
        # Save metrics
        save_a2c_metrics(data_name, epoch, actor_losses, critic_losses, train_bleu_scores, val_bleu_scores)


def save_a2c_metrics(data_name, epoch, actor_losses, critic_losses, train_bleu_scores, val_bleu_scores):
    """
    Save and plot A2C training metrics
    :param data_name: name of the dataset
    :param epoch: current epoch number
    :param actor_losses: list of actor losses
    :param critic_losses: list of critic losses
    :param train_bleu_scores: list of training BLEU scores
    :param val_bleu_scores: list of validation BLEU scores
    """
    output_dir = 'model_outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_file = os.path.join(output_dir, f'a2c_metrics_{data_name}.json')
    
    # Create epochs array based on the actual number of completed epochs
    # This handles cases where training might be resumed from different points
    num_epochs = len(actor_losses)
    if num_epochs == 0:
        print("Warning: No metrics to save")
        return
        
    # Create epochs array starting from the last epoch minus the number of recorded metrics
    # This ensures that if we're resuming training, the epoch numbers will be continuous
    start_epoch = epoch - num_epochs + 1
    epochs = list(range(start_epoch, epoch + 1))
    
    # Ensure all metrics arrays have the same length
    min_len = min(len(epochs), len(actor_losses), len(critic_losses), 
                 len(train_bleu_scores), len(val_bleu_scores))
    
    if min_len < num_epochs:
        print(f"Warning: Some metrics arrays have different lengths. Trimming to {min_len} elements.")
        epochs = epochs[-min_len:]
        actor_losses = actor_losses[-min_len:]
        critic_losses = critic_losses[-min_len:]
        train_bleu_scores = train_bleu_scores[-min_len:]
        val_bleu_scores = val_bleu_scores[-min_len:]
    
    # Save metrics with epoch numbers
    metrics_data = {
        'epochs': epochs,
        'actor_loss': actor_losses,
        'critic_loss': critic_losses,
        'train_bleu': train_bleu_scores,
        'val_bleu': val_bleu_scores
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f)
    
    # Plot actor and critic losses
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, actor_losses, marker='o', label='Actor Loss')
    plt.plot(epochs, critic_losses, marker='x', label='Critic Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'A2C Actor and Critic Losses for {data_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'a2c_losses_{data_name}.png'))
    plt.close()
    
    # Plot BLEU scores
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_bleu_scores, marker='o', label='Training BLEU')
    plt.plot(epochs, val_bleu_scores, marker='x', label='Validation BLEU')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU-4 Score')
    plt.title(f'A2C BLEU Scores for {data_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'a2c_bleu_{data_name}.png'))
    plt.close()
    
    print(f"A2C metrics saved to {metrics_file}")


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description='A2C Image Captioning')
    parser.add_argument('--data_folder', default='data_output', help='folder with data files')
    parser.add_argument('--data_name', default='flickr8k_5_5', help='base name shared by data files')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('--checkpoint', default='model_outputs/BEST_flickr8k_5_5.pth.tar', help='checkpoint to resume from')
    parser.add_argument('--fine_tune_encoder', action='store_true', help='fine-tune encoder')
    parser.add_argument('--freeze_encoder', action='store_true', help='completely freeze encoder (overrides fine_tune_encoder)')
    parser.add_argument('--resume', action='store_true', help='resume training from the A2C checkpoint')
    parser.add_argument('--workers', default=0, type=int, help='number of workers for data loading')
    parser.add_argument('--temperature', default=1.0, type=float, help='temperature for sampling (higher = more exploration)')
    parser.add_argument('--entropy_weight', default=0.05, type=float, help='weight for entropy regularization term')
    parser.add_argument('--value_loss_weight', default=0.2, type=float, help='weight for value loss term')
    parser.add_argument('--no_entropy_annealing', action='store_true', help='disable entropy weight annealing')
    args = parser.parse_args()
    
    # Train with A2C
    a2c_train(
        data_folder=args.data_folder,
        data_name=args.data_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        checkpoint=args.checkpoint,
        fine_tune_encoder=args.fine_tune_encoder,
        freeze_encoder=args.freeze_encoder,
        workers=args.workers,
        resume_training=args.resume,
        temperature=args.temperature,
        entropy_weight=args.entropy_weight,
        entropy_annealing=not args.no_entropy_annealing,
        value_loss_weight=args.value_loss_weight
    )


if __name__ == '__main__':
    main()