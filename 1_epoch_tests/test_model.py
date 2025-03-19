import time
import torch
import os
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from model import EncoderCNN, LSTMDecoderWithAttention
from dataset import CaptionDataset
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import json
from tqdm import tqdm

# Test configuration
data_folder = 'data_output'  # Folder with data files
data_name = 'flickr8k_5_5'  # Base name of data files

# Model parameters (same as training but with smaller batches)
emb_dim = 512
attention_dim = 512
decoder_dim = 512
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# Test parameters
batch_size = 4  # Small batch size for testing
workers = 0  # Set to 0 to avoid h5py pickling issues
encoder_lr = 1e-4
decoder_lr = 4e-4
grad_clip = 5.0
alpha_c = 1.0
fine_tune_encoder = False
print_freq = 2  # Print stats more frequently for testing

def test_model():
    """
    Test function to verify model functionality by running one epoch
    """
    print(f"Testing model on {device}")
    
    # Load word map
    try:
        word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
        with open(word_map_file, 'r') as j:
            word_map = json.load(j)
            print(f"Word map loaded with {len(word_map)} words")
    except FileNotFoundError:
        print(f"Word map file not found at {word_map_file}")
        print("Please run create_data_n_prep.py first to prepare the data")
        return
    
    # Initialize the models
    decoder = LSTMDecoderWithAttention(attention_dim, emb_dim, decoder_dim, len(word_map), dropout=dropout)
    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=decoder_lr)
    
    encoder = EncoderCNN()
    encoder.fine_tune(fine_tune_encoder)
    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()), lr=encoder_lr) if fine_tune_encoder else None
    
    # Move to GPU if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Normalization for preprocessing images
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Create data loaders
    try:
        # Only use a small subset of data for testing (10%)
        train_loader = DataLoader(
            CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
            batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True
        )
        
        val_loader = DataLoader(
            CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
            batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True
        )
        
        print(f"Data loaders created. Training set: {len(train_loader)} batches, Validation set: {len(val_loader)} batches")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please run create_data_n_prep.py first to prepare the data")
        return
    
    # Train for one epoch
    print("Starting training for one epoch...")
    train_one_epoch(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer)
    
    # Validate
    print("Validating model...")
    bleu_score = validate(val_loader, encoder, decoder, criterion, word_map)
    
    print(f"Test completed. BLEU-4 score: {bleu_score:.4f}")
    
    # Save test checkpoint
    save_checkpoint(
        data_name + '_test',
        0,  # epoch
        0,  # epochs since improvement
        encoder,
        decoder,
        encoder_optimizer,
        decoder_optimizer,
        bleu_score,
        is_best=True
    )
    
    print(f"Test model saved as checkpoint_{data_name}_test.pth.tar")

def train_one_epoch(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer):
    """
    Train the model for one epoch
    """
    decoder.train()
    encoder.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()
    
    start = time.time()
    
    # Only process a small number of batches for testing
    max_batches = min(10, len(train_loader))
    
    # Batches
    for i, (images, captions, caplens) in enumerate(train_loader):
        if i >= max_batches:
            break
            
        data_time.update(time.time() - start)
        
        # Move to GPU if available
        images = images.to(device)
        captions = captions.to(device)
        caplens = caplens.to(device)
        
        # Forward prop
        try:
            encoded_images = encoder(images)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(encoded_images, captions, caplens)
            
            # Create target captions for teacher forcing
            targets = caps_sorted[:, 1:]
            
            # Remove timesteps that we don't want to predict
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
            
            # Calculate loss
            loss = criterion(scores, targets)
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
            
            # Backprop
            decoder_optimizer.zero_grad()
            if encoder_optimizer is not None:
                encoder_optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            if grad_clip is not None:
                clip_gradient(decoder_optimizer, grad_clip)
                if encoder_optimizer is not None:
                    clip_gradient(encoder_optimizer, grad_clip)
            
            # Update weights
            decoder_optimizer.step()
            if encoder_optimizer is not None:
                encoder_optimizer.step()
                
            # Keep track of metrics
            top5 = accuracy(scores, targets, 5)
            losses.update(loss.item(), sum(decode_lengths))
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)
            start = time.time()
            
            # Print status
            if i % print_freq == 0:
                print(f'Batch: [{i}/{max_batches}]\t'
                      f'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Top-5 Accuracy {top5accs.val:.3f} ({top5accs.avg:.3f})')
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            continue

def validate(val_loader, encoder, decoder, criterion, word_map):
    """
    Performs validation on a small subset
    """
    decoder.eval()
    encoder.eval()
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()
    
    start = time.time()
    
    references = list()  # true captions
    hypotheses = list()  # predicted captions
    
    # Only process a small number of batches for testing
    max_batches = min(5, len(val_loader))
    
    with torch.no_grad():
        # Batches
        for i, (images, captions, caplens, allcaps) in enumerate(val_loader):
            if i >= max_batches:
                break
                
            # Move to GPU if available
            images = images.to(device)
            captions = captions.to(device)
            caplens = caplens.to(device)
            allcaps = allcaps.to(device)  # Make sure allcaps is also on the same device
            
            # Forward prop
            try:
                encoded_images = encoder(images)
                scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(encoded_images, captions, caplens)
                
                # Targets
                targets = caps_sorted[:, 1:]
                
                # Remove timesteps that we don't want to predict
                scores_copy = scores.clone()
                scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
                targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
                
                # Calculate loss
                loss = criterion(scores, targets)
                loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
                
                # Keep track of metrics
                losses.update(loss.item(), sum(decode_lengths))
                top5 = accuracy(scores, targets, 5)
                top5accs.update(top5, sum(decode_lengths))
                batch_time.update(time.time() - start)
                start = time.time()
                
                # Store references (true captions) and hypotheses (predictions)
                # Reference captions
                allcaps = allcaps[sort_ind]  # because images were sorted
                
                # Move sort_ind to the same device as allcaps
                sort_ind = sort_ind.to(allcaps.device)
                
                for j in range(allcaps.shape[0]):
                    img_caps = allcaps[j].tolist()
                    img_captions = list(
                        map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}], img_caps))
                    references.append(img_captions)
                
                # Hypotheses
                _, preds = torch.max(scores_copy, dim=2)
                preds = preds.tolist()
                temp_preds = list()
                for j, p in enumerate(preds):
                    temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
                preds = temp_preds
                hypotheses.extend(preds)
                
                if i % print_freq == 0:
                    print(f'Validation: [{i}/{max_batches}]\t'
                          f'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                          f'Top-5 Accuracy {top5accs.val:.3f} ({top5accs.avg:.3f})')
            except Exception as e:
                print(f"Error in validation batch {i}: {e}")
                continue
    
    # Calculate BLEU score
    print("Calculating BLEU score...")
    try:
        if references and hypotheses:
            bleu4 = corpus_bleu(references, hypotheses)
            print(f'BLEU-4 score: {bleu4:.4f}')
        else:
            print("No valid references or hypotheses for BLEU calculation.")
            bleu4 = 0
    except Exception as e:
        print(f"Error calculating BLEU score: {e}")
        bleu4 = 0
    
    return bleu4


if __name__ == '__main__':
    # Import necessary packages for packing sequences
    from torch.nn.utils.rnn import pack_padded_sequence
    
    # Test if NLTK is properly installed
    try:
        from nltk.translate.bleu_score import corpus_bleu
    except ImportError:
        print("NLTK not found. Installing...")
        import nltk
        nltk.download('punkt')
    
    # Create missing directories
    os.makedirs(data_folder, exist_ok=True)
    
    # Define global word_map for validation
    global word_map
    word_map = None
    
    # Run the test
    test_model() 