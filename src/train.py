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
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
import argparse
import os

# Set up argument parser for command line options
parser = argparse.ArgumentParser(description='Train image captioning model')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--fine_tune_encoder', '-f', action='store_true', help='fine tune encoder')
parser.add_argument('--epochs', '-e', default=120, type=int, help='number of epochs')
parser.add_argument('--batch_size', '-b', default=64, type=int, help='batch size')
parser.add_argument('--checkpoint', '-c', default=None, help='checkpoint to resume from')
parser.add_argument('--pretrained', '-p', action='store_true', help='use pretrained ResNet34 weights from torchvision')
parser.add_argument('--one_hot', '-o', action='store_true', help='use one-hot encoding for word embeddings')
args = parser.parse_args()

data_folder = 'data_output'
data_name = 'flickr8k_5_5' #change to flickr8k if using flickr8k dataset

#Model params
emb_dim = 512
attention_dim = 512
decoder_dim = 300 #512
dropout = 0.3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

#Train Params 
start_epoch = 0 
epochs = args.epochs
epochs_since_improvement = 0
batch_size = args.batch_size
workers = 0  # Set to 0 to avoid h5py pickling issues
encoder_lr = 1e-4
decoder_lr = 5.1e-4
grad_clip = 5.0
alpha_c = 1.0
best_bleu4 = 0
print_freq = 100

# Track metrics for plotting
bleu_scores = []
train_losses = []
train_top5accs = []
val_losses = []
val_top5accs = []

fine_tune_encoder = args.fine_tune_encoder
use_one_hot = args.one_hot

checkpoint = args.checkpoint

def main():
    '''Train and val'''

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map, bleu_scores
    global train_losses, train_top5accs, val_losses, val_top5accs, use_one_hot

    #Load word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    #Initialize or load the model 
    if checkpoint is None and not args.resume:
        # Set the encoder dimension (512 for ResNet34)
        encoder_dim = 512
        decoder = LSTMDecoderWithAttention(attention_dim, emb_dim, decoder_dim, len(word_map), 
                                          encoder_dim, dropout, num_layers=3, use_one_hot=use_one_hot)
        decoder_optimizer = torch.optim.Adam(params = filter(lambda p: p.requires_grad, decoder.parameters()), lr = decoder_lr)

        # Initialize encoder with or without pretrained weights
        pretrained_path = "torchvision" if args.pretrained else None
        encoder = EncoderCNN(pretrained_path=pretrained_path)
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params = filter(lambda p: p.requires_grad, encoder.parameters()), lr = encoder_lr) if fine_tune_encoder else None

    else: 
        # Load from checkpoint
        if args.resume and checkpoint is None:
            # Try to load the latest checkpoint
            try:
                checkpoint = os.path.join('model_outputs', f'checkpoint_{data_name}.pth.tar')
                print(f"Resuming from checkpoint {checkpoint}")
            except:
                print("No checkpoint found. Starting from scratch.")
                checkpoint = None
                
        if checkpoint is not None:
            checkpoint = torch.load(checkpoint)
            start_epoch = checkpoint['epoch'] + 1
            epochs_since_improvement = checkpoint['epochs_since_improvement']
            best_bleu4 = checkpoint['bleu-4']
            decoder = checkpoint['decoder']
            decoder_optimizer = checkpoint['decoder_optimizer']
            encoder = checkpoint['encoder']
            encoder_optimizer = checkpoint['encoder_optimizer']
            
            # Check if the checkpoint decoder has the use_one_hot attribute
            # If not, set it to False for backward compatibility
            if not hasattr(decoder, 'use_one_hot'):
                decoder.use_one_hot = False
                print("Checkpoint decoder doesn't have one-hot encoding setting. Setting to False.")
            else:
                # If the checkpoint has use_one_hot attribute, use that instead of command-line arg
                use_one_hot = decoder.use_one_hot
                print(f"Using one-hot encoding setting from checkpoint: {use_one_hot}")
            
            # Try to load BLEU scores if available
            bleu_scores_file = os.path.join('model_outputs', f'bleu_scores_{data_name}.json')
            if os.path.exists(bleu_scores_file):
                with open(bleu_scores_file, 'r') as f:
                    data = json.load(f)
                    bleu_scores = data['bleu_scores']
                    print(f"Loaded BLEU scores from previous training: {bleu_scores}")
            
            # Try to load training metrics if available
            metrics_file = os.path.join('model_outputs', f'training_metrics_{data_name}.json')
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    train_losses = data.get('train_loss', [])
                    train_top5accs = data.get('train_top5_accuracy', [])
                    val_losses = data.get('val_loss', [])
                    val_top5accs = data.get('val_top5_accuracy', [])
                    print(f"Loaded training metrics from previous training")
                    print(f"  - Train losses: {len(train_losses)} epochs")
                    print(f"  - Train accuracies: {len(train_top5accs)} epochs")
                    print(f"  - Validation losses: {len(val_losses)} epochs")
                    print(f"  - Validation accuracies: {len(val_top5accs)} epochs")

            print(f"Loaded checkpoint from epoch {start_epoch-1} with BLEU-4 score {best_bleu4:.4f}")

        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params = filter(lambda p: p.requires_grad, encoder.parameters()), lr = encoder_lr)
            print("Enabled fine-tuning of encoder.")

    #Move to GPU if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    #Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_loader = DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform = transforms.Compose([normalize])),
        batch_size = batch_size, shuffle = True, num_workers = workers, pin_memory = True
    )

    val_loader = DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform = transforms.Compose([normalize])),
        batch_size = batch_size, shuffle = True, num_workers = workers, pin_memory = True
    )
    
    print(f"Training model on {device}.")
    print(f"Training for {epochs} epochs from epoch {start_epoch}.")
    print(f"Fine-tuning encoder: {fine_tune_encoder}")
    print(f"Using one-hot encoding: {use_one_hot}")

    for epoch in range(start_epoch, epochs):
        if epochs_since_improvement == 20:
            print("No improvement for 20 epochs. Stopping training.")
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 5 == 0:
            adjust_learning_rate(decoder_optimizer, 0.7)
            #if encoder is trained, adjust learning rate for encoder
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)
        
        #One epoch's training
        train_loss, train_top5 = train(
            train_loader = train_loader,
            encoder = encoder,
            decoder = decoder,
            criterion = criterion,
            encoder_optimizer = encoder_optimizer,
            decoder_optimizer = decoder_optimizer,
            epoch = epoch,
        )
        
        # Store training metrics
        train_losses.append(train_loss)
        train_top5accs.append(train_top5)

        #one epochs validation
        val_loss, val_top5, recent_bleu4 = validate(
            val_loader = val_loader,
            encoder = encoder,
            decoder = decoder,
            criterion = criterion,
        )
        
        # Store validation metrics
        val_losses.append(val_loss)
        val_top5accs.append(val_top5)

        # Save bleu score
        bleu_scores.append(recent_bleu4)
        save_bleu_scores(data_name, bleu_scores, epoch)
        
        # Save training metrics
        save_training_metrics(data_name, train_losses, train_top5accs, val_losses, val_top5accs, epoch)

        #look for improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)

        if not is_best:
            epochs_since_improvement += 1
            print(f"\nEpochs since last improvement: {epochs_since_improvement}")
        else:
            epochs_since_improvement = 0
            print(f"\nNew best BLEU-4 score: {best_bleu4:.4f}")

        #Save checkpoint
        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best)
        
def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    '''
    Performs one epoch of training

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number 
    :return: average loss and top5 accuracy for this epoch
    '''

    decoder.train()
    encoder.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    #Batches
    for i, (images, captions, caplens) in tqdm(enumerate(train_loader)):
        data_time.update(time.time() - start)

        #Move to GPU if available
        images = images.to(device)
        captions = captions.to(device)
        caplens = caplens.to(device)

        #Forward prop
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(encoder(images), captions, caplens)

        #Create target captions for teacher forcing
        targetd = caps_sorted[:, 1:]

        #Remove timesteps that we don't want to predict (last target timestep)
        scores = pack_padded_sequence(scores, decode_lengths, batch_first = True)
        scores = scores.data
        targets = pack_padded_sequence(targetd, decode_lengths, batch_first = True)
        targets = targets.data

        #Compute loss
        loss = criterion(scores, targets)
        loss += alpha_c * ((1 - alphas.sum(dim = 1)) ** 2).mean()

        #Backprop
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        #Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)
        
        #Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        #keep metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))

        #keep track of metrics
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)
        start = time.time()

        #Print info
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                  data_time=data_time, loss=losses,
                                                                          top5=top5accs))
    
    # Return average loss and accuracy for the epoch
    return losses.avg, top5accs.avg

def validate(val_loader, encoder, decoder, criterion):
    '''
    Performs one epoch of validation

    :param val_loader: DataLoader for validation data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: average loss, top5 accuracy, and BLEU-4 score
    '''

    decoder.eval()
    if encoder is not None:
        encoder.eval()
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list() #true captions for calculating bleu score
    hypotheses = list() #predicted captions for calculating bleu score

    with torch.no_grad():
        #Batches
        for i, (images, captions, caplens, allcaps) in tqdm(enumerate(val_loader)):

            images = images.to(device)
            captions = captions.to(device)
            caplens = caplens.to(device)
            allcaps = allcaps.to(device)  # Move allcaps to the same device

            #Forward prop
            if encoder is not None:
                images = encoder(images)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(images, captions, caplens)
            
            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            #Remove timesteps that we don't want to predict (last target timestep)
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first = True)
            scores = scores.data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first = True)
            targets = targets.data

            #calc loss
            loss = criterion(scores, targets)
            loss += alpha_c * ((1 - alphas.sum(dim = 1)) ** 2).mean()

            #keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)
            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Ensure sort_ind is on the same device as allcaps
            sort_ind = sort_ind.to(allcaps.device)
            
            #Reference captions
            allcaps = allcaps[sort_ind] # because images were sorted in the decoder 
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}], img_caps)) #remove <start> and <pad>
                references.append(img_captions)

            #Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

    return losses.avg, top5accs.avg, bleu4


if __name__ == '__main__':
    main()

