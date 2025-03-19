##Fuctions for train.py

import torch
import os
import json
import matplotlib.pyplot as plt

def save_bleu_scores(data_name, bleu_scores, epoch):
    """
    Save BLEU scores to a file and plot them
    
    :param data_name: name of the dataset
    :param bleu_scores: list of BLEU scores
    :param epoch: current epoch number
    """
    # Use the model_outputs directory in the root folder
    output_dir = 'model_outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save scores to JSON
    scores_file = os.path.join(output_dir, f'bleu_scores_{data_name}.json')
    
    # Save scores with epoch numbers
    scores_data = {
        'epochs': list(range(1, epoch + 2)),  # +2 because epoch is 0-indexed and we want to include the current epoch
        'bleu_scores': bleu_scores
    }
    
    with open(scores_file, 'w') as f:
        json.dump(scores_data, f)
    
    # Plot scores
    plt.figure(figsize=(10, 5))
    plt.plot(scores_data['epochs'], scores_data['bleu_scores'], marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU-4 Score')
    plt.title(f'BLEU-4 Scores for {data_name}')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'bleu_scores_{data_name}.png'))
    plt.close()
    
    print(f"BLEU scores saved to {scores_file}")

def save_training_metrics(data_name, train_losses, train_top5accs, val_losses, val_top5accs, current_epoch):
    """
    Save and plot training metrics (loss and accuracy) to files
    Uses the model_outputs directory.
    """
    output_dir = 'model_outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    metrics_file = os.path.join(output_dir, f'training_metrics_{data_name}.json')
    
    # Make sure all arrays have consistent length
    # +1 because current_epoch is 0-indexed 
    epochs_needed = current_epoch + 1
    
    # Ensure all arrays have the correct length
    if len(train_losses) < epochs_needed:
        print(f"Warning: train_losses has {len(train_losses)} entries, expected {epochs_needed}")
        # Pad with last value if needed
        last_value = train_losses[-1] if train_losses else 0
        train_losses.extend([last_value] * (epochs_needed - len(train_losses)))
    
    if len(train_top5accs) < epochs_needed:
        print(f"Warning: train_top5accs has {len(train_top5accs)} entries, expected {epochs_needed}")
        last_value = train_top5accs[-1] if train_top5accs else 0
        train_top5accs.extend([last_value] * (epochs_needed - len(train_top5accs)))
    
    if len(val_losses) < epochs_needed:
        print(f"Warning: val_losses has {len(val_losses)} entries, expected {epochs_needed}")
        last_value = val_losses[-1] if val_losses else 0
        val_losses.extend([last_value] * (epochs_needed - len(val_losses)))
    
    if len(val_top5accs) < epochs_needed:
        print(f"Warning: val_top5accs has {len(val_top5accs)} entries, expected {epochs_needed}")
        last_value = val_top5accs[-1] if val_top5accs else 0
        val_top5accs.extend([last_value] * (epochs_needed - len(val_top5accs)))
    
    # Create epochs array (1-indexed for readability)
    epochs = list(range(1, current_epoch + 2))
    
    # Save metrics with epoch numbers
    metrics_data = {
        'epochs': epochs,
        'train_loss': train_losses,
        'train_top5_accuracy': train_top5accs,
        'val_loss': val_losses,
        'val_top5_accuracy': val_top5accs
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f)
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, marker='o', label='Training Loss')
    plt.plot(epochs, val_losses, marker='x', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss for {data_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'loss_plot_{data_name}.png'))
    plt.close()
    
    # Plot accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_top5accs, marker='o', label='Training Top-5 Accuracy')
    plt.plot(epochs, val_top5accs, marker='x', label='Validation Top-5 Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Top-5 Accuracy (%)')
    plt.title(f'Training and Validation Accuracy for {data_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'accuracy_plot_{data_name}.png'))
    plt.close()
    
    print(f"Training metrics saved to {metrics_file}")

def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrink learning rate by a specified factor
    """

    print("\nAdjusting learning rate.")

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor

    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum and count of a metric
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels
    """
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()
    return correct_total.item() * (100.0 / batch_size)

def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer, bleu4, is_best):
    """
    Saves model checkpoint
    """
    state = {
        'epoch': epoch,
        'epochs_since_improvement': epochs_since_improvement,
        'bleu-4': bleu4,
        'encoder': encoder,
        'decoder': decoder,
        "encoder_optimizer": encoder_optimizer,
        "decoder_optimizer": decoder_optimizer}
    
    # Use the model_outputs directory in the root folder
    output_dir = 'model_outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.join(output_dir, f'checkpoint_{data_name}.pth.tar')
    torch.save(state, filename)

    
    if is_best:
        best_filename = os.path.join(output_dir, f'BEST_{data_name}.pth.tar')
        torch.save(state, best_filename)
        
    
        
