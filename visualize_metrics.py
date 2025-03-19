"""
Visualize training metrics for the image captioning model.

This script loads and displays the training metrics (loss, accuracy, BLEU scores)
saved during training.
"""

import os
import json
import matplotlib.pyplot as plt
import argparse

def visualize_metrics(data_name='flickr8k_5_5'):
    """
    Load and visualize training metrics
    
    :param data_name: name of the dataset
    """
    print("=" * 60)
    print("VISUALIZING TRAINING METRICS".center(60))
    print("=" * 60)
    
    # Set up directories and file paths
    output_dir = 'model_outputs'
    metrics_file = os.path.join(output_dir, f'training_metrics_{data_name}.json')
    bleu_scores_file = os.path.join(output_dir, f'bleu_scores_{data_name}.json')
    
    # Check if metrics file exists
    if not os.path.exists(metrics_file):
        print(f"Error: Training metrics file not found at {metrics_file}")
        print("Please train the model first to generate metrics.")
        return False
    
    # Load metrics data
    with open(metrics_file, 'r') as f:
        metrics_data = json.load(f)
    
    epochs = metrics_data['epochs']
    train_loss = metrics_data['train_loss']
    train_acc = metrics_data['train_top5_accuracy']
    val_loss = metrics_data['val_loss']
    val_acc = metrics_data['val_top5_accuracy']
    
    # Set up figure for better visualization
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_loss, 'b-o', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-x', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_acc, 'b-o', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-x', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Top-5 Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Print summary statistics
    print("\nTraining Summary:")
    print(f"Total epochs: {len(epochs)}")
    print(f"Initial training loss: {train_loss[0]:.4f}")
    print(f"Final training loss: {train_loss[-1]:.4f}")
    print(f"Best training accuracy: {max(train_acc):.2f}%")
    print(f"Final training accuracy: {train_acc[-1]:.2f}%")
    print("\nValidation Summary:")
    print(f"Initial validation loss: {val_loss[0]:.4f}")
    print(f"Final validation loss: {val_loss[-1]:.4f}")
    print(f"Best validation accuracy: {max(val_acc):.2f}%")
    print(f"Final validation accuracy: {val_acc[-1]:.2f}%")
    
    # Check if BLEU scores file exists
    if os.path.exists(bleu_scores_file):
        with open(bleu_scores_file, 'r') as f:
            bleu_data = json.load(f)
        
        bleu_epochs = bleu_data['epochs']
        bleu_scores = bleu_data['bleu_scores']
        
        # Plot BLEU scores
        plt.subplot(2, 2, 3)
        plt.plot(bleu_epochs, bleu_scores, 'g-o')
        plt.xlabel('Epoch')
        plt.ylabel('BLEU-4 Score')
        plt.title('BLEU-4 Scores')
        plt.grid(True)
        
        # Print BLEU summary
        print("\nBLEU Score Summary:")
        print(f"Initial BLEU-4 score: {bleu_scores[0]:.4f}")
        print(f"Final BLEU-4 score: {bleu_scores[-1]:.4f}")
        print(f"Best BLEU-4 score: {max(bleu_scores):.4f}")
        
        # Find the epoch with the best BLEU score
        best_epoch = bleu_epochs[bleu_scores.index(max(bleu_scores))]
        print(f"Best model at epoch: {best_epoch}")
    
    # Plot training progress
    if len(epochs) > 1:
        plt.subplot(2, 2, 4)
        
        # Normalize values for better visualization
        norm_train_loss = [x/max(train_loss) for x in train_loss]
        norm_val_loss = [x/max(val_loss) for x in val_loss]
        norm_train_acc = [x/100 for x in train_acc]
        norm_val_acc = [x/100 for x in val_acc]
        
        plt.plot(epochs, norm_train_loss, 'b-', label='Train Loss (norm)')
        plt.plot(epochs, norm_val_loss, 'r-', label='Val Loss (norm)')
        plt.plot(epochs, norm_train_acc, 'b--', label='Train Acc (norm)')
        plt.plot(epochs, norm_val_acc, 'r--', label='Val Acc (norm)')
        
        if 'bleu_scores' in locals():
            norm_bleu = [x/max(bleu_scores) for x in bleu_scores]
            plt.plot(bleu_epochs, norm_bleu, 'g-', label='BLEU (norm)')
        
        plt.xlabel('Epoch')
        plt.ylabel('Normalized Value')
        plt.title('Overall Training Progress')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    
    # Save the visualization
    viz_path = os.path.join(output_dir, f'training_visualization_{data_name}.png')
    plt.savefig(viz_path)
    print(f"\nVisualization saved to: {viz_path}")
    
    # Show the plot
    plt.show()
    
    print("\n" + "=" * 60)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize training metrics')
    parser.add_argument('--data_name', default='flickr8k_5_5', help='dataset name')
    args = parser.parse_args()
    
    visualize_metrics(args.data_name) 