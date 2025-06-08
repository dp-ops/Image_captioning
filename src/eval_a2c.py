"""
A2C Model Evaluation Script

This script evaluates the trained A2C image captioning model on the test set.
Unlike standard evaluation which uses beam search, A2C evaluation uses the same
sampling strategy as training for consistency with the reinforcement learning approach.

Runs like this: python src/eval_a2c.py --num_samples 2 --max_length 15
"""

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import json
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import time

# Import modules handling both directory contexts
import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

# Fix the A2C import issue by handling the src module reference
try:
    # When running from src directory, we need to handle the internal imports
    import model as model_module
    sys.modules['src.model'] = model_module
    sys.modules['model'] = model_module
except ImportError:
    pass

try:
    # Try importing the required modules
    from dataset import CaptionDataset
    from utils import AverageMeter
    from A2C import A2CImageCaptioning, HybridCriticNetwork
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying alternative import paths...")
    try:
        from src.dataset import CaptionDataset
        from src.utils import AverageMeter  
        from src.A2C import A2CImageCaptioning, HybridCriticNetwork
    except ImportError as e2:
        print(f"Alternative import also failed: {e2}")
        sys.exit(1)

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

def load_a2c_model(checkpoint_path, device, data_name='flickr8k_5_5'):
    """
    Load A2C model from checkpoint
    :param checkpoint_path: path to the A2C checkpoint file
    :param device: device to load the model on
    :param data_name: dataset name for loading word map
    :return: loaded A2C model
    """
    print(f"Loading A2C model from: {checkpoint_path}")
    
    # Handle import compatibility issues
    if 'model' not in sys.modules:
        try:
            import src.model as model_module
            sys.modules['model'] = model_module
        except ImportError:
            print("Warning: Could not set up model module compatibility")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract word map from the data folder
    data_folder = 'data_output'
    word_map_file = os.path.join(data_folder, f'WORDMAP_{data_name}.json')
    
    if not os.path.exists(word_map_file):
        raise FileNotFoundError(f"Word map file not found: {word_map_file}")
    
    with open(word_map_file, 'r') as f:
        word_map = json.load(f)
    
    # Create A2C model instance
    a2c_model = A2CImageCaptioning(
        word_map=word_map,
        device=device,
        checkpoint=checkpoint_path,
        fine_tune_encoder=False  # Set to False for evaluation
    )
    
    # Set models to evaluation mode
    a2c_model.encoder.eval()
    a2c_model.decoder.eval()
    a2c_model.critic.eval()
    
    return a2c_model, word_map

def evaluate_a2c_on_test(a2c_model, word_map, data_folder, data_name, device, 
                        num_samples=5, temperature=1.0, max_length=20,
                        save_sample_captions=True):
    """
    Evaluate A2C model on test set using sampling
    :param a2c_model: loaded A2C model
    :param word_map: word to index mapping
    :param data_folder: folder containing data files
    :param data_name: base name of data files
    :param device: device to run evaluation on
    :param num_samples: number of caption samples per image
    :param temperature: sampling temperature
    :param max_length: maximum caption length
    :param save_sample_captions: whether to save sample captions
    :return: BLEU scores and sample results
    """
    
    # Set temperature for sampling
    a2c_model.temperature = temperature
    
    # Create reverse word map
    rev_word_map = {v: k for k, v in word_map.items()}
    
    # Data loader for test set
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_loader = DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    )
    
    # Metrics
    batch_time = AverageMeter()
    
    # Storage for BLEU calculation
    references = []
    hypotheses = []
    
    # Storage for sample results
    sample_results = []
    
    print(f"Evaluating A2C model on test set...")
    print(f"Number of samples per image: {num_samples}")
    print(f"Sampling temperature: {temperature}")
    print(f"Maximum caption length: {max_length}")
    
    start_time = time.time()
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="A2C Test Evaluation")):
            batch_start = time.time()
            
            # Handle batch format
            if len(batch) == 4:
                image, caption, caplen, allcaps = batch
            else:
                image, caption, caplen = batch
                allcaps = None
            
            # Move to device
            image = image.to(device)
            if allcaps is not None:
                allcaps = allcaps.to(device)
            
            # Get image features
            encoder_out = a2c_model.encoder(image)
            
            # Generate multiple caption samples for this image
            all_samples = []
            all_scores = []
            
            for sample_idx in range(num_samples):
                # Generate caption using sampling
                samples, log_probs, entropies, _ = a2c_model.generate_caption_with_sampling(
                    encoder_out, max_length=max_length
                )
                
                # Convert to words
                sample_tokens = samples[0].tolist()  # First (and only) item in batch
                
                # Find end token or use full length
                try:
                    end_idx = sample_tokens.index(word_map['<end>'])
                    sample_tokens = sample_tokens[:end_idx]
                except ValueError:
                    pass  # No end token found, use full sequence
                
                # Convert to words (excluding special tokens)
                sample_words = []
                for token in sample_tokens:
                    if token in rev_word_map and token not in [word_map['<start>'], word_map['<pad>']]:
                        sample_words.append(rev_word_map[token])
                
                all_samples.append(sample_words)
                
                # Calculate sample score (sum of log probabilities)
                valid_length = min(len(sample_tokens), log_probs.size(1))
                sample_score = log_probs[0, :valid_length].sum().item()
                all_scores.append(sample_score)
            
            # Select best sample based on log probability
            best_idx = np.argmax(all_scores)
            best_sample = all_samples[best_idx]
            
            # Add to hypotheses
            hypotheses.append(best_sample)
            
            # Get reference captions
            if allcaps is not None:
                # Multiple reference captions available
                img_captions = []
                for j in range(allcaps.size(1)):
                    ref_tokens = allcaps[0, j].tolist()
                    ref_words = [rev_word_map[token] for token in ref_tokens 
                               if token in rev_word_map and token not in 
                               [word_map['<start>'], word_map['<end>'], word_map['<pad>']] and token != 0]
                    if ref_words:  # Only add non-empty captions
                        img_captions.append(ref_words)
                references.append(img_captions)
            else:
                # Single reference caption
                ref_tokens = caption[0, :caplen[0]].tolist()
                ref_words = [rev_word_map[token] for token in ref_tokens 
                           if token in rev_word_map and token not in 
                           [word_map['<start>'], word_map['<end>'], word_map['<pad>']]]
                references.append([ref_words])
            
            # Store sample result for potential visualization
            if save_sample_captions and i < 10:  # Save first 10 samples
                sample_results.append({
                    'image_index': i,
                    'generated_captions': all_samples,
                    'caption_scores': all_scores,
                    'best_caption': best_sample,
                    'reference_captions': references[-1],
                    'best_caption_text': ' '.join(best_sample),
                    'reference_text': ' | '.join([' '.join(ref) for ref in references[-1]])
                })
            
            # Update timing
            batch_time.update(time.time() - batch_start)
            
            # Print progress every 100 images
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(test_loader)} images. "
                      f"Time: {batch_time.avg:.3f}s/image")
    
    # Calculate BLEU scores
    print("\nCalculating BLEU scores...")
    
    # BLEU-4 (primary metric)
    bleu4 = corpus_bleu(references, hypotheses)
    
    # Calculate individual BLEU scores for more detailed analysis
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    
    # Calculate sentence-level BLEU scores for statistics
    sentence_bleu_scores = []
    smoothing = SmoothingFunction().method1
    
    for ref, hyp in zip(references, hypotheses):
        if ref and hyp:  # Only calculate if both reference and hypothesis exist
            score = sentence_bleu(ref, hyp, smoothing_function=smoothing)
            sentence_bleu_scores.append(score)
    
    avg_sentence_bleu = np.mean(sentence_bleu_scores) if sentence_bleu_scores else 0.0
    std_sentence_bleu = np.std(sentence_bleu_scores) if sentence_bleu_scores else 0.0
    
    # Evaluation results
    results = {
        'bleu1': bleu1,
        'bleu2': bleu2,
        'bleu3': bleu3,
        'bleu4': bleu4,
        'avg_sentence_bleu': avg_sentence_bleu,
        'std_sentence_bleu': std_sentence_bleu,
        'num_images': len(test_loader),
        'num_samples_per_image': num_samples,
        'temperature': temperature,
        'total_time': time.time() - start_time,
        'avg_time_per_image': batch_time.avg
    }
    
    return results, sample_results

def save_evaluation_results(results, sample_results, data_name, output_dir='evaluation_outputs'):
    """
    Save evaluation results to files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save numerical results
    results_file = os.path.join(output_dir, f'a2c_test_results_{data_name}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save sample results
    samples_file = os.path.join(output_dir, f'a2c_sample_captions_{data_name}.json')
    with open(samples_file, 'w') as f:
        json.dump(sample_results, f, indent=2)
    
    # Print results summary
    print("\n" + "="*80)
    print("A2C MODEL EVALUATION RESULTS")
    print("="*80)
    print(f"Dataset: {data_name}")
    print(f"Number of test images: {results['num_images']}")
    print(f"Samples per image: {results['num_samples_per_image']}")
    print(f"Sampling temperature: {results['temperature']}")
    print("-"*80)
    print("BLEU SCORES:")
    print(f"  BLEU-1: {results['bleu1']:.4f}")
    print(f"  BLEU-2: {results['bleu2']:.4f}")
    print(f"  BLEU-3: {results['bleu3']:.4f}")
    print(f"  BLEU-4: {results['bleu4']:.4f}")
    print("-"*80)
    print("SENTENCE-LEVEL STATISTICS:")
    print(f"  Average sentence BLEU: {results['avg_sentence_bleu']:.4f}")
    print(f"  Std deviation: {results['std_sentence_bleu']:.4f}")
    print("-"*80)
    print("TIMING:")
    print(f"  Total evaluation time: {results['total_time']:.2f} seconds")
    print(f"  Average time per image: {results['avg_time_per_image']:.3f} seconds")
    print("="*80)
    
    # Print some sample captions
    if sample_results:
        print("\nSAMPLE GENERATED CAPTIONS:")
        print("-"*80)
        for i, sample in enumerate(sample_results[:5]):  # Show first 5 samples
            print(f"Image {sample['image_index'] + 1}:")
            print(f"  Generated: {sample['best_caption_text']}")
            print(f"  Reference: {sample['reference_text']}")
            print()
    
    print(f"Detailed results saved to: {results_file}")
    print(f"Sample captions saved to: {samples_file}")

def create_evaluation_plots(results, data_name, output_dir='evaluation_outputs'):
    """
    Create visualization plots for evaluation results
    """
    # BLEU scores comparison
    bleu_scores = [results['bleu1'], results['bleu2'], results['bleu3'], results['bleu4']]
    bleu_labels = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(bleu_labels, bleu_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.title(f'A2C Model BLEU Scores on {data_name} Test Set')
    plt.ylabel('BLEU Score')
    plt.ylim(0, max(bleu_scores) * 1.1)
    
    # Add value labels on bars
    for bar, score in zip(bars, bleu_scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'a2c_bleu_scores_{data_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"BLEU scores plot saved to: {os.path.join(output_dir, f'a2c_bleu_scores_{data_name}.png')}")

def main():
    """
    Main evaluation function
    """
    parser = argparse.ArgumentParser(description='A2C Image Captioning Model Evaluation')
    parser.add_argument('--data_folder', default='data_output', help='folder with data files')
    parser.add_argument('--data_name', default='flickr8k_5_5', help='base name shared by data files')
    parser.add_argument('--checkpoint', default='model_outputs/a2c_BEST_flickr8k_5_5.pth.tar', 
                       help='path to A2C model checkpoint')
    parser.add_argument('--num_samples', default=5, type=int, help='number of samples per image')
    parser.add_argument('--temperature', default=1.0, type=float, help='sampling temperature')
    parser.add_argument('--max_length', default=20, type=int, help='maximum caption length')
    parser.add_argument('--output_dir', default='evaluation_outputs', help='directory to save results')
    parser.add_argument('--create_plots', action='store_true', help='create visualization plots')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    
    print(f"Using device: {device}")
    print(f"Evaluating checkpoint: {args.checkpoint}")
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        print("Available A2C checkpoints in model_outputs:")
        model_dir = 'model_outputs'
        if os.path.exists(model_dir):
            a2c_files = [f for f in os.listdir(model_dir) if f.startswith('a2c_') and f.endswith('.pth.tar')]
            for f in a2c_files:
                print(f"  {os.path.join(model_dir, f)}")
        return
    
    try:
        # Load A2C model
        a2c_model, word_map = load_a2c_model(args.checkpoint, device, args.data_name)
        
        # Evaluate on test set
        results, sample_results = evaluate_a2c_on_test(
            a2c_model=a2c_model,
            word_map=word_map,
            data_folder=args.data_folder,
            data_name=args.data_name,
            device=device,
            num_samples=args.num_samples,
            temperature=args.temperature,
            max_length=args.max_length,
            save_sample_captions=True
        )
        
        # Save results
        save_evaluation_results(results, sample_results, args.data_name, args.output_dir)
        
        # Create plots if requested
        if args.create_plots:
            create_evaluation_plots(results, args.data_name, args.output_dir)
        
        print(f"\nEvaluation completed successfully!")
        print(f"Main result: BLEU-4 = {results['bleu4']:.4f}")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 