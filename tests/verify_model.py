"""
Comprehensive verification script for the image captioning model.

This script verifies that:
1. Pretrained ResNet34 weights are loaded correctly
2. Training works with pretrained weights
3. BLEU scores are tracked and saved
4. Model outputs are saved to the correct directory
"""

import os
import sys
import torch
import shutil
import json
import matplotlib.pyplot as plt
from src.model import EncoderCNN

def verify_model():
    """Run a comprehensive verification of the image captioning model"""
    print("=" * 80)
    print("COMPREHENSIVE IMAGE CAPTIONING MODEL VERIFICATION".center(80))
    print("=" * 80)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check model_outputs directory
    verify_model_outputs_directory()
    
    # Verify pretrained weights loading
    verify_pretrained_weights()
    
    # Optional: Run a test training epoch if requested
    if '--run-training' in sys.argv:
        run_test_training()
    else:
        print("\nSkipping test training. Add --run-training flag to execute training test.")
    
    # Check if there are BLEU scores saved
    check_bleu_scores()
    
    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE".center(80))
    print("=" * 80)
    
def verify_model_outputs_directory():
    """Verify that the model_outputs directory exists and is writable"""
    print("\n[1] VERIFYING MODEL_OUTPUTS DIRECTORY")
    print("-" * 50)
    
    output_dir = 'model_outputs'
    
    # Check if directory exists
    if not os.path.exists(output_dir):
        print(f"Creating directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    else:
        print(f"✅ Directory exists: {output_dir}")
    
    # Test writing to directory
    test_file = os.path.join(output_dir, 'test_verification.txt')
    try:
        with open(test_file, 'w') as f:
            f.write("Verification test")
        print(f"✅ Successfully wrote test file: {test_file}")
        os.remove(test_file)  # Clean up
    except Exception as e:
        print(f"❌ Error writing to directory: {e}")
        return False
    
    print("✅ Model outputs directory verification passed")
    return True

def verify_pretrained_weights():
    """Verify that pretrained weights are loaded correctly"""
    print("\n[2] VERIFYING PRETRAINED WEIGHTS LOADING")
    print("-" * 50)
    
    # Create two encoders - one with pretrained weights, one without
    print("Creating encoder with pretrained weights...")
    try:
        encoder_pretrained = EncoderCNN(pretrained_path="torchvision")
        print("Creating encoder with random weights...")
        encoder_random = EncoderCNN(pretrained_path=None)
        
        # Move to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder_pretrained = encoder_pretrained.to(device)
        encoder_random = encoder_random.to(device)
        
        # Compare weights
        print("Comparing weights between models...")
        param_count = 0
        diff_count = 0
        diff_total = 0.0
        
        for (name_p, param_p), (name_r, param_r) in zip(encoder_pretrained.resnet.named_parameters(),
                                                      encoder_random.resnet.named_parameters()):
            if 'bn' not in name_p:  # Skip batch norm layers
                param_count += 1
                diff = (param_p.data - param_r.data).abs().mean().item()
                diff_total += diff
                
                if diff > 0.01:
                    diff_count += 1
        
        avg_diff = diff_total / param_count if param_count > 0 else 0
        diff_percentage = (diff_count / param_count) * 100 if param_count > 0 else 0
        
        print(f"Total layers compared: {param_count}")
        print(f"Layers with significant differences: {diff_count} ({diff_percentage:.1f}%)")
        print(f"Average difference across all layers: {avg_diff:.4f}")
        
        if diff_percentage > 50:
            print("✅ Pretrained weights verification passed")
            return True
        else:
            print("❌ Pretrained weights don't seem to be loaded correctly")
            return False
            
    except Exception as e:
        print(f"❌ Error verifying pretrained weights: {e}")
        return False

def run_test_training():
    """Run a test training epoch"""
    print("\n[3] RUNNING TEST TRAINING")
    print("-" * 50)
    
    import subprocess
    
    try:
        # Run the training script with pretrained weights for one batch only
        command = ["python", "train.py", "--pretrained", "--epochs", "1"]
        
        print(f"Executing command: {' '.join(command)}")
        print("Running test training (this may take a few minutes)...")
        
        # Run the training process
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Test training completed successfully")
            
            # Check if BLEU scores were saved
            if "BLEU scores saved to" in result.stdout:
                print("✅ BLEU scores were saved during training")
            else:
                print("❌ BLEU scores were not saved during training")
                
            return True
        else:
            print(f"❌ Test training failed with error code {result.returncode}")
            print("Error output:")
            print(result.stderr[:500] + "..." if len(result.stderr) > 500 else result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running test training: {e}")
        return False

def check_bleu_scores():
    """Check if BLEU scores are saved and plot them"""
    print("\n[4] CHECKING BLEU SCORES")
    print("-" * 50)
    
    data_name = 'flickr8k_5_5'
    bleu_scores_file = os.path.join('model_outputs', f'bleu_scores_{data_name}.json')
    bleu_plot_file = os.path.join('model_outputs', f'bleu_scores_{data_name}.png')
    
    if os.path.exists(bleu_scores_file):
        print(f"✅ BLEU scores file exists: {bleu_scores_file}")
        
        # Load and display BLEU scores
        try:
            with open(bleu_scores_file, 'r') as f:
                data = json.load(f)
            
            epochs = data.get('epochs', [])
            scores = data.get('bleu_scores', [])
            
            print(f"Found {len(scores)} BLEU score entries")
            if len(scores) > 0:
                print(f"Latest BLEU score: {scores[-1]:.4f}")
                
            # Check if plot exists
            if os.path.exists(bleu_plot_file):
                print(f"✅ BLEU scores plot exists: {bleu_plot_file}")
            else:
                print(f"❌ BLEU scores plot not found: {bleu_plot_file}")
                
            return True
            
        except Exception as e:
            print(f"❌ Error reading BLEU scores: {e}")
            return False
    else:
        print(f"❌ BLEU scores file not found: {bleu_scores_file}")
        print("Run training to generate BLEU scores")
        return False

if __name__ == "__main__":
    verify_model() 