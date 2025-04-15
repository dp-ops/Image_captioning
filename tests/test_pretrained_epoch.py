"""
Script to run one epoch of training with pretrained ResNet34 weights.
This will help verify that the pretrained weights are properly used during training.
"""

import os
import subprocess
import sys

def run_one_epoch_with_pretrained():
    """
    Run one epoch of training with pretrained weights
    """
    print("=" * 60)
    print("TESTING ONE EPOCH TRAINING WITH PRETRAINED RESNET34 WEIGHTS")
    print("=" * 60)
    
    # Verify that pretrained weights are loaded correctly
    print("\nStep 1: Verifying pretrained weights are loaded correctly...")
    try:
        subprocess.run(["python", "test_pretrained.py"], check=True)
    except subprocess.CalledProcessError:
        print("❌ Pretrained weights test failed. Please check the error above.")
        return
    
    # Run one epoch of training with pretrained weights
    print("\nStep 2: Running one epoch of training with pretrained weights...")
    print("This will take some time depending on your dataset size and hardware.")
    
    try:
        # Run the training script with pretrained weights for one epoch
        command = ["python", "train.py", "--pretrained", "--epochs", "1"]
        
        print(f"Executing command: {' '.join(command)}")
        print("\nTraining output:")
        print("-" * 60)
        
        # Run the training process
        result = subprocess.run(command, check=True)
        
        print("-" * 60)
        if result.returncode == 0:
            print("\n✅ Successfully completed one epoch of training with pretrained weights!")
            print("\nTo continue training with more epochs, run:")
            print("python train.py --pretrained --epochs <number_of_epochs>")
            
            print("\nTo use pretrained weights and fine-tune the encoder, run:")
            print("python train.py --pretrained --fine_tune_encoder")
        else:
            print("\n❌ Training failed. Please check the error above.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error code {e.returncode}")
        print("Please check the error output above.")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    run_one_epoch_with_pretrained() 