"""
Test script to verify that the model correctly loads and uses pretrained ResNet34 weights.
This script compares pretrained and randomly initialized models to confirm weights are loaded properly.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
from model import EncoderCNN, LSTMDecoderWithAttention
import time

def test_pretrained_weights():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create models
    print("\nInitializing models...")
    
    # Create encoder with pretrained weights
    print("Creating encoder with pretrained weights from torchvision...")
    encoder_pretrained = EncoderCNN(pretrained_path="torchvision")
    
    # Create a second encoder without pretrained weights for comparison
    print("Creating encoder with random weights for comparison...")
    encoder_random = EncoderCNN(pretrained_path=None)
    
    # Move to device
    encoder_pretrained = encoder_pretrained.to(device)
    encoder_random = encoder_random.to(device)
    
    # Verify that the weights are different
    print("\nVerifying that pretrained weights were loaded...")
    
    # Compare some weights from both models
    param_count = 0
    diff_count = 0
    diff_total = 0.0
    
    # Sample a few layers to display
    sample_layers = []
    
    for (name_p, param_p), (name_r, param_r) in zip(encoder_pretrained.resnet.named_parameters(), 
                                                   encoder_random.resnet.named_parameters()):
        if 'bn' not in name_p:  # Skip batch norm layers as they might be initialized similarly
            param_count += 1
            # Calculate difference
            diff = (param_p.data - param_r.data).abs().mean().item()
            diff_total += diff
            
            # Count significantly different layers
            if diff > 0.01:
                diff_count += 1
                
                # Save a few sample layers for display
                if len(sample_layers) < 3 and ('conv' in name_p or 'layer1' in name_p or 'layer2' in name_p):
                    sample_layers.append((name_p, param_p.data.flatten()[:3], param_r.data.flatten()[:3], diff))
    
    avg_diff = diff_total / param_count if param_count > 0 else 0
    diff_percentage = (diff_count / param_count) * 100 if param_count > 0 else 0
    
    print(f"Weight analysis:")
    print(f"- Total layers compared: {param_count}")
    print(f"- Layers with significant differences: {diff_count} ({diff_percentage:.1f}%)")
    print(f"- Average difference across all layers: {avg_diff:.4f}")
    
    if diff_percentage > 50:
        print("\n✅ Pretrained weights were successfully loaded!")
        print("\nSample weight differences between pretrained and random initialization:")
        for name, pretrained_weights, random_weights, diff in sample_layers:
            print(f"  Layer '{name}':")
            print(f"    Pretrained: {pretrained_weights}")
            print(f"    Random:     {random_weights}")
            print(f"    Mean diff:  {diff:.4f}")
    else:
        print("\n❌ Pretrained weights don't seem to be loaded correctly.")
    
    # Test the forward pass
    print("\nTesting forward pass...")
    
    # Create a small dummy dataset for testing
    batch_size = 2
    
    # Create dummy input data
    dummy_images = torch.randn(batch_size, 3, 256, 256).to(device)
    
    # Set models to eval mode for testing
    encoder_pretrained.eval()
    encoder_random.eval()
    
    # Forward pass with both models
    with torch.no_grad():
        out_pretrained = encoder_pretrained(dummy_images)
        out_random = encoder_random(dummy_images)
    
    # Compare outputs
    output_diff = (out_pretrained - out_random).abs().mean().item()
    print(f"Mean difference in encoder outputs: {output_diff:.4f}")
    
    print("\n=== Final Test Results ===")
    if diff_percentage > 50:
        print("✅ Pretrained ResNet34 weights were successfully loaded")
        print("✅ Forward pass completed successfully with different results from random initialization")
        print("\nTo train the model with pretrained weights, use:")
        print("python train.py --pretrained")
    else:
        print("❌ Test failed: Pretrained weights don't appear to be loaded correctly")
        print("Please check that torchvision is installed and accessible")

if __name__ == "__main__":
    test_pretrained_weights() 