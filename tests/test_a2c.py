import torch
from src.A2C import A2CImageCaptioning
import json
import os

# Load word map
data_folder = 'data_output'
data_name = 'flickr8k_5_5'

word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
with open(word_map_file, 'r') as j:
    word_map = json.load(j)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model
checkpoint = 'model_outputs/BEST_flickr8k_5_5.pth.tar'
print(f"Loading checkpoint from {checkpoint}")

try:
    model = A2CImageCaptioning(
        word_map=word_map,
        device=device,
        checkpoint=checkpoint,
        fine_tune_encoder=False,
        entropy_weight=0.05,
        value_loss_weight=0.2,
        temperature=1.2
    )
    print("Model initialized successfully!")
    print(f"Model structure: Encoder, Decoder (Actor), and Critic")
    print(f"Critic type: {type(model.critic).__name__}")
    
    # Print model properties
    print(f"Encoder output dimension: {model.encoder.enc_dim}")
    print(f"Decoder dimension: {model.decoder.decoder_dim}")
    
    # Create a dummy batch with correct dimensions
    # The encoder output should be batch_size × num_pixels × encoder_dim
    # where num_pixels = height * width (e.g., 14*14 = 196)
    batch_size = 2
    encoder_dim = model.encoder.enc_dim  # Should be 512 for ResNet34
    height, width = 14, 14  # Common output size from ResNet34
    num_pixels = height * width
    
    # Create dummy encoder output (batch_size, encoder_dim, height, width)
    dummy_encoder_out = torch.randn(batch_size, encoder_dim, height, width).to(device)
    print(f"Dummy encoder output shape: {dummy_encoder_out.shape}")
    
    print("Testing forward pass of HybridCriticNetwork directly...")
    # Use a sample hidden state for critic testing
    dummy_hidden_state = torch.randn(batch_size, model.decoder.decoder_dim).to(device)
    critic_output = model.critic(dummy_encoder_out, dummy_hidden_state)
    print(f"Critic output shape: {critic_output.shape}")
    
    print("Testing caption generation with sampling...")
    samples, log_probs, entropies, hidden_states = model.generate_caption_with_sampling(dummy_encoder_out)
    print(f"Generated captions shape: {samples.shape}")
    print(f"Hidden states shape: {hidden_states.shape}")
    
    # Now test the full a2c_train_step with the correct input sizes
    print("\nTesting critic with hidden states from generate_caption_with_sampling...")
    
    # Expand encoder out for each time step
    seq_length = hidden_states.size(1)
    expanded_encoder_out = dummy_encoder_out.unsqueeze(1).expand(-1, seq_length, -1, -1, -1)
    flat_encoder_out = expanded_encoder_out.reshape(-1, *dummy_encoder_out.shape[1:])
    flat_hidden_states = hidden_states.view(-1, hidden_states.size(-1))
    
    print(f"Expanded encoder shape: {expanded_encoder_out.shape}")
    print(f"Flattened encoder shape: {flat_encoder_out.shape}")
    print(f"Flattened hidden states shape: {flat_hidden_states.shape}")
    
    # Test critic with expanded data
    values = model.critic(flat_encoder_out, flat_hidden_states)
    print(f"Critic values shape: {values.shape}")
    values_reshaped = values.view(batch_size, seq_length)
    print(f"Reshaped values shape: {values_reshaped.shape}")
    
    print("All tests passed successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 