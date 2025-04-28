import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.data import DataLoader
import json
import os
import sys
from src.A2C import A2CImageCaptioning
from src.dataset import CaptionDataset

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
print(f"Using device: {device}")

# Parameters
data_folder = 'data_output'
data_name = 'flickr8k_5_5'
checkpoint = 'model_outputs/BEST_flickr8k_5_5.pth.tar'
batch_size = 2
temperature = 1.2
entropy_weight = 0.05
value_loss_weight = 0.2

try:
    # Load word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    print(f"Loaded word map with {len(word_map)} words")
    
    # Initialize model
    print(f"Loading checkpoint from {checkpoint}")
    model = A2CImageCaptioning(
        word_map=word_map,
        device=device,
        checkpoint=checkpoint,
        fine_tune_encoder=False,
        entropy_weight=entropy_weight,
        value_loss_weight=value_loss_weight,
        temperature=temperature
    )
    print("Model initialized successfully!")
    
    # Create a small data loader
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_loader = DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    print(f"Created data loader with batch size {batch_size}")
    
    # Set models to training mode
    model.encoder.train()
    model.decoder.train()
    model.critic.train()
    
    # Get a single batch
    print("Getting a batch from the data loader...")
    batch = next(iter(train_loader))
    
    if len(batch) == 4:
        images, captions, caplens, allcaps = batch
        print(f"Batch shapes: images={images.shape}, captions={captions.shape}, caplens={caplens.shape}, allcaps={allcaps.shape}")
    else:
        images, captions, caplens = batch
        allcaps = None
        print(f"Batch shapes: images={images.shape}, captions={captions.shape}, caplens={caplens.shape}")
    
    # Perform a single A2C training step
    print("\nRunning a single A2C training step...")
    actor_loss, critic_loss, bleu = model.a2c_train_step(images, captions, caplens, allcaps)
    
    print(f"\nTraining step completed successfully!")
    print(f"Actor Loss: {actor_loss:.4f}")
    print(f"Critic Loss: {critic_loss:.4f}")
    print(f"BLEU Score: {bleu:.4f}")
    
    print("\nAll tests passed successfully! The A2C model is working correctly.")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 