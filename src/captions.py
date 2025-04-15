import torch
import torch.nn.functional as F
import numpy as np
import json
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import skimage.transform
import argparse
import cv2
from cv2 import imread, resize
from PIL import Image
import warnings
import os
import math
import os.path as osp

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image transformation pipeline
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

def read_image(image_path):
    """
    Read an image using either PIL or OpenCV
    
    :param image_path: path to the image file
    :return: RGB image as numpy array of shape (height, width, 3)
    """
    # Try using PIL first
    try:
        img = Image.open(image_path).convert('RGB')
        # Convert to numpy array and ensure RGB order
        img = np.array(img)
        return img
    except Exception as e:
        print(f"Error loading image with PIL: {e}")
        print("Trying OpenCV instead...")
        
    # Fallback to OpenCV
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise Exception(f"OpenCV couldn't read the image at {image_path}")
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as e:
        print(f"Error loading image with OpenCV: {e}")
        return None

def find_image_captions(image_path, captions_file):
    """
    Find the original captions for an image from Karpathy's dataset JSON file
    
    :param image_path: path to the image
    :param captions_file: path to Karpathy's dataset JSON file
    :return: list of original captions or None if not found
    """
    # Get just the filename without path
    image_filename = osp.basename(image_path)
    
    try:
        with open(captions_file, 'r') as f:
            data = json.load(f)
            
        # Check the format of the JSON file
        if 'images' in data:
            # Karpathy's format
            for img in data['images']:
                if image_filename in img.get('filename', ''):
                    if 'sentences' in img:
                        # Extract raw captions
                        return [sent['raw'] for sent in img['sentences']]
        
        # If we couldn't find the image or captions
        print(f"Could not find captions for {image_filename} in the provided JSON file.")
        return None
        
    except Exception as e:
        print(f"Error loading captions file: {e}")
        return None

def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=3):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each step
    :return: caption, weights for visualization
    """
    k = beam_size
    vocab_size = len(word_map)
    
    # Read image and process
    img = read_image(image_path)
    if img is None:
        raise Exception(f"Could not read image at {image_path}")
    
    print(f"Image shape before transforms: {img.shape}")
    
    # Encode
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension (1, 3, 256, 256)
    img = img.to(device)
    encoder_out = encoder(img)  # (1, enc_image_size, enc_image_size, encoder_dim)
    print(f"Encoder output shape: {encoder_out.shape}")
    
    # Get dimensions
    if encoder_out.dim() == 4:
        # If output is (batch_size, enc_image_size, enc_image_size, encoder_dim)
        encoder_dim = encoder_out.size(-1)
        enc_image_size = encoder_out.size(1)
        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    else:
        # If output is already (batch_size, num_pixels, encoder_dim)
        encoder_dim = encoder_out.size(-1)
        enc_image_size = int(math.sqrt(encoder_out.size(1)))
    
    num_pixels = encoder_out.size(1)
    
    # Expand encoder_out for beam search
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)
    
    # Initialize LSTM state
    h_list, c_list = decoder.init_hidden_state(encoder_out)
    
    # Expand states for beam search
    h_list = [h.expand(k, -1) for h in h_list]
    c_list = [c.expand(k, -1) for c in c_list]
    
    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)
    seqs = k_prev_words  # (k, 1)
    top_k_scores = torch.zeros(k, 1).to(device)
    
    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_scores = list()
    complete_alphas = list()
    
    # Collect alphas at each step for visualization
    all_alphas = torch.zeros(k, 1, num_pixels).to(device)
    
    # Decoding
    step = 1
    
    while True:
        # Check if the model uses one-hot encoding
        if hasattr(decoder, 'use_one_hot') and decoder.use_one_hot:
            # Convert to one-hot vectors first
            one_hot = decoder.one_hot_encoder(k_prev_words)
            embeddings = decoder.embedding(one_hot.float()).squeeze(1)
        else:
            # Traditional embedding lookup
            embeddings = decoder.embedding(k_prev_words).squeeze(1)
        
        # Use last layer's hidden state for attention
        awe, alpha = decoder.attention(encoder_out, h_list[-1])
        gate = decoder.sigmoid(decoder.f_beta(h_list[-1]))
        awe = gate * awe
        
        h_new = []
        c_new = []
        
        # Process through each LSTM layer
        for i in range(decoder.num_layers):
            if i == 0:
                lstm_input = torch.cat([embeddings, awe], dim=1)
            else:
                lstm_input = h_new[-1]
                
            h, c = decoder.lstm_layers[i](
                lstm_input,
                (h_list[i], c_list[i])
            )
            h_new.append(h)
            c_new.append(c)
            
        # Update states
        h_list = h_new
        c_list = c_new
        
        # Generate prediction using last layer
        scores = decoder.fc(decoder.dropout(h_list[-1]))
        scores = F.log_softmax(scores, dim=1)
        
        # Add scores
        scores = top_k_scores.expand_as(scores) + scores
        
        # For the first step, all k sequences are equally likely, so we can just take the best k
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)
        
        # Convert scores and words to lists
        prev_word_inds = top_k_words / vocab_size
        next_word_inds = top_k_words % vocab_size
        
        # Add new words to sequences
        seqs = torch.cat([seqs[prev_word_inds.long()], next_word_inds.unsqueeze(1)], dim=1)
        all_alphas = torch.cat([all_alphas[prev_word_inds.long()], alpha[prev_word_inds.long()].unsqueeze(1)], dim=1)
        
        # Find incomplete sequences
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
        
        # Store complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
            complete_alphas.extend(all_alphas[complete_inds])
        
        k = len(incomplete_inds)
        
        if k == 0:
            break
        
        # Proceed with incomplete sequences
        seqs = seqs[incomplete_inds]
        all_alphas = all_alphas[incomplete_inds]
        h_list = [h[prev_word_inds[incomplete_inds].long()] for h in h_list]
        c_list = [c[prev_word_inds[incomplete_inds].long()] for c in c_list]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds].long()]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
        
        if step > 50:
            break
        step += 1
    
    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_alphas[i]
    
    return seq, alphas

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show, Attend and Tell - Generate Captions")

    parser.add_argument('--img', '-i', help='path to image')
    parser.add_argument('--model', '-m', help='path to model')
    parser.add_argument('--word_map', '-wm', help='path to word map JSON')
    parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--visualize_attention', '-v', action='store_true', help='visualize attention weights')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')
    parser.add_argument('--captions_json', '-cj', help='path to original captions JSON file (Karpathy\'s format)')

    args = parser.parse_args()

    #load model
    checkpoint = torch.load(args.model, map_location=str(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    #load word map
    with open(args.word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()} #inverse word map

    # Try to load original captions if JSON file is provided
    original_captions = None
    if args.captions_json:
        original_captions = find_image_captions(args.img, args.captions_json)
        if original_captions:
            print("Original captions:")
            for i, cap in enumerate(original_captions):
                print(f"{i+1}: {cap}")

    # Generate caption
    seq, alphas = caption_image_beam_search(encoder, decoder, args.img, word_map, args.beam_size)
    
    # Convert word indices to words
    words = [rev_word_map[ind] for ind in seq if ind not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
    
    # Create the caption
    caption = ' '.join(words)
    print('Generated caption:', caption)
    
    # Display the image
    img = cv2.imread(args.img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    
    # Create combined title with generated and original caption if available
    if original_captions:
        plt.title(f"Generated: {caption}\nOriginal: {original_captions[0]}", fontsize=10)
    else:
        plt.title(f"Generated: {caption}", fontsize=12)
        
    plt.axis('off')
    
    # Create output directory if it doesn't exist
    os.makedirs('model_outputs', exist_ok=True)
    
    # Save captioned image
    plt.savefig(os.path.join('model_outputs', 'caption_result.png'), bbox_inches='tight')
    
    # Show the image
    plt.show()
    
    # If original captions are available, create a comparison visualization
    if original_captions and len(original_captions) > 0:
        plt.figure(figsize=(12, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("Generated Caption", fontsize=12)
        plt.text(0, img.shape[0] + 20, caption, fontsize=10, wrap=True)
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.title("Original Captions", fontsize=12)
        
        # Display up to 5 original captions
        caption_text = "\n".join(original_captions[:5])
        plt.text(0, img.shape[0] + 20, caption_text, fontsize=10, wrap=True)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join('model_outputs', 'caption_comparison.png'), bbox_inches='tight')
        plt.show()
    
    # Visualize attention if requested
    if args.visualize_attention:
        try:
            # Check if alphas is already a tensor or a list
            if isinstance(alphas, list):
                # If it's a list, convert to tensor
                alphas = torch.FloatTensor(alphas)
            else:
                # If it's already a tensor, just ensure it's a FloatTensor
                alphas = alphas.float()
            
            # Make sure everything is on CPU
            if alphas.is_cuda:
                print("Moving attention weights from GPU to CPU...")
                alphas = alphas.cpu()
            
            # Debug the alphas shape
            print(f"Alphas shape: {alphas.shape}")
            
            # Remove <start> token if needed
            if len(words) + 1 == len(alphas):  # +1 for <start> token
                alphas = alphas[1:]  # Remove the first attention map for <start> token
            
            # Create attention visualization
            plt.figure(figsize=(15, 10))
            
            # Calculate size of attention map - checking if it's already in the right format
            if alphas.dim() == 3 and alphas.shape[1] == alphas.shape[2]:
                # If alphas is already in format [len_words, height, width] where height=width
                att_size = alphas.shape[1]
                print(f"Using existing attention map size: {att_size}x{att_size}")
            else:
                # Otherwise calculate from the flattened dimension
                att_size = int(math.sqrt(alphas.shape[-1])) if alphas.dim() > 2 else 14
                print(f"Calculated attention map size: {att_size}x{att_size}")
            
            for t in range(len(words)):
                if t >= len(alphas):
                    print(f"Warning: More words ({len(words)}) than attention maps ({len(alphas)})")
                    break
                    
                plt.subplot(((len(words)-1) // 5) + 1, min(5, len(words)), t + 1)
                
                plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
                plt.imshow(img)
                
                # Reshape the attention map based on its dimension
                if alphas.dim() == 3 and alphas.shape[1] == alphas.shape[2]:
                    # alphas is already in the right shape [len_words, att_size, att_size]
                    alpha = alphas[t].detach().cpu().numpy()
                    print(f"Word '{words[t]}': Using pre-shaped attention map of size {alpha.shape}")
                else:
                    try:
                        # Reshape to square attention map
                        alpha = alphas[t].view(att_size, att_size).detach().cpu().numpy()
                        print(f"Word '{words[t]}': Reshaped attention map to {alpha.shape}")
                    except RuntimeError as e:
                        print(f"Error reshaping attention map for '{words[t]}': {e}")
                        print(f"Attention map shape: {alphas[t].shape}, attempting to reshape to {att_size}x{att_size}")
                        # Try alternative approach - reshape first, then extract
                        if alphas[t].numel() == att_size * att_size:
                            alpha = alphas[t].reshape(att_size, att_size).detach().cpu().numpy()
                        else:
                            print(f"Cannot reshape attention map for '{words[t]}', using fallback")
                            alpha = np.ones((att_size, att_size)) * 0.5  # Fallback - uniform attention
                
                if args.smooth:
                    try:
                        alpha = skimage.transform.pyramid_expand(alpha, upscale=16, sigma=8)
                    except Exception as e:
                        print(f"Error smoothing attention map: {e}")
                        # Use a simpler upsampling method if pyramid_expand fails
                        alpha = cv2.resize(alpha, (alpha.shape[0]*16, alpha.shape[1]*16))
                
                plt.imshow(alpha, alpha=0.6, cmap=cm.hot)
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join('model_outputs', 'attention_visualization.png'))
            plt.show()
        except Exception as e:
            print(f"Error visualizing attention: {e}")
            print("Skipping attention visualization. Full error details:")
            import traceback
            traceback.print_exc()