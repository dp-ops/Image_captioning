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
    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image to caption
    :param word_map: word map
    :param beam_size: beam size
    :return: caption
    """
    k = beam_size
    vocab_size = len(word_map)

    # Read image and process
    try:
        # Method 1: Try using PIL and convert to numpy
        img_pil = Image.open(image_path)
        image = np.array(img_pil)
        
        # If image is grayscale, convert to RGB
        if len(image.shape) == 2:
            image = np.stack((image,) * 3, axis=-1)
        elif image.shape[2] == 1:
            image = np.concatenate([image, image, image], axis=2)
        
        # Convert from RGB to BGR if needed for cv2
        if image.shape[2] == 3:  # Color image
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"PIL loading failed: {e}, trying OpenCV directly")
        # Method 2: Fallback to OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")
    
    # Convert back to RGB for processing (OpenCV loads as BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize and normalize
    image = cv2.resize(image, (256, 256))
    image = image.transpose(2, 0, 1)  # (H,W,C) -> (C,H,W)
    image = image / 255.0
    image = torch.FloatTensor(image).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(image) #(3, 256, 256)

    #Encode image
    image = image.unsqueeze(0) #(1, 3, 256, 256)
    encoder_out = encoder(image) #(1, encoder_dim, enc_image_size, enc_image_size)
    
    # Get dimensions of encoder output
    if encoder_out.dim() == 4:  # (batch_size, channels, height, width)
        # ResNet outputs (batch_size, channels, height, width)
        batch_size, encoder_dim, enc_image_size, _ = encoder_out.size()
        # Reshape to (batch_size, height*width, channels)
        encoder_out = encoder_out.permute(0, 2, 3, 1)  # (batch_size, height, width, channels)
    else:
        # Handle the case where the encoder output is differently shaped
        print(f"Encoder output shape: {encoder_out.size()}")
        if encoder_out.dim() == 3:
            # If it's a 3D tensor like (batch_size, num_pixels, encoder_dim)
            batch_size, num_pixels, encoder_dim = encoder_out.size()
            enc_image_size = int(math.sqrt(num_pixels))  # Assuming it's square
        else:
            # Default fallback
            batch_size = encoder_out.size(0)
            encoder_dim = encoder_out.size(-1)  # Last dimension is usually the feature dimension
            enc_image_size = 14  # Default for ResNet
    
    #Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim) #(1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim) #(k, num_pixels, encoder_dim)

    #tensor to store top k sequences; now they are just <start> tokens
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device) #(k, 1)
    seqs = k_prev_words
    top_k_scores = torch.zeros(k, 1).to(device) #(k, 1)
    attention_weights = torch.zeros(k, 1, enc_image_size, enc_image_size).to(device) #(k, 1, enc_image_size, enc_image_size)

    #list to store complete sequences, alphas and scores
    complete_seqs = list()
    complete_seqs_scores = list()
    complete_seqs_alphas = list()

    #start decoding 
    step = 1
    h_list, c_list = decoder.init_hidden_state(encoder_out)  # Initialize states
    
    # Expand h_list and c_list for beam size
    h_list = [h.expand(k, -1) for h in h_list]  # Expand each hidden state
    c_list = [c.expand(k, -1) for c in c_list]  # Expand each cell state

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)
        # Use last layer's hidden state for attention
        awe, alpha = decoder.attention(encoder_out, h_list[-1])  # Use h_list[-1] instead of h
        alpha = alpha.view(-1, enc_image_size, enc_image_size)
        gate = decoder.sigmoid(decoder.f_beta(h_list[-1]))  # Use h_list[-1]
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
                (h_list[i], c_list[i])  # Remove the batch size slicing
            )
            h_new.append(h)
            c_new.append(c)

        # Update hidden and cell states
        h_list = h_new
        c_list = c_new

        scores = decoder.fc(h_list[-1])  # Use last layer's hidden state
        scores = F.log_softmax(scores, dim=1)

        #add new scores
        scores = top_k_scores.expand_as(scores) + scores #(k, vocab_size)

        #for the first step, all k sequences are equally likely, so we can just take the best (k, 1) sequences
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True) #(k, 1)
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True) #(k, )
        
        #convert scores and words to lists
        prev_word_inds = top_k_words / vocab_size #(k, )
        next_word_inds = top_k_words % vocab_size #(k, )

        #add new words to sequences
        seqs = torch.cat([seqs[prev_word_inds.long()], next_word_inds.unsqueeze(1)], dim=1) #(k, step)
        attention_weights = torch.cat([attention_weights[prev_word_inds.long()], alpha[prev_word_inds.long()].unsqueeze(1)], dim=1) #(k, step, enc_image_size, enc_image_size)

        #find incomplete sequences
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map['<end>']]

        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
            complete_seqs_alphas.extend(attention_weights[complete_inds])

        k = len(incomplete_inds) #number of incomplete sequences

        if k == 0:
            break

        #proceed with incomplete sequences
        seqs = seqs[incomplete_inds]
        attention_weights = attention_weights[incomplete_inds]
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
    alphas = complete_seqs_alphas[i]

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