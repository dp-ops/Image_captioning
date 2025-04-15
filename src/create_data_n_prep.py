#create dataset with the caption for training and evaluation
#captions use Andrej Karpathy's training, validation, and test splits. This zip file contains the captions. You will also find splits and captions for the Flicker8k and Flicker30k datasets, so feel free to use these instead of MSCOCO if the latter is too large for your computer.


import os
import json
from random import seed, choice, sample
from tqdm import tqdm
import cv2
from cv2 import imread, resize
import numpy as np
import h5py
from collections import Counter


def create_input_data(dataset, json_path, image_folder, captions_per_image, min_word_freq, output_folder, max_len=100):

    assert dataset in {'coco', 'flickr8k', 'flickr30k'}

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Read image path and captions
    train_img_paths = []
    train_img_captions = []
    val_img_paths = []
    val_img_captions = []
    test_img_paths = []
    test_img_captions = []
    word_freq = Counter()

    print(f"Loading data from {json_path}")
    # Print the keys in the JSON file to understand its structure
    print(f"JSON keys: {list(data.keys())}")
    
    # Handle different JSON structures (Karpathy's format)
    if 'images' in data and 'annotations' not in data:
        # Format is likely a dictionary with 'images' that contains both image info and captions
        print("Using Karpathy's dataset format...")
        
        # Create a mapping from image ID to captions
        image_to_captions = {}
        
        # Process images and their captions directly
        for img in data['images']:
            img_id = img['cocoid'] if 'cocoid' in img else img.get('id', 0)
            filename = img['filename']
            filepath = os.path.join(image_folder, filename)
            
            # Check if the image file exists
            if not os.path.exists(filepath):
                print(f"Warning: Image file {filepath} does not exist. Skipping.")
                continue
            
            # Get sentences/captions
            captions = []
            if 'sentences' in img:
                for sent in img['sentences']:
                    caption = sent['tokens'] if isinstance(sent['tokens'], list) else sent['tokens'].split()
                    word_freq.update(caption)
                    if len(caption) <= max_len:
                        captions.append(caption)
            
            # Skip images without captions
            if not captions:
                continue
                
            # Assign to appropriate split
            split = img.get('split', '')
            if split in {'train', 'restval'}:
                train_img_paths.append(filepath)
                train_img_captions.append(captions)
            elif split in {'val'}:
                val_img_paths.append(filepath)
                val_img_captions.append(captions)
            elif split in {'test'}:
                test_img_paths.append(filepath)
                test_img_captions.append(captions)
    
    else:
        # COCO format with separate 'annotations' list
        print("Using COCO format with separate annotations...")
        
        # Create a mapping from image ID to captions
        image_to_captions = {}
        for annotation in data['annotations']:
            img_id = annotation['image_id']
            if img_id not in image_to_captions:
                image_to_captions[img_id] = []
            
            caption = annotation['caption'].split()  # Assuming caption is a string, split into words
            word_freq.update(caption)
            
            if len(caption) <= max_len:
                image_to_captions[img_id].append(caption)

        # Go through each image
        for img in data['images']:
            img_id = img['id']
            
            # Skip images without captions
            if img_id not in image_to_captions or len(image_to_captions[img_id]) == 0:
                continue
                
            captions = image_to_captions[img_id]
            path = os.path.join(image_folder, img['filename'])
            
            # Check if the image file exists
            if not os.path.exists(path):
                print(f"Warning: Image file {path} does not exist. Skipping.")
                continue

            # Assign to appropriate split
            if img['split'] in {'train', 'restval'}:
                train_img_paths.append(path)
                train_img_captions.append(captions)
            elif img['split'] in {'val'}:
                val_img_paths.append(path)
                val_img_captions.append(captions)
            elif img['split'] in {'test'}:
                test_img_paths.append(path)
                test_img_captions.append(captions)

    # Sanity check
    assert len(train_img_paths) == len(train_img_captions)
    assert len(val_img_paths) == len(val_img_captions)
    assert len(test_img_paths) == len(test_img_captions)
    
    print(f"Found {len(train_img_paths)} training images, {len(val_img_paths)} validation images, {len(test_img_paths)} test images")

    # Shortlist words by frequency
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}  # Start at index 1
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0
    
    print(f"Vocabulary size: {len(word_map)} words")

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_' + str(min_word_freq)

    # Save word map to a json file 
    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Save images to hdf5 file and captions to json files
    seed(100)

    for impaths, imcaps, split in [(train_img_paths, train_img_captions, 'TRAIN'),
                                  (val_img_paths, val_img_captions, 'VAL'),
                                  (test_img_paths, test_img_captions, 'TEST')]:
        
        if not impaths:
            print(f"No images found for {split} split. Skipping.")
            continue
            
        print(f"\nProcessing {len(impaths)} {split} images...")
        
        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'w') as h:
            # Create dataset for images
            h.attrs['captions_per_image'] = captions_per_image
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print(f"\nReading {split} images and captions, storing to file...\n")

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):
                # Select captions for this image
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                assert len(captions) == captions_per_image

                # Read image
                try:
                    img = imread(path)
                    if img is None:
                        print(f"Warning: Could not read image {path}. Skipping.")
                        continue
                        
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # If image is grayscale, add the depth dimension
                    if len(img.shape) == 2:
                        img = img[:, :, np.newaxis]
                        img = np.concatenate([img, img, img], axis=2)

                    # Resize image
                    img = resize(img, (256, 256))
                    img = img.transpose(2, 0, 1)

                    assert img.shape == (3, 256, 256)
                    assert np.max(img) <= 255 and np.min(img) >= 0
                    
                    images[i] = img
                except Exception as e:
                    print(f"Error processing image {path}: {e}. Skipping.")
                    continue

                # Encode captions
                for j, c in enumerate(captions): 
                    # Encode caption words using word_map
                    enc_c = [word_map['<start>']] + [word_map.get(w, word_map['<unk>']) for w in c] + [word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))
                    c_len = len(c) + 2  # +2 for <start> and <end>
                    
                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Verify counts
            print(f"Number of images: {images.shape[0]}, Number of captions: {len(enc_captions)}")
            print(f"Expected: {images.shape[0] * captions_per_image}")
            
            # Save encoded captions and their lengths to json files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '_enc_captions.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '_caplens.json'), 'w') as j:
                json.dump(caplens, j)


if __name__ == '__main__':
    #Create input files with their word map 
    #change the dataset to flickr8k or flickr30k
    create_input_data(
        dataset='flickr8k',
        json_path='data/caption_datasets/dataset_flickr8k.json',
        image_folder='data/flickr8k/Images',
        captions_per_image=5,
        min_word_freq=5,
        output_folder='data_output',
        max_len=50
    )