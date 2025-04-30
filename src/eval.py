import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from dataset import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
import math

#Params
data_folder = 'data_output' # files saved by create_input_files.py
data_name = 'flickr8k_5_5'  # base name shared by data files
checkpoint = 'model_outputs/BEST_flickr8k_5_5.pth.tar' # model checkpoint
word_map = 'data_output/WORDMAP_flickr8k_5_5.json' # word map, ensure it's the same the data was encoded with and the model was trained with
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Load model
checkpoint = torch.load(checkpoint, weights_only=False)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

# Load word map (word2ix)
with open(word_map, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def evaluate(beam_size):


    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    
    references = list() # references (true captions) for calculating BLEU-4 score
    hypotheses = list() # hypotheses (predictions)

    for i, (image, caption, caplens, allcaps) in enumerate(tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size

        image = image.to(device)  # (1, 3, 256, 256)

        # Encode
        encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        
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

        #Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)
        seqs = k_prev_words  # (k, 1)
        top_k_scores = torch.zeros(k, 1).to(device)

        #Lists to store completed sequences, their alphas and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        #Decoding
        step = 1

        while True:
            # Check if decoder uses one-hot encoding
            if hasattr(decoder, 'use_one_hot') and decoder.use_one_hot:
                # Convert indices to one-hot vectors before embedding
                one_hot = decoder.one_hot_encoder(k_prev_words)
                embeddings = decoder.embedding(one_hot.float()).squeeze(1)
            else:
                # Standard embedding lookup for indices
                embeddings = decoder.embedding(k_prev_words).squeeze(1)
            
            # Use last layer's hidden state for attention
            awe, _ = decoder.attention(encoder_out, h_list[-1])
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
            scores = decoder.fc(h_list[-1])
            scores = F.log_softmax(scores, dim=1)

            #ADD
            scores = top_k_scores.expand_as(scores) + scores

            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
            else: 
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)
            
            #convert to indices
            prev_word_inds = (top_k_words / vocab_size).long()  # Convert to LongTensor
            next_word_inds = (top_k_words % vocab_size).long()  # Convert to LongTensor

            #Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1) # (s, step+1)

            #Which sequences are incomplete
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            #Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds) #reduce beam length accordingly

            #Proceed with incomplete seqs
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h_list = [h[prev_word_inds[incomplete_inds]] for h in h_list]
            c_list = [c[prev_word_inds[incomplete_inds]] for c in c_list]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            #Break if it takes too long
            if step > 50:
                break
            step += 1
        
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        #References
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))
        references.append(img_captions)

        #Hypotheses
        hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

        assert len(references) == len(hypotheses)

    #Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)

    return bleu4

if __name__ == '__main__':
    beam_size = int(input("Enter beam size: "))
    print("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_size, evaluate(beam_size)))