#%%
#Used #
# https://github.com/dwayne99/Image_Captioning/blob/master/models.py
#https://www.youtube.com/watch?v=DkNIBBBvcPs
#https://github.com/cengineer13/Resnet34-Deep-Residual-Learning-for-Image-Recognition-from-scratch-in-pytorch/blob/master/resnet34.py


import torch
import torch.nn as nn
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#block class for resNet34
class block(nn.Module):
    """
    Basic ResNet block with residual connections
    Each block has two conv layers with batch normalization and ReLU activation
    The residual connection allows gradients to flow directly through the network
    """
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1) -> None:
        super(block, self).__init__()
        self.expansion = 1
        # First convolutional layer (may downsample with stride)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # Second convolutional layer (always keeps dimensions)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Optional downsampling for the identity mapping if dimensions change
        self.identity_downsample = identity_downsample


    def forward(self, x):
        # Save input for the residual connection
        identity = x.clone()

        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)

        # Apply identity mapping, with optional downsampling if needed
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        # Add residual connection and apply ReLU
        x += identity
        x = self.relu(x)
        
        return x
    
    
#resNet encoder class
class ResNet(nn.Module): #[3, 4, 6, 3]
    """
    ResNet model with configurable depth based on the layers parameter
    ResNet34 uses [3, 4, 6, 3] configuration for the four layer groups
    """
    def __init__(self, block, layers, image_channels, num_classes=None) -> None:
        super(ResNet, self).__init__()

        #initial conv layer
        self.in_channels = 64
        # Initial 7x7 convolution that reduces spatial dimensions by half
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU() #inpalce=True
        # Max pooling further reduces spatial dimensions by half
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Four ResNet layer groups with increasing channels and decreasing spatial dimensions
        self.layer1 = self.make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self.make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self.make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self.make_layer(block, layers[3], out_channels=512, stride=2)

        # Only create the final classification layers if num_classes is provided
        # These won't be used for the encoder in image captioning
        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(512, num_classes)

    #forward pass
    def forward(self, x):
        # Initial convolution and pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Pass through the four ResNet layer groups
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Only use classification layers if they exist
        if hasattr(self, 'avgpool') and hasattr(self, 'fc'):
            x = self.avgpool(x)
            x = x.reshape(x.shape[0], -1)
            x = self.fc(x)

        return x

    def make_layer(self, block, num_residual_blocks, out_channels, stride):
        """
        Creates a layer group with multiple residual blocks
        The first block may include downsampling if stride > 1 or channels change
        """
        identity_downsample = None
        layers = []

        # Create downsampling projection if stride or channels change
        if stride != 1 or self.in_channels != out_channels: 
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size = 1, stride = stride), 
                nn.BatchNorm2d(out_channels) 
            )
        
        # First block may have different input channels and stride
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels   

        # Remaining blocks keep the same dimensions
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))  

        return nn.Sequential(*layers)
    
    def fine_tune(self, fine_tune=True):
        """
        Enable or disable fine-tuning of the model parameters
        :param fine_tune: boolean indicating whether to allow fine-tuning
        """
        for param in self.parameters():
            param.requires_grad = fine_tune
            
    def load_pretrained_weights(self, weights_path):
        """
        Load pretrained weights from a file path
        :param weights_path: path to the pretrained weights file
        """
        state_dict = torch.load(weights_path, weights_only=True)
        # Filter out avgpool and fc layers if we're using as encoder
        if not hasattr(self, 'fc'):
            state_dict = {k: v for k, v in state_dict.items() if 'avgpool' not in k and 'fc' not in k}
        self.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {weights_path}")
    
# def ResNet34(img_channel=3, num_classes=1000):
#     """
#     Creates a ResNet34 model with specified input channels and output classes
#     """
#     return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)

class EncoderCNN(nn.Module):
    """
    Encoder CNN using ResNet34 for feature extraction in image captioning
    Extracts spatial features from images for the attention mechanism
    """
    def __init__(self, encoded_image_size=14, pretrained_path=None, fine_tune=False):
        """
        :param encoded_image_size: size of the feature maps after encoding
        :param pretrained_path: path to pretrained weights file or "torchvision" to use torchvision's pretrained weights
        :param fine_tune: whether to fine-tune the encoder
        """
        super(EncoderCNN, self).__init__()
        
        # Load ResNet without classification layers
        self.resnet = ResNet(block, [3, 4, 6, 3], 3, num_classes=None)
        
        # Load pretrained weights if provided
        if pretrained_path:
            if pretrained_path == "torchvision":
                # Load weights from torchvision's pretrained models
                try:
                    from torchvision.models import resnet34, ResNet34_Weights
                    # Try to load newest version first (for PyTorch 1.13+)
                    try:
                        pretrained_model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
                    except:
                        # Fall back to older version
                        pretrained_model = resnet34(pretrained=True)
                    
                    # Copy weights from pretrained model to our model
                    pretrained_dict = pretrained_model.state_dict()
                    
                    # Filter out avgpool and fc layers
                    filtered_dict = {k: v for k, v in pretrained_dict.items() 
                                    if 'avgpool' not in k and 'fc' not in k}
                    
                    # Match keys between our model and pretrained model
                    model_dict = self.resnet.state_dict()
                    
                    # Check which keys overlap
                    matching_keys = {k: v for k, v in filtered_dict.items() if k in model_dict}
                    
                    # Update our model with pretrained weights
                    model_dict.update(matching_keys)
                    self.resnet.load_state_dict(model_dict)
                    
                    print(f"Loaded pretrained ResNet34 weights from torchvision")
                    print(f"Transferred {len(matching_keys)}/{len(model_dict)} layers")
                except ImportError:
                    print("Torchvision not available. Using random initialization.")
            else:
                self.resnet.load_pretrained_weights(pretrained_path)
        else:
            print("No pretrained weights specified. Using random initialization.")
        
        # Fine-tune or freeze the encoder
        self.resnet.fine_tune(fine_tune)
        
        # Adaptive pooling to get fixed size output
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        
        # Get the encoding dimension (512 for ResNet34)
        self.enc_dim = 512

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images with shape (batch_size, num_pixels, encoder_dim)
                 where num_pixels = encoded_image_size^2
        """
        # Get features from ResNet
        features = self.resnet(images)  # (batch_size, 512, h/32, w/32)
        
        # Adaptive pooling to get fixed size
        features = self.adaptive_pool(features)  # (batch_size, 512, encoded_image_size, encoded_image_size)
        
        # Reshape for attention: (batch_size, encoded_image_size, encoded_image_size, 512)
        features = features.permute(0, 2, 3, 1)
        batch_size = features.size(0)
        # Flatten spatial dimensions: (batch_size, num_pixels, 512)
        features = features.view(batch_size, -1, self.enc_dim)
        
        return features
        
    def fine_tune(self, fine_tune=True):
        """
        Enable or disable fine-tuning of the encoder
        :param fine_tune: boolean indicating whether to fine-tune the encoder
        """
        self.resnet.fine_tune(fine_tune)

class Attention(nn.Module): 
    '''Attention network mechanism for focusing on relevant image regions'''
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        Initialize attention network
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        # Transform image features to attention dimension
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        # Transform decoder hidden state to attention dimension
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        # Calculate attention scores
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        # Softmax to calculate the attention weights
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, attention weights
        """
        # Transform encoder outputs: (batch_size, num_pixels, attention_dim)
        att1 = self.encoder_att(encoder_out)
        # Transform decoder hidden state: (batch_size, attention_dim)
        att2 = self.decoder_att(decoder_hidden)
        # Combine by element-wise addition: (batch_size, num_pixels, attention_dim)
        # att2 needs to be unsqueezed to be broadcastable
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        
        # Normalize attention weights: (batch_size, num_pixels)
        alpha = self.softmax(att)

        # Weight encoder outputs with attention weights: (batch_size, encoder_dim)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)

        return attention_weighted_encoding, alpha
    
class LSTMDecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=512, dropout=0.5, num_layers=3):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding dimension for words
        :param decoder_dim: size of decoder's RNN (300 for each layer)
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout probability
        :param num_layers: number of LSTM layers (default: 3)
        """
        super(LSTMDecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.num_layers = num_layers

        # Create an Attention Network Instance
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=self.dropout)
        
        # Create multiple LSTM layers
        self.lstm_layers = nn.ModuleList([
            nn.LSTMCell(embed_dim + encoder_dim if i == 0 else decoder_dim, 
                       decoder_dim, 
                       bias=True)
            for i in range(num_layers)
        ])

        # Initialize hidden and cell states from image features for each layer
        self.init_h = nn.ModuleList([
            nn.Linear(encoder_dim, decoder_dim) for _ in range(num_layers)
        ])
        self.init_c = nn.ModuleList([
            nn.Linear(encoder_dim, decoder_dim) for _ in range(num_layers)
        ])

        # Gating scalar for attention
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()

        # Linear layer to produce vocabulary distribution
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.init_weights()
    
    def init_weights(self):
        '''
        Initializes parameters with values from the uniform distribution, for easier convergence
        '''
        # Initialize embedding weights
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        
        # Initialize LSTM layers
        for lstm in self.lstm_layers:
            for name, param in lstm.named_parameters():
                if 'weight' in name:
                    param.data.uniform_(-0.1, 0.1)
                elif 'bias' in name:
                    param.data.fill_(0)
        
        # Initialize final linear layer
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        '''
        Initialize the hidden and cell states of all LSTM layers
        :param encoder_out: encoded images, tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden states, cell states (both lists of length num_layers with size (batch_size, decoder_dim))
        '''
        mean_encoder_out = encoder_out.mean(dim=1)
        h = []
        c = []
        for i in range(self.num_layers):
            h.append(self.init_h[i](mean_encoder_out))
            c.append(self.init_c[i](mean_encoder_out))
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten encoder_out
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)

        # Initialize LSTM state
        h_list, c_list = self.init_hidden_state(encoder_out)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # For each time step
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            
            # Calculate attention using the last layer's hidden state
            attention_weighted_encoding, alpha = self.attention(
                encoder_out[:batch_size_t], 
                h_list[-1][:batch_size_t]
            )

            gate = self.sigmoid(self.f_beta(h_list[-1][:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding

            h_new = []
            c_new = []
            
            # Process through each LSTM layer
            for i in range(self.num_layers):
                if i == 0:
                    # First layer receives embeddings and attention
                    lstm_input = torch.cat(
                        [embeddings[:batch_size_t, t, :], attention_weighted_encoding], 
                        dim=1
                    )
                else:
                    # Other layers receive previous layer's hidden state
                    lstm_input = h_new[-1]

                h, c = self.lstm_layers[i](
                    lstm_input,
                    (h_list[i][:batch_size_t], c_list[i][:batch_size_t])
                )
                h_new.append(h)
                c_new.append(c)

            # Update hidden and cell states
            h_list = [h_new[i] for i in range(self.num_layers)]
            c_list = [c_new[i] for i in range(self.num_layers)]

            # Generate word predictions using the last layer
            preds = self.fc(self.dropout(h_list[-1]))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

# # Example of creating a complete image captioning model
# class ImageCaptioningModel(nn.Module):
#     """
#     Full image captioning model combining encoder and decoder
#     """
#     def __init__(self, encoder_dim=512, attention_dim=512, embed_dim=512, decoder_dim=512, 
#                  vocab_size=10000, encoded_image_size=14, pretrained_path=None, fine_tune_encoder=False):
#         super(ImageCaptioningModel, self).__init__()
        
#         # Initialize encoder and decoder
#         self.encoder = EncoderCNN(encoded_image_size=encoded_image_size, 
#                                   pretrained_path=pretrained_path, 
#                                   fine_tune=fine_tune_encoder)
        
#         self.decoder = LSTMDecoderWithAttention(attention_dim=attention_dim,
#                                                embed_dim=embed_dim,
#                                                decoder_dim=decoder_dim,
#                                                vocab_size=vocab_size,
#                                                encoder_dim=encoder_dim)
        
#     def forward(self, images, encoded_captions, caption_lengths):
#         """
#         Forward propagation through the full model
#         """
#         # Encode the images
#         encoder_out = self.encoder(images)
        
#         # Decode with attention
#         predictions, encoded_captions, decode_lengths, alphas, sort_ind = self.decoder(
#             encoder_out, encoded_captions, caption_lengths
#         )
        
#         return predictions, alphas, encoded_captions, decode_lengths, sort_ind

