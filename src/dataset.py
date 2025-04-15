import torch
from torch.utils.data import Dataset
import h5py
import json
import os

class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to wrap the Flickr8k dataset.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder with data files
        :param data_name: base name of processed data files
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        #load in data
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.images = self.h['images']

        self.cpi = self.h.attrs['captions_per_image']

        #load in captions
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '_enc_captions.json'), 'r') as j:
            self.enc_captions = json.load(j)

        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '_caplens.json'), 'r') as j:
            self.caplens = json.load(j)
            
        self.transform = transform
        self.dataset_size = len(self.enc_captions)

    def __getitem__(self, index):
        # Remember, the Nth caption corresponds to the ( N // captions_per_image)th image
        img = torch.FloatTensor(self.images[index // self.cpi] / 255)

        if self.transform is not None:
            img = self.transform(img)
        
        caption = torch.LongTensor(self.enc_captions[index])
        caplen = torch.LongTensor([self.caplens[index]])

        if self.split == 'TRAIN':
            return img, caption, caplen
        else:
            # For validation and test set, also return all captions
            all_captions = torch.LongTensor(
                self.enc_captions[((index // self.cpi) * self.cpi):(((index // self.cpi) * self.cpi) + self.cpi)]
            )
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size
