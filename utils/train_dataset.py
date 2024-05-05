import torch.utils.data as data
import os
import numpy as np
import numpy.random as random
from utils.dataset_utils import *

################################################################
#                    Train Dataset
################################################################

class TrainDataset(data.Dataset):
    def __init__(self, filenames, captions, att_masks, attr_label, split="train", args=None):

        self.captions_per_image = args.captions_per_image
        self.data_dir = args.data_dir
        self.dataset = args.dataset
        self.en_type = args.en_type
        self.model_type = args.model_type
        self.split = split 

        self.filenames = filenames
        self.captions = captions 
        self.att_masks = att_masks
        self.attr_label = attr_label
        self.word_num = args.bert_words_num 

        split_dir = os.path.join(self.data_dir, self.split)
        self.class_id = load_class_id(split_dir)


    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')

        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)

        # pad with 0s (i.e., '<end>')
        x = np.zeros((self.word_num, 1), dtype='int64')
        x_len = num_words
        if num_words <= self.word_num:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:self.word_num]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = self.word_num

        return x, x_len


    def get_attr_vector(self, attr_file):
        with open(attr_file, "r") as f:
            line = f.read()

        return torch.Tensor([0.0 if int(l) == -1 else 1.0 for l in line.split(",")])



    def __getitem__(self, index):
        key = self.filenames[index]
        cls_id = self.class_id[index]
        cap_attr = torch.Tensor(self.attr_label[index]) 
        
        attr_file = os.path.join(self.data_dir, "attributes", self.split, key + ".txt")
        img_extension = ".jpg" # works for all dataset 
        img_name = os.path.join(self.data_dir, "images", self.split, key + img_extension)
        
        imgs = get_imgs(img_name, self.split, self.model_type)
        attr_vec = self.get_attr_vector(attr_file)

        # random select a sentence
        sent_ix = random.randint(0, self.captions_per_image - 1)
        new_sent_ix = index * self.captions_per_image + sent_ix

        caps, mask = self.captions[new_sent_ix], self.att_masks[new_sent_ix]
        return imgs, caps, mask, key, cap_attr, attr_vec, cls_id 


    def __len__(self):
        return len(self.filenames)