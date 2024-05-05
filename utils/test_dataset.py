import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np 
import os
import numpy.random as random
from utils.dataset_utils import * 


################################################################
#                    Test Dataset
################################################################
class TestDataset(data.Dataset):
    def __init__(self, filenames, captions, att_masks, split, args=None):
        
       
        self.split= split
        self.data_dir = args.data_dir
        self.dataset = args.dataset
        self.captions_per_image = args.captions_per_image
        self.model_type = args.model_type 

        #self.is_ident = args.is_ident 
        self.filenames = filenames
        self.captions = captions 
        self.en_type = args.en_type 
        self.att_masks = att_masks

        print("\n############## Loading %s dataset ################" % self.dataset)
        self.class_id = load_class_id(os.path.join(self.data_dir, self.split))

        if split == "test":
            if args.is_ident == True: self.test_pair_list = args.test_ver_acc_list
            else: self.test_pair_list = args.test_ver_list

        elif split == "valid":
            if args.is_ident == True: self.test_pair_list = args.valid_ver_acc_list
            else: self.test_pair_list = args.valid_ver_list

        self.imgs_pair, self.pair_label = self.get_test_list()


    def get_test_list(self):
        with open(self.test_pair_list, 'r') as fd:
            pairs = fd.readlines()
        imgs_pair = []
        pair_label = []
        for pair in pairs:
            splits = pair.split(" ")
            imgs = [splits[0], splits[1]]
            imgs_pair.append(imgs)
            pair_label.append(int(splits[2]))
        return imgs_pair, pair_label
    
    def get_best_caption_id(self, real_idx):
        start = real_idx * self.captions_per_image
        end = start + self.captions_per_image
        ls = [torch.sum(self.att_masks[idx]).numpy() for idx in range(start, end)]
        return start + np.argmax(ls)


    def get_attr_vector(self, attr_file):
        with open(attr_file, "r") as f:
            line = f.read()

        return torch.Tensor([0.0 if int(l) == -1 else 1.0 for l in line.split(",")])


    def __getitem__(self, index):
        imgs = self.imgs_pair[index]
        pair_label = self.pair_label[index]

        data_dir = os.path.join(self.data_dir, "images")

        img1_name = os.path.join(imgs[0].split("_")[0], imgs[0])
        img2_name = os.path.join(imgs[1].split("_")[0], imgs[1])

        img1_path = os.path.join(data_dir, self.split, img1_name)
        img2_path = os.path.join(data_dir, self.split, img2_name)

        key1 = img1_name[:-4]
        key2 = img2_name[:-4]

        attr_file1 = os.path.join(self.data_dir, "attributes", self.split, key1 + ".txt")
        attr_file2 = os.path.join(self.data_dir, "attributes", self.split, key2 + ".txt")

        attr_vec1 = self.get_attr_vector(attr_file1)
        attr_vec2 = self.get_attr_vector(attr_file2)

        img1 = get_imgs(img1_path, self.split, self.model_type)
        img2 = get_imgs(img2_path, self.split, self.model_type)

        img1_h = do_flip_test_images(img1_path, self.model_type)
        img2_h = do_flip_test_images(img2_path, self.model_type)

        real_index1 = self.filenames.index(key1)
        real_index2 = self.filenames.index(key2)
       
        new_sent_ix1 = self.get_best_caption_id(real_index1)
        new_sent_ix2 = self.get_best_caption_id(real_index2)

        cap1, mask1 = self.captions[new_sent_ix1], self.att_masks[new_sent_ix1]
        cap2, mask2 = self.captions[new_sent_ix2], self.att_masks[new_sent_ix2]

        return img1, img1_h, img2, img2_h, cap1, cap2, mask1, mask2,  attr_vec1, attr_vec2, pair_label

    def __len__(self):
        return len (self.imgs_pair)