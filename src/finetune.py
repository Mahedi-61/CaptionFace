import os, sys, random
import os.path as osp
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm 
import itertools
from datetime import datetime 
ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)

from utils.utils import mkdir_p,  merge_args_yaml
from utils.prepare import prepare_adaface, prepare_arcface
from cfg.config_space import face2text_cfg, celeba_cfg, celeba_dialog_cfg, setup_cfg, LFW_cfg
from types import SimpleNamespace

from utils.modules import test 
from models import metrics, losses 
from utils.dataset_utils import *
from utils.modules import (calculate_scores, calculate_identification_acc)

today = datetime.now() 
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    cfg_file = "train_bert.yml"
    print("loading %s" % cfg_file)
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--cfg', dest='cfg_file', type=str, 
                        default='./cfg/%s' %cfg_file)
    
    file_args = parser.parse_args()
    return file_args


class ClsDataset(torch.utils.data.Dataset):
    def __init__(self, filenames, split="train", args=None):

        print("\n############## Loading %s dataset ################" % split)
        self.data_dir = args.data_dir
        self.dataset_name = args.dataset_name
        self.model_type = args.model_type
        self.split = split 

        self.filenames = filenames
        split_dir = os.path.join(self.data_dir, self.split)
        self.class_id = load_class_id(split_dir)


    def __getitem__(self, index):
        key = self.filenames[index]
        cls_id = self.class_id[index]
        
        img_extension = ".jpg" # works for all dataset 
        img_name = os.path.join(self.data_dir, "images", self.split, key + img_extension)
        
        imgs = get_imgs(img_name, self.split, self.model_type)

        return imgs, cls_id 

    def __len__(self):
        return len(self.filenames)


class ValidDataset(torch.utils.data.Dataset):
    def __init__(self, split="test", args=None):
        
        print("\n############## Loading %s dataset ################" % split)
        self.split= split
        self.data_dir = args.data_dir
        self.dataset_name = args.dataset_name
        self.model_type = args.model_type 

        #self.class_id = load_class_id(os.path.join(self.data_dir, self.split))
        self.valid_pair_list = args.test_pair_list
        self.imgs_pair, self.pair_label = self.get_test_list()


    def get_test_list(self):
        with open(self.valid_pair_list, 'r') as fd:
            pairs = fd.readlines()
        imgs_pair = []
        pair_label = []
        for pair in pairs:
            splits = pair.split(" ")
            imgs = [splits[0], splits[1]]
            imgs_pair.append(imgs)
            pair_label.append(int(splits[2]))
        return imgs_pair, pair_label
    

    def __getitem__(self, index):
        imgs = self.imgs_pair[index]
        pair_label = self.pair_label[index]

        data_dir = os.path.join(self.data_dir, "images")

        img1_name = os.path.join(imgs[0].split("_")[0], imgs[0])
        img2_name = os.path.join(imgs[1].split("_")[0], imgs[1])

        img1_path = os.path.join(data_dir, self.split, img1_name)
        img2_path = os.path.join(data_dir, self.split, img2_name)

        img1 = get_imgs(img1_path, self.split, self.model_type)
        img2 = get_imgs(img2_path, self.split, self.model_type)

        return img1, img2, pair_label


    def __len__(self):
        return len (self.imgs_pair)



def get_data_loader(args, split):
    if split == "train":
        train_filenames = load_filenames(args.data_dir, 'train')
        train_ds = ClsDataset(train_filenames, 
                            split="train", 
                            args=args)
        
        dl = torch.utils.data.DataLoader(
            train_ds, 
            batch_size=args.batch_size, 
            drop_last=True,
            num_workers=args.num_workers, 
            shuffle=True)
        
    elif split == "valid":
        valid_ds = ValidDataset("test", args)
        
        dl = torch.utils.data.DataLoader(
            valid_ds, 
            batch_size=args.batch_size, 
            drop_last=False,
            num_workers=args.num_workers, 
            shuffle=False)
    return dl



class Train:
    def __init__(self, args):
        self.args = args 
        self.device = args.device
        self.model_type = args.model_type

        # prepare dataloader
        #self.train_dl = get_data_loader(self.args, split="train")
        self.valid_dl = get_data_loader(self.args, split="valid")
        #self.total_steps = len(self.train_dl) * self.args.max_epoch
        
        print("Loading training and valid data ...")
        self.criterion = losses.FocalLoss(gamma=2)

        # building image encoder
        self.build_image_encoders() 
        

        # set the optimizer
        self.get_optimizer() 

        #resume checkpoint
        self.start_epoch = 1
    

    def save_models(self):
        save_dir = os.path.join(self.args.checkpoints_path, 
                                self.args.dataset_name,  
                                self.args.en_type + "_" + self.args.model_type + "_" + self.args.architecture,
                                self.args.fusion_type,
                                today.strftime("%m-%d-%y-%H:%M"))
        mkdir_p(save_dir)

        name = '%s_%d.pth' % (args.model_type, self.args.current_epoch )
        
        state_path = os.path.join(save_dir, name)
        state = {"image_encoder": self.image_encoder.state_dict()}
        torch.save(state, state_path)


    def build_image_encoders(self):
        if self.model_type == "arcface":
            self.image_encoder = prepare_arcface(self.args, train_mode="fixed") #finetune
            self.image_cls = metrics.ArcMarginProduct(self.args.fusion_final_dim, 
                                    self.args.num_classes, 
                                    s=30.0, 
                                    m=0.5, 
                                    easy_margin=False)
            
        elif self.model_type == "adaface":
            self.image_encoder = prepare_adaface(self.args, train_mode="fixed")  #finetune
    
            self.image_cls = metrics.AdaFace(self.args.fusion_final_dim,
                                            self.args.num_classes,  
                                            m=0.9, 
                                            h=0.333,   
                                            s=40.0)
 
        self.image_cls = torch.nn.DataParallel(self.image_cls, 
                    device_ids=self.args.gpu_id).to(self.args.device)


    def get_optimizer(self):
        params_cls = [{"params": self.image_cls.parameters(), 
                      "lr" : 0.01, 
                      "weight_decay" : 0.0005}]
        
        self.optimizer_cls = torch.optim.SGD(params_cls, momentum=0.9)

        params_en = [{"params": self.image_encoder.parameters(), 
                      "lr" : 0.01, 
                      "weight_decay" : 0.0005}]
        
        self.optimizer_en = torch.optim.SGD(params_en, momentum=0.9)



    def test(self, valid_dl, model, args):
        device = args.device
        model.eval()
        preds = []
        labels = []

        loop = tqdm(total = len(valid_dl))

        for step, data in enumerate(valid_dl, 0):
            img1, img2, pair_label = data 
            
            # upload to cuda
            img1 = img1.to(device)
            img2 = img2.to(device)
            pair_label = pair_label.to(device)

            # get global and local image features from COTS model
            if args.model_type == "arcface" or args.model_type == "magface":
                global_feat1,  local_feat1 = model(img1)
                global_feat2,  local_feat2 = model(img2)

            elif args.model_type == "adaface":
                global_feat1,  local_feat1, norm = model(img1)
                global_feat2,  local_feat2, norm = model(img2)

            cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
            pred = cosine_sim(global_feat1, global_feat2)
            preds += pred.data.cpu().tolist()
            labels += pair_label.data.cpu().tolist()

            # update loop information
            loop.update(1)
            loop.set_postfix()

        loop.close()

        #a = ["pred: " + str(i) + " true: " + str(j) for i, j in zip(preds, labels)]
        if not args.is_ident: 
            calculate_scores(preds, labels, args)
        else:
            #calculate_identification_acc(preds, args)
            calculate_scores(preds, labels, args)


    def train(self):
        self.image_encoder.train()
        self.image_cls.train()

        epoch = self.args.current_epoch 
        total_length = len(self.train_dl) * self.args.batch_size
        total_loss = 0
        correct = 0

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        loop = tqdm(total = len(self.train_dl))

        for imgs, label in  self.train_dl:   
            imgs = imgs.to(self.device)
            label = label.to(self.device)
        
            if self.model_type == "adaface":
                gl_img_features, words_features, norm = self.image_encoder(imgs)
                output = self.image_cls(gl_img_features, norm, label)
            else:
                gl_img_features, words_features = self.image_encoder(imgs)
                output = self.image_cls(gl_img_features, label)
    
            self.optimizer_cls.zero_grad()
            if epoch > 8: self.optimizer_en.zero_grad() 
            loss =  self.criterion(output, label)

            total_loss += loss.item()
            loss.backward()

            if epoch > 8: self.optimizer_en.step()
            self.optimizer_cls.step()

            out_ind = torch.argmax(output, axis=1)
            correct += sum(out_ind == label)

            # update loop information
            loop.update(1)
            loop.set_description(f'Training Epoch [{epoch}/{self.args.max_epoch}]')
            loop.set_postfix()

        loop.close()
        print(' | epoch {:3d} |' .format(self.args.current_epoch))
        print("Identity loss: {:5.4f} ".format(total_loss / total_length))
        acc = correct / total_length
        print("accuracy {:5.4f} ".format(acc*100))
   

    def main(self):
        LR_change_seq = [6, 10]
        gamma = 0.75
        lr = 0.01

        self.test(self.valid_dl, self.image_encoder, args)
        """
        for epoch in range(self.start_epoch, self.args.max_epoch + 1):
            self.args.current_epoch = epoch

            self.train()
            
            if epoch % self.args.save_interval==0:
                self.save_models()

            if epoch in LR_change_seq:
                lr = lr * gamma 
                for g in self.optimizer_cls.param_groups:
                    g['lr'] = lr 

                for k in self.optimizer_en.param_groups:
                    k['lr'] = lr

                print("Learning Rate change to: {:0.5f}".format(lr))

            if (epoch > 8 and self.args.do_test == True):
                print("\nLet's validate the model")
                self.test(self.valid_dl, self.image_encoder, args)
        """


if __name__ == "__main__":
    file_args = merge_args_yaml(parse_args())
    args = SimpleNamespace(**file_args.__dict__, **setup_cfg.__dict__)

    if args.dataset_name == "face2text": 
        args =  SimpleNamespace(**face2text_cfg.__dict__, **args.__dict__)
    
    elif args.dataset_name == "celeba":
        args =  SimpleNamespace(**celeba_cfg.__dict__, **args.__dict__)
    
    elif args.dataset_name == "celeba_dialog":
        args  = SimpleNamespace(**celeba_dialog_cfg.__dict__, **args.__dict__)

    elif args.dataset_name == "LFW":
        args  = SimpleNamespace(**LFW_cfg.__dict__, **args.__dict__)
    else:
        print("Error: New Dataset !!, dataset doesn't have config file!!")

    # set seed
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    torch.cuda.manual_seed_all(args.manual_seed)
    args.batch_size = 64
    Train(args).main()