import os, sys, random
import os.path as osp
import argparse, itertools
import torch
import numpy as np
import pprint 

import torch
import torch.nn as nn
from tqdm import tqdm 
ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)

from utils.prepare import prepare_adaface, prepare_arcface
from cfg.config_space import face2text_cfg, celeba_cfg, celeba_dialog_cfg, setup_cfg, LFW_cfg
from types import SimpleNamespace

from utils.modules import test 
from models import metrics, losses 
from utils.dataset_utils import *
from utils.modules import calculate_scores

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class ClsDataset:
    def __init__(self, filenames, args=None):        
        self.data_dir = args.data_dir
        self.dataset_name = args.dataset
        self.model_type = args.model_type
        self.split = "train" 

        print("\n############## Loading %s dataset ################" % self.split)
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


class ValidDataset:
    def __init__(self, split="test", args=None):
        
        print("\n############## Loading %s dataset ################" % split)
        self.split= split
        self.data_dir = args.data_dir
        self.dataset_name = args.dataset
        self.model_type = args.model_type 

        #self.class_id = load_class_id(os.path.join(self.data_dir, self.split))
        if split == "test": self.pair_list = args.test_ver_list
        elif split == "valid": self.pair_list = args.valid_ver_list
        self.imgs_pair, self.pair_label = self.get_test_list()


    def get_test_list(self):
        with open(self.pair_list, 'r') as fd:
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


class Trainer:
    def __init__(self, args):
        self.args = args 
        self.args.device = my_device
        self.model_type = args.model_type
        self.get_data_loader()
  
        #self.total_steps = len(self.train_dl) * self.args.max_epoch
        
        print("Loading training and valid data ...")
        self.criterion = losses.FocalLoss(gamma=2)

        # building image encoder
        self.build_image_encoders() 
        self.get_optimizer() 
    

    def get_data_loader(self):
        train_filenames = load_filenames(self.args.data_dir, 'train')
        train_ds = ClsDataset(train_filenames, 
                            args = self.args)
        
        self.train_dl = torch.utils.data.DataLoader(
            train_ds, 
            batch_size=self.args.batch_size, 
            drop_last=True,
            num_workers=self.args.num_workers, 
            shuffle=True)
        

        valid_ds = ValidDataset("test", self.args)
        
        self.valid_dl = torch.utils.data.DataLoader(
            valid_ds, 
            batch_size=self.args.batch_size, 
            drop_last=False,
            num_workers=self.args.num_workers, 
            shuffle=False)



    def save_models(self):
        save_dir = os.path.join(self.args.weights_path, "finetuned", self.args.dataset)        
        os.makedirs(save_dir, exist_ok=True)

        name = 'image_%s_%d.pth' % (self.args.model_type, self.args.current_epoch)
        state_path = os.path.join(save_dir, name)
        state = {"image_encoder": self.image_encoder.state_dict()}
        torch.save(state, state_path)


    def build_image_encoders(self):
        if self.model_type == "arcface":
            self.image_encoder = prepare_arcface(self.args, train_mode="finetune") 
            self.image_cls = metrics.ArcMarginProduct(512, 
                                    self.args.num_classes, 
                                    s=30.0, 
                                    m=0.5, 
                                    easy_margin=False).to(my_device)
            
        elif self.model_type == "adaface":
            self.image_encoder = prepare_adaface(self.args, train_mode="finetune")  
    
            self.image_cls = metrics.AdaFace(512,
                                            self.args.num_classes,  
                                            m=0.9, 
                                            h=0.333,   
                                            s=40.0).to(my_device)


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
        model.eval()
        preds = []
        labels = []

        loop = tqdm(total = len(valid_dl))

        for step, data in enumerate(valid_dl, 0):
            img1, img2, pair_label = data 
            
            # upload to cuda
            img1 = img1.to(my_device)
            img2 = img2.to(my_device)
            pair_label = pair_label.to(my_device)

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

        if not args.is_ident: 
            calculate_scores(preds, labels, args)
        else:
            #calculate_identification_acc(preds, args)
            calculate_scores(preds, labels, args)


    def train_epoch(self):
        self.image_encoder.train()
        self.image_cls.train()

        epoch = self.args.current_epoch 
        total_length = len(self.train_dl) * self.args.batch_size
        total_loss = 0

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        loop = tqdm(total = len(self.train_dl))

        for imgs, label in  self.train_dl:   
            imgs = imgs.to(my_device)
            label = label.to(my_device)
        
            if self.model_type == "adaface":
                gl_img_features, words_features, norm = self.image_encoder(imgs)
                output = self.image_cls(gl_img_features, norm, label)
            else:
                gl_img_features, words_features = self.image_encoder(imgs)
                output = self.image_cls(gl_img_features, label)
    
            self.optimizer_cls.zero_grad()
            if epoch > self.args.freeze: self.optimizer_en.zero_grad() 
            loss =  self.criterion(output, label)

            total_loss += loss.item()
            loss.backward()

            if epoch > self.args.freeze: self.optimizer_en.step()
            self.optimizer_cls.step()

            # update loop information
            loop.update(1)
            loop.set_description(f'Training Epoch [{epoch}/{self.args.epochs}]')
            loop.set_postfix()

        loop.close()
        print(' | epoch {:3d} |' .format(self.args.current_epoch))
        print("Identity loss: {:5.4f} ".format(total_loss / total_length))

   

    def train(self):
        LR_change_seq = [6, 10]
        gamma = 0.75
        lr = 0.01

        for epoch in range(0, self.args.epochs):
            self.args.current_epoch = epoch

            self.train_epoch()
            
            if epoch  > self.args.save_interval:
                self.save_models()

            if epoch in LR_change_seq:
                lr = lr * gamma 
                for g in self.optimizer_cls.param_groups:
                    g['lr'] = lr 

                for k in self.optimizer_en.param_groups:
                    k['lr'] = lr

                print("Learning Rate change to: {:0.5f}".format(lr))

            if (epoch > self.args.valid_interval):
                print("\nLet's validate the model")
                self.test(self.valid_dl, self.image_encoder, args)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',          dest="train",       help='train the pretrained resent model',   action='store_true')
    parser.add_argument('--test',           dest="train",       help='evaluate the pretrained resent model',action='store_false')
    parser.set_defaults(train=False)

    parser.add_argument('--dataset',            type=str,   default="celeba",    help='Name of the datasets celeba|face2text|celeba_dialog')
    parser.add_argument('--batch_size',             type=int,   default=128,         help='Batch size')
    parser.add_argument('--epochs',             type=int,   default=36,          help='Number of epochs')

    parser.add_argument('--architecture',       type=str,   default="ir_18",     help='iResNet Architecture 18|50|101')
    parser.add_argument('--model_type',         type=str,   default="arcface",   help='architecture of the model: arcface | adaface | magface')
    parser.add_argument('--freeze',             type=int,   default=6,           help='Number of epoch pretrained model frezees')

    parser.add_argument('--save_interval',      type=int,   default=7,           help='saving intervals (epochs)')
    parser.add_argument('--valid_interval',     type=int,   default=7,           help='valid (epochs)')

    parser.add_argument('--checkpoint_path',    type=str,   default="./checkpoints", help='model directory')
    parser.add_argument('--weights_path',       type=str,   default="./weights/pretrained", help='model directory')

    parser.add_argument('--test_file',          type=str,   default="test_ver.txt",          help='Name of the test list file')
    parser.add_argument('--valid_file',         type=str,   default="valid_ver.txt",         help='Name of the test list file')

    return  parser.parse_args(argv)


# Face2Text dataset
face2text_cfg = SimpleNamespace(
    num_classes = 4500,
    captions_per_image = 4,
    test_sub = 1193 
)

# CelebA dataset
celeba_cfg = SimpleNamespace(
    num_classes= 4500, 
    captions_per_image= 10,
    test_sub = 1217
)


# CelebA-Dialog dataset
celeba_dialog_cfg = SimpleNamespace(
    num_classes= 8000, 
    captions_per_image = 1,
    test_sub = 1677
)


setup_cfg = SimpleNamespace(
    weights_adaface_18 = "./weights/pretrained/adaface_ir18_webface4m.ckpt",
    weights_arcface_18 = "./weights/pretrained/arcface_ir18_ms1mv3.pth", 

    metric = "arc_margin", 
    loss = "focal_loss", 
    use_se = False,
    manual_seed = 61,
    num_workers = 4,

    en_type = "BERT",        
    embedding_dim = 256,
    bert_type = "bert",

    bert_config=  "bert-base-uncased",
    align_config= "kakaobrain/align-base",
    clip_config= "openai/clip-vit-base-patch32",
    blip_config= "Salesforce/blip-image-captioning-base",
    is_ident = False,
)


if __name__ == "__main__":
    c_args = parse_arguments(sys.argv[1:])

    if c_args.dataset == "celeba":
        args = SimpleNamespace(**c_args.__dict__, **setup_cfg.__dict__, **celeba_cfg.__dict__)

    elif c_args.dataset == "face2text":
        args = SimpleNamespace(**c_args.__dict__, **setup_cfg.__dict__, **face2text_cfg.__dict__)
    
    elif c_args.dataset == "celeba_dialog":
        args = SimpleNamespace(**c_args.__dict__, **setup_cfg.__dict__, **celeba_dialog_cfg.__dict__)


    print("******** Dataset Name: %s ***** " % args.dataset)
    # set seed
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    torch.cuda.manual_seed_all(args.manual_seed)
    args.data_dir = os.path.join("./data", args.dataset)
    args.test_ver_list = os.path.join(args.data_dir, "images", args.test_file)
    args.valid_ver_list = os.path.join(args.data_dir, "images", args.valid_file)
    #pprint.pp(args)
    
    t = Trainer(args)
    print("start training ...")
    t.train()