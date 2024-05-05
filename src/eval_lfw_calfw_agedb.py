import os, sys, random
import os.path as osp
import argparse
import numpy as np

import torch
import torch.nn as nn
import pprint
from tqdm import tqdm 
from datetime import datetime 

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)

from utils.prepare import prepare_adaface, prepare_arcface, prepare_magface
from types import SimpleNamespace
from utils.dataset_utils import *
from utils.modules import calculate_acc

os.environ["TOKENIZERS_PARALLELISM"] = "false"
my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TestDataset:
    def __init__(self, args=None):
        self.split= "test"
        self.data_dir = args.data_dir
        self.dataset_name = args.dataset
        self.model_type = args.model_type 

        print("\n############## Loading %s dataset ################" % self.split)
        self.imgs_pair, self.pair_label = self.get_test_list(args.test_ver_list)


    def get_test_list(self, test_ver_list):
        with open(test_ver_list, 'r') as fd:
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

        img1_name = imgs[0] #os.path.join(imgs[0].split("_")[0], imgs[0])
        img2_name = imgs[1] #os.path.join(imgs[1].split("_")[0], imgs[1])

        img1_path = os.path.join(data_dir, self.split, img1_name)
        img2_path = os.path.join(data_dir, self.split, img2_name)

        img1 = get_imgs(img1_path, self.split, self.model_type)
        img2 = get_imgs(img2_path, self.split, self.model_type)

        img1_h = do_flip_test_images(img1_path, self.model_type)
        img2_h = do_flip_test_images(img2_path, self.model_type)

        return img1, img2, img1_h, img2_h, pair_label


    def __len__(self):
        return len (self.imgs_pair)


class Evaluate:
    def __init__(self, args):
        self.test_ds = TestDataset(args)
        self.test_dl = self.get_data_loader()
        args.device = my_device

        if args.model_type == "adaface":
            self.model = prepare_adaface(args, train_mode="fixed")
            
        elif args.model_type == "arcface":
            self.model = prepare_arcface(args, train_mode="fixed")

        elif args.model_type == "magface":
            self.model = prepare_magface(args, train_mode="fixed") 


    def get_data_loader(self,):
        return torch.utils.data.DataLoader(
                self.test_ds, 
                batch_size = args.batch_size, 
                drop_last = False,
                num_workers = 4, 
                shuffle = False)
   

    def test(self, args):
        self.model.eval()
        preds = []
        labels = []

        loop = tqdm(total = len(self.test_dl))
        cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

        with torch.no_grad():
            for step, data in enumerate(self.test_dl, 0):
                img1, img2, img1_h, img2_h, pair_label = data 
                
                img1 = img1.to(my_device)
                img2 = img2.to(my_device)

                img1_h = img1_h.to(my_device)
                img2_h = img2_h.to(my_device)
                pair_label = pair_label.to(my_device)

                # get global and local image features from COTS model
                if args.model_type == "arcface" or args.model_type == "magface":
                    global_feat1,  _ = self.model(img1)
                    global_feat2,  _ = self.model(img2)

                    global_feat1_h,  _ = self.model(img1_h)
                    global_feat2_h,  _ = self.model(img2_h)

                elif args.model_type == "adaface":
                    global_feat1,  _, norm = self.model(img1)
                    global_feat2,  _, norm = self.model(img2)

                    global_feat1_h,  _, norm = self.model(img1_h)
                    global_feat2_h,  _, norm = self.model(img2_h)

                gf1 = torch.cat((global_feat1, global_feat1_h), dim=1)
                gf2 = torch.cat((global_feat2, global_feat2_h), dim=1)

                pred = cosine_sim(gf1, gf2)
                preds += pred.data.cpu().tolist()
                labels += pair_label.data.cpu().tolist()

                # update loop information
                loop.update(1)
                loop.set_postfix()

        loop.close()
        calculate_acc(preds, labels, args)



setup_cfg = SimpleNamespace(
    weights_adaface_18 = "./weights/pretrained/adaface_ir18_webface4m.ckpt",
    weights_adaface_50 = "./weights/pretrained/adaface_ir50_ms1mv2.ckpt",
    weights_adaface_101 = "./weights/pretrained/adaface_ir101_ms1mv2.ckpt",

    weights_arcface_18 = "./weights/pretrained/arcface_ir18_ms1mv3.pth", 
    weights_arcface_50 = "./weights/pretrained/arcface_ir50_ms1mv3.pth", 
    weights_arcface_101= "./weights/pretrained/arcface_ir101_ms1mv3.pth", 

    weights_magface_50 = "./weights/pretrained/magface_ir50_ms1mv2.pth", 
    weights_magface_101 ="./weights/pretrained/magface_ir101_ms1mv2.pth", 

    metric = "arc_margin", 
    easy_margin = False,
    loss = "focal_loss", 
    use_se = False,

    manual_seed= 61
)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',       type=str,   default="LFW",               help='Name of the datasets LFW | CALFW | AGEDB')
    parser.add_argument('--batch_size',    type=int,   default=64,                  help='Batch size')
    parser.add_argument('--architecture',  type=str,   default="ir_50",             help='iResNet Architecture 18|50|101')
    parser.add_argument('--model_type',    type=str,   default="arcface",           help='architecture of the model: arcface | adaface | magface')
    parser.add_argument('--test_file',     type=str,   default="test_pairs.txt",    help='Name of the test list file')

    return  parser.parse_args(argv)



if __name__ == "__main__":
    c_args = parse_arguments(sys.argv[1:])
    args = SimpleNamespace(**c_args.__dict__, **setup_cfg.__dict__)


    print("************* Dataset Name : ", args.dataset)
    # set seed
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    torch.cuda.manual_seed_all(args.manual_seed)
    args.data_dir = os.path.join("./data", args.dataset)
    args.test_ver_list = os.path.join(args.data_dir, "images", args.test_file)
    #pprint.pp(args)
    
    eval = Evaluate(args)
    eval.test(args)


"""
RUN THE CODE
python3 src/eval_lfw_calfw_agedb.py  --architecture ir_101 --model_type adaface --dataset CALFW
"""