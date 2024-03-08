import sys
import os.path as osp
import random
import argparse
import numpy as np
import torch

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from cfg.config_space import face2text_cfg, celeba_cfg, celeba_dialog_cfg, setup_cfg
from types import SimpleNamespace
from utils.utils import merge_args_yaml
from utils.prepare import (prepare_test_loader, 
                           prepare_arcface, prepare_adaface, prepare_image_head, prepare_image_text_attr,
                           prepare_text_encoder, 
                           prepare_fusion_net)
from utils.modules import test


def parse_args():
    # Training settings
    print("loading celeba.yml")
    cfg_file = "test.yml"
    parser = argparse.ArgumentParser(description='Testing TGFR model')
    parser.add_argument('--cfg', dest='cfg_file', type=str, 
                        default='./cfg/%s' % cfg_file,
                        help='optional config file')
    args = parser.parse_args()
    return args


class Test:
    def __init__(self, args):
        self.args = args 
        self.test_dl = prepare_test_loader(args)

        # preapare model
        self.text_encoder, self.text_head = prepare_text_encoder(self.args)
        
        if self.args.model_type == "arcface":
            self.image_encoder = prepare_arcface(self.args, train_mode="fixed") 
            
        elif self.args.model_type == "adaface":
            self.image_encoder = prepare_adaface(self.args, train_mode="fixed")

        if self.args.printing_attr == True: 
            self.image_text_attr = prepare_image_text_attr(self.args)

        self.image_head = prepare_image_head(self.args)
        self.fusion_net = prepare_fusion_net(self.args) 

    
    def main(self):
        #pprint.pprint(self.args)
        print("\nLet's test the model")
        test(self.test_dl, 
            self.image_encoder, self.image_head, self.image_text_attr,
            self.fusion_net, 
            self.text_encoder, self.text_head, 
            self.args)
    

if __name__ == "__main__":
    file_args = merge_args_yaml(parse_args())
    args = SimpleNamespace(**file_args.__dict__, **setup_cfg.__dict__)

    if args.dataset_name == "face2text": 
        args =  SimpleNamespace(**face2text_cfg.__dict__, **args.__dict__)
    
    elif args.dataset_name == "celeba":
        args =  SimpleNamespace(**celeba_cfg.__dict__, **args.__dict__)
    
    elif args.dataset_name == "celeba_dialog":
        args  = SimpleNamespace(**celeba_dialog_cfg.__dict__, **args.__dict__)
    else:
        print("Error: New Dataset !!, dataset doesn't have config file!!")


    # set seed
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    torch.cuda.manual_seed_all(args.manual_seed)

    Test(args).main()