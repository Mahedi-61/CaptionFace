import os, sys
import os.path as osp
import torch
import numpy as np
from torch import nn 

import argparse, sys 
import pprint

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from fgic_models import get_resnet_model, TopLayer
from dataset import CUBDataset


class Trainer:
    def __init__(self, config):
        self.config = config
        self.my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = get_resnet_model(config).to(self.my_device)
        embed_dim = 512 if self.config.resnet_layer == 18 else 2048
        self.cls_model = TopLayer(input_dim=embed_dim, 
                                num_classes=self.config.num_classes).to(self.my_device)

        train_ds = CUBDataset(train = True, args = config) 
        test_ds = CUBDataset(train = False, args = config)  

        self.train_dl = torch.utils.data.DataLoader(
            train_ds, 
            batch_size = config.b_size, 
            drop_last = False,
            num_workers = 4, 
            shuffle = True)
        
        self.test_dl = torch.utils.data.DataLoader(
            test_ds, 
            batch_size = config.b_size, 
            drop_last = False,
            num_workers = 4, 
            shuffle = False)

        self.optimizer_model = torch.optim.Adam(self.model.parameters(), 
                                lr = config.lr_model, 
                                betas=(0.9, 0.99), 
                                weight_decay=5e-5)
        
        self.optimizer_cls = torch.optim.Adam(self.cls_model.parameters(), 
                                lr = config.lr_cls, 
                                betas=(0.9, 0.99), 
                                weight_decay=6e-4)
        
        self.lrs_model = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_model, gamma=0.97)
        self.lrs_cls = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_cls, gamma=0.95)
        self.criterion = torch.nn.CrossEntropyLoss()


    def save_model(self, epoch):
        name = 'model_epoch_%d.pth' % (epoch)
        state = {"model": self.model.state_dict(), 
                "cls":   self.cls_model.state_dict()}

        state_path = os.path.join(self.config.model_path, name)
        torch.save(state, state_path)
        print("saving ... ", name)


    def load_model(self,):
        print("\nloading finetuned resent model .....")
        loading_dir = os.path.join(self.config.model_path, self.config.saved_model_file)
        checkpoint = torch.load(loading_dir)

        self.model.load_state_dict(checkpoint["model"])
        self.cls_model.load_state_dict(checkpoint["cls"])


    def valid_epoch(self, ):
        self.model.eval()
        self.cls_model.eval()
        correct = 0
        total = 0
        total_ce_loss = 0
        
        with torch.no_grad():
            for inputs, targets, _ in self.test_dl:
                inputs, targets = inputs.to(self.my_device), targets.to(self.my_device)
                gl_feat, l_feat = self.model(inputs)
                out = self.cls_model(gl_feat)

                ce_loss = self.criterion(out, targets)
                total_ce_loss += ce_loss.item()

                _, predicted = torch.max(out.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum().item()

        test_acc = correct/total
        print('test_acc = %.5f,  valid loss %.7f\n' %  (test_acc, total_ce_loss / total))


    def train_epoch(self, epoch):
        self.model.train()
        self.cls_model.train()

        total_ce_loss = 0
        correct = 0
        total = 0
        
        for inputs, _, targets, _ in self.train_dl:
            inputs, targets = inputs.to(self.my_device), targets.to(self.my_device)

            gl_img, l_img = self.model(inputs)
            out = self.cls_model(gl_img)
            
            self.optimizer_model.zero_grad()
            self.optimizer_cls.zero_grad()

            ce_loss = self.criterion(out, targets)
            ce_loss.backward()

            if epoch > self.config.freeze: self.optimizer_model.step()
            self.optimizer_cls.step()

            total_ce_loss += ce_loss.item()

            _, predicted = torch.max(out.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()

        train_acc = correct/total
        print('Epoch %d, train_acc = %.5f, CE loss %.7f' % (epoch + 1, train_acc, total_ce_loss / total))


    def train(self, ):
        for epoch in range(1, self.config.epochs):

            self.train_epoch(epoch)
            self.lrs_cls.step()
            if epoch > self.config.freeze: self.lrs_model.step()

            if (epoch % config.save_interval==0 and epoch > 28) :
                self.save_model(epoch)


            if (epoch % config.valid_interval == 0 and epoch > 2):
                self.valid_epoch()
        

    def test(self, ):
        self.model.eval()
        self.cls_model.eval()
        correct = 0
        total = 0
        self.load_model()
        
        with torch.no_grad():
            for inputs, _, targets, _ in self.test_dl:
                inputs, targets = inputs.to(self.my_device), targets.to(self.my_device)
                gl_feat, l_feat = self.model(inputs)
                out = self.cls_model(gl_feat)

                _, predicted = torch.max(out.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum().item()

        test_acc = correct/total
        print('test_acc = %.5f' %  test_acc)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',          dest="train",       help='train the pretrained resent model',   action='store_true')
    parser.add_argument('--test',           dest="train",       help='evaluate the pretrained resent model',action='store_false')
    parser.add_argument('--is_valid',       dest="is_valid",     help='Employing attribute loss', action='store_true')

    parser.set_defaults(train=False)
    parser.set_defaults(is_valid=False)

    parser.add_argument('--dataset',            type=str,   default="cub",    help='Name of the datasets')
    parser.add_argument('--b_size',             type=int,   default=128,         help='Batch size')
    parser.add_argument('--epochs',             type=int,   default=36,          help='Number of epochs')
    parser.add_argument('--saved_model_file',   type=str,   default="resnet18_cub.pth", help='The resent model to load for test')

    parser.add_argument('--lr_model',           type=float, default=0.0002,  help='Learning rate during pretrained resnet model')
    parser.add_argument('--lr_cls',             type=float, default=0.001,   help='Learning rate of the classifier')
    parser.add_argument('--captions_per_image', type=int,   default=10,      help='Captions per image')

    parser.add_argument('--resnet_layer',       type=int,   default=18,     help='Number of ResNet layers 18|50')
    parser.add_argument('--freeze',             type=int,   default=6,      help='Number of epoch pretrained model frezees')

    parser.add_argument('--num_classes',        type=float, default=200,    help='Number of classes')
    parser.add_argument('--model_path',         type=str,   default="./weights/fgic", help='model directory')
    parser.add_argument('--weights_path',       type=str,   default="./weights/fgic", help='pretrained weights directory')

    parser.add_argument('--save_interval',      type=int,   default=1,    help='saving intervals (epochs)')
    parser.add_argument('--valid_interval',     type=int,   default=1,    help='valid (epochs)')

    parser.add_argument('--train_imgs_list',  type=str,   default="./data/cub/train_images_shuffle.txt", help='train image list of CUB dataset')
    parser.add_argument('--test_imgs_list',  type=str,    default="./data/cub/test_images_shuffle.txt",  help='test image list of CUB dataset')
       
    return  parser.parse_args(argv)



if __name__ == '__main__':
    config = parse_arguments(sys.argv[1:])
    pprint.pp(config)

    t = Trainer(config)

    if config.train == True:
        t.train()

    elif config.train == False:
        t.test()

"""
RUN THE CODE
python3 fgic/train_fgic_resnet.py --test --dataset cub --resnet_layer 18 --saved_model_file resnet18_cub.pth
"""