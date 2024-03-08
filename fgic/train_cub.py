import os, sys, random
import os.path as osp
import torch
import numpy as np
import torch, torchvision
from torch import nn 
import torch.nn.functional as F 

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from resnet import resnet18, resnet50
from dataset import CUBDataset



my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TopLayer(nn.Module):  
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self._dropout = nn.Dropout(p=0.3).cuda()
        self._fc = nn.Linear(input_dim, num_classes).cuda()

    def forward(self, x):
        x = self.flatten(x)
        return self._fc(self._dropout(x))

        

def save_model(model, cls_model, epoch):
    save_dir = os.path.join("./checkpoints")

    name = 'model_epoch_%d.pth' % (epoch)
    state = {"model": model.state_dict(), 
             "cls":   cls_model.state_dict()}

    state_path = os.path.join(save_dir, name)
    torch.save(state, state_path)
    print("saving ... ", name)


def load_model(model, cls_model, model_name):
    print("loading model of my own.....")
    loading_dir = os.path.join("./weights/finetuned", model_name)

    checkpoint = torch.load(loading_dir)
    model.load_state_dict(checkpoint["model"])
    cls_model.load_state_dict(checkpoint["cls"])
    return model, cls_model  


def train_epoch(epoch, train_dl, model, cls_model, criterion, optimizer_model, optimizer_cls):
    model.train()
    cls_model.train()

    total_ce_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in train_dl:
        inputs, targets = inputs.cuda(), targets.cuda()

        gl_img, l_img = model(inputs)
        out = cls_model(gl_img)
        
        optimizer_model.zero_grad()
        optimizer_cls.zero_grad()

        ce_loss = criterion(out, targets)
        ce_loss.backward()

        if epoch > 6: optimizer_model.step()
        optimizer_cls.step()

        total_ce_loss += ce_loss.item()

        _, predicted = torch.max(out.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()

    train_acc = correct/total

    print('Epoch %d, train_acc = %.5f, CE loss %.7f' % (epoch + 1, train_acc, total_ce_loss / total))



def valid_epoch(test_dl, model, model_cls, criterion):
    model.eval()
    model_cls.eval()
    correct = 0
    total = 0
    total_ce_loss = 0
    
    with torch.no_grad():
        for inputs, targets in test_dl:
            inputs, targets = inputs.cuda(), targets.cuda()
            gl_feat, l_feat = model(inputs)
            out = model_cls(gl_feat)

            ce_loss = criterion(out, targets)
            total_ce_loss += ce_loss.item()

            _, predicted = torch.max(out.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()

    test_acc = correct/total
    print('test_acc = %.5f,  valid loss %.7f\n' %  (test_acc, total_ce_loss / total))


def main():
    lr_model = 0.0002
    lr_cls = 0.001
    max_epoch = 36
    #student_model = 'efficientnet-b0', 'efficientnet-b4'
    do_valid = True
    save_interval = 1
    valid_interval = 1 
    b_size = 128

    model = resnet50().to(my_device)
    cls_model = TopLayer(input_dim=2048, num_classes=200).to(my_device)

    train_ds = CUBDataset(train=True)
    test_ds = CUBDataset(train = False) 

    optimizer_model = torch.optim.Adam(model.parameters(), 
                             lr = lr_model, 
                             betas=(0.9, 0.99), 
                             weight_decay=5e-5)
    
    optimizer_cls = torch.optim.Adam(cls_model.parameters(), 
                             lr = lr_cls, 
                             betas=(0.9, 0.99), 
                             weight_decay=6e-4)
    
    ls_model = torch.optim.lr_scheduler.ExponentialLR(optimizer_model, gamma=0.97)
    ls_cls = torch.optim.lr_scheduler.ExponentialLR(optimizer_cls, gamma=0.95)
    criterion = torch.nn.CrossEntropyLoss()

    train_dl = torch.utils.data.DataLoader(
        train_ds, 
        batch_size = b_size, 
        drop_last = False,
        num_workers = 4, 
        shuffle = True)
    
    test_dl = torch.utils.data.DataLoader(
        test_ds, 
        batch_size = b_size, 
        drop_last = False,
        num_workers = 4, 
        shuffle = False)

    model, cls_model = load_model(model, cls_model, "resnet50_cub.pth")
    valid_epoch(test_dl, model, cls_model, criterion)

    """
    for epoch in range(0, max_epoch):

        train_epoch(epoch, train_dl, model, cls_model, criterion, optimizer_model, optimizer_cls)
        ls_cls.step()
        if epoch > 6: ls_model.step()

        if (epoch % save_interval==0) and (epoch > 28) :
            save_model(model, cls_model, epoch)


        if ((do_valid == True) and (epoch % valid_interval == 0) and (epoch > 2)):
            valid_epoch(test_dl, model, cls_model, criterion)
    """

if __name__ == '__main__':
    main() 