import sys
import os.path as osp
import random
import argparse
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from PIL import Image
import torch 
import numpy as np 
from torchvision import transforms
from matplotlib import pyplot as plt 

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from types import SimpleNamespace

from utils.prepare import (prepare_arcface, prepare_adaface, prepare_image_head, 
                           prepare_text_encoder, prepare_fusion_net, prepare_train_val_loader)

from utils.dataset_utils import encode_Bert_tokens


class ArcModel(torch.nn.Module):
    def __init__(self, image_encoder, image_head, fusion_net):
        super(ArcModel, self).__init__()
        self.model = image_encoder.module
        self.head = image_head.module
        self.fusion_net = fusion_net.module
        
    def forward(self, x):
        img, sent_emb = x[0], x[1]
        img, local_feats = self.model(img)
        img, local_feats = self.head(img, local_feats)
        out = self.fusion_net(local_feats, img, sent_emb)
        return out 


class ArcFace(torch.nn.Module):
    def __init__(self, image_encoder):
        super(ArcFace, self).__init__()
        self.model = image_encoder.module
        
    def forward(self, x):
        img, local_feats = self.model(x)
        return img  


class Test:
    def __init__(self, args):
        self.args = args 
        self.train_dl, self.valid_dl = prepare_train_val_loader(self.args)
        self.args.len_train_dl = len(self.train_dl)

        # preapare model
        self.text_encoder, self.text_head = prepare_text_encoder(self.args)
        
        if self.args.model_type == "arcface":
            self.image_encoder = prepare_arcface(self.args) 
            
        elif self.args.model_type == "adaface":
            self.image_encoder = prepare_adaface(self.args)

        for name, param in self.image_encoder.named_parameters():
            print(name)
        self.image_head = prepare_image_head(self.args)
        self.fusion_net = prepare_fusion_net(self.args) 


    def main(self):
        for  data in  self.train_dl:   
            imgs, caps, masks, keys, cap_attr, attr_vec, class_ids = data
            words_emb, sent_emb = encode_Bert_tokens(self.text_encoder, self.text_head, 
                                                caps, masks)


            model_arc = ArcFace(self.image_encoder).cuda()
            target_layers = [model_arc.model.layer4[-1]] 
            cam_arc =  GradCAM(model=model_arc, target_layers=target_layers)

            model = ArcModel(self.image_encoder, self.image_head, self.fusion_net).cuda()
            target_layers = [model.model.layer4[-1]] 
            cam_tgfr = GradCAM(model=model, target_layers=target_layers)
                    

            #img = Image.open("./src/44_1.jpg").convert("RGB")
            #trans = transforms.Compose([transforms.ToTensor()])
            #input_tensor = trans(img)
            #input_tensor = input_tensor.unsqueeze(dim=0).cuda()
            imgs = torch.autograd.Variable(imgs, requires_grad=True).cuda()
            sent_emb = torch.autograd.Variable(sent_emb, requires_grad=True).cuda()
            

            targets = None #[ClassifierOutputTarget(281)]
            # ploting original image
            dis_org_img = imgs[0].detach().cpu().numpy()
            dis_org_img = dis_org_img.transpose((1, 2, 0))

            # ploting arcface image
            # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
            grayscale_cam = cam_arc(input_tensor=imgs, targets=targets)
            grayscale_cam = grayscale_cam[0, :]

            dis_arc_img = imgs[0].detach().cpu().numpy()
            vis_arc_img = show_cam_on_image(dis_arc_img.transpose((1, 2, 0)), 
                                            grayscale_cam, use_rgb=True)


            # ploing the output of our TGFR model
            x = (imgs, sent_emb)
            grayscale_cam = cam_tgfr(input_tensor=x, targets=targets)
            grayscale_cam = grayscale_cam[0, :]

            print(grayscale_cam.size())
            print(grayscale_cam)
            dis_tgfr_img = imgs[0].detach().cpu().numpy()
            vis_tgfr_img = show_cam_on_image(dis_tgfr_img.transpose((1, 2, 0)), 
                                            grayscale_cam, use_rgb=True)
            
            f, axarr = plt.subplots(1,3) 
            axarr[0].imshow(dis_org_img)
            axarr[1].imshow(vis_arc_img)
            axarr[2].imshow(vis_tgfr_img)
            #plt.show()


if __name__ == "__main__":
    args = SimpleNamespace(
        weights_adaface = "./weights/pretrained/adaface_ir18_webface4m.ckpt",
        weights_arcface = "./weights/pretrained/arcface_ir18_ms1mv3.pth", #celeba_caption.pt",
        text_encoder_path= "./checkpoints/celeba/TGFR/BERT_arcface/linear/encoder_BERT_linear_16.pth",
        image_encoder_path= "./checkpoints/celeba/TGFR/BERT_arcface/linear/fusion_linear_arcface_16.pth",

        en_type= "BERT",
        bert_type = "bert",
        dataset_name = "celeba",
        data_dir = "./data/celeba", 
        valid_pair_list= "./data/celeba/images/valid_199_sub.txt", 
        num_classes = 4500, 
        bert_words_num = 32,
        captions_per_image= 10,
        img_size= 112,
        ch_size= 3,

        # fusion arch
        fusion_type = "linear",
        split = "valid",
        fusion_final_dim = 512,   
        gl_img_dim = 256,
        gl_text_dim = 256,
        model_type = "arcface", 
        device = ("cuda" if torch.cuda.is_available() else "cpu"),
        bert_config =  "bert-base-uncased", #distilbert-base-uncased 
        blip_config = "Salesforce/blip-image-captioning-base",

        # machine setup
        num_workers = 1, 
        batch_size = 1,
        gpu_id= [0], 
        manual_seed= 100,
        CUDA= True
    )

    Test(args)#.main()