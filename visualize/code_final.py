from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import torch 
import numpy as np 
from torchvision import transforms
from matplotlib import pyplot as plt 
import cv2, sys 
import os.path as osp
from transformers import  AutoTokenizer
from torch.autograd import Variable

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from types import SimpleNamespace
from utils.prepare import (prepare_arcface, prepare_text_encoder, prepare_image_head, prepare_fusion_net)

#48_1
def get_sent_emb(args):
    f = "She has high cheekbones, big lips, straight hair, black hair, rosy cheeks, and arched eyebrows. She wears lipstick. She is attractive, and smiling."
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    cap = f.encode('utf-8').decode('utf8')
    cap = cap.replace("\ufffd\ufffd", " ")
    
    encoding = tokenizer.encode_plus(
                cap,
                add_special_tokens=True,
                max_length = args.bert_words_num,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt')

    input_ids=encoding["input_ids"].flatten()
    mask=encoding["attention_mask"].flatten()

    caption = Variable(input_ids).cuda()
    mask = Variable(mask).cuda()

    caption = caption.unsqueeze(dim=0)
    mask = mask.unsqueeze(dim=0)  
    text_encoder, text_head = prepare_text_encoder(args)
    words_emb, sent_emb = text_encoder(caption, mask)
    words_emb, sent_emb = text_head(words_emb, sent_emb)
    return sent_emb


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



if __name__ == "__main__":
    args = SimpleNamespace()
    args.gpu_id = [0]
    args.device = "cuda"
    args.weights_arcface = "./weights/pretrained/arcface_ir18_ms1mv3.pth"

    args.num_workers = 1 
    args.bert_words_num = 32
    args.bert_config =  "bert-base-uncased"
    args.batch_size = 1
    args.bert_type = "bert"
    args.fusion_type = "linear"
    args.fusion_final_dim = 512
      
    args.gl_img_dim = 256
    args.gl_text_dim = 256


    args.split = "train" 
    arc_model = prepare_arcface(args)
    for name, param in arc_model.named_parameters():
        param.requires_grad = True 

    #arcface 
    trans = transforms.Compose([transforms.ToTensor()])
    img = Image.open("./data/celeba/images/train/62/62_8.jpg").convert("RGB")
    input_tensor = trans(img)
    input_tensor = input_tensor.unsqueeze(dim=0).cuda()
    targets = None #[ClassifierOutputTarget(281)]

    model = ArcFace(arc_model)
    target_layers = [model.model.layer4[-1]]

    cam = GradCAMPlusPlus(model=model, target_layers=target_layers) #, use_cuda=True
    
    #  aug_smooth=True and eigen_smooth=True
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    img =  np.divide(np.array(img), 255.0)
    visualization_arc = show_cam_on_image(img, grayscale_cam, use_rgb=True)


    ###################### Ours ####################
    models = [1, 2, 3, 4]
    #models = [8, 9, 10, 11, 12, 13, 14]
    visualization_tgfr = []
    for model_num in models:
        args.text_encoder_path = "./checkpoints/celeba/TGFR/BERT_arcface/linear/hobe_insh/encoder_BERT_linear_%d.pth" % model_num
        args.image_encoder_path = "./checkpoints/celeba/TGFR/BERT_arcface/linear/hobe_insh/fusion_linear_arcface_%d.pth" % model_num

        args.split = "test" 
        sent_emb = get_sent_emb(args)
        input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True).cuda()
        sent_emb = torch.autograd.Variable(sent_emb, requires_grad=True).cuda()
        
        x = (input_tensor, sent_emb)
        image_head = prepare_image_head(args)
        fusion_net = prepare_fusion_net(args) 

        image_encoder = prepare_arcface(args)
        for name, param in image_encoder.named_parameters():
            param.requires_grad = True 

            model_tfgr = ArcModel(image_encoder, image_head, fusion_net)
            target_layers = [model_tfgr.model.layer4[-1]]
            cam_tgfr = GradCAMPlusPlus(model=model_tfgr, target_layers=target_layers)

            grayscale_cam_tg = cam_tgfr(input_tensor=x, targets=targets)
            grayscale_cam_tg = grayscale_cam_tg[0, :]
            visualization_tgfr.append(show_cam_on_image(img, grayscale_cam_tg, use_rgb=True)) 

    f, axarr = plt.subplots(2, 3)
    axarr = axarr.flatten()
    axarr[0].imshow(img)
    axarr[1].imshow(visualization_arc)
    axarr[2].imshow(visualization_tgfr[0])
    axarr[3].imshow(visualization_tgfr[1])
    axarr[4].imshow(visualization_tgfr[2])
    axarr[5].imshow(visualization_tgfr[3])

    #plt.savefig("r/65_2_She is wearing necklace, and earrings. She has brown hair, rosy cheeks, arched eyebrows, and mouth slightly open. She is young, and smiling..svg")
    plt.show()