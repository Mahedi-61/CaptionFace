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

This woman has high cheekbones, bangs, oval face, and rosy cheeks and wears lipstick. She is smiling.

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from types import SimpleNamespace
from utils.prepare import prepare_arcface

class ArcFace(torch.nn.Module):
    def __init__(self, image_encoder):
        super(ArcFace, self).__init__()
        self.model = image_encoder.module
        
    def forward(self, x):
        img, local_feats = self.model(x)
        return img  

args = SimpleNamespace()
args.gpu_id = [0]
args.device = "cuda"
args.weights_arcface = "./weights/pretrained/arcface_ir18_ms1mv3.pth"

image_encoder = prepare_arcface(args) 
trans = transforms.Compose([transforms.ToTensor()])
model = ArcFace(image_encoder)
target_layers = [model.model.layer4[-1]]

cam = GradCAMPlusPlus(model=model, target_layers=target_layers) #, use_cuda=True
img = Image.open("./src/4500_2.jpg").convert("RGB")
input_tensor = trans(img)
input_tensor = input_tensor.unsqueeze(dim=0).cuda()

targets = None #[ClassifierOutputTarget(281)]


# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
img =  np.divide(np.array(img), 255.0)
visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)

f, axarr = plt.subplots(1, 2)
axarr[0].imshow(img)
axarr[1].imshow(visualization)

#plt.imshow(visualization, cmap='jet')
plt.savefig("r/code.svg")