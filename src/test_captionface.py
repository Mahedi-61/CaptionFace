import os, sys, random
import os.path as osp
import argparse
import numpy as np
import torch

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from types import SimpleNamespace
from utils.prepare import (prepare_test_loader, 
                           prepare_arcface, prepare_adaface, prepare_image_head, prepare_image_text_attr,
                           prepare_text_encoder, 
                           prepare_fusion_net)
from utils.modules import test

os.environ["TOKENIZERS_PARALLELISM"] = "false"
my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Test:
    def __init__(self, args):
        self.args = args 
        args.device = my_device
        self.test_dl = prepare_test_loader(args)

        # preapare model
        self.text_encoder, self.text_head = prepare_text_encoder(self.args)
        
        print("loading image encoder: ", args.image_encoder_path)
        if self.args.model_type == "arcface":
            self.image_encoder = prepare_arcface(self.args, train_mode="fixed") 
            
        elif self.args.model_type == "adaface":
            self.image_encoder = prepare_adaface(self.args, train_mode="fixed")

        self.image_text_attr = prepare_image_text_attr(self.args)
        self.image_head = prepare_image_head(self.args)
        self.fusion_net = prepare_fusion_net(self.args) 

    def eval(self):
        #pprint.pprint(self.args)
        print("\nLet's test the model")
        test(self.test_dl, self.image_encoder, self.image_head, self.image_text_attr, 
             self.fusion_net, self.text_encoder, self.text_head, args)
    

# Face2Text dataset
face2text_cfg = SimpleNamespace(
    data_dir = "./data/face2text",  
    test_ver_acc_list= "./data/face2text/images/test_ver_acc.txt",
    num_classes = 4500,
    bert_words_num = 40, 
    captions_per_image = 4,
    test_sub = 1193 
)

# CelebA dataset
celeba_cfg = SimpleNamespace(
    data_dir= "./data/celeba",  
    test_ver_acc_list= "./data/celeba/images/test_ver_acc.txt",
    num_classes= 4500, 
    bert_words_num = 32,
    captions_per_image= 10,
    test_sub = 1217
)


# CelebA-Dialog dataset
celeba_dialog_cfg = SimpleNamespace(
    data_dir= "./data/celeba_dialog",  
    test_ver_acc_list= "./data/celeba_dialog/images/test_ver_acc.txt",
    num_classes= 8000, 
    bert_words_num = 32,
    captions_per_image = 1,
    test_sub = 1677
)


setup_cfg = SimpleNamespace(
    weights_adaface_18 = "./weights/pretrained/adaface_ir18_webface4m.ckpt",
    weights_arcface_18 = "./weights/pretrained/arcface_ir18_ms1mv3.pth", 

    metric = "arc_margin", 
    easy_margin = False,
    loss = "focal_loss", 
    use_se = False,
    manual_seed = 61,
    num_workers = 4,

    en_type = "BERT",        
    embedding_dim = 256,
    gl_img_dim = 256,
    gl_text_dim = 256,
    bert_type = "bert",

    bert_config=  "bert-base-uncased",
    align_config= "kakaobrain/align-base",
    clip_config= "openai/clip-vit-base-patch32",
    blip_config= "Salesforce/blip-image-captioning-base",
    falva_config= "facebook/flava-full",
    architecture = "ir_18"
)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_ident',       dest="is_ident",      help='identification',   action='store_true')
    parser.add_argument('--do_test',        dest="do_test",       help='perform test',     action='store_true')
    parser.add_argument('--printing_attr',   dest="printing_attr",  help='perform test',     action='store_true')
    parser.set_defaults(is_ident=False)
    parser.set_defaults(do_test=True)
    parser.set_defaults(printing_attr=False)

    parser.add_argument('--dataset',       type=str,   default="celeba",                help='Name of the datasets celeba | face2text | celeba_dialog')
    parser.add_argument('--batch_size',    type=int,   default=128,                      help='Batch size')
    parser.add_argument('--model_type',    type=str,   default="arcface",               help='architecture of the model: arcface | adaface | magface')
    parser.add_argument('--test_file',     type=str,   default="test_ver.txt",          help='Name of the test list file')
    parser.add_argument('--valid_file',    type=str,   default="valid_ver.txt",         help='Name of the test list file')

    parser.add_argument('--fusion_final_dim',   type=int,   default=576,     help='Final fusion dimension')
    parser.add_argument('--freeze',             type=int,   default=5,      help='Number of epoch pretrained model frezees')
    parser.add_argument('--fusion_type',        type=str,   default="CMF_FR",  help='Type of Fusion block CMF|linear')
    
    parser.add_argument('--checkpoint_path',    type=str,   default="./checkpoints", help='model directory')
    parser.add_argument('--weights_path',       type=str,   default="./weights/pretrained", help='model directory')
    parser.add_argument('--text_encoder',  type=str,   default="text_arcface_BERT_CMF_20.pth", help='text encoder file')
    parser.add_argument('--image_encoder', type=str,   default="image_arcface_BERT_CMF_20.pth", help='image encoder file')

    return  parser.parse_args(argv)


if __name__ == "__main__":
    c_args = parse_arguments(sys.argv[1:])

    if c_args.dataset == "celeba":
        args = SimpleNamespace(**c_args.__dict__, **setup_cfg.__dict__, **celeba_cfg.__dict__)

    elif c_args.dataset == "face2text":
        args = SimpleNamespace(**c_args.__dict__, **setup_cfg.__dict__, **face2text_cfg.__dict__)
    
    elif c_args.dataset == "celeba_dialog":
        args = SimpleNamespace(**c_args.__dict__, **setup_cfg.__dict__, **celeba_dialog_cfg.__dict__)

    print("************* Dataset Name : ", args.dataset)
    # set seed
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    torch.cuda.manual_seed_all(args.manual_seed)
    args.data_dir = os.path.join("./data", args.dataset)
    args.test_ver_list = os.path.join(args.data_dir, "images", args.test_file)
    args.valid_ver_list = os.path.join(args.data_dir, "images", args.valid_file)

    args.text_encoder_path = os.path.join(args.checkpoint_path, args.dataset, "CaptionFace", args.text_encoder)
    args.image_encoder_path = os.path.join(args.checkpoint_path, args.dataset, "CaptionFace", args.image_encoder)
    #pprint.pp(args)
    
    t = Test(args)
    print("start testing ...")
    t.eval()

"""
RUN THE CODE
python3 src/test_captionface.py  --model_type adaface --dataset celeba
"""