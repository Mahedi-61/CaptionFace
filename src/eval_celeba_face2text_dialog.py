import os, sys, random
import os.path as osp
import argparse
import numpy as np
import torch
from torch import nn 
from tqdm import tqdm 
import pprint 
ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from types import SimpleNamespace
from utils.prepare import prepare_test_loader, prepare_arcface, prepare_adaface, prepare_magface
from utils.modules import calculate_scores, calculate_acc,  do_dis_plot
os.environ["TOKENIZERS_PARALLELISM"] = "false"
my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class Evaluate:
    def __init__(self, args):
        self.test_dl = prepare_test_loader(args)
        args.device = my_device

        if args.model_type == "adaface":
            self.model = prepare_adaface(args, train_mode="fixed")
            
        elif args.model_type == "arcface":
            self.model = prepare_arcface(args, train_mode="fixed")


    def test(self, args):
        self.model = self.model.eval()
        preds = []
        labels = []

        loop = tqdm(total = len(self.test_dl))
        cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

        with torch.no_grad():
            for step, data in enumerate(self.test_dl, 0):
                img1, img1_h, img2, img2_h,  caption1, caption2, mask1, mask2, attr_vec1, attr_vec2, pair_label = data 
                
                # upload to cuda
                img1 = img1.to(my_device)
                img2 = img2.to(my_device)
                img1_h = img1_h.to(my_device)
                img2_h = img2_h.to(my_device)
                pair_label = pair_label.to(my_device)

                # get global and local image features from COTS model
                if args.model_type == "arcface":
                    global_feat1,  _ = self.model(img1)
                    global_feat2,  _ = self.model(img2)

                    global_feat1_h,  _ = self.model(img1_h)
                    global_feat2_h,  _ = self.model(img2_h)

                elif args.model_type == "adaface":
                    global_feat1,  _, norm = self.model(img1)
                    global_feat2,  _, norm = self.model(img2)

                    global_feat1_h,  _, norm = self.model(img1_h)
                    global_feat2_h,  _, norm = self.model(img2_h)

                #gf1 = torch.cat((global_feat1, global_feat1_h), dim=1)
                #gf2 = torch.cat((global_feat2, global_feat2_h), dim=1)

                pred = cosine_sim(global_feat1, global_feat2)
                preds += pred.data.cpu().tolist()
                labels += pair_label.data.cpu().tolist()

                # update loop information
                loop.update(1)
                loop.set_postfix()

        loop.close()

        if not args.is_ident: 
            #do_dis_plot(preds, labels, args)
            calculate_scores(preds, labels, args)
        else:
            calculate_acc(preds, labels, args)


# Face2Text dataset
face2text_cfg = SimpleNamespace(
    data_dir = "./data/face2text",  
    test_ver_acc_list= "./data/face2text/images/test_ver_acc.txt",
    num_classes = 4500,
    bert_words_num = 48, 
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
    bert_type = "bert",

    bert_config=  "bert-base-uncased",
    align_config= "kakaobrain/align-base",
    clip_config= "openai/clip-vit-base-patch32",
    blip_config= "Salesforce/blip-image-captioning-base",

    is_ident = False,
    architecture = "ir_18"
)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',       type=str,   default="celeba",                help='Name of the datasets celeba | face2text | celeba_dialog')
    parser.add_argument('--batch_size',    type=int,   default=128,                      help='Batch size')
    parser.add_argument('--model_type',    type=str,   default="arcface",               help='architecture of the model: arcface | adaface | magface')
    parser.add_argument('--test_file',     type=str,   default="test_ver.txt",          help='Name of the test list file')
    parser.add_argument('--valid_file',    type=str,   default="valid_ver.txt",         help='Name of the test list file')

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
    #pprint.pp(args)
    
    eval = Evaluate(args)
    print("start testing ...")
    eval.test(args)

"""
RUN THE CODE
python3 src/eval_celeba_face2text_dialog.py  --model_type adaface --dataset celeba
"""