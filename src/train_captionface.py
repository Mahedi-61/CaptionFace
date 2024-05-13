import os, sys, random
import os.path as osp
import argparse, itertools
import torch
import numpy as np
import pprint 
import torch.nn as nn
from torch.nn import functional as F 
from torch.autograd import Variable
from tqdm import tqdm 
from types import SimpleNamespace
from transformers import get_linear_schedule_with_warmup
ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)

from models.losses import sent_loss, words_loss, CMPLoss 
from utils.prepare import (prepare_train_val_loader, prepare_test_loader, 
                           prepare_adaface, prepare_arcface)

from models.fusion_nets import LinearFusion, FCFM, CMF, CMF_FR
from utils.modules import test 
from models.models import (TextEncoder, TextHeading, ImageHeading)
from models import metrics, losses 
from utils.dataset_utils import encode_Bert_tokens

os.environ["TOKENIZERS_PARALLELISM"] = "false"
my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer:
    def __init__(self, args):
        self.args = args 
        args.device = my_device
        self.model_type = args.model_type

        # prepare dataloader
        self.train_dl, self.valid_dl = prepare_train_val_loader(self.args)

        self.args.len_train_dl = len(self.train_dl)
        self.total_steps = len(self.train_dl) * self.args.max_epoch

        if self.args.is_CMP == True:
            self.cmpm_loss = CMPLoss(num_classes = self.args.num_classes, 
                                    feature_dim = self.args.gl_img_dim)
            self.cmpm_loss.to(my_device)

        # building image encoder
        self.build_image_encoders() 
        
        # building text encoder
        self.build_text_encoders()

        # building fusion net
        self.build_fusion_net() 

        # set the optimizer
        self.get_optimizer() 

    
    def save_models(self):
        save_dir = os.path.join(self.args.checkpoint_path, self.args.dataset,  "CaptionFace")        
        os.makedirs(save_dir, exist_ok=True)

        name = 'image_%s_%s_%s_%d.pth' % (self.args.model_type, 
                                          self.args.en_type, 
                                          self.args.fusion_type, 
                                          self.args.current_epoch)
        state_path = os.path.join(save_dir, name)

        state = {'net': self.fusion_net.state_dict(), 
                 "image_encoder": self.image_encoder.state_dict(),
                 'image_head': self.image_head.state_dict(),
                 'attr_head': self.image_text_attr.state_dict()
                }
        
        torch.save(state, state_path)
        checkpoint_text_en = {
            'model': self.text_encoder.state_dict(),
            'head': self.text_head.state_dict()
        }

        torch.save(checkpoint_text_en, '%s/text_%s_%s_%s_%d.pth' % 
                    (save_dir, self.args.model_type,self.args.en_type, 
                               self.args.fusion_type, self.args.current_epoch))


    def print_losses(self, s_total_loss, w_total_loss, 
                     total_cl_loss, total_cmp_loss, 
                     total_idn_loss, total_attr_loss, total_itc_loss, 
                     total_kd_loss, f_loss, total_length):
        
        print(' | epoch {:3d} |' .format(self.args.current_epoch))
        if self.args.is_DAMSM == True:
            total_damsm_loss = (s_total_loss + w_total_loss) / total_length
            s_total_loss = s_total_loss / total_length
            w_total_loss = w_total_loss / total_length
            print('s_loss {:5.5f} | w_loss {:5.5f} | DAMSM loss {:5.5}'.format(s_total_loss, w_total_loss, total_damsm_loss)) 

        if self.args.is_CLIP == True:
            print("Clip loss: {:5.6f} ".format(total_cl_loss / total_length))

        if self.args.is_CMP == True: 
            print("CMP Mloss: {:5.6f} ".format(total_cmp_loss / total_length))            

        if self.args.is_attr_loss == True: 
            print("Attribute loss: {:5.6f} ".format(total_attr_loss / total_length))

        if self.args.is_itc == True: 
            print("ITC loss: {:5.6f} ".format(total_itc_loss / total_length))

        if self.args.is_KD == True: 
            print("KD loss: {:5.6f} ".format(total_kd_loss)) # / total_length

        print("Identity loss: {:5.6f} ".format(total_idn_loss / total_length))
        print("Fusion loss: {:5.6f} ".format(f_loss / total_length))


    def build_image_encoders(self):
        if self.model_type == "arcface":
            self.image_encoder = prepare_arcface(self.args, train_mode="fixed")

        elif self.model_type == "adaface":
            self.image_encoder = prepare_adaface(self.args, train_mode="fixed")
        
        self.image_head = ImageHeading(self.args).to(my_device)
            
 
    def build_text_encoders(self):
        self.text_encoder = TextEncoder(self.args).to(my_device)
        self.text_head = TextHeading(self.args).to(my_device)
        
        self.image_text_attr = metrics.TopLayer(
                                self.args.gl_text_dim, 
                                40).to(my_device)
        
        self.image_text_cls = metrics.TopLayer(
                                self.args.gl_img_dim, 
                                self.args.num_classes).to(my_device)


    def build_fusion_net(self):
        if self.args.fusion_type == "linear":
            self.fusion_net = LinearFusion(self.args).to(my_device)

        elif self.args.fusion_type == "CMF_FR": 
            self.fusion_net = CMF_FR(args).to(my_device)

        elif self.args.fusion_type == "fcfm": 
            self.fusion_net = FCFM(args.gl_img_dim).to(my_device)

        self.metric_fc = metrics.TopLayer(512, #self.args.fusion_final_dim, 
                        self.args.num_classes).to(my_device)
        

    def get_optimizer(self):
        if self.args.is_CMP == True:
            params_align = [
                {"params": itertools.chain(self.text_head.parameters(),
                                        self.image_head.parameters(),
                                        self.image_text_cls.parameters(),
                                        self.image_text_attr.parameters(),
                                        self.cmpm_loss.parameters()),
                "lr": 0.001} 

            ]
        elif self.args.is_CMP == False:
            params_align = [
                {"params": itertools.chain(self.text_head.parameters(),
                                        self.image_head.parameters(),
                                        self.image_text_cls.parameters(),
                                        self.image_text_attr.parameters()),

                "lr": 0.001}
            ] 

        params_fusion = [
            {"params": itertools.chain(self.metric_fc.parameters(),
                                        self.fusion_net.parameters()),
                "lr": 0.002}
            ]
        self.optimizer = torch.optim.AdamW(self.text_encoder.parameters(),
                                eps=1e-8,
                                lr=self.args.min_lr_bert)
        
        self.optimizer_align = torch.optim.Adam(params_align)  
        self.optimizer_fusion = torch.optim.Adam(params_fusion) 

        self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                num_warmup_steps= int(self.total_steps * 0.01), #1% of total training
                                num_training_steps=self.total_steps)

        self.lrs_align = torch.optim.lr_scheduler.ExponentialLR(
                                        self.optimizer_align, 
                                        gamma=0.98)
        
        self.lrs_fusion = torch.optim.lr_scheduler.ExponentialLR(
                                        self.optimizer_fusion, 
                                        gamma=0.98)


    def get_identity_loss(self, sent_emb, img_features, class_ids):
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        class_ids = class_ids.to(my_device)

        # for text branch
        output = self.image_text_cls(sent_emb)
        tid_loss = criterion(output, class_ids)

        # for image branch (parameter sharing)
        output = self.image_text_cls(img_features)
        iid_loss = criterion(output, class_ids)

        loss = self.args.lambda_id * (tid_loss + iid_loss) / 2
        self.idn_loss += loss.item() 
        return loss
    

    def get_cap_attr_loss(self, sent_emb, img_features, cap_attr, attr_vec):
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")
        criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
        cap_attr = cap_attr.to(my_device)
        attr_vec = attr_vec.to(my_device)

        cap_attr.requires_grad_()
        attr_vec.requires_grad_()

        # for text branch
        #output = self.image_text_attr(sent_emb)
        #ta_loss = criterion(output, cap_attr)

        # for image branch (parameter sharing)
        output = self.image_text_attr(img_features)
        ia_loss = bce_loss(output, attr_vec)

        total_attr_loss = self.args.lambda_attr *  ia_loss #(ta_loss + ia_loss) / 2
        self.attr_loss += total_attr_loss.item() 
        return total_attr_loss


    def get_DAMSM_loss(self, sent_emb, words_emb, img_features, words_features, class_ids):
        batch_size = self.args.batch_size 
        labels = self.prepare_labels(self.args.batch_size)
        cap_lens = None 
    
        w_loss0, w_loss1, attn_maps = words_loss(words_features, words_emb, labels, 
                    cap_lens, class_ids.numpy(), batch_size, self.args)

        s_loss0, s_loss1 = sent_loss(img_features, sent_emb, labels, 
                            class_ids.numpy(), batch_size, self.args)

        total_damsm_loss = w_loss0 + w_loss1 + s_loss0 + s_loss1 
        self.w_loss += ((w_loss0 + w_loss1).data).item()
        self.s_loss += ((s_loss0 + s_loss1).data).item()
        return total_damsm_loss


    def get_clip_loss(self, sent_emb, img_features):
        #clip_loss = global_loss(img_features, sent_emb) 
        clip_loss = losses.clip_loss(sent_emb, img_features, self.args) 

        total_cl_loss = self.args.lambda_clip * clip_loss
        self.cl_loss += total_cl_loss.item()
        return total_cl_loss


    def get_itc_loss(self, sent_emb, img_features, logit_scale):
        itc_loss = losses.compute_itc(img_features, sent_emb, logit_scale)
        
        total_itc_loss = self.args.lambda_itc * itc_loss
        self.itc_loss += total_itc_loss.item()
        return total_itc_loss


    def get_CMP_loss(self, sent_emb, img_features, class_ids):    
        class_ids = class_ids.to(my_device)
        total_cmp_loss, _ = self.cmpm_loss(sent_emb, img_features, class_ids)
        self.cmp_loss += self.args.lambda_cmp *  total_cmp_loss.item()
        return self.args.lambda_cmp * total_cmp_loss 


    def get_CPA(self, sent_emb, img_features):
        T = 5
        soft_targets = nn.functional.softmax(img_features / T, dim=-1)
        soft_prob = nn.functional.log_softmax(sent_emb / T, dim=-1)

        # Calculate the true label loss
        KL_loss = nn.KLDivLoss(reduction="batchmean", log_target=False)
        total_kd_loss = self.args.lambda_kd * KL_loss(soft_prob, soft_targets)
        self.kd_loss += total_kd_loss.item()
        return total_kd_loss


    def prepare_labels(self, batch_size):
        match_labels = Variable(torch.LongTensor(range(batch_size)))
        return match_labels.to(my_device)


    def get_fusion_loss(self, local_feats, words_emb, global_feats, sent_emb, gl_img, class_ids):
        fusion_loss = losses.FocalLoss(gamma=2) #torch.nn.CrossEntropyLoss() 

        output = self.fusion_net(local_feats, words_emb, global_feats, sent_emb, gl_img)
        output = self.metric_fc(output)
  
        loss = (args.lambda_f * fusion_loss(output, class_ids.to(my_device)))
        self.f_loss += loss.item() 
        return loss   


    def train_epoch(self):
        self.text_encoder.train()
        self.text_head.train() 
        self.image_text_cls.train()
        self.image_text_attr.train()
        self.image_head.train()
        self.fusion_net.train()

        epoch = self.args.current_epoch 
        total_length = len(self.train_dl) * self.args.batch_size

        self.s_loss = 0
        self.w_loss = 0
        
        self.cl_loss = 0
        self.cmp_loss = 0
        self.idn_loss = 0
        self.attr_loss = 0
        self.itc_loss = 0
        self.kd_loss = 0
        self.f_loss = 0
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        loop = tqdm(total = len(self.train_dl))

        for  data in  self.train_dl:   
            imgs, caps, masks, keys, cap_attr, attr_vec, class_ids = data
            words_emb, sent_emb = encode_Bert_tokens(self.text_encoder, self.text_head, caps, masks, self.args)

            if self.model_type == "adaface":
                gl_img, local_img, norm = self.image_encoder(imgs.to(my_device))
            else:
                gl_img, local_img = self.image_encoder(imgs.to(my_device))

            img_features, local_features = self.image_head(gl_img, local_img)
            
            self.optimizer.zero_grad()
            self.optimizer_align.zero_grad()
            self.optimizer_fusion.zero_grad()
            total_loss = 0
            
            if self.args.is_DAMSM == True:
                 total_loss += self.get_DAMSM_loss(sent_emb, words_emb, img_features, local_features, class_ids)

            if self.args.is_attr_loss == True: 
                total_loss += self.get_cap_attr_loss(sent_emb, img_features, cap_attr, attr_vec)

            if self.args.is_CLIP == True:
                total_loss += self.get_clip_loss(sent_emb, img_features)

            # cross-modal projection loss
            if self.args.is_CMP == True: 
                total_loss += self.get_CMP_loss(sent_emb, img_features, class_ids)

            if self.args.is_itc == True:
                total_loss += self.get_itc_loss(sent_emb, img_features, self.logit_scale)

            if self.args.is_KD == True:
                total_loss += self.get_CPA(sent_emb, img_features)

            # fusion & identity loss
            total_loss += self.get_identity_loss(sent_emb, img_features, class_ids)
            total_loss += self.get_fusion_loss(local_features, words_emb, 
                                               img_features, sent_emb, gl_img, class_ids) 
            
            # update
            total_loss.backward()
            if (epoch > self.args.freeze): self.optimizer.step()
            self.optimizer_align.step()
            self.optimizer_fusion.step()

            #`clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.    
            torch.nn.utils.clip_grad_norm_(itertools.chain(self.text_encoder.parameters()), 
                                                           self.args.clip_max_norm)

            # update loop information
            loop.update(1)
            loop.set_description(f'Training Epoch [{epoch}/{self.args.max_epoch}]')
            loop.set_postfix()
        
        loop.close()
        self.print_losses(self.s_loss, self.w_loss, self.cl_loss, self.cmp_loss, 
                self.idn_loss, self.attr_loss, self.itc_loss, self.kd_loss, self.f_loss, total_length)
       

    def train(self):
        for epoch in range(0, self.args.max_epoch):
            self.args.current_epoch = epoch

            self.train_epoch()
            if (epoch > self.args.freeze): self.lr_scheduler.step()
            self.lrs_align.step()
            self.lrs_fusion.step()

            if epoch > self.args.save_interval:
                print("saving image, text encoder, and fusion block\n")
                self.save_models()
            
            if epoch > self.args.test_interval:
                print("\nLet's validate the model")
                test(self.valid_dl, 
                    self.image_encoder, self.image_head, self.image_text_attr,
                    self.fusion_net, 
                    self.text_encoder, self.text_head, 
                    self.args)


face2text_cfg = SimpleNamespace(
    data_dir = "./data/face2text",  
    test_ver_acc_list= "./data/face2text/images/test_ver_acc.txt",
    num_classes = 4500,
    bert_words_num = 40, 
    captions_per_image = 4,
    test_sub = 1193 
)


celeba_cfg = SimpleNamespace(
    data_dir= "./data/celeba",  
    test_ver_acc_list= "./data/celeba/images/test_ver_acc.txt",
    num_classes= 4500, 
    bert_words_num = 32,
    captions_per_image= 10,
    test_sub = 1217
)


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
    blip_config= "Salesforce/blip-image-captioning-base",
    falva_config= "facebook/flava-full",
    is_CLIP = False ,
    is_CMP = False,
    architecture = "ir_18",        

    temperature = 2.0,
    GAMMA1 = 4.0,  
    GAMMA2 = 5.0,
    GAMMA3 = 10.0,

    clip_max_norm = 1.0,
    is_ident = False,
    split = "train"
)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--printing_attr',   dest="printing_attr",  help='perform test',     action='store_true')
    parser.set_defaults(do_test=True)
    parser.set_defaults(printing_attr=False)

    parser.add_argument('--is_itc',         dest="is_itc",        help='including image-text cls. loss',     action='store_true')
    parser.add_argument('--is_KD',          dest="is_KD",         help='including knowledge dist. loss',     action='store_true')
    parser.add_argument('--is_attr_loss',   dest="is_attr_loss",  help='including attribute loss',           action='store_true')
    parser.add_argument('--is_DAMSM',       dest="is_DAMSM",      help='including DAMSM loss',               action='store_true')
    parser.set_defaults(is_itc=False)
    parser.set_defaults(is_DAMSM=False)
    parser.set_defaults(is_KD=False)
    parser.set_defaults(is_attr_loss=False)

    parser.add_argument('--dataset',       type=str,   default="celeba",                help='Name of the datasets celeba | face2text | celeba_dialog')
    parser.add_argument('--batch_size',    type=int,   default=8,                       help='Batch size')
    parser.add_argument('--max_epoch',     type=int,   default=18,                      help='Maximum epochs')
    parser.add_argument('--model_type',    type=str,   default="arcface",               help='architecture of the model: arcface | adaface')
    parser.add_argument('--test_file',     type=str,   default="test_ver.txt",          help='Name of the test list file')
    parser.add_argument('--valid_file',    type=str,   default="valid_ver.txt",         help='Name of the test list file')

    parser.add_argument('--fusion_final_dim',   type=int,   default=576,     help='Final fusion dimension')
    parser.add_argument('--freeze',             type=int,   default=4,      help='Number of epoch pretrained model frezees')
    parser.add_argument('--fusion_type',        type=str,   default="CMF_FR",  help='Type of Fusion block CMF|linear')
    
    parser.add_argument('--checkpoint_path',    type=str,   default="./checkpoints", help='checkpoints directory')
    parser.add_argument('--weights_path',       type=str,   default="./weights/pretrained", help='pretrained model directory')

    parser.add_argument('--lambda_f',       type=float,   default=3,    help='weight value of the fusion loss')
    parser.add_argument('--lambda_itc',     type=float,   default=1,    help='weight value of the ITC loss')
    parser.add_argument('--lambda_cmp',     type=float,   default=1,    help='weight value of the identity loss')
    parser.add_argument('--lambda_kd',      type=float,   default=1,   help='weight value of the KD loss')
    parser.add_argument('--lambda_attr',    type=float,   default=12,   help='weight value of the attribute loss')
    parser.add_argument('--lambda_clip',    type=float,   default=1,   help='weight value of the KD loss')
    parser.add_argument('--lambda_id',      type=float,   default=1,   help='weight value of the attribute loss')

    parser.add_argument('--save_interval',     type=int,   default=13,    help='saving intervals (epochs)')
    parser.add_argument('--test_interval',     type=int,   default=17,    help='tester intervals (epochs)')
    parser.add_argument('--min_lr_bert',       type=float,  default=0.00003,    help='minimum learning rate for bert optimization')
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
    
    t = Trainer(args)
    print("start training ...")
    t.train()

"""
RUN THE CODE
python3 src/test_tgfr.py  --model_type adaface --dataset celeba
"""