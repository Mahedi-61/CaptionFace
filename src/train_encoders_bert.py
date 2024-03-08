import os, sys, random
import os.path as osp
import pprint
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm 
import itertools
from datetime import datetime 
from transformers import get_linear_schedule_with_warmup
ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)

from utils.utils import mkdir_p,  merge_args_yaml
from models.losses import sent_loss, words_loss, CMPLoss, global_loss 
from utils.prepare import (prepare_train_val_loader, prepare_test_loader, prepare_adaface, prepare_arcface)
from cfg.config_space import face2text_cfg, celeba_cfg, celeba_dialog_cfg, setup_cfg
from types import SimpleNamespace

from models.fusion_nets import LinearFusion, FCFM, CMF
from utils.modules import test 
from models.models import (TextEncoder, TextHeading, ImageHeading)
from models import metrics, losses 
from utils.dataset_utils import encode_Bert_tokens

today = datetime.now() 
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    cfg_file = "train_bert.yml"
    print("loading %s" % cfg_file)
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--cfg', dest='cfg_file', type=str, 
                        default='./cfg/%s' %cfg_file)
    
    file_args = parser.parse_args()
    return file_args


class Train:
    def __init__(self, args):
        self.args = args 
        self.model_type = args.model_type

        # prepare dataloader
        self.train_dl, self.valid_dl = prepare_train_val_loader(self.args)
        self.valid_dl = prepare_test_loader(args)

        self.args.len_train_dl = len(self.train_dl)
        self.total_steps = len(self.train_dl) * self.args.max_epoch

        if self.args.is_CMP == True:
            self.cmpm_loss = CMPLoss(num_classes = self.args.num_classes, 
                                    feature_dim = self.args.gl_img_dim)
            self.cmpm_loss.to(self.args.device)
        
        print("Loading training and valid data ...")

        # building image encoder
        self.build_image_encoders() 
        
        # building text encoder
        self.build_text_encoders()

        # building fusion net
        self.build_fusion_net() 

        # set the optimizer
        self.get_optimizer() 

        #resume checkpoint
        self.start_epoch = 1
    

    def save_models(self):
        save_dir = os.path.join(self.args.checkpoints_path, 
                                self.args.dataset_name,  
                                self.args.en_type + "_" + self.args.model_type + "_" + self.args.architecture,
                                self.args.fusion_type,
                                today.strftime("%m-%d-%y-%H:%M"))
        mkdir_p(save_dir)

        name = 'fusion_%s_%s_%d.pth' % (args.fusion_type,
                                        args.model_type, 
                                        self.args.current_epoch )
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

        torch.save(checkpoint_text_en, '%s/encoder_%s_%s_%d.pth' % 
                    (save_dir, args.en_type, args.fusion_type, self.args.current_epoch ))


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

        if self.args.is_ident_loss == True: 
            print("Identity loss: {:5.6f} ".format(total_idn_loss / total_length))

        if self.args.is_attr_loss == True: 
            print("Attribute loss: {:5.6f} ".format(total_attr_loss / total_length))

        if self.args.is_ITC == True: 
            print("ITC loss: {:5.6f} ".format(total_itc_loss / total_length))

        if self.args.is_KD == True: 
            print("KD loss: {:5.6f} ".format(total_kd_loss)) # / total_length

        print("Fusion loss: {:5.6f} ".format(f_loss / total_length))


    def build_image_encoders(self):
        if self.model_type == "arcface":
            self.image_encoder = prepare_arcface(self.args, train_mode="my_own")

        elif self.model_type == "adaface":
            self.image_encoder = prepare_adaface(self.args, train_mode="fixed")
        
        self.image_head = ImageHeading(self.args)
        self.image_head = nn.DataParallel(self.image_head, 
                                device_ids=self.args.gpu_id).to(self.args.device)
            
 
    def build_text_encoders(self):
        self.text_encoder = TextEncoder(self.args)
        self.text_encoder = nn.DataParallel(self.text_encoder, 
                                device_ids=self.args.gpu_id).to(self.args.device)
        
        self.text_head = TextHeading(self.args)
        self.text_head = nn.DataParallel(self.text_head, 
                                device_ids=self.args.gpu_id).to(self.args.device)

        
        self.image_text_attr = metrics.Classifier(
                                self.args.gl_text_dim, 
                                40)
    
        self.image_text_attr = torch.nn.DataParallel(
                                self.image_text_attr, 
                                device_ids=self.args.gpu_id).to(self.args.device)
        
        self.image_text_cls = metrics.Classifier(
                                self.args.gl_img_dim, 
                                self.args.num_classes)
        
        self.image_text_cls = torch.nn.DataParallel(
                                self.image_text_cls, 
                                device_ids=self.args.gpu_id).to(self.args.device)


        """
        self.text_cls = metrics.Classifier(
                                self.args.gl_text_dim, 
                                self.args.num_classes)
        
        self.text_cls = torch.nn.DataParallel(
                                self.text_cls, 
                                device_ids=self.args.gpu_id).to(self.args.device)
        """


    def build_fusion_net(self):
        if self.args.fusion_type == "linear":
            self.fusion_net = LinearFusion(self.args)

        elif self.args.fusion_type == "CMF": 
            self.fusion_net = CMF(args)

        elif self.args.fusion_type == "fcfm": 
            self.fusion_net = FCFM(args.gl_img_dim)


        self.fusion_net = torch.nn.DataParallel(self.fusion_net, 
                                device_ids=self.args.gpu_id).to(self.args.device)

        self.metric_fc = metrics.Classifier(self.args.fusion_final_dim, 
                        self.args.num_classes)
        
        self.metric_fc = torch.nn.DataParallel(self.metric_fc, 
                                device_ids=self.args.gpu_id).to(self.args.device)


    def get_optimizer(self):
        if self.args.is_CMP == True:
            params_align = [
                {"params": itertools.chain(self.text_head.parameters(),
                                        self.image_head.parameters(),
                                        self.image_text_cls.parameters(),
                                        #self.text_cls.parameters(),
                                        self.image_text_attr.parameters(),
                                        self.cmpm_loss.parameters()),
                "lr": self.args.lr_head} 

            ]
        elif self.args.is_CMP == False:
            params_align = [
                {"params": itertools.chain(self.text_head.parameters(),
                                        self.image_head.parameters(),
                                        self.image_text_cls.parameters(),
                                        #self.text_cls.parameters(),
                                        self.image_text_attr.parameters()),

                "lr": self.args.lr_head}
            ] 

        params_fusion = [
            {"params": itertools.chain(self.metric_fc.parameters(),
                                        self.fusion_net.parameters()),
                "lr": self.args.lr_head}
            ]
        self.optimizer = torch.optim.AdamW(self.text_encoder.parameters(),
                                eps=1e-8,
                                lr=self.args.min_lr_bert)
        
        self.optimizer_align = torch.optim.Adam(params_align) #betas=(0.5, 0.999) 

        self.optimizer_fusion = torch.optim.Adam(params_fusion) #betas=(0.5, 0.999) 

        self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                num_warmup_steps= int(self.total_steps * 0.01), #1% of total training
                                num_training_steps=self.total_steps)

        self.lr_scheduler_align = torch.optim.lr_scheduler.ExponentialLR(
                                        self.optimizer_align, 
                                        gamma=0.97)
        
        self.lr_scheduler_fusion = torch.optim.lr_scheduler.ExponentialLR(
                                        self.optimizer_fusion, 
                                        gamma=0.97)


    def get_identity_loss(self, sent_emb, img_features, class_ids):
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        class_ids = class_ids.to(self.args.device)

        # for text branch
        output = self.image_text_cls(sent_emb)
        tid_loss = criterion(output, class_ids)

        # for image branch (parameter sharing)
        output = self.image_text_cls(img_features)
        iid_loss = criterion(output, class_ids)

        total_idn_loss = self.args.lambda_id * (tid_loss + iid_loss) / 2
        self.idn_loss += total_idn_loss.item() 
        return total_idn_loss
    

    def get_cap_attr_loss(self, sent_emb, img_features, cap_attr, attr_vec):
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")
        criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
        cap_attr = cap_attr.to(self.args.device)
        attr_vec = attr_vec.to(self.args.device)

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
        class_ids = class_ids.to(self.args.device)
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
        return match_labels.to(self.args.device)


    def get_fusion_loss(self, sent_emb, words_emb, global_feats, local_feats, class_ids):
        label = class_ids.to(self.args.device)
        criterion = losses.FocalLoss(gamma=2) #torch.nn.CrossEntropyLoss() 

        if self.args.fusion_type == "linear":
            output = self.fusion_net(local_feats, global_feats, sent_emb)

        elif self.args.fusion_type == "fcfm":
            output = self.fusion_net(local_feats, words_emb, global_feats, sent_emb)

        elif self.args.fusion_type == "CMF":
            output = self.fusion_net(local_feats, words_emb, global_feats, sent_emb)

        output = self.metric_fc(output)      
        total_fusion_loss = args.lambda_f * criterion(output, label)
        self.f_loss += total_fusion_loss.item() 

        return total_fusion_loss     


    def train(self):
        self.text_encoder.train()
        self.text_head.train() 
        self.image_text_cls.train()
        #self.text_cls.train()
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
                gl_img_features, words_features, norm = self.image_encoder(imgs)
            else:
                gl_img_features, words_features = self.image_encoder(imgs)

            img_features, words_features = self.image_head(gl_img_features, words_features)
            
            self.optimizer.zero_grad()
            self.optimizer_align.zero_grad()
            self.optimizer_fusion.zero_grad()
            total_loss = 0
            
            if self.args.is_DAMSM == True:
                 total_loss += self.get_DAMSM_loss(sent_emb, words_emb, img_features, words_features, class_ids)


            if self.args.is_ident_loss == True: 
                total_loss += self.get_identity_loss(sent_emb, img_features, class_ids)

            if self.args.is_attr_loss == True: 
                total_loss += self.get_cap_attr_loss(sent_emb, img_features, cap_attr, attr_vec)

            if self.args.is_CLIP == True:
                total_loss += self.get_clip_loss(sent_emb, img_features)

            # cross-modal projection loss
            if self.args.is_CMP == True: 
                total_loss += self.get_CMP_loss(sent_emb, img_features, class_ids)

            if self.args.is_ITC == True:
                total_loss += self.get_itc_loss(sent_emb, img_features, self.logit_scale)

            if self.args.is_KD == True:
                total_loss += self.get_CPA(sent_emb, img_features)

            # fusion loss
            total_loss += self.get_fusion_loss(sent_emb, words_emb, img_features, words_features, class_ids) #img_features
            

            # update
            total_loss.backward()
            self.optimizer.step()
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
  
        
    def main(self):
        for epoch in range(self.start_epoch, self.args.max_epoch + 1):
            self.args.current_epoch = epoch

            self.train()
       
            self.lr_scheduler.step()
            self.lr_scheduler_align.step()
            self.lr_scheduler_fusion.step()

            if epoch % self.args.save_interval==0:
                print("saving image, text encoder, and fusion block\n")
                self.save_models()
            
            if (epoch > 6):
                if ((self.args.do_test == True) and 
                        (epoch % self.args.test_interval == 0 and epoch !=0)):
                    print("\nLet's validate the model")
                    test(self.valid_dl, 
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
    Train(args).main()