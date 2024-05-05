import os, sys, random
import os.path as osp
import argparse, itertools
import torch
import numpy as np
from torch import nn 
from tqdm import tqdm 
from torch.optim.lr_scheduler import ExponentialLR
from transformers import  AutoTokenizer
from torch.autograd import Variable
from transformers import get_linear_schedule_with_warmup
from types import SimpleNamespace
import pprint
from torch.nn import functional as F 

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from fgic_models import get_resnet_model, TopLayer
from dataset import CUBDataset
from models.models import TextHeading, TextEncoder, ImageHeading
from models import  losses
from models.fusion_nets import LinearFusion, CMF  

os.environ["TOKENIZERS_PARALLELISM"] = "false"
my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer:
    def __init__(self, config):
        self.config = config  
        self.get_loaders()

        self.total_steps = len(self.train_dl) * self.config.epochs
        self.total_length = len(self.train_dl) * self.config.batch_size

        self.build_models()
        self.get_optimizer()


    def save_models(self):
        save_dir = os.path.join(self.config.model_path, self.config.dataset)
        os.makedirs(save_dir, exist_ok=True)

        name = 'image_res%d_%s_%s_%d.pth' % (self.config.resnet_layer,
                                                   self.config.bert_type,
                                                   self.config.fusion_type, 
                                                   self.epoch)

        state = {'net': self.fusion_net.state_dict(), 
                 "image_encoder": self.model.state_dict(),
                 'image_head': self.image_head.state_dict(),
                 'metric_fc': self.metric_fc.state_dict()
                }
        
        torch.save(state, os.path.join(save_dir, name))
        checkpoint_text_en = {
            'model': self.text_encoder.state_dict(),
            'head': self.text_head.state_dict()
        }

        torch.save(checkpoint_text_en, '%s/text_res%d_%s_%s_%d.pth' % (save_dir, 
                                                                    self.config.resnet_layer,
                                                                    self.config.bert_type,
                                                                    self.config.fusion_type,
                                                                    self.epoch))


    def load_all_models(self):
        print("loading all models")
        state_dict = torch.load(os.path.join(self.config.model_path, self.config.image_encoder_path))
        self.model.load_state_dict(state_dict['image_encoder'])
        self.image_head.load_state_dict(state_dict["image_head"])
        self.metric_fc.load_state_dict(state_dict["metric_fc"])
        self.fusion_net.load_state_dict(state_dict["net"])

        checkpoint = torch.load(os.path.join(self.config.model_path, self.config.text_encoder_path))
        self.text_encoder.load_state_dict(checkpoint['model'])
        self.text_head.load_state_dict(checkpoint['head'])


    def get_optimizer(self):
        params_align = [
                {"params": itertools.chain(self.text_head.parameters(),
                                        self.image_head.parameters(),
                                        self.image_text_cls.parameters(),
                                        self.fusion_net.parameters()),

                "lr": 0.001}
            ] 
      
        self.optimizer_model = torch.optim.Adam(self.model.parameters(), 
                                lr = self.config.lr_model, 
                                betas=(0.9, 0.99), 
                                weight_decay=5e-5)
        
        self.optimizer_en = torch.optim.AdamW(self.text_encoder.parameters(),
                                eps=1e-8, lr=self.config.min_lr_bert)
        
        self.optimizer_align = torch.optim.Adam(params_align) 
        self.optimizer_fusion = torch.optim.Adam(self.metric_fc.parameters(), lr=0.001) 

        self.ls_model = ExponentialLR(self.optimizer_model, gamma=0.90)        
        self.lrs_en = get_linear_schedule_with_warmup(self.optimizer_en,
                                num_warmup_steps= int(self.total_steps * 0.01), #1% of total training
                                num_training_steps = self.total_steps)

        self.lrs_align = ExponentialLR(self.optimizer_align, gamma=0.97)       
        self.lrs_fusion = ExponentialLR(self.optimizer_fusion, gamma=0.97)


    def get_text_emb(self, sent):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        all_caption = []
        all_mask = []

        bs = len(sent)
        for i in range(bs):
            cap = sent[i].encode('utf-8').decode('utf8')
            cap = cap.replace("\ufffd\ufffd", " ")
        
            encoding = tokenizer.encode_plus(
                        cap,
                        add_special_tokens=True,
                        max_length = self.config.bert_words_num,
                        return_token_type_ids=False,
                        padding='max_length',
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors='pt')

            input_ids=encoding["input_ids"].flatten()
            mask=encoding["attention_mask"].flatten()

            caption = Variable(input_ids).cuda()
            mask = Variable(mask).cuda()

            #caption = caption.unsqueeze(dim=0)
            #mask = mask.unsqueeze(dim=0)  

            all_caption.append(caption)
            all_mask.append(mask)

        all_caption = torch.stack(all_caption, dim=0)
        all_mask = torch.stack(all_mask, dim=0)

        words_emb, sent_emb = self.text_encoder(all_caption, all_mask)
        words_emb, sent_emb = self.text_head(words_emb, sent_emb)
 
        return words_emb, sent_emb


    def get_loaders(self):
        train_ds = CUBDataset(train = True, args=self.config)
        test_ds = CUBDataset(train = False, args=self.config)

        self.train_dl = torch.utils.data.DataLoader(
            train_ds, 
            batch_size = self.config.batch_size, 
            drop_last = False,
            num_workers = 4, 
            shuffle = True)
        
        self.test_dl = torch.utils.data.DataLoader(
            test_ds, 
            batch_size = self.config.batch_size * 4, 
            drop_last = False,
            num_workers = 4, 
            shuffle = False)
        

    def build_models(self):
        self.model = get_resnet_model(config).to(my_device)
        self.image_text_cls = TopLayer(input_dim=256, 
                            num_classes=self.config.num_classes).to(my_device)
            
        print("loading pretrained resent%s weights" % self.config.resnet_layer)
        loading_dir = os.path.join(self.config.model_path, self.config.saved_model_file)
        checkpoint = torch.load(loading_dir)

        self.model.load_state_dict(checkpoint["model"])

        print("loading pretrained foundation model")
        self.text_encoder =  TextEncoder(self.config).to(my_device)
        self.text_head = TextHeading(self.config).to(my_device)

        if self.config.resnet_layer == 18:
            config.fusion_final_dim = 576
            feature_dim = 512

        elif self.config.resnet_layer == 50:
            config.fusion_final_dim = 2112
            feature_dim = 2048

        self.image_head = ImageHeading(self.config, feature_dim).to(my_device)
        
        if self.config.fusion_type == "linear":
            self.fusion_net = LinearFusion(self.config).to(my_device)

        elif self.config.fusion_type == "CMF": 
            self.fusion_net = CMF(self.config).to(my_device)

        self.metric_fc = TopLayer(
                                self.config.fusion_final_dim, 
                                self.config.num_classes).to(my_device)
        

    def get_cls_loss(self, sent_emb, img_features, class_ids):
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        class_ids = class_ids.to(my_device)

        # for text branch
        output = self.image_text_cls(sent_emb)
        tid_loss = criterion(output, class_ids)

        # for image branch (parameter sharing)
        output = self.image_text_cls(img_features)
        iid_loss = criterion(output, class_ids)

        loss = (self.config.lambda_cls * (tid_loss + iid_loss) / 2)
        self.cls_loss += loss.item() 
        return loss
    

    def get_fusion_loss(self, gl_img, words_features, img_features, words_emb, sent_emb, targets):
        fusion_loss = losses.FocalLoss(gamma=2) 
        output = self.fusion_net(words_features, words_emb)  #(words_features, img_features, sent_emb)

        gl_img = F.normalize(gl_img, p=2, dim=1)
        output = F.normalize(output, p=2, dim=1)
        output = torch.cat((gl_img, output), dim=1)
        output = self.metric_fc(output)

        loss = (config.lambda_f * fusion_loss(output, targets))
        self.f_loss += loss.item() 
        return loss 
    

    def get_itc_loss(self, sent_emb, img_features):
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        itc_loss = losses.compute_itc(img_features, sent_emb, logit_scale)
        
        total_itc_loss = self.config.lambda_itc * itc_loss
        self.itc_loss += total_itc_loss.item()
        return total_itc_loss


    def get_CPA(self, sent_emb, img_features):
        T = 2
        soft_targets = nn.functional.softmax(img_features / T, dim=-1)
        soft_prob = nn.functional.log_softmax(sent_emb / T, dim=-1)

        # Calculate the true label loss
        KL_loss = nn.KLDivLoss(reduction="batchmean", log_target=False)
        total_kd_loss = self.config.lambda_kd * KL_loss(soft_prob, soft_targets)
        self.kd_loss += total_kd_loss.item()
        return total_kd_loss


    def print_losses(self):
        print(' | epoch {:3d} |' .format(self.epoch))
        print("KD loss: {:5.6f} ".format(self.kd_loss)) # / total_length
        print("Identity loss: {:5.6f} ".format(self.cls_loss / self.total_length))
        print("ITC loss: {:5.6f} ".format(self.itc_loss / self.total_length))
        print("Fusion loss: {:5.6f} ".format(self.f_loss / self.total_length))


    def train_epoch(self):
        self.model.eval()
        self.image_text_cls.train()
        self.image_head.train()
        self.text_encoder.train()
        self.text_head.train()
        self.fusion_net.train()
        self.metric_fc.train()

        self.s_loss = 0
        self.w_loss = 0
        self.cls_loss = 0
        self.kd_loss = 0
        self.f_loss = 0
        self.itc_loss = 0
        
        loop = tqdm(total = len(self.train_dl))

        for inputs, targets, caption in self.train_dl:
            inputs, targets = inputs.to(my_device), targets.to(my_device)

            gl_img, l_img = self.model(inputs)
            img_features, words_features = self.image_head(gl_img, l_img)
            words_emb, sent_emb = self.get_text_emb(caption)
            
            self.optimizer_model.zero_grad()
            self.optimizer_en.zero_grad()
            self.optimizer_align.zero_grad()
            self.optimizer_fusion.zero_grad()
            total_loss = 0

            total_loss += self.get_cls_loss(sent_emb, img_features, targets)
            if self.config.is_KD: total_loss += self.get_CPA(sent_emb, img_features)
            total_loss += self.get_itc_loss(sent_emb, img_features)
            total_loss += self.get_fusion_loss(gl_img, words_features, img_features, words_emb, sent_emb, targets) 

            # update
            total_loss.backward()

            if self.epoch > self.config.freeze: 
                self.optimizer_model.step()
                self.optimizer_en.step()

            self.optimizer_align.step()
            self.optimizer_fusion.step()

            #`clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.    
            torch.nn.utils.clip_grad_norm_(itertools.chain(self.text_encoder.parameters()), 
                                                           self.config.clip_max_norm)

            # update loop information
            loop.update(1)
            loop.set_description(f'Training Epoch [{self.epoch}/{self.config.epochs}]')
            loop.set_postfix()
        
        loop.close()
        self.print_losses()


    def valid_epoch(self):
        self.model.eval()
        self.image_head.eval()
        self.text_encoder.eval()
        self.text_head.eval()
        self.fusion_net.eval()
        self.metric_fc.eval()

        correct = 0
        total = 0
        
        loop = tqdm(total = len(self.test_dl))
        with torch.no_grad():
            for inputs, targets, caption in self.test_dl:
                inputs, targets = inputs.cuda(), targets.cuda()
                gl_img, l_img = self.model(inputs)

                img_features, words_features = self.image_head(gl_img, l_img)
                words_emb, sent_emb = self.get_text_emb(caption)
                output = self.fusion_net(words_features, words_emb) 

                #concatenation
                gl_img = F.normalize(gl_img, p=2, dim=1)
                output = F.normalize(output, p=2, dim=1)
                output = torch.cat((gl_img, output), dim=1)
                output = self.metric_fc(output)

                _, predicted = torch.max(output.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum().item()
                loop.update(1)
                loop.set_postfix()
             
            loop.close()

        test_acc = correct/total
        print('test_acc = %.5f\n' %  (test_acc))


    def train(self):
        for epoch in range(self.config.epochs):
            self.epoch = epoch 
            self.train_epoch()

            if epoch > self.config.freeze: 
                self.ls_model.step()
                self.lrs_en.step()

            self.lrs_align.step()
            self.lrs_fusion.step()

            if (epoch % self.config.save_interval==0) and (epoch > 5) :
                print("saving model for epoch: ", self.epoch)
                self.save_models()
                pass 

            if ( epoch % self.config.valid_interval == 0 and epoch > 5):
                print("lets validate: ")
                self.valid_epoch()


    def test(self):
        self.load_all_models()
        self.valid_epoch()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',          dest="train",       help='train the pretrained resent model',   action='store_true')
    parser.add_argument('--test',           dest="train",       help='evaluate the pretrained resent model',action='store_false')
    parser.set_defaults(train=False)

    parser.add_argument('--dataset',            type=str,   default="cub",    help='Name of the datasets')
    parser.add_argument('--batch_size',         type=int,   default=8,         help='Batch size')
    parser.add_argument('--epochs',             type=int,   default=20,          help='Number of epochs')

    parser.add_argument('--resnet_layer',       type=int,   default=18,     help='Number of ResNet layers 18|50')
    parser.add_argument('--freeze',             type=int,   default=5,      help='Number of epoch pretrained model frezees')
    parser.add_argument('--fusion_type',        type=str,   default="CMF",  help='Type of Fusion block CMF|linear')
    
    parser.add_argument('--model_path',    type=str,   default="./weights/fgic/", help='model directory')
    parser.add_argument('--weights_path',    type=str,   default="./weights/fgic/", help='model directory')
    parser.add_argument('--text_encoder_path',    type=str,   default="text_res18_bert_CMF_17.pth", help='text encoder directory')
    parser.add_argument('--image_encoder_path',   type=str,   default="image_res18_bert_CMF_17.pth", help='image encoder directory')

    parser.add_argument('--lambda_f',       type=float,   default=1,    help='weight value of the fusion loss')
    parser.add_argument('--lambda_itc',     type=float,   default=1,    help='weight value of the ITC loss')
    parser.add_argument('--lambda_cls',     type=float,   default=1,    help='weight value of the identity loss')
    parser.add_argument('--lambda_kd',      type=float,   default=0.1,    help='weight value of the KD loss')

    parser.add_argument('--save_interval',      type=int,   default=1,    help='saving intervals (epochs)')
    parser.add_argument('--valid_interval',     type=int,   default=1,    help='valid (epochs)')

    parser.add_argument('--train_imgs_list',  type=str,   default="./data/cub/train_images_shuffle.txt", help='train image list of CUB dataset')
    parser.add_argument('--test_imgs_list',  type=str,    default="./data/cub/test_images_shuffle.txt",  help='test image list of CUB dataset')
    parser.add_argument('--saved_model_file',   type=str,   default="resnet18_cub.pth", help='The resent model to load for test')
       
    return  parser.parse_args(argv)


setup_cfg = SimpleNamespace(
    num_classes = 200, 
    bert_words_num = 32,
    captions_per_image= 10,
    en_type = "BERT",        
    embedding_dim = 256,
    bert_type = "bert", 
    bert_config = "bert-base-uncased",

    min_lr_bert = 0.00002,
    lr_model = 0.00002, 
    clip_max_norm = 1.0,

    # model arch         
    is_KD = False,       

    # fusion arch
    manual_seed = 61,   
    gl_img_dim = 256,
    gl_text_dim = 256
)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    config = SimpleNamespace(**args.__dict__, **setup_cfg.__dict__)
    #pprint.pp(config)

    random.seed(config.manual_seed)
    np.random.seed(config.manual_seed)
    torch.manual_seed(config.manual_seed)
    torch.cuda.manual_seed_all(config.manual_seed)

    t = Trainer(config)
    if config.train == True:
        t.train()

    elif config.train == False:
        t.test()