from torch.cuda.amp import autocast, GradScaler 
from tqdm.auto import tqdm 
import torch 
import numpy as np
import os, sys
import os.path as osp
import gc
from transformers import GPT2TokenizerFast
from PIL import Image
import pandas as pd 
import matplotlib.pyplot as plt 


ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from utils import attribute as a 
from models.fc_model import VisionGPT2Model
from utils.utils import Img2CapDataset, TestImg2CapDataset
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.belu_1 = 0
        self.belu_2 = 0
        self.belu_3 = 0
        self.belu_4 = 0

        self.rougeL = 0
        self.meteor = 0
        self.count = 0

    def update(self, belu_1, belu_2, belu_3, belu_4, rougeL, meteor):
        self.belu_1 += belu_1
        self.belu_2 += belu_2
        self.belu_3 += belu_3
        self.belu_4 += belu_4
    
        self.rougeL += rougeL
        self.meteor  += meteor 
        self.count += 1

    def average(self):
        self.belu_1  = self.belu_1 / self.count
        self.belu_2  = self.belu_2 / self.count
        self.belu_3  = self.belu_3 / self.count
        self.belu_4  = self.belu_4 / self.count

        self.rougeL = self.rougeL / self.count
        self.meteor  = self.meteor / self.count


def collate_fn(batch):
    image = [i[0] for i in batch]
    input_ids = [i[1] for i in batch]
    labels = [i[2] for i in batch]
    captions = [i[3] for i in batch]

    image = torch.stack(image, dim=0)
    input_ids = tokenizer.pad(
        {'input_ids':input_ids},
        padding='longest',
        return_attention_mask=False,
        return_tensors='pt'
    )['input_ids']
    labels = tokenizer.pad(
        {'input_ids':labels},
        padding='longest',
        return_attention_mask=False,
        return_tensors='pt'
    )['input_ids']
    mask = (input_ids!=tokenizer.pad_token_id).long()
    labels[mask==0] = -100
    return image, input_ids, labels, captions


class Trainer:
    def __init__(self, config):
        
        self.config = config
        self.device = self.config.device
        
        self.model = VisionGPT2Model.from_pretrained(self.config).to(self.device)
        self.model.trainable_gpt_layers(trainable=False)
                
        self.tokenizer = tokenizer        
        self.scaler = GradScaler()
        
        self.train_ds =  Img2CapDataset(self.config.dataset, exp_type="train")
        self.val_ds =  Img2CapDataset(self.config.dataset, exp_type="valid")
        self.test_ds = TestImg2CapDataset(self.config.dataset, self.config.gen_text)

        self.train_dl = torch.utils.data.DataLoader(self.train_ds,
                                       batch_size=config.batch_size,
                                       shuffle=True,
                                       pin_memory=True,
                                       num_workers=4,
                                       collate_fn=collate_fn,
                                       persistent_workers=True)

        self.val_dl = torch.utils.data.DataLoader(self.val_ds,
                                     batch_size=config.batch_size,
                                     shuffle=False,
                                     pin_memory=True,
                                     num_workers=4,
                                     collate_fn=collate_fn,
                                     persistent_workers=True)
        
       
        steps_per_epoch = len(self.train_dl)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.config.lr / 25.)

        self.sched = torch.optim.lr_scheduler.OneCycleLR(
            self.optim,
            max_lr = self.config.lr,
            epochs = self.config.epochs,
            steps_per_epoch = steps_per_epoch
        )
                
        self.metrics = pd.DataFrame()
        self.metrics[['train_loss','train_perplexity','val_loss','val_perplexity']] = None
        
            
    def save_model(self,):
        os.makedirs(self.config.model_path, exist_ok=True)
        sd = self.model.state_dict()
        torch.save(sd, os.path.join(self.config.model_path, self.config.saved_model_file))


    def save_arcface_model(self,):
        arcface = iresnet.iresnet18(pretrained=False, progress=True)
        sd = self.model.state_dict()
        for key in list(sd.keys()):
            sd[key.replace('arcface.', '')] = sd.pop(key)

        print(list(sd.keys()))
        arcface.load_state_dict(sd, strict=False)

        os.makedirs(self.config.model_path, exist_ok=True)
        torch.save(arcface.state_dict(), 
                   os.path.join(self.config.model_path, 'arcface_celeba.pt'))


    def load_best_model(self,):
        saved_file = "model_%s_%s_%s_w_attr_full.pth" % (self.config.dataset, 
                                                            self.config.arch, 
                                                            self.config.resnet_layer)
        
        sd = torch.load(os.path.join(self.config.model_path, saved_file))
        self.model.load_state_dict(sd)
        print("loading saved model: ")
    
    
    def get_attr_loss(self, images, caption):
        ac_atrr_ls = [torch.Tensor(a.get_attr_vector(cap)) for cap in caption]
        ac_atrr_ls = torch.stack(ac_atrr_ls, dim=0)
        
        gen_caption = self.generate_caption(images)
        gen_arr_ls = [torch.tanh(torch.Tensor(a.get_attr_vector(gen_cap))) for gen_cap in gen_caption]

        gen_arr_ls = torch.stack(gen_arr_ls, dim=0)
        mse_loss =  torch.nn.MSELoss()(gen_arr_ls, ac_atrr_ls)
        self.model.train()
        return mse_loss


    def train_one_epoch(self, epoch, is_attr):
        prog = tqdm(self.train_dl, total=len(self.train_dl))
        running_loss = 0.
        mse_loss = 0.0

        for image, input_ids, labels, caption in prog:
            with autocast():
                image = image.to(self.device)
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                cap_loss = self.model(image, input_ids, labels)
                loss = cap_loss 

                if is_attr and epoch > 0:
                    attr_loss =  self.get_attr_loss(image, caption)
                    loss = cap_loss + attr_loss

                self.optim.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
                self.sched.step()
                
                if is_attr and epoch > 3:
                    running_loss += cap_loss.item()
                    mse_loss += attr_loss.item()
                    prog.set_description(f"train loss: {cap_loss.item():.3f} and mse loss: {attr_loss.item():.3f}")
                else:
                    running_loss += cap_loss.item()
                    prog.set_description(f'train loss: {cap_loss.item():.3f}')
                
            del image, input_ids, labels, loss

        train_loss = running_loss / len(self.train_dl)
        train_pxp = np.exp(train_loss)
        
        if is_attr:
            total_mse_loss = mse_loss / len(self.train_dl)
            self.metrics.loc[epoch, ['train_loss', 'mse_loss', 'train_perplexity']] = (train_loss, total_mse_loss, train_pxp)

        else:
            self.metrics.loc[epoch, ['train_loss',  'train_perplexity']] = (train_loss, train_pxp)


    @torch.no_grad()
    def valid_one_epoch(self, epoch):
        
        prog = tqdm(self.val_dl,total=len(self.val_dl))
        running_loss = 0.
        
        for image, input_ids, labels, caption in prog:

            with autocast():
                image = image.to(self.device)
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                
                loss = self.model(image,input_ids,labels)
                running_loss += loss.item()
                
                prog.set_description(f'valid loss: {loss.item():.3f}')
                
            del image, input_ids, labels, loss
            
        val_loss = running_loss / len(self.val_dl)
        val_pxp = np.exp(val_loss)
        
        self.metrics.loc[epoch,['val_loss','val_perplexity']] = (val_loss,val_pxp)
        return val_pxp
        
        
    def clean(self):
        gc.collect()
        torch.cuda.empty_cache()


    def plot_graph(self,):
        plt.plot(self.metrics['train_loss'],color='red', label='train loss')
        plt.plot(self.metrics['val_loss'],color='orange',label='valid loss')
        plt.title('loss, lower=better')
        plt.legend()
        plt.show()


    def train(self,):
        best_pxp = 1e9
        best_epoch = -1
        prog = tqdm(range(self.config.epochs))
        
        for epoch in prog:
            if epoch >= self.config.freeze_epochs:
                print("************* GPT layers are trainable now *********")
                self.model.trainable_gpt_layers(trainable=True)

            self.model.train()
            prog.set_description('training')
            
            self.train_one_epoch(epoch, self.config.is_attr)
            self.clean()
            
            self.model.eval()
            prog.set_description('validating')
            pxp = self.valid_one_epoch(epoch)
            self.clean()
            
            print("\n")
            print(self.metrics.tail(1))
            
            if pxp < best_pxp:
                best_pxp = pxp
                best_epoch = epoch
                print('saving best model...')
                self.save_model()
                
        print("best_perplexity: %.5f, best_epoch: %d" %(best_pxp, best_epoch))
        self.plot_graph()


    @torch.no_grad()
    def generate_caption(self, 
                         images, 
                         max_tokens=48, 
                         temperature=1.0, 
                         deterministic=False):
        
        self.model.eval()
        sequence = torch.ones(1,1).to(device=self.device).long() * self.tokenizer.bos_token_id
        
        #_, embed = self.model.arcface(image)
        #embed = nnf.normalize(self.model.proj(embed), p=2, dim=-1)
        caption = [self.model.generate(
            image.unsqueeze(0),
            sequence,
            self.tokenizer, 
            max_tokens=max_tokens,
            temperature=temperature,
            deterministic=deterministic
        ).numpy() for image in images]

        return self.tokenizer.batch_decode(caption, skip_special_tokens=True)
