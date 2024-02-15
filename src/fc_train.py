
from torch.cuda.amp import autocast, GradScaler 
from tqdm.auto import tqdm 
import torch 
import numpy as np
import os, sys
import os.path as osp
import gc
from transformers import GPT2TokenizerFast
import albumentations as A
from PIL import Image
import pandas as pd 
from types import SimpleNamespace
import torch.nn.functional as nnf
from tqdm import trange
import matplotlib.pyplot as plt 
from albumentations.pytorch import ToTensorV2


ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from utils import attribute as a 
from models import fc_iresnet as iresnet
from models.fc_model import VisionGPT2Model
from utils.utils import Img2CapDataset, get_dataframes
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

def collate_fn(batch):
    image = [i[0] for i in batch]
    input_ids = [i[1] for i in batch]
    labels = [i[2] for i in batch]
    captions = [i[3] for i in batch]

    image = torch.stack(image,dim=0)
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
    def __init__(self, model_config, train_config):
        
        self.train_config = train_config
        self.model_config = model_config
        self.device = self.train_config.device
        
        self.model = VisionGPT2Model.from_pretrained(self.model_config).to(self.device)
        self.model.pretrained_layers_trainable(trainable=False)
        
        print(f'trainable parameters: {sum([p.numel() for p in self.model.parameters() if p.requires_grad])}')
        
        self.tokenizer = tokenizer        
        self.scaler = GradScaler()
        
        self.train_ds =  Img2CapDataset(self.train_config.dataset, train=True)
        self.val_ds =  Img2CapDataset(self.train_config.dataset, train=False)

        self.train_dl = torch.utils.data.DataLoader(self.train_ds,
                                       batch_size=train_config.batch_size,
                                       shuffle=True,
                                       pin_memory=True,
                                       num_workers=4,
                                       collate_fn=collate_fn,
                                       persistent_workers=True)

        self.val_dl = torch.utils.data.DataLoader(self.val_ds,
                                     batch_size=train_config.batch_size,
                                     shuffle=False,
                                     pin_memory=True,
                                     num_workers=4,
                                     collate_fn=collate_fn,
                                     persistent_workers=True)
       
        total_steps = len(self.train_dl)
        print("total steps: ", total_steps)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.train_config.lr / 25.)
        self.sched = torch.optim.lr_scheduler.OneCycleLR(
            self.optim,
            max_lr=self.train_config.lr,
            epochs=self.train_config.epochs,
            steps_per_epoch=total_steps
        )
                
        self.metrics = pd.DataFrame()
        self.metrics[['train_loss','train_perplexity','val_loss','val_perplexity']] = None
        
        self.gen_tfms = A.Compose([
            A.Resize(112, 112),
            A.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5],always_apply=True),
            ToTensorV2()
        ])
            
    def save_model(self,):
        os.makedirs(self.train_config.model_path, exist_ok=True)
        sd = self.model.state_dict()
        torch.save(sd, os.path.join(self.train_config.model_path, 'model_celeba_arc_w_caption.pt'))
        

    def save_arcface_model(self,):
        arcface = iresnet.iresnet18(pretrained=False, progress=True)
        sd = self.model.state_dict()
        for key in list(sd.keys()):
            sd[key.replace('arcface.', '')] = sd.pop(key)

        print(list(sd.keys()))
        arcface.load_state_dict(sd, strict=False)

        os.makedirs(self.train_config.model_path, exist_ok=True)
        torch.save(arcface.state_dict(), os.path.join(self.train_config.model_path, 'arcface_celeba.pt'))


    def load_best_model(self,):
        sd = torch.load(os.path.join(self.train_config.model_path, "model_celeba_arc_w_caption.pt"))
        self.model.load_state_dict(sd)
    
    
    def get_attr_loss(self, image, ac_caption):
        self.model.eval()
        gen_arr_ls = []
        ac_atrr_ls = [torch.Tensor(a.get_attr_vector(cap)) for cap in ac_caption]
        ac_atrr_ls = torch.stack(ac_atrr_ls, dim=0)
        sequence = torch.ones(1, 1).to(device=self.device).long() * self.tokenizer.bos_token_id
        
        for img in image:
            img = img.unsqueeze(0)
            gen_caption = self.model.generate(
                img,
                sequence,
                max_tokens=48,
                temperature=1.0,
                deterministic=True
            )

            gen_caption = self.tokenizer.decode(gen_caption, skip_special_tokens=True)
            gen_arr_ls.append(torch.tanh(torch.Tensor(a.get_attr_vector(gen_caption))))

        gen_arr_ls = torch.stack(gen_arr_ls, dim=0)
        mse_loss =  torch.nn.MSELoss()(gen_arr_ls, ac_atrr_ls)
        self.model.train()
        return mse_loss



    def train_one_epoch(self, epoch):
        prog = tqdm(self.train_dl, total=len(self.train_dl))
        running_loss = 0.
        mse_loss = 0.0

        for image, input_ids, labels, caption in prog:
            with autocast():
                image = image.to(self.device)
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                cap_loss = self.model(image, input_ids, labels)

                if epoch > 3:
                    attr_loss = self.get_attr_loss(image, caption)
                    loss = cap_loss + attr_loss
                else:
                    loss = cap_loss 

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
                self.sched.step()
                self.optim.zero_grad(set_to_none=True)
                
                if epoch > 3:
                    running_loss += cap_loss.item()
                    mse_loss += attr_loss.item()
                    prog.set_description(f'train loss: {cap_loss.item():.3f} and mse loss: {attr_loss.item():.3f}')
                else:
                    running_loss += cap_loss.item()
                    prog.set_description(f'train loss: {cap_loss.item():.3f}')
                
            del image, input_ids, labels, loss
            
        train_loss = running_loss / len(self.train_dl)
        train_pxp = np.exp(train_loss)

        total_mse_loss = mse_loss / len(self.train_dl)
        self.metrics.loc[epoch,['train_loss', 'mse_loss', 'train_perplexity']] = (train_loss, total_mse_loss, train_pxp)
       
        
    @torch.no_grad()
    def valid_one_epoch(self,epoch):
        
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
       
    
    def fit(self,):
        best_pxp = 1e9
        best_epoch = -1
        prog = tqdm(range(self.train_config.epochs))
        
        for epoch in prog:
            
            if epoch == self.train_config.freeze_epochs_gpt:
                self.model.unfreeze_gpt_layers()
                print('unfreezing GPT2 entirely...')
                
            if epoch == self.train_config.freeze_epochs_all:
                self.model.pretrained_layers_trainable(trainable=True)
            
            self.model.train()
            prog.set_description('training')
            
            self.train_one_epoch(epoch)
            self.clean()
            
            self.model.eval()
            prog.set_description('validating')
            pxp = self.valid_one_epoch(epoch)
            self.clean()
            
            print(self.metrics.tail(1))
            
            if pxp < best_pxp:
                best_pxp = pxp
                best_epoch = epoch
                print('saving best model...')
                self.save_model()
                
        return {
            'best_perplexity': best_pxp,
            'best_epoch': best_epoch
        }
           
        
    @torch.no_grad()
    def generate_caption(self,image, max_tokens=48, temperature=1.0, deterministic=False):
        
        self.model.eval()
        image = Image.open(image).convert('RGB')
        image = np.array(image)
        image = self.gen_tfms(image=image)['image']
        image = image.unsqueeze(0).to(self.device)
        sequence = torch.ones(1,1).to(device=self.device).long() * self.tokenizer.bos_token_id
        
        #_, embed = self.model.arcface(image)
        #embed = nnf.normalize(self.model.proj(embed), p=2, dim=-1)
        caption = self.model.generate(
            image,
            sequence,
            max_tokens=max_tokens,
            temperature=temperature,
            deterministic=deterministic
        )
        return self.tokenizer.decode(caption.numpy(), skip_special_tokens=True)
        #return self.generate_colab(image, embed = embed, temperature = temperature)


    def generate_colab(self,
        image,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count=1,
        max_tokens=48,  # maximum number of words
        top_p=0.8,
        temperature=1.0):

        self.model.eval()
        generated_list = []
        stop_token_index = self.tokenizer.eos_token_id
        filter_value = -float("Inf")
        sequence = torch.ones(1,1).to(device=self.device).long() * self.tokenizer.bos_token_id

        with torch.no_grad():
            generated = embed 

            for i in range(max_tokens):

                out = self.model(image, generated.long().cuda())
                out = out[:,-1,:] / temperature
                logits = nnf.softmax(out,dim=-1)

                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = self.model.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = self.tokenizer.decode(output_list, skip_special_tokens=True)
            generated_list.append(output_text)

        return generated_list[0]



def get_configs():
    model_config = SimpleNamespace(
        vocab_size = 50_257,
        embed_dim = 768,
        num_heads = 12,
        seq_len = 1024,
        depth = 12,
        attention_dropout = 0.1,
        residual_dropout = 0.1,
        mlp_ratio = 4,
        mlp_dropout = 0.1,
        emb_dropout = 0.1,
        weights_adaface= "./weights/pretrained/adaface_ir18_webface4m.ckpt",
        weights_arcface= "./weights/pretrained/arcface_ir18_ms1mv3.pth",
    )

    train_config = SimpleNamespace(
        dataset = "celeba", 
        epochs = 6,
        freeze_epochs_gpt = 2,
        freeze_epochs_all = 3,
        lr = 1e-4,
        device = 'cuda',
        model_path = "./weights",
        batch_size = 64
    )
    return model_config, train_config


def train(model_config, train_config):
    trainer = Trainer(model_config, train_config)

    #trainer.load_best_model()
    trainer.fit()
    plt.plot(trainer.metrics['train_loss'],color='red',label='train loss')
    plt.plot(trainer.metrics['val_loss'],color='orange',label='valid loss')
    plt.title('loss, lower=better')
    plt.legend()
    plt.show()


def test(model_config, train_config):
    val_df = get_dataframes(train_config.dataset, train == False)
    trainer = Trainer(model_config, train_config)
    trainer.load_best_model()
    trainer.save_arcface_model()

    """
    for i in range(5):
        t = np.random.uniform(0.5, 1.5)
        test = val_df.sample(n=1).values[0]
        test_img, test_caption = test[0],test[1]

        det = True
        gen_caption = "This" + trainer.generate_caption(test_img, temperature=t, deterministic=det)
        plt.imshow(Image.open(test_img).convert('RGB'))
        v = a.get_attr_vector(gen_caption)

        plt.title(f"actual: {test_caption}\n model: {gen_caption}\n caption_vector: {v}")
        plt.axis('off')
        plt.show()
    """


if __name__ == "__main__": 
    model_config, train_config = get_configs()
    #train(model_config, train_config)
    #test(model_config, train_config)