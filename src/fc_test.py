from torch.cuda.amp import GradScaler 
from tqdm.auto import tqdm 
import torch 
import numpy as np
import os, sys
import os.path as osp
from transformers import GPT2TokenizerFast
import evaluate
import argparse

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from models.fc_model import VisionGPT2Model
from utils.utils import TestImg2CapDataset
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

from nltk.tokenize import word_tokenize
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



def test_collate_fn(batch):
    image = [i[0] for i in batch]
    captions = [i[1] for i in batch]
    gen_file = [i[2] for i in batch]

    image = torch.stack(image, dim=0)
    return image, captions, gen_file


class Tester:
    def __init__(self, config):
        
        self.config = config
        self.device = self.config.device
        
        self.model = VisionGPT2Model.from_pretrained(self.config).to(self.device)
        self.model.trainable_gpt_layers(trainable=False)
                
        self.tokenizer = tokenizer        
        self.scaler = GradScaler()
        
        self.test_ds = TestImg2CapDataset(self.config.dataset, self.config.gen_text)
        
        bs = 1 if self.config.gen_text else 32
        self.test_dl = torch.utils.data.DataLoader(self.test_ds,
                                     batch_size=bs,
                                     shuffle=False,
                                     pin_memory=True,
                                     collate_fn=test_collate_fn,
                                     num_workers=4)

    def load_best_model(self,):
        
        sd = torch.load(os.path.join(self.config.model_path, self.config.saved_model_file))
        self.model.load_state_dict(sd)
        print("loading saved model: ", self.config.saved_model_file)
    

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


    def test(self, ):
        self.load_best_model()
        if self.config.gen_text == True:
            os.makedirs(os.path.join("./data", self.config.dataset, 
                                     "annotations", "gen_text"), exist_ok=True)
        else:
            bleu = evaluate.load("bleu")
            rouge = evaluate.load("rouge")
            meteor = evaluate.load('meteor')
            meter = AverageMeter()

        loop = tqdm(self.test_dl, total=len(self.test_dl))
        
        for image, ref_caption, gen_file in loop:
            
            image = image.to(self.device)
            if self.config.gen_text == False:    
                t = np.random.uniform(0.5, 1.5)
                gen_caption = self.generate_caption(image, 
                                                    temperature=t, 
                                                    deterministic=True)

                gen_caption = ["This" + gen_cap for gen_cap in gen_caption]

                b1 = bleu.compute(predictions=gen_caption, references= ref_caption, max_order=1) #word_tokenize
                b2 = bleu.compute(predictions=gen_caption, references= ref_caption, max_order=2)
                b3 = bleu.compute(predictions=gen_caption, references= ref_caption, max_order=3)
                b4 = bleu.compute(predictions=gen_caption, references= ref_caption, max_order=4)

                rL = rouge.compute(predictions=gen_caption, references= ref_caption)
                m = meteor.compute(predictions=gen_caption, references= ref_caption)

                # upading metrics for each sample
                meter.update(b1["bleu"], b2["bleu"], b3["bleu"], b4["bleu"], rL["rougeL"], m["meteor"])

            elif self.config.gen_text == True:
                t = np.random.uniform(0.5, 1.5)
                gen_caption = self.generate_caption(image, 
                            temperature=t, 
                            deterministic=True)
                    
                full_text = "This" + gen_caption[0]
                with open(gen_file[0], "w") as file:
                    file.write(full_text)
        
        if self.config.gen_text == False: 
            meter.average()
            print("BLEU@1: ", meter.belu_1)
            print("BLEU@2: ", meter.belu_2)
            print("BLEU@3: ", meter.belu_3)
            print("BLEU@4: ", meter.belu_4)

            print("rougeL: ", meter.rougeL)
            print("METEOR: ", meter.meteor)