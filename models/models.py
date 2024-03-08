import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torchvision import models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import (BertModel, AlignTextModel, CLIPTextModel, 
                          FlavaTextModel, BlipTextModel, GroupViTTextModel)
import torch.nn.functional as F
from models.fusion_nets import SelfAttention 
from torchsummary import summary
import numpy as np 
import math 


def l2_norm(input, axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output


def get_CLS_embedding(layer):
    return layer[:, 0, :]


############### Encoder-Decoder ###################
class ProjectionHead(nn.Module):
    def __init__(
        self,
        input_dim,
        projection_dim,
        dropout = 0.2
    ):
        super().__init__()
        self.projection = nn.Linear(input_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        #self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        x = self.dropout(x)
        x = self.projection(x)
        #x = self.gelu(projected)
        #x = self.fc(x)
        #x = x + projected
        #x = self.layer_norm(x)
        x = F.normalize(x, p=2, dim=-1)
        return x


def get_encoder(args):
    if args.bert_type == "bert":
        return BertModel.from_pretrained(args.bert_config)

    elif args.bert_type == "align":
        return AlignTextModel.from_pretrained(args.align_config)

    elif args.bert_type == "clip": #512 text dim
        return CLIPTextModel.from_pretrained(args.clip_config)
    
    elif args.bert_type == "blip":
        return BlipTextModel.from_pretrained(args.blip_config)
    
    elif args.bert_type == "falva":
        return FlavaTextModel.from_pretrained(args.falva_config)
    
    elif args.bert_type == "groupvit": #256 text dim
        return GroupViTTextModel.from_pretrained(args.groupvit_config)



class TextEncoder(nn.Module):
    def __init__(self, args):
        super(TextEncoder, self).__init__()
        self.encoder = get_encoder(args)

        print("Loading : ", args.bert_type)

        unfreeze_layers = ['layer.8','layer.9','layer.10', 'layer.11', 'pooler']
        
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True


    def forward(self, captions, mask):
        outputs = self.encoder(input_ids=captions, attention_mask=mask)

        ### Sentence features
        # outputs -> (last_hidden_state, pooler_output, hidden_states)
        # hidden_states -> tuple of lenght 13
        # another way to calculate sentece features
        # sent_feat = (word_feat * mask.unsqueeze(-1)).sum(1) / mask.sum(1).unsqueeze(-1)

        #embeddings = outputs[2][1:]
        words_emb = outputs.last_hidden_state
        sent_emb = outputs.pooler_output
    
        #sent_emb = outputs[0][:,0,:]
        #words_emb = outputs[0][:,1:,:] #outputs[0]
        return words_emb, sent_emb


class Bert_Word_Mapping(nn.Module):
    def __init__(self, feat_dim):
        super(Bert_Word_Mapping, self).__init__()
        Ks = [2, 3, 4]
        in_channel = 1
        out_channel = feat_dim #* 4
        self.convs1 = nn.ModuleList([nn.Conv2d(in_channel, out_channel, (K, 768)) for K in Ks]) #512 for clip

        self.dropout = nn.Dropout(0.1)
        #self.mapping = nn.Linear(out_channel, feat_dim)

    def forward(self, words_emb):
        x = words_emb.unsqueeze(1)  # (batch_size, 1, token_num, embedding_dim)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(batch_size, out_channel, W), ...]*len(Ks)
        return x


class TextHeading(nn.Module):
    def __init__(self, args):
        super(TextHeading, self).__init__()
        self.feat_dim = args.gl_text_dim
        self.bwm = Bert_Word_Mapping(self.feat_dim)
        self.sentence_feat = ProjectionHead(768, args.gl_text_dim)
        self.args = args 

    def get_each_word_feature(self, x):
        bs = x[0].size(0)
        a = x[0].transpose(2, 1)
        b = x[1].transpose(2, 1)
        c = x[2].transpose(2, 1)
        code = []
        for i in range(bs):
            seq = self.args.bert_words_num - 1 - 3 #removing [CLS] token and two positions (for 1->2, 2->3)
            t = [torch.amax(torch.stack((a[i, j], b[i, j], c[i, j])), dim=0) for j in range(seq)]
            t +=  [torch.amax(torch.stack((a[i, seq], b[i, seq])), dim=0)]
            t += [torch.cuda.FloatTensor(a[i, seq+1])]  
            t = torch.stack(t)
            code.append(t)

        code = torch.stack(code)
        code = F.normalize(code, p=2, dim=2)
        return code 


    def get_word_feature(self, x):
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        output = torch.stack((x[0], x[1], x[2])).mean(dim=0)
        output = F.normalize(output, p=2, dim=1)
        return output 


    def forward(self, words_emb, sent_emb):
        sent_emb =  self.sentence_feat(sent_emb) #batch_size x 64
        #words_emb = self.word_feat(words_emb) #batch_size x 20 x 256
        
        x = self.bwm(words_emb)
        words_emb = self.get_each_word_feature(x) 
        #sent_emb = self.get_word_feature(x)

        words_emb = words_emb.transpose(1, 2)
        return words_emb, sent_emb


class ImageHeading(nn.Module):
    def __init__(self, args):
        super(ImageHeading, self).__init__()
        self.project_global = ProjectionHead(input_dim=512, projection_dim=args.gl_img_dim)
        self.imim = IMIM(args, channel_dim = 256)
        
    def forward(self, global_image, local_image):
        local_image = self.imim(local_image)
        global_image = self.project_global(global_image) #batch_size x 256

        return  global_image, local_image, 



class IMIM(nn.Module):
    def __init__(self, args, channel_dim):
        super(IMIM, self).__init__()
        self.channel_dim = channel_dim
        self.project_local =  nn.Conv2d(self.channel_dim, args.gl_img_dim, kernel_size=(1, 1), padding=0) 
        #ProjectionHead(input_dim=256, projection_dim=args.gl_img_dim)
        
        self.bn_img = nn.BatchNorm2d(self.channel_dim)
        self.sa = SelfAttention(channel_dim = self.channel_dim, scale=1)
        self.conv1x1_1 = nn.Conv2d(self.channel_dim, self.channel_dim//2, kernel_size=(1, 1)) 
        self.relu = nn.ReLU()
        self.conv1x1_2 = nn.Conv2d(self.channel_dim//2, self.channel_dim, kernel_size=(1, 1)) 
        self.ln = nn.LayerNorm([self.channel_dim, 14, 14])
        
    def forward(self, img):
        img = self.bn_img(img)
        #img = self.sa(img, img)
        #img = self.ln(img)
    
        #img = self.relu(self.conv1x1_1(img))
        #img = self.relu(self.conv1x1_2(img))

        #img = img.permute((0, 2, 3, 1))
        #img = self.project_local(img) #batch_size x 14 x 14 x 256
        img = F.normalize(img, p=2, dim=-1)
        #img = img.permute((0, 3, 1, 2))
        return img



if __name__ == "__main__":
    from easydict import EasyDict as edict
    args = edict()
    args.gl_img_dim = 256
    x = torch.randn(128, 512, 14, 14)
    net = ImageHeading(args)
    y = net(x)
    print(y.shape)
