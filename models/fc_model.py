import torch.nn as nn
import torch 
import torch.nn.functional as F 
import numpy as np
import os
from models import iresnet
from transformers import GPT2LMHeadModel
import torch.nn.functional as F


class GPT2Attention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.n_heads = config.num_heads
        assert self.embed_dim % self.n_heads == 0, 'embedding dimension by be divisible by number of heads'
        self.head_size = self.embed_dim // self.n_heads
        self.seq_len = config.seq_len
        
        self.c_attn = nn.Linear(self.embed_dim, self.head_size * self.n_heads * 3,bias=True)
        self.scale = self.head_size ** -0.5
        
        self.register_buffer('mask',torch.tril(torch.ones(1,1,self.seq_len,self.seq_len)))
        
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)
        
        
    def forward(self, x):
        b,t,c = x.shape
        # q,k,v shape individually: batch_size x seq_len x embed_dim
        # we know that qk_t = q x k_t, where q=bxtxhead_dim, k_t=bxhead_timxt
        q,k,v = self.c_attn(x).chunk(3,dim=-1)
        q = q.view(b,t,self.n_heads,self.head_size).permute(0,2,1,3) # batch x n_heads x seq_len x head_dim
        k = k.view(b,t,self.n_heads,self.head_size).permute(0,2,1,3)
        v = v.view(b,t,self.n_heads,self.head_size).permute(0,2,1,3)
        
        qk_t = (q@k.transpose(-2,-1)) * self.scale
        qk_t = qk_t.masked_fill(self.mask[:,:,:t,:t]==0,float('-inf'))
        qk_t = F.softmax(qk_t,dim=-1)
        weights = self.attn_dropout(qk_t)
        
        attention = weights @ v # batch x n_heads x t x head_size
        attention = attention.permute(0,2,1,3).contiguous().view(b,t,c) # batch x t x embed_dim
        
        out = self.c_proj(attention)
        out = self.resid_dropout(out)
        
        return out


class GPT2CrossAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.n_heads = config.num_heads
        assert self.embed_dim % self.n_heads == 0, 'embedding dimension by be divisible by number of heads'
        self.head_size = self.embed_dim // self.n_heads
        self.seq_len = config.seq_len
        
        self.q = nn.Linear(self.embed_dim,self.embed_dim)
        self.k = nn.Linear(self.embed_dim,self.embed_dim)
        self.v = nn.Linear(self.embed_dim,self.embed_dim)
        self.scale = self.head_size ** -0.5
        
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        
        
    def forward(self, q,k,v):
        b,t,c = q.shape
        
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)
        
        q = q.view(b,q.size(1),self.n_heads,self.head_size).permute(0,2,1,3) # batch x n_heads x seq_len x head_dim
        k = k.view(b,k.size(1),self.n_heads,self.head_size).permute(0,2,1,3)
        v = v.view(b,v.size(1),self.n_heads,self.head_size).permute(0,2,1,3)
        
        qk_t = (q@k.transpose(-2,-1)) * self.scale
        qk_t = F.softmax(qk_t,dim=-1)
        weights = self.attn_dropout(qk_t)
        
        attention = weights @ v # batch x n_heads x t x head_size
        attention = attention.permute(0,2,1,3).contiguous().view(b,t,c) # batch x t x embed_dim
        
        out = self.c_proj(attention)
        out = self.resid_dropout(out)
        
        return out


class GPT2MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.mlp_ratio = config.mlp_ratio
        self.mlp_dropout = config.mlp_dropout
        
        self.c_fc = nn.Linear(self.embed_dim,self.embed_dim*self.mlp_ratio)
        self.c_proj = nn.Linear(self.embed_dim*self.mlp_ratio,self.embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(self.mlp_dropout)
        
    def forward(self,x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GPT2Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.ln_1 = nn.LayerNorm(self.embed_dim)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(self.embed_dim)
        self.mlp = GPT2MLP(config)
        self.ln_3 = nn.LayerNorm(self.embed_dim)
        self.cross_attn = GPT2CrossAttention(config)
        
    def forward(self,x, enc_out):
        x = x+self.attn(self.ln_1(x))
        x = x+self.cross_attn(self.ln_2(x),enc_out, enc_out)
        x = x+self.mlp(self.ln_3(x))
        return x


class SelfAttention(nn.Module):
    def __init__(self, ):
        super(SelfAttention, self).__init__()
        self.scale_dim = 64
        self.query_proj = nn.Linear(256, self.scale_dim)
        self.key_proj =  nn.Linear(256, self.scale_dim)
        self.value_proj =  nn.Linear(256, 256)
        self.sqrt_dim = np.sqrt(self.scale_dim)


    def forward(self, x):
        N, D = x.size()
        C = D // 3
        x = x.contiguous().view(N, 3, C)
        query = self.query_proj(x) 
        query = query.contiguous().view(N, self.scale_dim, 3) #N, Scale, HW

        key = self.key_proj(x)
        key = key.contiguous().view(N, self.scale_dim, 3)
        key = key.transpose(2,1) #N, HW, Scale

        # compute attention
        attention = torch.bmm(key, query) / self.sqrt_dim 

        assert attention.size() == (N, 3, 3)
        attention = F.softmax(attention, dim=-1)

        # g transform
        value = self.value_proj(x) #x --> image
        value = value.contiguous().view(N, C, -1)
        value = value.transpose(2, 1) #N, HW, C
        
        # final response
        response = torch.bmm(attention, value)
        response = response.contiguous().view(N, -1)
        return response
    


def load_pretrained_arcface(config):

    if config.resnet_layer == 18:
        model = iresnet.iresnet18(pretrained=False, progress=True)

    elif config.resnet_layer == 50:
        model = iresnet.iresnet50(pretrained=False, progress=True)

    if config.arch == "arcface" and config.resnet_layer == 18:
        weight = config.weight_arcface_18

    elif config.arch == "arcface" and config.resnet_layer == 50:
        weight = config.weight_arcface_50

    elif config.arch == "adaface" and config.resnet_layer == 18:
        weight = config.weight_adaface_18

    elif config.arch == "adaface" and config.resnet_layer == 50:
        weight = config.weight_adaface_50

    weight_dir = os.path.join(config.weights_path, weight)
    checkpoint = torch.load(weight_dir)
    print("loading image encoder: ", weight_dir)

    model.load_state_dict(checkpoint)
    for p in model.parameters():
        p.requires_grad = False 

    return model 


class VisionGPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.arcface = load_pretrained_arcface(self.config)
        self.proj = nn.Linear(768, 768)
        self.l_norm1 = nn.LayerNorm(768)

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.embed_dim),
            wpe = nn.Embedding(config.seq_len,config.embed_dim),
            drop = nn.Dropout(config.emb_dropout),
            h = nn.ModuleList([GPT2Block(config) for _ in range(config.depth)]),
            ln_f = nn.LayerNorm(config.embed_dim)
        ))
        self.lm_head = nn.Linear(config.embed_dim,config.vocab_size,bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        if self.config.is_sa:
            self.sa = SelfAttention()
            self.l_norm2 = nn.LayerNorm(768)
        

    def trainable_gpt_layers(self, trainable=False):  
        layers = [
            self.transformer.wte, self.transformer.wpe,
            self.transformer.ln_f, self.lm_head
        ]
        gpt_layers = [[
            self.transformer.h[i].ln_1,self.transformer.h[i].ln_2,
            self.transformer.h[i].attn,self.transformer.h[i].mlp
        ] for i in range(self.config.depth)]

        for l in gpt_layers:
            layers.extend(l)
        
        unfreeze_layers = ['layer3', 'layer4', 'bn2', 'dropout', 'fc', 'features'] 

        for name, param in self.arcface.named_parameters():
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = trainable

        for layer in layers:
            if not isinstance(layer, nn.Parameter):
                for p in layer.parameters():
                    p.requires_grad = trainable
            else:
                layer.requires_grad = trainable


    @classmethod    
    def from_pretrained(self, config):
        model = VisionGPT2Model(config)
        sd = model.state_dict()
        keys = sd.keys()
        ignore_matches = ['blocks.','cross_attn.','ln_3','.attn.mask', 'proj']
        vit_keys = [key for key in keys if any(match in key for match in ignore_matches)]
        gpt_keys = [key for key in keys if key not in vit_keys]
        
        gpt2_small = GPT2LMHeadModel.from_pretrained('gpt2')
        sd_hf = gpt2_small.state_dict()
        hf_keys = sd_hf.keys()
        hf_keys = [k for k in hf_keys if not k.endswith('.attn.masked_bias')]
        hf_keys = [k for k in hf_keys if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        for k in hf_keys:
            if any(match in k for match in ignore_matches):
                continue
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
            
        model.load_state_dict(sd)
        return model
    


    def forward(self,image, input_ids, labels=None):
        
        gl_feats, lc_feats = self.arcface(image)
        lc_feats = F.adaptive_avg_pool2d(lc_feats, (1, 1))
        lc_feats = lc_feats.view(lc_feats.size(0), -1)

        image = torch.concat((gl_feats, lc_feats), dim=1)
        image = self.proj(image)
        image = self.l_norm1(image) 

        if self.config.is_sa:
            x = self.sa(image)
            image = self.l_norm2(image + x)

        image = torch.unsqueeze(image, dim=1)

        token_embeddings = self.transformer.wte(input_ids) # batch x seq_len
        pos_embs = torch.arange(0,input_ids.size(1)).to(input_ids.device)
        positional_embeddings = self.transformer.wpe(pos_embs)
        input_ids = self.transformer.drop(token_embeddings+positional_embeddings)
        
        for i in range(self.config.depth):
            input_ids = self.transformer.h[i](input_ids, image)
        
        input_ids = self.transformer.ln_f(input_ids)
        
        if labels is not None:
            lm_logits = self.lm_head(input_ids)
            loss = F.cross_entropy(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
            return loss
        
        lm_logits = self.lm_head(input_ids[:,[-1],:])
        return lm_logits


    def generate(self,  
                 image,  
                 sequence, 
                 tokenizer, 
                 max_tokens=48,  
                 temperature=1.0, 
                 deterministic=False):
        
        for _ in range(max_tokens):
            out = self(image, sequence)
            out = out[:,-1,:] / temperature
            probs = F.softmax(out, dim=-1)
            if deterministic:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs,num_samples=1)
            sequence = torch.cat([sequence,next_token],dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break
            
        return sequence.cpu().flatten()



if __name__ == "__main__":
    from types import SimpleNamespace

    config = SimpleNamespace(
        vocab_size = 50_257,
        embed_dim = 768,
        num_heads = 12,
        seq_len = 1024,
        depth = 12,
        attention_dropout = 0.1,
        residual_dropout = 0.1,
        mlp_ratio = 4,
        arch = "arcface",
        resnet_layer = 18,
        mlp_dropout = 0.1,
        emb_dropout = 0.1, 
        weights_path = "./weights/pretrained",
        weight_arcface_18 = "arcface_ir18_ms1mv3.pth")

    arcface = load_pretrained_arcface(config)
    for name, m in arcface.named_modules():
        print(name) 