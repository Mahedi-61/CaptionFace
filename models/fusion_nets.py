import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Optional, Tuple
from torchsummary import summary
from types import SimpleNamespace

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values

    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked

    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked

    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float('Inf'))

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn


############### Fusion ###################
class LinearFusion(nn.Module):
    def __init__(self, args):
        super(LinearFusion, self).__init__()
        self.fc_out = nn.Linear(512, 64) # change 512, 576, 640, 704, 768, args.fusion_final_dim
        self.dropout = nn.Dropout(0.2)


    def forward(self, local_feats, words_emb, global_feats, sent_emb):
        concat_features =  torch.cat((global_feats, sent_emb), dim=1)
        x = self.fc_out(concat_features)
        self.dropout(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, channel_dim, scale=2):
        super(SelfAttention, self).__init__()
        self.inplanes = channel_dim
        self.query_proj = nn.Conv2d(self.inplanes, self.inplanes // scale, 1)
        self.key_proj = nn.Conv2d(self.inplanes,  self.inplanes // scale, 1)
        self.value_proj = nn.Conv2d(self.inplanes, self.inplanes, 1)

        self.sqrt_dim = np.sqrt(channel_dim / scale)


    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        query = self.query_proj(y) # y--> text
        N,C,W,H = query.size()
        query = query.contiguous().view(N, C, H*W) #.transpose(2,1)

        key = self.key_proj(x) # x-->image
        key = key.contiguous().view(N, C, -1)
        key = key.transpose(2,1) #N, HW, C

        # compute attention
        attention = torch.bmm(key, query) / self.sqrt_dim 

        assert attention.size() == (N,H*W,H*W)
        attention = F.softmax(attention, dim=-1)

        # g transform
        value = self.value_proj(x) #x --> image
        N, C, W, H = y.size()
        value = value.contiguous().view(N, C, -1)
        value = value.transpose(2, 1) #N, HW, C
        
        # final response
        response = torch.bmm(attention, value)
        response = response.permute(0, 2, 1) #N, C, HW
        response = response.contiguous().view(N, C, W, H)
        return response
        


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 32, num_heads: int = 1):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model % num_heads should be zero."

        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_head)
        self.query_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.key_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.value_proj = nn.Linear(d_model, self.d_head * num_heads)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)  # BxQ_LENxNxD
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head)      # BxK_LENxNxD
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head)  # BxV_LENxNxD

        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)  # BNxQ_LENxD
        key = key.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)      # BNxK_LENxD
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)  # BNxV_LENxD

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # BxNxQ_LENxK_LEN

        context, attn = self.scaled_dot_attn(query, key, value, mask)

        context = context.view(self.num_heads, batch_size, -1, self.d_head)
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.d_head)  # BxTxND
        return context


class FCFM(nn.Module):
    def __init__(self, channel_dim):
        super(FCFM,self).__init__()
        channel_dim = 36
        self.bn_img = nn.BatchNorm2d(channel_dim)
        self.bn_word = nn.BatchNorm2d(channel_dim)
        self.projection = nn.Linear(256, channel_dim)

        self.sa = SelfAttention(channel_dim, scale=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv = nn.Conv2d(256, channel_dim, kernel_size=(3, 3), padding=0)
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm([channel_dim, 6, 6])
        self.ln_gl_image = nn.LayerNorm([256])
        self.ln_sent = nn.LayerNorm([256])
        self.ln_iw = nn.LayerNorm([32])
        self.linear = nn.Linear(324, 32)


    def forward(self, img, word, gl_img, sent):
        img = self.maxpool(self.relu(self.conv(img)))
        img = self.bn_img(img)
        #img = F.normalize(img, p=2, dim=1)

        word = self.projection(word.transpose(1, 2)) #cap_len, 64
        word = torch.bmm(word.transpose(1, 2), word) / np.sqrt(36) #batch x 256 x 256
        word = word.unsqueeze(-1).view(word.size(0), word.size(1), 6, 6)
        word = self.bn_word(word)
        #word = F.normalize(word, p=2, dim=1)

        #img = self.sa(img, img)
        #iw = self.ln(img)
        iw = word + self.sa(word, img) #img, word
        iw = self.ln(iw)
        iw = self.maxpool(iw)
        iw = iw.view(iw.size(0), -1) #batch_size x 1024

        #img_sent = self.pl_cfa(gl_img, sent)
        #iw = torch.cat((iw, img_sent), dim=1)
        iw =  self.ln_iw(self.linear(iw))
        gl_img = self.ln_gl_image(gl_img) 
        sent = self.ln_sent(sent)
        return torch.concat((iw, gl_img, sent), dim=1) 


class ParagraphLevelCFA(nn.Module):
    def __init__(self):
        super(ParagraphLevelCFA, self).__init__()

        self.mha = torch.nn.MultiheadAttention(embed_dim = 128, num_heads = 1, dropout=0.1, batch_first=True) #64
        self.linear_project = torch.nn.Linear(768, 128)
        self.ln = nn.LayerNorm(64)

    def forward(self, img: Tensor, sent_emb: Tensor) -> Tensor:
        bs = img.size(0)
        img = img.contiguous().view(bs, 8, 64)  #8, 64
        sent_emb = sent_emb.contiguous().view(bs, 1, 64)  #1, 64

        sent_feats = self.mha(sent_emb, img, img)
        sent_feats = sent_feats[0].contiguous().view(bs, -1) #batch_size x 64
        self.ln(sent_feats)
        return sent_feats  



class GPT2Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, seq_len, attention_dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = num_heads
        assert self.embed_dim % self.n_heads == 0, 'embedding dimension by be divisible by number of heads'

        self.head_size = self.embed_dim // self.n_heads
        self.seq_len = seq_len
        
        self.c_attn = nn.Linear(self.embed_dim, self.head_size * self.n_heads * 3,bias=True)
        self.scale = self.head_size ** -0.5
                
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.attn_dropout = nn.Dropout(attention_dropout)
        #self.resid_dropout = nn.Dropout(config.residual_dropout)
        
        
    def forward(self, x):
        b,t,c = x.shape
        # q,k,v shape individually: batch_size x seq_len x embed_dim
        # we know that qk_t = q x k_t, where q=bxtxhead_dim, k_t=bxhead_timxt
        q,k,v = self.c_attn(x).chunk(3, dim=-1)

        q = q.view(b,t,self.n_heads,self.head_size).permute(0,2,1,3) # batch x n_heads x seq_len x head_dim
        k = k.view(b,t,self.n_heads,self.head_size).permute(0,2,1,3)
        v = v.view(b,t,self.n_heads,self.head_size).permute(0,2,1,3)
        
        qk_t = (q@k.transpose(-2,-1)) * self.scale
        qk_t = F.softmax(qk_t,dim=-1)
        weights = self.attn_dropout(qk_t)
        
        attention = weights @ v # batch x n_heads x t x head_size
        attention = attention.permute(0,2,1,3).contiguous().view(b,t,c) # batch x t x embed_dim
        
        #out = self.c_proj(attention)
        #out = self.resid_dropout(out)
        return attention


class GPT2CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attention_dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = num_heads
        assert self.embed_dim % self.n_heads == 0, 'embedding dimension by be divisible by number of heads'
        self.head_size = self.embed_dim // self.n_heads
        
        self.q = nn.Linear(self.embed_dim, self.embed_dim)
        self.k = nn.Linear(self.embed_dim, self.embed_dim)
        self.v = nn.Linear(self.embed_dim, self.embed_dim)
        self.scale = self.head_size ** -0.5
        
        #self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        
        self.attn_dropout = nn.Dropout(attention_dropout)
        #self.resid_dropout = nn.Dropout(config.residual_dropout)
        
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
        
        #out = self.c_proj(attention)
        #out = self.resid_dropout(out)
        
        return attention


class MLP(nn.Module):
    def __init__(self, embd_dim, final_out):
        super().__init__()
        embed_dim = embd_dim
        self.mlp_dropout = 0.40
        self.c_proj = nn.Linear(embed_dim, final_out)
        self.dropout = nn.Dropout(self.mlp_dropout)
        
    def forward(self,x):
        x = self.c_proj(x)
        x = self.dropout(x)
        return F.normalize(x, p=2, dim=1)


class GPT2Block(nn.Module):
    def __init__(self, channel_dim, num_tokens, embed_dim):
        super().__init__()
        config = SimpleNamespace()
        emb_dropout = 0.2
        attention_dropout = 0.3
        depth = 1
        num_heads = 2 
        seq_len_text = num_tokens
        seq_len_image = channel_dim

        self.sa_img = GPT2Attention(embed_dim, num_heads, seq_len_image , attention_dropout)
        self.sa_words = GPT2Attention(embed_dim, num_heads, seq_len_text, attention_dropout)
        self.cross_attn = GPT2CrossAttention(embed_dim, num_heads, attention_dropout)

        self.ln_img1 = nn.LayerNorm(embed_dim)
        self.ln_text1 = nn.LayerNorm(embed_dim)

        self.ln_img = nn.LayerNorm(embed_dim)
        self.ln_text = nn.LayerNorm(embed_dim)
        self.ln_3 = nn.LayerNorm(embed_dim)
        
    def forward(self, img, word):
        #x = x+self.attn(self.ln_1(x))
        img = self.sa_img(img)
        word = self.sa_words(word)
        x = word + self.cross_attn(self.ln_text(word), self.ln_img(img), self.ln_img(img))   
        return self.ln_3(x)
    

class CMF(nn.Module):
    def __init__(self, args):
        super(CMF, self).__init__()
        embed_dim = 36
        channel_dim = 30 #48, 64, 144, 196, 256
        num_tokens = args.bert_words_num - 2
        self.mlp_dropout = 0.3
        r_out = 64

        self.bn_img = nn.BatchNorm2d(channel_dim)
        self.bn_word = nn.BatchNorm1d(num_tokens)

        self.word_projection = nn.Linear(256, embed_dim)
        self.conv = nn.Conv2d(256, channel_dim, kernel_size=(3, 3), bias=False, stride=2, padding="valid")
        self.relu = nn.ReLU()
        self.block = GPT2Block(channel_dim, num_tokens, embed_dim)

        self.dropout = nn.Dropout(self.mlp_dropout)
        dim = (num_tokens + channel_dim) * embed_dim  
        self.mlp_r = MLP(dim, r_out)


    def forward(self, local_feats, word_emb, global_feats, sent_emb):     
        img = self.relu(self.conv(local_feats))
        img = self.bn_img(img)
        img = torch.reshape(img, (img.size(0), img.size(1), -1)) 

        word_emb = self.word_projection(word_emb.transpose(1, 2)) 
        word_emb = self.bn_word(word_emb)

        x = self.block(img, word_emb)
        x = torch.reshape(x, (x.size(0), -1))
        img = torch.reshape(img, (img.size(0), -1))
        x = torch.cat((x, img), dim=-1)
        x = self.mlp_r(x) 
        return x 
        #y = torch.cat((gl_img, sent), dim=1)
        #return torch.cat((x, y), dim=1)
            #gl_img = F.normalize(gl_img, p=2, dim=1)
        #output = F.normalize(output, p=2, dim=1)
        #output = torch.cat((gl_img, output), dim=1)


class CMF_FR(nn.Module):
    def __init__(self, args):
        super(CMF_FR, self).__init__()
        embed_dim = 36
        channel_dim = 30 #48, 64, 144, 196, 256
        num_tokens = args.bert_words_num - 2
        self.mlp_dropout = 0.3
        r_out = 64

        self.bn_img = nn.BatchNorm2d(channel_dim)
        self.bn_word = nn.BatchNorm1d(num_tokens)

        self.word_projection = nn.Linear(256, embed_dim)
        self.conv = nn.Conv2d(256, channel_dim, kernel_size=(3, 3), bias=False, stride=2, padding="valid")
        self.relu = nn.ReLU()
        self.block = GPT2Block(channel_dim, num_tokens, embed_dim)

        self.dropout = nn.Dropout(self.mlp_dropout)
        dim = (num_tokens + channel_dim) * embed_dim  
        self.mlp_r = MLP(dim, r_out)
        self.fusion_final = nn.Linear(args.fusion_final_dim, 512)


    def forward(self, local_feats, word_emb, global_feats, sent_emb, gl_img):     
        img = self.relu(self.conv(local_feats))
        img = self.bn_img(img)
        img = torch.reshape(img, (img.size(0), img.size(1), -1)) 

        word_emb = self.word_projection(word_emb.transpose(1, 2)) 
        word_emb = self.bn_word(word_emb)

        x = self.block(img, word_emb)
        x = torch.reshape(x, (x.size(0), -1))
        img = torch.reshape(img, (img.size(0), -1))
        x = torch.cat((x, img), dim=-1)
        x = self.mlp_r(x) 

        gl_img = F.normalize(gl_img, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)
        output = torch.cat((gl_img, x), dim=1)
        return self.fusion_final(output)


if __name__ == "__main__":
    bs = 16
    img = torch.randn((bs, 256, 14, 14))
    word_emb = torch.randn((bs, 256, 30))
    gl_img = torch.randn((bs, 512))
    sent = torch.randn((bs, 512))
    w = CMF(channel_dim = 256)
    w(img, word_emb)