import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math

class Relu(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return torch.max(0,x)

class NewGelu(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return 0.5*x*(1+torch.tanh(math.sqrt(2/math.pi)*(x*+0.044715*(torch.pow(x,3)))))

class LayerNorm(nn.Module):
    def __init__(self,ndim,bias):
        super().__init__()
        self.ndim=ndim
        self.weight=nn.Parameter(torch.ones(ndim))
        self.bias=nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self,x,norm_shape,eps=1e-05,):
        return F.layer_norm(x,norm_shape,self.weight,self.bias,eps)
    
class AttentionHeads(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.emb_size%config.num_head==0,"there is somthing wrong with the emb_size and n_head assignment"
        #input of the attention of the model
        self.c_attn=torch.nn.Linear(config.emb_size,3*config.emb_size,bias=config.bias)  #the shape is (batch size,number of elements and 3*embedding dimensions)
        #the output projection
        self.out_proj=torch.nn.Linear(config.num_head*(config.emb_size//config.num_head),config.emb_size,bias=config.bias)
        #drop out layers
        self.dropout=nn.Dropout(config.dropout) #this is normal dropout layer
        self.resid_dropout=nn.Dropout(config.resid_dropout)  #this is the residuals output

        self.num_head=config.num_head  #number of heads of the transformer
        self.emb_size=config.emb_size  #embedding of the input

        #registering a buffer of bias type here
        self.register_buffer("bias",torch.ones(config.block_size,config.block_size).view(1,1,config.block_size,config.block_size))

    def forward(self,x):
        B,T,C=x.size()  #batch size,number of sequence ,embedding size
        #print(f"the shapes are {B.shape},{T.shape},{C.shape}")
        
        q,k,v=self.c_attn(x).split(self.emb_size,dim=2)  # => ()
        q=q.view(B,T,self.num_head,C//self.num_head).transpose(1,2)  #this becomes of shape batch size,num heads,number of sequnces and head size
        k=k.view(B,T,self.num_head,C//self.num_head).transpose(1,2)
        v=v.view(B,T,self.num_head,C//self.num_head).transpose(1,2)   #the final shape is (B,nh,T,hs)

        #how are the shapes supposed to be handled...-> (B,nh,T,hs)  * (B,nh,hs,T)  -> (B,nh,T,T)

        attn=q@k.transpose(-2,-1)*(1/math.sqrt(k.shape[-1]))  #shape is->  (B,nh,T,T)
        attn=attn.masked_fill(self.bias[:,:,:T,:T]==0,float("-inf"))  
        attn=F.softmax(attn,dim=-1)
        attn=self.dropout(attn)
        y=attn@v  #this the final output   #(B,nh,T,T) * (B,nh,T,hs) => (B,nh,T,hs)
        #the current shape is (B,nh,T,hs)  => (B,T,C) input => output both must have the same shape
        y=y.transpose(1,2).contiguous().view(B,T,C)
   
        #time for the dropout
        y=self.resid_dropout(self.out_proj(y))
        return y

#this is the defualt state of the transformer
@dataclass
class default_config:
    block_size=1024
    emb_size=768
    num_head=12
    dropout=0.0
    resid_dropout=0.1
    bias=True
    num_mlp=6


class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln1=nn.LayerNorm(config.emb_size,config.bias)
        self.attn=AttentionHeads(config)
        self.ln2=nn.LayerNorm(config.emb_size,config.bias)
        #elements of the multi layer perceptron
        
        self.mlp=nn.ModuleDict(dict(
            l1= nn.Linear(config.emb_size,4*config.emb_size), #i chose the shape of the hidden layer 4*config.emb_size by chance 
            act=NewGelu(),
            l2=nn.Linear(config.emb_size*4,config.emb_size),
            dropout=nn.Dropout(config.resid_dropout)
        ))
        m=self.mlp  #this is the multi layer perceptron
        #this is the feed forward format of out inptut
        self.mlpf=lambda x:m.dropout(m.l2(m.act(m.l1(x))))
    def forward(self,x):
        """this is done with the residual connections"""
        r1=self.ln1(self.attn(x))+x
        r2=self.mlpf(r1)+r1
        return r2
class 
