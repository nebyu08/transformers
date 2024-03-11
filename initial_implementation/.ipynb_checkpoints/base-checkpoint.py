import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LayerNorm(nn.Module):
    def __init__(self,ndim,bias):
        super().__init__()
        self.ndim=ndim
        self.weight=nn.Parameter(torch.ones(ndim))
        self.bias=nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self,x,norm_shape,eps=1e-05,):
        return F.layer_norm(x,norm_shape,self.weight,self.bias,eps)
    
class Attention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd%config.n_head==0
        #input of the attention of the model
        self.c_attn=torch.Linear(config.emb_size,3*config.emb_size)  #input dimension X and output dimension
        #the output projection
        self.out_proj=torch.Linear(config.n_head,config.emb_size)
        #drop out layers
        self.dropout=nn.Dropout(config.dropout) #this is normal dropout layer
        self.resid_dropout=nn.Dropout(config.resid_dropout)  #this is the residuals output

        self.num_head=config.num_head  #number of heads of the transformer
        self.emb_size=config.emb_size  #embedding of the input

        #registering a buffer of bias type here
        self.register_buffer("bias",torch.ones(config.block_size,config.block_size).view(1,1,config.block_size,config.block_size))

    def forward(self,x):
        B,T,C=x.size()  #batch size,number of sequence ,embedding size
        q,k,v=self.c_attn(x).split(self.emb_size,dim=2)
        q=q.view(B,T,self.num_head,C//self.num_head).transpose(1,2)  #this becomes of shape batch size,num heads,number of sequnces and head size
        k=k.view(B.T,self.num_head,C//self.num_head).transpose(1,2)
        v=v.view(B,T,self.num_head,C//self.num_head).transpose(1,2)   #the final shape is (B,nh,T,hs)

        #how are the shapes supposed to be handled...-> (B,nh,T,hs)  * (B,nh,hs,T)  -> (B,nh,T,T)

        #lets do some attention scores
        # attention_logits=torch.matmul(q,k)
        # attention_logits=attention_logits/math.sqrt(self.emb_size)
        # attention_masked=attention_logits.masked_fill(mask=0,-float("inf"))
        # attention_softmaxed=F.softmax(attention_masked,dim=-1)
        # attention_values=torch.matmul(attention_softmaxed,v)
        

        attn=q@k.transpose(-2,-1)*(1/math.sqrt(k.shape[-1]))  #shape is->  (B,nh,T,T)
        attn=attn.masked_fill(self.bias[:,:,:T,:T]==0,-float("-inf"))  
        attn=F.softmax(attn,dim=attn.shape[-1])
        attn=self.dropout(attn)
        y=attn@v  #this the final output   #(B,nh,T,T) * (B,nh,T,hs) => (B,nh,T,hs)
        #the current shape is (B,nh,T,hs)  => (B,T,C) input => output both must have the same shape
        y=y.reshape(1,2).contiguous().view(B,T,C)

    
        #time for the dropout
        y=self.resid_dropout(self.out_proj(attn))



        return attention_values
    