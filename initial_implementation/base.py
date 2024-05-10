import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math
from utils import CfgNonde as CN  #CfgNonde

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
    def forward(self,x,norm_shape,eps=1e-05):
        return F.layer_norm(x,norm_shape,self.weight,self.bias,eps)
    
class AttentionHeads(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert (config.emb_size%config.num_heads)==0,"there is somthing wrong with the emb_size and n_head assignment"
        #input of the attention of the model
        self.c_attn=torch.nn.Linear(config.emb_size,3*config.emb_size,bias=config.bias)  #the shape is (batch size,number of elements and 3*embedding dimensions)
        #the output projection
        self.out_proj=torch.nn.Linear(config.num_heads*(config.emb_size//config.num_heads),config.emb_size,bias=config.bias)
        #adding the regulrization methods
        self.att_drop=nn.Dropout(config.att_dropout)
        self.resid_dropout=nn.Dropout(config.resid_dropout)  #this is the residuals output

        self.num_head=config.num_heads  #number of heads of the transformer
        self.emb_size=config.emb_size  #embedding of the input

        #registering a buffer of bias type here
        self.register_buffer("bias",torch.ones(config.block_size,config.block_size).view(1,1,config.block_size,config.block_size))

    def forward(self,x):
        B,T,C=x.size()  #batch size,number of sequence ,embedding size
        #print(f"the shapes are {B.shape},{T.shape},{C.shape}")
        
        q,k,v=self.c_attn(x).split(self.emb_size,dim=2)  # => ()
        q=q.view(B,T,self.num_heads,C//self.num_head).transpose(1,2)  #this becomes of shape batch size,num heads,number of sequnces and head size
        k=k.view(B,T,self.num_heads,C//self.num_head).transpose(1,2)
        v=v.view(B,T,self.num_heads,C//self.num_head).transpose(1,2)   #the final shape is (B,nh,T,hs)

        #how are the shapes supposed to be handled...-> (B,nh,T,hs)  * (B,nh,hs,T)  -> (B,nh,T,T)

        attn=q@k.transpose(-2,-1)*(1/math.sqrt(k.shape[-1]))  #shape is->  (B,nh,T,T)
        attn=attn.masked_fill(self.bias[:,:,:T,:T]==0,float("-inf"))  
        attn=F.softmax(attn,dim=-1)
        attn=self.att_drop(attn)
        y=attn@v  #this the final output   #(B,nh,T,T) * (B,nh,T,hs) => (B,nh,T,hs)
        #the current shape is (B,nh,T,hs)  => (B,T,C) input => output both must have the same shape
        y=y.transpose(1,2).contiguous().view(B,T,C)
   
        #time for the dropout
        y=self.resid_dropout(self.out_proj(y))
        return y

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

    
class GPT(nn.Module):
    """this is the all the where all the configuration comes to build the model.
    """
    @staticmethod
    def get_default_config():    
        C=CN()  #an instance of the yacs(yet another configuration)
        C.n_layer=2
        C.emb_size=300
        C.vocab_size=123 #None
        C.num_heads=5
        C.att_dropout=0.1
        C.emb_drop=0.0
        C.resid_dropout=0.0
        C.bias=True
        C.num_mlp=6
        C.block_size=123 #None
        C.model_type=None  #this is the equivalent of none for string present here 
        return C

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config=GPT.get_default_config()

        assert config.block_size is not None,"must setup the block size" 
        assert config.vocab_size is not None,"must setup the vocab size"
        self.block_size=config.block_size

        params_given=all([config.n_layer is not None,config.num_heads is not None,config.emb_size is not None]) #raised if all are true or all are false
        type_given=config.model_type is not None
        assert type_given^params_given,"either specify the modl type or give the hyper-parameters." #this makes either the model is given or the parameters of the model is given
  
        self.transformer=nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size,config.emb_size),  #this is the word to embdding dimension
            etp=nn.Embedding(config.block_size,config.emb_size),  #this is adding of the positional embedding to each tokens(embdding know)
            drop=nn.Dropout(config.emb_drop),
            blocks=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ly_n=nn.LayerNorm(config.emb_size)  #change from dropout to layernorm
        ))
        #this is the last layer(the language model head)
        self.lm_head=nn.Linear(config.emb_size,config.vocab_size)

        if type_given:
            config.merge_from_dict({
                'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
                # GPT-2 configs
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
                # Gophers
                'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
                # (there are a number more...)
                # I made these tiny models up
                'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
            }[config.model_type])
        
        if config.model_type is None:
            print("the modely type is not defineds or its custom made.")

        #parameter initialization
        self.apply(self._init_weight)
        
        #weight initilization for the output projection of the transformer
        for pn,p in self.named_parameters():
            if pn.endswith("out_proj.weight"):
                torch.nn.init.normal_(p,mean=0,std=1)
        num_params=sum(p.numel() for p in self.parameters())
        print(f"the number of parameters is {num_params/1e6}M")

    def _init_weight(self,module):
        "different part of the configuration get to be initialized"
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0,std=0.2)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0,std=0.02)
        elif isinstance(module,nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def confiure_optimizer(self,trainer_config):
        """for is used for making the optimization process of torch faster.
        """
        decay=set()
        no_decay=set()
        grad_modules=[nn.Linear,]  #this has been open on purpose
        nograd_modules=[nn.Embedding,nn.LayerNorm]
        for mn,m in self.named_modules():
            for pn,p in m.named_parameters():
                #there seems to be open for change here in the fpn=pn

                #the full path name of the parameters and modules is
                fpn="%s.%s"%(mn,pn) if mn else pn
                if pn.ends_with("bias"):
                    no_decay.add(fpn)
                elif pn.ends_with("weight") and isinstance(m,grad_modules):  #if its is part of the grad list
                    decay.add(fpn)
                elif pn.ends_with("weight") and isinstance(m,nograd_modules):   #if it is not part of the no grad list
                    no_decay.add(fpn)
        
        tot_params={pn:p for pn,p in self.named_parameters()}
        union_params=decay | no_decay
        inter_params=decay & no_decay
        
        assert len(inter_params)==0,"there seems to be an overlap between the gradient and non gradient taking weights"
        assert len(tot_params.keys()) - len(union_params)==0,"the total number of params is not eequal with the union of parameters"
        #creating a dictionary with the necssary info.

        optim_groups = [
            {"params": [tot_params[pn] for pn in sorted(list(decay))], "weight_decay": trainer_config.weight_decay},
            {"params": [tot_params[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0}
        ]

        optimizer=torch.optim.Adam(optim_groups,lr=trainer_config.lr)
        return optimizer

    def forward(self,ids,targets=None):
        b,t=ids.size(0),ids.size(1)  #shape is b,t
        assert t<=self.block_size,"the dimension of the inputs is much larger than the block size"
        device=ids.device
        pos=torch.arange(0,t,dtype=torch.long,device=device).unsqueeze(0) #shape is (1,t)..note make sure to make the data type torch.long
        emb_token=self.transformer.wte(ids) #shape is (b,t,emb_size)
        pos_emb=self.transformer.etp(pos) #shape is (1,t,emb_size)
        x=self.transformer.drop(emb_token+pos_emb)
        for block in self.transformer.blocks:
            x=block(x)
        x=self.transformer.ly_n(x)  

        logits=self.lm_head(x)  #this is the logits
        loss=None

        if targets is not None:
            loss=F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1)) 

        return logits,loss

    torch.no_grad()
    def generate(self,ids,temprature=1.0,max_tokens=200,top_k=None,do_samples=False):
        """for generating new tokens from given tokens,during the process the output of this
        is contineuosly fedback to the neural nets.

        Args:
            ids (_type_): _description_
            temprature (_type_): _description_
            max_tokens (_type_): _description_
            tok_k (_type_, optional): _description_. Defaults to None.
        """
        for _ in range(max_tokens):
            ids=ids if ids.size(1) <= self.block_size else ids[:,:-self.block_size,:]
            logits,_=self(ids)  #the logists are of shape (B,N,vocab)
            logits=logits[:,-1,:]/temprature  #dividing along the last element of the second dimension
            if top_k is not None:
                top_log,_=torch.topk(logits,top_k)   
                logits[logits<top_log[:,[-1]]]=float("-inf")
            
            #normalizing the probability distribution
            probs=F.softmax(logits,dim=-1)
            if do_samples:
                ids_next=torch.multinomial(logits,dim=-1)
            else:
                _,ids_next=torch.topk(logits,k=top_k)
            
            #concatinate the values
            ids=torch.cat((ids,ids_next),dim=1)  #to do:add unsqeeze along the second dim here
        return ids