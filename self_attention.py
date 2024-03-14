import numpy as np
import torch
import torch.nn as nn

class MultiheadAttention(nn.Module):
    def __init__(self,num_head,d_model,batch_first=True,drop_out=0):
        "assume that the size of the key and query are (3,3)"
        super().__init__()
        self.q=torch.rand(size=(64,64))
        self.k=torch.rand(size=(64,64))
        self.v=torch.rand(size=(64,64))
        self.batch_first=batch_first
        self.drop_out=drop_out
        self.dmodel=d_model

    def encoder(self,inputs):
        temp1=(self.q*(self.k.T))
        temp2=temp1/self.k.shape[0]
        tmep3=temp2*self.v.shape[0]
        

        pass
    def decoder():
        pass