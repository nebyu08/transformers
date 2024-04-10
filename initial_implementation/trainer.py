from torch.utils.data import DataLoader
from utils import CfgNonde as CN
from base import GPT
import torch
import torch.nn as nn
import time

class Trainer:
    @staticmethod
    def get_default():
        C=CN()
        # C.n_layers=12
        # C.block_size=512
        # C.emb_size=512
        # C.vocab_size=50000
        # C.num_heads=3
        # C.att_dropout=0.0
        # C.emb_drop=0.0
        # # C.resid_dropout=0.1
        # C.bias=False
        # C.num_mlp=3
        # C.model_type=None
        C.device="auto"
        C.max_iter=None
        C.num_workers=2  #todo change to the number of workers available
        C.batch_size=32

    def __init__(self,model,train_dataset,config=None):
        if config.device=="auto":
            self.device="cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device=config.device

        self.config=config
        self.num_workers=config.num_workers  #? isolated?

        self.model=model.to(self.device)
        self.dataloader=DataLoader(train_dataset,
                                   num_workers=self.num_workers,
                                   batch_size=self.config.batch_size,
                                   shuffle=False
                                   )  
        self.iter_num=0
        self.iter_time=0  
    def add_callbacks(self,onevent,callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self,onevent,callback):
        self.add_callbacks[onevent]=[callback]

    def triger_callback(self,callback):
        for callback in self.set_callback.get(callback,[]):
            callback(self)

    def run(self):
        iter_num=0
        start_time=time.now()  #this is the starting time of the training
        #lets train the model
        model,config=self.model,self.config
        model.train()
        iter_num+=1
        #self.optimier=self.model.optim()  #setup right?
        self.loss=nn.CrossEntropyLoss()
        self.data_iter=iter(self.dataloader)
        while True:
            try:
                batch=next(self.data_iter)
            #stop iteration and extract the individual data's
            except StopIteration:
                #once we reached end of data reinitiaze the data
                self.data_iter=iter(self.dataloader)
                batch=next(self.data_iter)

            self.optimier=model.configure_optimizer()
            #move the data into the device
            batch=[i.to(self.device) for i in batch]
            x,y=iter(batch)

            #calculate the loss
            probs,loss=self.model(x,y)
            model.zero_grad()  #avoiding accumulation of gradient
            #optimizer and a step
            self.optimier.step()
            if config.max_iter is not None and iter_num<=config.max_iter:
                break

    #todo add:1)add timing 
            #2)add callbacks
            