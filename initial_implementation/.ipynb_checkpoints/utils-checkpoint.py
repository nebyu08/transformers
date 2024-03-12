import numpy as np
import torch
import random
from ast import literal_eval

def seed(seed_num=42):
    """the default value for the seeding number is 42"""
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)

class CfgNonde:
    """this is the implementaion of the YACS:yet another configuration."""
    def __init__(self,**kwargs):
        self.__dict__.update(kwargs)
    def __str__(self):
        return self.__return_string(0)
    def __return_string(self,indent):
        "this is used for prity output representation of the model"
        parts=[]
        for k,v in  self.__dict__.items():
            if isinstance(v,CfgNonde):
                parts.append("%s:\n"%k)
                parts.append(v.__return_string(indent+1))
            else:
                parts.append("%s: %s\n"%(k,v))
        parts=[" "*(indent*4) + p for p in parts]
        return "".join(parts)
    
    def to_dict(self):
        "output the elements of the configuration in dictionary format"
        return {k:v.to_dict() if isinstance(v,CfgNonde) else v for k,v in self.__dict__.items()}
    def merge_from_dict(self,d):
        self.__dict__.update(d)

    def merge_from_args(self,args):
        "updating the values that come from lists of values like [model.lr=1.0]..."
        for arg in args:
            keyval=arg.split("=")
            assert len(keyval)==2,"the length of the key and val must be 2."
            key,val=keyval
            try:
                val=literal_eval(val) #turning it to appropriate value
            except:
                raise ValueError
            
            assert key[:2]=="--","there is something wrong with the structure of the input"
            keys=key[2:]
            keys=keys.split(".")
            obj=self
            for k in keys[:-1]:
                obj=getattr(obj,k)

            leaf_key=keys[-1]
            #final check is for checking the value of the leaf
            assert hasattr(obj,leaf_key),f"the value {keys[-1]} doesn't exist in the attribute of the structure"
            print(f"the command line attribut that has been set is {(key,val)}")
            setattr(obj,leaf_key,val)