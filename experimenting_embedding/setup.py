### this is my implementation of byte pair encoding...
import unicodedata

def get_stats(ids,counts=None):
    counts={} if counts is None else counts
    for pair in zip(ids,ids[1:]):
        counts[pair]=counts.get(pair,0)+1
    return counts

def merge(ids,pair,idx):
    new_ids=[]
    i=0
    while i<len(ids):
        if (i<len(ids)-1 and ids[i]==pair[0] and ids[i+1]==pair[1]):
            new_ids.append(idx)
            i+=2
        else:
            new_ids.append(ids[i])
            i+=1
    return new_ids

def handle_control_charachters(text:str)->str:
    chars=[]
    for s in text:
        if unicodedata.category(s)[0] != "C":
            chars.append(s)
        else:
            chars.append(f"{ord(s):4x}")
    return chars

def back_to_token(s:bytes):
    """ turn byte into an appropriate strings."""
    text=s.decode("utf-8",errors="replace")
    text=handle_control_charachters(text)
    return text

class Tokenizer:
    def __init__(self):
        self.merges={}  #contains the new merged strings
        self.pattern={}
        self.special_tokens={}  #this are
        self.vocab=self._built_vocab()  #contains the vocabulary of the embedder

    def train(self,text,vocab_size,verbose):
        raise NotImplementedError
    
    def _built_vocab(self):
        """this is the vocabulary of the model"""
        #this part is of the origianl encoding value from 0-255 and the merges value we obtained
        vocab={idx:bytes([idx]) for idx in range(256)}
        for (p1,p2),idx in self.merges.items():
            vocab[idx]=vocab[p1]+vocab[p2]
        #this is for the special tokens of the embedder
        for special_value,idx in self.special_tokens:
            vocab[idx]=special_value.encode("utf-8")
        return vocab

    
    def encode(self,text,encoding_number):
        raise NotImplementedError

    def decode():
        raise NotImplementedError

    def save(self,file_path):
        """saves the vocabulary and  model in the specified directory
        Args:
            file_path (dir): this is the prefix path of the model and the vocabulary we are goining to save into
        """
        model_path=file_path+".model"   #the embedding model
        #this part write the neccsary info which are the pattern,special token and merges
        with open("model_path","w") as f:
            f.write("bpe\n")
            f.write(f"{self.pattern}\n")
        
            #this saves the special tokens
            f.write(f"{len(self.special_tokens)} \n")
            
            #the following saves the special token
            for special,idx in self.special_tokens:
                f.write(f"{special},{idx}\n")
            #the following saves the merges
            for id1,id2 in self.merges:
                f.write(f"{id1} {id2}\n")

        #lets save the vocabulary a human like you and i can look at  
        vocab_file=file_path+".vocab"
        #inverted_merges
        inverted_merges={idx:pair for pair,idx in self.merges.items()}
        with open(vocab_file,"w",encoding="utf-8"):
            for idx,token in self.vocab:
                s=back_to_token(token)
                #what is the element has childeren...or is the result of some combinatin(encoding)
                if idx in inverted_merges:
                    idx0,idx1=inverted_merges[idx]
                    s0=back_to_token(idx0)
                    s1=back_to_token(idx1)
                    f.write(f"[{s0}][{s1}]->[{idx}],[{s}]\n")   
                else:
                    f.write(f"[{s}]{idx}\n")  
        

    def load(self,model_path):
        """loading the .model  of the model into a file

        Args:
            file_path (filepath): the path to the .model would be saved
        """
        special_values={}
        merges={}
        idx=256  

        assert model_path.endswith(".model"),"the directory is not correct"
        with open(model_path,"w") as f:
            assert f.readline().strip()=="bpe"
            self.pattern=f.readline().strip()  
            num_specials=f.readline().strip()
            for _ in range(num_specials):
                specials,idx=model_path.readline().strip().split()
                special_values[specials]=int(idx)
            
            #merges in the vocab
            for line in f:
                id1,id2=map(int,line.split())
                merges[[id1,id2]]=idx
                idx+=1
        self.merges=merges
        self.special_values=special_values