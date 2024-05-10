import torch
import regex as re
import os
import requests
import json

def byte_to_unicode():
    """mapping the "ugly" charachters into pretty ones in the unicode structure.
    """
    #this is the list charachter that are not ugly
    usfull_char=list(range(ord("!"),ord("~")+1))+list(range(ord("¡"),ord("¬")+1))+list(range(ord("®"),ord("ÿ")+1))
    all_chrs=usfull_char[:]  #mapping this value into the usefull_char
    inc=0
    for c in range(2**8):
        if c not in usfull_char:
            usfull_char.append(c)
            all_chrs.append(2**8+inc)
            inc+=1
    
    all_chrs=[chr(c) for c in all_chrs]  #turning the integer into charachters
    d=dict(zip(usfull_char,all_chrs))  #this is a mapping of bytes and there respcective characters that are usefull
    return d

def get_pairs(words):
    #precondition before execution
   # assert type(words)==bytes,"the inputed text is not byte type."
    assert len(words)>0,"empty input has been given."

    f_ch=words[0]
    pairs=set()
    for i in words[1:]:
        pairs.add((f_ch,i))
        f_ch=i
    return pairs

class Encoder:
    def __init__(self,encoder,byte_merges):
        #byte merge ranked
        self.byte_ranks=dict(zip(byte_merges,range(len(byte_merges))))
        
        #this is byte operation
        self.byte_encoder=byte_to_unicode()
        self.byte_decoder={v:k for k,v in self.byte_encoder.items()}
        
        #another encoder and decoder on a word level
        self.encoder=encoder
        self.decoder={v:k for k,v in self.encoder.items()}

        #the pattern that is going to be extracted
        self.re = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")



        #lets make some cache here
        self.cache={}

    def bpe(self,token):
        #check wether the word exists in our cache
        if token in self.cache:
            return self.cache[token]
        
        words=tuple(token)
        words_pairs=get_pairs(words)

        #if it can't be paired
        if not words_pairs:
            return token
        
        while True:
            biagram=min(words_pairs,key= lambda p:self.byte_ranks.get(p,float("inf")))
            #condition for exit is that ther are no more pairs that are registered in our vocab
            if biagram not in self.byte_ranks:
                break
            i=0
            first,second=biagram 
            new_word=[]
            #merging the most repeated part of the code
            while i < len(words):
                try:
                    j=words.index(first,i)
                    new_word.extend(words[i:j]) #charachter that are not out of paired value
                    i=j
                except:
                    new_word.extend(words[i:])
                    break

                if i<len(words)-1 and words[i]==first and words[i+1]==second:
                    new_word.append(first+second)
                    i+=2
                else:
                    new_word.append(words[i])
                    i+=1

            #lets update the merging values here
            word=tuple(new_word)  
            if len(word)==1:
                break
            else:
                words_pairs=get_pairs(word)

        final_form="".join(word)
        #lets register our token in our cache
        self.cache[token]=final_form
        return final_form
    
    def encode(self,text):
        """word comes in numbers comes out
        Args:
            text (str): text to be encoded
        """
        tokens=re.findall(self.re,text)
        #tokens=self.re.findall(text)

        #tokens=text.split(' ')
        print(f"inside the enocde fn: this are the tokens:{tokens}")
        bpe_idx=[]  #byte that have been turned into index(nums basically)
        for token in tokens:
            stream_bytes=token.encode("utf-8")   #turn it into bytes
            token_translated="".join(self.byte_encoder[i] for i in stream_bytes)   #turn it into unicode
            #lets merge the tokens
            token_merged=self.bpe(token_translated).split('') 
            #turning it back to words
            token_words=[self.encoder[i] for i in token_merged]  #using GPT-2 encoder to turn text into numbers
            bpe_idx.extend(token_words)   #making one single encoded text format

        return bpe_idx
    
    def encode_and_show_work(self,text):
        """this is used as the encode that is above but its used for debugging codes

        Args:
            text (str): inputed text
        """
       # print("this is the text to encode that the model sees:",text)
        tokens=re.findall(self.re,text)
        #tokens=self.re.findall(text)
        bpe_idx=[]
        parts=[]
        for token in tokens:
            token_bytes=token.encode("utf-8")
            token_byte_translated="".join(self.byte_encoder[i] for i in token_bytes)
            token_merged=self.bpe(token_byte_translated).split(' ')
            token_words=[self.encoder[token_byte] for token_byte in token_merged]
            bpe_idx.extend(token_words)
            parts.append({
                "token":token,
                "token_bytes":token_bytes,
                "token_merged":token_merged,
                "token_words":token_words,
                
            })
        outs={
            "token_idx":bpe_idx,
            "parts":parts,
            "tokens":tokens
        }
        return outs
    
    def decode(self,byte_stream):
        #byte decode
        decode_bytes=[self.decoder(token) for token in byte_stream]
        #map some bytes into strings
        byte_flat="".join(decode_bytes)
        #decode at the byte level
        byte_decode=bytearray([self.byte_decoder(i) for i in byte_flat])
        #decode using the utf-8
        text=byte_decode.decode("utf-8",errors="replace")
        return text 
    
#this are helper functions that are gonna help in vocab,dicitonary and model building stuff
def get_file(remote_file,local_file):
    """get files from some remote file and it moves it into my local repo"""
    # #check whether the file exitst exitst 
    # if not os.path.isfile(local_file):
    #        raise FileNotFoundError("the directory you given me doesn't exist.")

    #check if the parent directory of
    parent_dir=os.path.dirname(local_file)
    if not os.path.exists(parent_dir):
        raise FileNotFoundError(f"the parent directory {parent_dir} doesn't exist")
    

    print(f"downloading files... {os.path.basename(local_file)}")
    with open(local_file,"wb") as f:
        file=requests.get(remote_file)
        f.write(file.content)

def get_encoder():
    home_dir=os.getcwd()
    cached_dir=os.path.join(home_dir,".cache","chatgpt")
    os.makedirs(cached_dir,exist_ok=True)
    
    #lets make the encoder 
    encoder_local_file=os.path.join(cached_dir,"encoder.json")
    remote_encoder_path="https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json"
    #get_file(remote_encoder_path,encoder_local_file)      
    
    #lets load our encoder into our class
    with open(encoder_local_file,"r") as f:
        encoder=json.load(f)
    assert len(encoder)==50257,"there is something wrong with the length of your encoder." #0-256 are taken while the 50,000 arethe vocab size

    #handle the vocabulary 
    local_vocab=os.path.join(cached_dir,"vocab.bpe")
    remote_vocab="https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe"
    #os.makedirs(local_vocab,exist_ok=True)

    #get_file(remote_vocab,local_vocab)
    #save the BPE into a merge bytes 
    with open(local_vocab,"r",encoding="utf-8") as f:
        byte_data=f.read()
    
    #first and last line are blanks so we need to remove them
    byte_merges=[tuple(pairs.split()) for pairs in byte_data.split("\n")[1:-1]]
    assert len(byte_merges)==50000,"there is something wrong with loaded number of merges."
    #lets make a contructor 
    cn=Encoder(encoder,byte_merges)  #returns the classs

    return cn  
    
class BPETokinizer:
    def __init__(self):
        self.encoder=get_encoder()

    def __call__(self,text):
        assert isinstance(text,str)
        ids=self.encoder.encode(text)
        print(f"this are the ids that are returned from the fn:{ids}")
        out=torch.tensor([ids],dtype=torch.long)
        return out

    def decode(self,ids):
        text=self.encoder.decode(ids.tolist())
        return text

text="hello my name is nebiyu"
bp_tokenizer=BPETokinizer()
out=bp_tokenizer(text)
print(out)
