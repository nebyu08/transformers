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
    f_ch=words[0]
    pairs=set()
    for i in words[1:]:
        pairs.add((f_ch,i))
        f_ch=i
    return pairs

class Encoder:
    def __init__(self,encoder,byte_merges):
        self.byte_ranks=dict(zip(byte_merges,range(len(byte_merges))))
        #this is byte operation
        self.byte_encoder=byte_to_unicode()
        self.byte_decoder={j:i for i,j in self.byte_encoder.items()}
        
        #another encoder and decoder on a word level
        self.encoder=encoder
        self.decoder={v:k for k,v in self.encoder.items()}

        #self.byte_ranks
        #lets do some preprocessing here
        self.re=re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{N}+| ?[\{L}+|?\s\p{L}\p{N}]+\s+(?!\S|\s)""")

        #lets make some cache here
        self.cache={}

    def bpe(self,token):
        #check wether the word exists in our cache
        if token in self.cache:
            return self.cache[token]
        
        words=tuple(token)
        words_pairs=get_pairs(words)

        #if it can't be paire
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
            word=tuple(new_word)  #the new
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
        bpe_idx=[]  #byte that have been turned into index
        for token in tokens:
            stream_bytes=token.encode("utf-8")
            token_translated=["".join(self.byte_encoder(i)) for i in stream_bytes]
            #lets merge the tokens
            token_merged=self.bpe(token_translated).split(" ")
            #turning it back to words
            token_words=[self.encoder[i] for i in token_merged]
            bpe_idx.extend(token_words)
        return bpe_idx
    
    def encode_and_show_work(self,text):
        """this is used as the encode that is above but its used for debugging codes

        Args:
            text (str): inputed text
        """
        tokens=re.findall(self.re,text)
        bpe_idx=[]
        parts={}
        for token in tokens:
            token_bytes=token.encode("utf-8")
            token_byte_translated=["".join(self.byte_encoder[i] for i in token_bytes)]
            token_merged=self.bpe(token_byte_translated).split(" ")
            token_words=[self.encoder(token_byte) for token_byte in token_merged]
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
        return parts
    
    def decode(self,byte_stream):
        #byte decode
        decode_bytes=(self.decoder(token) for token in byte_stream)
        #map some "ugly values to normal strings"
        byte_flat="".join(decode_bytes)
        #decode at the byte level
        byte_decode=[self.byte_decoder(i) for i in byte_flat]
        #decode using the utf-8
        text=byte_decode.decode("utf-8",errors="replace")
        return text 
    
def get_file(remote_path,local_path):
    """get files from some remote file and it moves it into my local repo"""
    #check whether a directory exitst 
    if not os.path.file(local_path):
         raise FileNotFoundError("the directory you given me doesn't exist.")
    print("downloading files...")
    with open(local_path,"wb") as f:
        file=requests.get(remote_path)
        f.write(file.content)

def get_encoder():
    home_dir=os.getcwd()
    cached_dir=os.path.join(home_dir,".cache","chatgpt")
    os.makedirs(cached_dir,exist_ok=True)
    
    #lets make the encoder 
    encoder_local_path=os.path.join(cached_dir,"encoder.json")
    remote_encoder_path="https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json"
    get_file(remote_encoder_path,encoder_local_path)      
    
    #lets load our encoder into our class
    with open(encoder_local_path,"r") as f:
        encoder=json.load(f)
    assert len(encoder)==50257,"there is something wrong with the length of your encoder." #0-256 are taken while the 50,000 arethe vocab size

    #handle the vocabulary 
    local_vocab=os.path.join(cached_dir,"vocab.bpe")
    remote_vocab="https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe"
    #os.mkdir(local_vocab)
    get_file(remote_vocab,local_vocab)
    #save the BPE into a merge bytes 
    with open(local_vocab,"r",) as f:
        byte_data=f.read()
    
    #first and last line are blanks so we need to remove them
    byte_merges=[tuple(pairs.split()) for pairs in byte_data.split("\n")[1:-1]]
    assert byte_merges==50257
    #lets make a contructor 
    cn=Encoder(encoder,byte_merges)  #returns the
    return cn  
    

class BPETokinizer:
    def __inti__(self):
        self.encoder=get_encoder()

    def __call__(self,text):
        assert isinstance(text,str)
        ids=[self.encoder.encode(text)]
        out=torch.tensor(ids,dtype=torch.long)

    def decode(self,ids):
        text=self.encoder.decode(ids.tolist())
        return text


#lets experiment
text="my name is nebiyu youhannes."
# e=get_encoder()
# r=e.encode_and_show_work(text)

# print(f"text is:{text}")
# tokens=r["token"]

# print(f"token is {tokens}")
# for part in r["parts"]:
#     print(part)

# print(f"the final output is: {r['token_idx']}")
#
bpe_tokenizer=BPETokinizer()
bpe_tokenizer(text)

#manually integrate the values
vocab=json