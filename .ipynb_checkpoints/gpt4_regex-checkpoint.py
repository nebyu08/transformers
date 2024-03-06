import regex as re
from base import BaseTokenizer
from setup import merge,get_stats

gpt2_regexpression=r"""(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
gpt4_regexpression=r"""(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(BaseTokenizer):
    def __inti__(self,pattern):
        super().__init__()
        self.special_tokens={}
        self.inverse_special_tokens={}
        self.pattern=gpt4_regexpression if pattern is None else pattern
        self.pattern_compile=re.compile(self.pattern)  #this is compiled object of the regex

    def train(self,text,vocab_size,versbose=False):
        assert vocab_size>=256,"make a good an appropriate vocab size"
        num_embedding=vocab_size-256
        chunks=list(re.findall(self.pattern_compile,text))
        ids=[list(i.encode("utf-8")) for i in chunks]  #list of encoding numbers it s a list of list
       
        merges={}
        vocab={idx:bytes([idx]) for idx in range(256)}
        #find the pairs counts across the chunks and encode them
        for i in range(num_embedding):
            stats={}
            for chunks_ids in range(len(ids)):
                get_stats(chunks_ids,stats)
            pair=max(stats,key=stats.get)
            idx=256+i
            #when u merge your doing it for every pair
            ids=[merge(chunk_ids,pair,idx) for chunk_ids in ids]
            #lets save the merges
            merges[pair]=idx
            vocab[idx]=vocab[pair[0]] + vocab[pair[1]]
            if versbose:
                print(f"{i+1}/{num_embedding},{pair} ==> {idx} ({vocab[idx]} had {stats[idx]} occurances")

        self.merges=merges
        self.vocab=vocab

    def register_special_token(self,special_tokens):
        self.special_tokens=special_tokens
        self.inverse_special_tokens={v:k for k,v in self.special_tokens.items()}
        
    def decode(self,ids):
        byte_values=[]
        for chunks_ids in ids:
            if chunks_ids in self.vocab:
                byte_values.append(byte_values)
            elif chunks_ids in self.inverse_special_tokens:
                byte_values.append(self.inverse_special_tokens[chunks_ids].encode("utf-8"))
            else:
                raise ValueError("invalid token id: {idx}")
        text_bytes=b"".join(byte_values)
        text=text_bytes.decode("utf-8",errors="repalce")
        return text
    
    def _encode_chunk(self,text_bytes):
        ids=list(text_bytes)
        while len(ids)>2:
            stats=get_stats(ids)
            #this finds the merges with the lowes merge index
            pair=min(stats,key=lambda p:self.merges.get(p,float("inf")))
            if pair not in self.merges:
                break
            idx=self.merges[pair]
            ids=merge(ids,pair,idx)
        return ids
    
    def _encode_ordinary(self,text):
        "encoding that ignores special tokens"
        text_chunk=re.findall(self.pattern_compile,text)
        ids=[]
        for chunk in text_chunk:
            chunk_bytes=chunk.encode("utf-8")
            chunk_ids=self._chunk_encode(chunk_bytes)
            ids.extend(chunk_ids)
        return ids
    
    def encode(self, text,special_allowed):
        """this encodint handles special tokens
        Args:
            text (string): inputed text to be encoded
            special (bool): allow for special tokens
        """
        special=None
        if special_allowed=="all":
            special=self.special_tokens
        elif special_allowed=="none":
            special={}
        elif special_allowed=="none_raise":
            special={}
        elif isinstance(special_allowed,set):
            special={v:k for k,v in special_allowed.items() if k in self.special_tokens}
        else:
            raise f"the special_allowed={special_allowed} Not allowed"
        
        #if there are no special allowed then treat it as normal encode or or encode oridnary
        if not special_allowed:
            return self._encode_ordinary(text)
        special_pattern="("+"|".join(re.escape(k) for k in special)
        special_chunks=re.split(text,special_pattern)
        ids=[]
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.append(self._encode_ordinary(part))

        return ids
