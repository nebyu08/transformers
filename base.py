from .setup import Tokenizer,get_stats,merge

class BaseTokenizer(Tokenizer):
    def __init__(self,text):
        super().__inti__()
    def train(self,text,vocabsize,verbose):
        assert vocabsize>=256,"vocab size must be large than 256."
        
        num_embedding=vocabsize-256
        encoded=text.encode("utf-8")
        ids=list(encoded)
        
        vocab={idx:bytes([idx]) for idx in range(256)}
        merges={}
        i=0
        while i<num_embedding:
            ids=get_stats(ids)  #get the pair of numbers
            pair=max(ids,key=ids.get)
            idx=256+i
            
            ids=merges(ids,pair,idx)
            merges[pair]=idx
            vocab[idx]=vocab[pair[0]]+vocab[pair[1]]

            if verbose:   #ready for update
                print(f"{i}/{num_embedding} {pair[0]} {pair[1]}-->{idx}")
        self.merge=merges
        self.vocab=vocab
    
    def decode(self,ids):
        text_in_bytes=b"".join(self.vocab[i] for i in ids)
        text=text_in_bytes.decode("utf-8",errors="replace")
        return text
    
    def encode(self,text):
        ids=list(text.encode("utf-8"))
        while len(stats)>2:
            stats=get_stats(ids)
            pair=min(stats,key=lambda a:self.merge.get(a,float("inf")))
            stats=merge(stats,pair,)
            if pair not in self.merge:
                break
            idx=self.merge[pair]
            stats=merge(stats,pair,idx)
        return stats
