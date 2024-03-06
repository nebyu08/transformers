from base import BaseTokenizer

with open("data/swifty.txt","r") as f:
    swifty=f.read()
    
#print(swifty)
tokenizer=BaseTokenizer() 
vocab_size=300
tokenizer.train(swifty,vocabsize=vocab_size,verbose=False) #train the embed
print(f"the tokenizer values are: {tokenizer.merge}")