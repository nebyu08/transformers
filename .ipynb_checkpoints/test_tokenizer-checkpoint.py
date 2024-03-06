from base import BaseTokenizer

with open("data/swifty.txt","r") as f:
    swifty=f.read()

tokenizer=BaseTokenizer() 
vocab_size=300
tokenizer.train(swifty,vocabsize=vocab_size,verbose=False) #train the embed
