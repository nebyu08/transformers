from bpe import Encoder,get_encoder,BPETokinizer,get_pairs,byte_to_unicode

My_Encoder=get_encoder()  #initializing the encoder

text="hello my name is nebiyu. and I "

#encoded_text=Encoder.encode_and_show_work(text=text)
print(type(My_Encoder))

print(My_Encoder.bpe("text"))
