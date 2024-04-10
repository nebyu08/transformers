import regex as re

class tokenizer:
    def __init__(self):
        self.pattern=re.compile(r"'ve|'ll")

    def token(self,text):
        # print("text:",text)
        tokenized=re.findall(self.pattern,text)
        print(tokenized)

        return tokenized


temp=tokenizer()
text="this is just random text ha've"
print(list(temp.token(text)))

# tokenize=re.compile(r"""'ve|'er|'ll|""")
# print(re.findall(tokenize,text))