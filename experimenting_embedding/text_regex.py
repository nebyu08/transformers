from gpt4_regex import RegexTokenizer

reg=RegexTokenizer()
text="train deez nuts"
vocab_sz=270
reg.train(text,vocab_sz)