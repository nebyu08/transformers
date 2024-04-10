import os
import requests
import regex as re

current_dir=os.getcwd()
new_dir="ned_with_dreads"
new_dir=os.path.join(current_dir,new_dir)
os.makedirs(new_dir,exist_ok=True)

#print("created new directory.")


#downloading files into a file
output_dir=new_dir
#file_path=os.path.join(output_dir,"ima.png")
to_download="C:/Users/nebiy/Documents/deep_learning_scratch"
# with open(to_download,"wb") as f:
#     file=requests.get("https://github.com/karpathy/nanoGPT/blob/master/assets/nanogpt.jpg")
#     f.write(file.content)

#os.path.isdir(new_dir)
#print(os.path.isdir(new_dir))

path_1="C:/Users/nebiy/Documents/deep_learning_scratch/Transformer_implementation"
print(os.getcwd())
p=re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
temp="this is crax'ill u9"
clean_temp=re.findall(p,temp)
print(clean_temp)