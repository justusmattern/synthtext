import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")#.to('cuda')

# add the EOS token as PAD token to avoid warnings
model = GPT2LMHeadModel.from_pretrained("model2 train_model.txt", pad_token_id=tokenizer.eos_token_id)#.to('cuda:0')

input_ids = tokenizer.encode('<BOS> Write a positive review about a great movie:', return_tensors='pt')#.to('cuda:0')

# set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3

sample_outputs = model.generate(
    input_ids,
    do_sample=True, 
    max_length=300, 
    top_k=20, 
    top_p=0.97, 
    num_return_sequences=100
)


print("Output:\n" + 100 * '-')
with open(f'generated_reviews{sys.argv[1]}.txt', 'w') as f:
    for i, sample_output in enumerate(sample_outputs):
      f.write("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True))+ "\n")

