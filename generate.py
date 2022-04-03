import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
import sys
import argparse
from utils import *

def generate(model, tokenizer, prompt, num_sequences, gen_data_dir, gen_data_filename):

    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer)
    model = GPT2LMHeadModel.from_pretrained(model, pad_token_id=tokenizer.eos_token_id)
    model.parallelize()

    input_ids = tokenizer.encode('<BOS> ' + prompt + ' ', return_tensors='pt')
    
    sample_outputs = model.generate(
        input_ids,
        do_sample=True, 
        max_length=200, 
        top_k=20, 
        top_p=0.97, 
        num_return_sequences=num_sequences
    )

    final_samples = []
    for sample in sample_outputs:
        final_samples.append(sample.replace('<BOS> ' + prompt + ' ', '').replace('<EOS>', '').replace('\n' ' '))
    
    write_texts_to_file(final_samples, dir=gen_data_dir, filename=gen_data_filename)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str)
    parser.add_argument('--tokenizer')
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--num-sequences', type=int, default=1)
    parser.add_argument('--gen-data-dir', type=str, default='generated_data')
    parser.add_argument('--gen-data-filename')

    args = parser.parse_args()

    generate(*args)
