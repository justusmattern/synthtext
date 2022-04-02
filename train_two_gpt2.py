from transformers import GPT2Tokenizer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


class TorchTokenizer(GPT2Tokenizer):

    def __call__(self, sentence):
        return [self.encode(s, max_length=1024, truncation=True, return_tensors='pt') for s in sentence]

tokenizer = TorchTokenizer.from_pretrained('gpt2-large')
special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>', 'pad_token': '<PAD>'}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

#print(tokenizer(['<|endoftext|> This is my cute dog']))
#print(tokenizer.encode('<|endoftext|> This is my cute dog', max_length=1024, truncation=True, return_tensors='pt'))


from transformers import (
AutoModelWithLMHead,
AutoConfig,
Trainer,
AutoTokenizer,
TextDataset,
DataCollatorForLanguageModeling,
TrainingArguments)

import os

def modelTrainer(text_path, epochs, model="gpt2-large", batch_size=1, cache_dir = "cache"):

    model = AutoModelWithLMHead.from_pretrained(model)
    #for name, param in model.named_parameters():
         #nums = list(range(35,48))
         #if not any(str(num) in name for num in nums):
         #    param.requires_grad = False
         #    print(name)

    tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
    tokenizer = TorchTokenizer.from_pretrained('gpt2-large')
    special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>', 'pad_token': '<PAD>'}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataset = TextDataset(
      tokenizer=tokenizer,
      file_path=text_path,
      block_size=256
    )
    
    training_args =TrainingArguments(
    output_dir="model2 {}".format(os.path.basename(text_path)),
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    warmup_steps=500,
    save_steps=2000,
    logging_steps=10,
    prediction_loss_only=True
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model()


with open('data/imdb_train.txt', 'r') as f:
    with open('data/train_model.txt', 'w') as f2:
        count = 0
        for line in f:
            count += 1
            if int(line.split(' ')[0]) == 1:
                f2.write('<BOS> Write a positive review about a great movie: '+' '.join(line.split(' ')[1:]).replace('\n', '')+' <EOS>\n\n')
            else:
                f2.write('<BOS> Write a negative review about a bad movie: '+' '.join(line.split(' ')[1:]).replace('\n', '')+' <EOS>\n\n')
            if count > 50:
                break


modelTrainer('data/train_model.txt', 3, model='gpt2-large')

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')

test_pos = []
test_neg = []

dataset = []
with open('data/test.txt', 'r') as f:
    for line in f:
        dataset.append((' '.join(line.split(' ')[1:]).replace('\n', ''), int(line.split(' ')[0])))

"""
import math
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import random
# Load pre-trained model (weights)
model_pos = GPT2LMHeadModel.from_pretrained('./model yelp_positive_train.txt')

test_data = test_pos+test_neg
random.shuffle(test_data)

model_pos.eval()
model_neg.eval()

# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def score(sentence, model):
    tensor_input = tokenizer.encode(sentence, max_length=1024, truncation=True, return_tensors='pt')
    loss=model(tensor_input, labels=tensor_input)[0]
    return math.exp(loss)

print('len test', len(test_data))

pos_scores = []
neg_scores = []
rights = 0
preds = []
for i, (rev, lab) in enumerate(test_data[:1000]):
    if i % 100 == 0:
      print(i)
    print(rev)
    print(lab)
    pos = score(rev, model_pos)
    neg = score(rev, model_neg)
    pos_scores.append(pos)
    neg_scores.append(neg)
    if pos > neg:
      preds.append(0)
      if lab == 0:
        rights += 1
    else:
      preds.append(1)
      if lab == 1:
        rights +=1
    

print('acc', rights/len(test_data[:1000]))


"""
