import torch
from torch import nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class CausalClassifier(nn.Module):
    def __init__(self, gpt2_model='gpt2', gpt2_tokenizer = 'gpt2', device_id=0, prompt="Write a <LABEL> comment mentioning <DOMAIN> people:"):
        super(CausalClassifier, self).__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model)
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_tokenizer)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.prompt = prompt
        self.device_id = device_id
        self.loss_fn_cls = nn.CrossEntropyLoss(reduction='mean')

    def generate_prompts(self, domains_list):
        prompts_neutral = []
        prompts_toxic = []
        for domains in domains_list:
            neutral = self.prompt.replace('<LABEL>', 'neutral').replace('<DOMAIN>', " and ".join(domains))
            toxic = self.prompt.replace('<LABEL>', 'toxic').replace('<DOMAIN>', " and ".join(domains))
            
            prompts_neutral.append(neutral)
            prompts_toxic.append(toxic)
        
        return prompts_neutral, prompts_toxic

    def forward(self, texts, domains_list, labels):
        prompts_neutral, prompts_toxic = self.generate_prompts(domains_list)

        neutral_whole = [f'{prompt} {text}' for prompt, text in zip(prompts_neutral, texts)]
        toxic_whole = [f'{prompt} {text}' for prompt, text in zip(prompts_toxic, texts)]

        neutral_loss = self.gpt2(neutral_whole, labels=neutral_whole).loss - self.gpt2(prompts_neutral, labels=prompts_neutral).loss
        toxic_loss = self.gpt2(toxic_whole, labels=toxic_whole).loss - self.gpt2(prompts_toxic, labels=prompts_toxic).loss

        loss_scores = [neutral_loss, toxic_loss]

        label_probs = -1* torch.stack(loss_scores).unsqueeze(dim=0)
        #print(label_probs)
        cls_loss = self.loss_fn_cls(label_probs.cpu(), labels)

        predictions = torch.argmax(label_probs.cpu(), dim=1)
        
        return cls_loss, label_probs, predictions





