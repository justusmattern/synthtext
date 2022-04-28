
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
        self.loss_fn_lm = nn.CrossEntropyLoss(reduction='none')
        self.loss_fn_cls = nn.CrossEntropyLoss(reduction='mean')
        
    
    def lm_loss(self, input):
        logits = self.gpt2(input_ids=input).logits.permute(0,2,1)
        loss = self.loss_fn_lm(logits, input)

        return torch.sum(loss, dim=1)

    
    def generate_prompts(self, domains_list):
        prompts_neutral = []
        prompts_toxic = []
        for domains in domains_list:
            neutral = self.prompt.replace('<LABEL>', 'neutral').replace('<DOMAIN>', " and ".join(domains))
            toxic = self.prompt.replace('<LABEL>', 'toxic').replace('<DOMAIN>', " and ".join(domains))
            
            prompts_neutral.append(neutral)
            prompts_toxic.append(toxic)
        
        return prompts_neutral, prompts_toxic

    
    def get_losses(self, prompt_list, whole_text_list):
        tokenized_all = self.tokenizer(whole_text_list, return_tensors='pt', padding=True, truncation=True, max_length=512).input_ids.to(f'cuda:{self.device_id}')
        tokenized_prompt = self.tokenizer(prompt_list, return_tensors='pt', padding=True, truncation=True).input_ids.to(f'cuda:{self.device_id}')

        language_loss = self.lm_loss(tokenized_all) - self.lm_loss(tokenized_prompt)
        return language_loss


    def forward(self, texts, domains_list, labels):
        prompts_neutral, prompts_toxic = self.generate_prompts(domains_list)

        neutral_whole = [f'{prompt} {text}' for prompt, text in zip(prompts_neutral, texts)]
        toxic_whole = [f'{prompt} {text}' for prompt, text in zip(prompts_toxic, texts)]

        loss_scores = [self.get_losses(prompts_neutral, neutral_whole), self.get_losses(prompts_toxic, toxic_whole)]

        label_probs = torch.softmax(-1* torch.stack(loss_scores).permute(1,0), dim=1)
        cls_loss = self.loss_fn_cls(label_probs.cpu(), labels)

        predictions = torch.argmax(label_probs.cpu(), dim=1)
        
        return cls_loss, label_probs, predictions





