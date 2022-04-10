
import torch
from torch import nn
from transformers import DistilBertModel

class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.classification_head = nn.Linear(768,2)
        self.drop = nn.Dropout(p=0.3)

    def forward(self, input_tokens):
        rep = self.drop(self.bert(input_tokens).last_hidden_state[:,0,:].squeeze(dim=1))
        pred = self.classification_head(rep)

        return torch.softmax(pred, dim=1)
