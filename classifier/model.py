import torch
from torch import nn, optim
from transformers import BertModel

class ReviewClassifier(nn.Module):

  def __init__(self, n_classes):
    super(ReviewClassifier, self).__init__()

    # layers
    self.bert_layer = BertModel.from_pretrained('bert-base-cased')
    self.dropout_layer = nn.Dropout(p=0.3)
    self.fc_layer = nn.Linear(self.bert_layer.config.hidden_size, n_classes)
    self.softmax_layer = nn.Softmax(dim=1)

  def forward(self, encodings, attention_mask):
    out = self.bert_layer(input_ids=encodings, attention_mask=attention_mask)
    bert_pooled_out = out[1]
    out = self.dropout_layer(bert_pooled_out)
    out = self.fc_layer(out)
    out = self.softmax_layer(out)
    
    return out