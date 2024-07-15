import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


class BERTClassifier(nn.Module):
	def __init__(self, bert_model_name, num_classes):
		super(BERTClassifier, self).__init__()
		self.bert = BertModel.from_pretrained(bert_model_name)
		self.dropout = nn.Dropout(0.1)
		self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
		
	def forward(self, input_ids, attention_mask):
		outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
		pooled_output = outputs.pooler_output
		x = self.dropout(pooled_output)
		logits = self.fc(x)
		return logits
		
		
model_name = 'bert-base-uncased'
num_classes = 2
device = torch.device('cpu')

model = BERTClassifier(model_name, num_classes)
model.load_state_dict(torch.load('bert_classifier.pth', map_location=device))


