import os
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (BertTokenizer, BertModel, 
						  AutoTokenizer, 
						  AutoModelForSequenceClassification,
						  get_linear_schedule_with_warmup,
						  )
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from tqdm import tqdm
from codalab_utils.get_names import Names
import sys

names = Names()

def load_hope_data(data_file, label_map, classification_type):
    df = pd.read_csv(data_file)
    texts = df['text'].tolist()
    labels = [label_map[sentiment] for sentiment in df[classification_type].tolist()]
    return texts, labels
    
polyhope_binary_labels = {'Hope':1, 'Not Hope':0}
polyhope_inv_binary_labels = {v: k for k, v in polyhope_binary_labels.items()}

polyhope_multi_labels = {'Not Hope':0, 'Generalized Hope':1, 'Realistic Hope':2, 'Unrealistic Hope':3}
polyhope_inv_multi_labels = {v: k for k, v in polyhope_multi_labels.items()}

edi_labels = {'hs':1, 'nhs':0}
inv_edi_labels = {v: k for k, v in edi_labels.items()}

binary_polyhope = 'binary'
multi_polyhope = 'multiclass'

binary_classes = 2
multi_classes = 4

classification_type = {'0':'binary', '1':'multiclass', '2':'hopeedi'}
label_type = {'0': polyhope_binary_labels, '1':polyhope_multi_labels, '2':edi_labels}


val_file = names.val_path + names.select_file('val')
test_file = val_file.replace('val', 'test')
print('Using test file: ', test_file)

for k, v in classification_type.items():
	print(f'\t{k} : {v}')
c_type = input('Select classification type: ')


val_texts, val_labels = load_hope_data(val_file, label_type[c_type], classification_type[c_type])
test_texts, test_labels = load_hope_data(test_file, label_type[c_type], classification_type[c_type])

class TextClassificationDataset(Dataset):
	def __init__(self, texts, labels,tokenizer, max_length):
		self.texts = texts
		self.labels = labels
		self.tokenizer = tokenizer
		self.max_length = max_length
        
	def __len__(self):
		return len(self.texts)
        
	def __getitem__(self, idx):
		text = self.texts[idx]
		label = self.labels[idx]
		encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
		return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}


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
        
def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)


def predict_sentiment(label_map, text, model, tokenizer, device, max_length=128):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
    	outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    	_, preds = torch.max(outputs, dim=1)
    return label_map[preds.item()]
		
		
#bert_model_name = names.select_bert_model()
bert_model_name = name.models_path + name.select_hf_model()
num_classes = len(label_type[c_type].keys())
max_length = 128
batch_size = 16
num_epochs = 3
learning_rate = 2e-5

#train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

tokenizer = BertTokenizer.from_pretrained(bert_model_name)

val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, max_length)


val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTClassifier(bert_model_name, num_classes).to(device)

ml = names.saved_models()
for i, n in enumerate(ml):
	print(f'\t{i} : {n}')
mn = int(input('Select model to load: '))
load_model_name = names.models_path + ml[mn]
if load_model_name.endswith('.pth'):
	model.load_state_dict(torch.load(load_model_name, map_location=device))

	print('----------------------------------')
	print('Running Validation:')
	acc, rep = evaluate(model, val_dataloader, device)
	print('Validation accuracy: ', acc)
	print(rep)
	print('----------------------------------')
	print('Running Testing:')
	acc, rep = evaluate(model, test_dataloader, device)
	print('Testing accuracy: ', acc)
	print(rep)
else:
	pass


