import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import pandas as pd
import os
from codalab_utils.get_names import Names

def load_hope_data(data_file, label_map, classification_type):
    df = pd.read_csv(data_file)
    texts = df['text'].tolist()
    labels = [label_map[sentiment] for sentiment in df[classification_type].tolist()]
    return texts, labels

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
             
names = Names()

model_list = names.pretrained_models()
print('Model list.')
for i, f in enumerate(model_list):
    print(f'\t{i}] {f}')
m = int(input('Select model: '))
model_name = model_list[m]

if 'spanish' in model_name:
	train_file_list = names.train_files('spanish')
	val_file_list = names.val_files('spanish')
	test_file_list = names.test_files('spanish')
else:
	train_file_list = names.train_files('eng')
	val_file_list = names.val_files('eng')
	test_file_list = names.test_files('eng')
	
print('Train file list')
for i, f in enumerate(train_file_list):
	print(f'\t{i}) {f}')
tf = int(input('Select train file: '))
train_file = names.train_path + train_file_list[tf]

print('Validation file list')
for i, f in enumerate(val_file_list):
	print(f'\t{i}) {f}')
vf = int(input('Select validation file: '))
val_file = names.val_path + val_file_list[vf]

print('Testing file list')
for i, f in enumerate(test_file_list):
	print(f'\t{i}) {f}')
tstf = int(input('Select testing file: '))
val_file = names.test_path + test_file_list[tstf]

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)



