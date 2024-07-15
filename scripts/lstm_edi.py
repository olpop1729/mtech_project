from transformers import T5ForConditionalGeneration, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd
from tqdm import tqdm

class TextClassificationDataset(Dataset):
	def __init__(self, texts, tokenizer, max_length):
		self.texts = texts
		self.tokenizer = tokenizer
		self.max_length = max_length
        
	def __len__(self):
		return len(self.texts)
        
	def __getitem__(self, idx):
		text = self.texts[idx]
		encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
		return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten()}

device = torch.device('cuda')
model_name = 'google/flan-t5-small'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

df = pd.read_csv('../data/train/train_polyhope_spanish.csv')
texts = df['text'].tolist()
tds = TextClassificationDataset(texts, tokenizer, 128)
tdd = DataLoader(val_dataset, batch_size=32)

tran_texts = []

for batch in tdd:
	input_ids = batch['input_ids'].to(device)
	output = model.generate(input_ids)
	for i in output:
		tran_texts.append(tokenizer.decode(i, skip_special_tokens=True))

