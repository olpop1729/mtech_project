from transformers import BertModel, BertTokenizer
import torch
import pandas as pd
from tqdm import tqdm, trange


file_name = '../data/val/val_polyhope_english_cleaned.csv'
model_name = 'bert-base-uncased'

df = pd.read_csv(file_name)
texts = df['text'].tolist()

model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

device = torch.device('cuda')

model.to(device)
embeds = []
with torch.no_grad():
	for text in tqdm(texts):
		enc = tokenizer(text, return_tensors='pt', max_length=128, padding='max_length', truncation=True).to(device)
		output = model(**enc)
		embeds.append(output.pooler_output.cpu())
	
torch.save(embeds, '../data/embeds/polyhope_val_english_cleaned_bbu_0shot.pt')
