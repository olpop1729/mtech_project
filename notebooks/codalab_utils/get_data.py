from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch

class GetData():
    def __init__(self, file_name, tokenizer, label_map=None, classification_type=None, batch_size=16):
        self.file_name = file_name
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.batch_size = batch_size
        self.classification_type = classification_type

    def load_text(self):
        df = pd.read_csv(self.file_name)
        return df['text'].tolist()

    def load_data(self, data_file, label_map, classification_type):
        df = pd.read_csv(data_file)
        texts = df['text'].tolist()
        labels = [label_map[sentiment] for sentiment in df[classification_type].tolist()]
        return texts, labels

    def get_dataset(self):
    	t, l = self.load_data(self.file_name, self.label_map, self.classification_type)
    	return TextClassificationDataset(texts=t, tokenizer=self.tokenizer, max_length=128, labels=l)

    def get_text_dataset(self):
        return TextClassificationDataset(texts=self.load_text(), tokenizer=self.tokenizer, max_length=128)

    def get_text_dataloader(self):
        return DataLoader(self.get_text_dataset(), batch_size=self.batch_size)



class TextClassificationDataset(Dataset):
	def __init__(self, texts, tokenizer, max_length, labels=None):
		self.texts = texts
		self.labels = labels
		self.tokenizer = tokenizer
		self.max_length = max_length
        
	def __len__(self):
		return len(self.texts)
        
	def __getitem__(self, idx):
		text = self.texts[idx]
		if self.labels:
			label = self.labels[idx]
		else:
			label = [0]
		encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
		return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}
