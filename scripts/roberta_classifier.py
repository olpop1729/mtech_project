import os
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def load_hope_data(data_file):
    df = pd.read_csv(data_file)
    texts = df['text'].tolist()
    labels = [1 if sentiment == "Hope" else 0 for sentiment in df['binary'].tolist()]
    return texts, labels
    
train_file = "train.csv"
val_file = "val.csv"

train_texts, train_labels = load_hope_data(train_file)
val_texts, val_labels = load_hope_data(val_file)

class TextClassificationDataset(Dataset):
	def __init__(self, texts, labels, tokenizer, max_length):
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


class RoBERTaClassifier(nn.Module):
	def __init__(self, roberta_model_name, num_classes):
		super(RoBERTaClassifier, self).__init__()
		self.bert = RobertaModel.from_pretrained(roberta_model_name)
		self.dropout = nn.Dropout(0.1)
		self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
		
	def forward(self, input_ids, attention_mask):
		outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
		pooled_output = outputs.pooler_output
		x = self.dropout(pooled_output)
		logits = self.fc(x)
		return logits
        
def train(model, data_loader, optimizer, scheduler, device):
    model.train()
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)


def predict_sentiment(text, model, tokenizer, device, max_length=128):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
    	outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    	_, preds = torch.max(outputs, dim=1)
    return "Hope" if preds.item() == 1 else "Not Hope"
		
		
roberta_model_name = 'roberta-base'
num_classes = 2
max_length = 128
batch_size = 16
num_epochs = 4
learning_rate = 5e-6

#train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

tokenizer = RobertaTokenizer.from_pretrained(roberta_model_name)
train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RoBERTaClassifier(roberta_model_name, num_classes).to(device)

optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train(model, train_dataloader, optimizer, scheduler, device)
    accuracy, report = evaluate(model, val_dataloader, device)
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(report)
    
    
torch.save(model.state_dict(), "roberta_polyhope_multi_classifier.pth")

test_text = "The movie was great and I really enjoyed the performances of the actors."
sentiment = predict_sentiment(test_text, model, tokenizer, device)
print("The movie was great and I really enjoyed the performances of the actors.")
print(f"Predicted sentiment: {sentiment}")
