import os
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from tqdm import tqdm
from codalab_utils.classification_dataset import TextClassificationDataset
from codalab_utils.get_names import Names, GenNames
import json
import sys


names = Names()


def load_hope_data(data_file, label_map, classification_type):
    df = pd.read_csv(data_file)
    texts = df['text'].tolist()
    labels = [label_map[sentiment] for sentiment in df[classification_type].tolist()]
    return texts, labels

def get_tag(txt):
    tok = txt.split('.')[0].split('_')
    if len(tok) > 3:
        return tok[-1]
    return 'none'

def get_lang_dataset(txt):
    if 'english' in txt and 'polyhope' in txt:
        return 'eng', 'poly'
    elif 'spanish' in txt and 'polyhope' in txt:
        return 'esp', 'poly'
    else:
        return 'edi'
    
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

train_file = names.train_path + names.select_file('train')
# val_file = names.val_path + names.select_file('val')
val_file = train_file.replace('train', 'val')
# test_file = names.test_path + names.select_file('test')
test_file = train_file.replace('train', 'test')

for k, v in classification_type.items():
	print(f'\t{k} : {v}')
c_type = input('Select classification type: ')


train_texts, train_labels = load_hope_data(train_file, label_type[c_type], classification_type[c_type])
val_texts, val_labels = load_hope_data(val_file, label_type[c_type], classification_type[c_type])
test_texts, test_labels = load_hope_data(test_file, label_type[c_type], classification_type[c_type])


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
        
def train(model, data_loader, optimizer, scheduler, device):
    model.train()
    for batch in tqdm(data_loader):
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
bert_model_name = names.models_path + names.select_saved_model('hf')
num_classes = len(label_type[c_type].keys())
max_length = 128
batch_size = 16
num_epochs = 3
learning_rate = 2e-5

#train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

tokenizer = BertTokenizer.from_pretrained(bert_model_name)

train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, max_length)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTClassifier(bert_model_name, num_classes).to(device)

optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


with open('codalab_utils/short_names.json', 'r') as fp:
    temp = json.load(fp)
inv_temp = {v: k for k, v in temp.items()}

lang, ds = get_lang_dataset(train_file)
gen_names = GenNames(clf=classification_type[c_type],
                     lang=lang, dataset = ds,
                     tag=get_tag(train_file),
                     short_name=inv_temp[bert_model_name]
                    )
# print('**', gen_names.torch_model_name(), '**')
# if input('Continue: ') != 'y':
#     sys.exit()


for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train(model, train_dataloader, optimizer, scheduler, device)
    accuracy, report = evaluate(model, val_dataloader, device)
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(report)
    

print('----------------------------------')
print('Test set evaluation.')
acc, rep = evaluate(model, test_dataloader, device)
print('Testing accuracy: ', acc)
print(rep)

#########################
print(f'Suggested name: {gen_names.torch_model_name()}')
save_file_name = input('Enter save-file name(.pth): ')
if save_file_name != 'exit':
	save_file_name = save_file_name if save_file_name.endswith('.pth') or save_file_name.endswith('.pt') else save_file_name+'pth'
	torch.save(model.state_dict(), save_file_name)

#model.load_state_dict(torch.load('../models/poly_multi_english_uncased.pth', #map_location=device))
#accuracy, report, predictions = evaluate(model, val_dataloader, device)
#print(f"Accuracy : {accuracy}")
#print(report)


