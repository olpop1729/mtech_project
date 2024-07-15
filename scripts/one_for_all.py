#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.optim import AdamW
from transformers import (AutoTokenizer,
                          AutoConfig,
                          AutoModelForSequenceClassification,
                          TrainingArguments,
                          Trainer,
                          get_linear_schedule_with_warmup,)
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import os
import evaluate
import numpy as np


# In[2]:


def load_hope_data(data_file, label_map, classification_type):
    df = pd.read_csv(data_file)
    texts = df['text'].tolist()
    labels = [label_map[sentiment] for sentiment in df[classification_type].tolist()]
    return texts, labels


# In[3]:


polyhope_binary_labels = {'Hope':1, 'Not Hope':0}
polyhope_inv_binary_labels = {v: k for k, v in polyhope_binary_labels.items()}

polyhope_multi_labels = {'Not Hope':0, 'Generalized Hope':1, 'Realistic Hope':2, 'Unrealistic Hope':3}
polyhope_inv_multi_labels = {v: k for k, v in polyhope_multi_labels.items()}

train_path = '../data/train/'
val_path = '../data/val/'


# In[4]:


class_type = input('Select Classification type: \n\t1] Binary.\n\t2] Multiclass.')
if class_type == '1':
    num_labels = 2
    label_type = 'binary'
    label_map = polyhope_binary_labels
else:
    num_labels = 4
    label_type =  'multiclass'
    label_map = polyhope_multi_labels


# In[5]:


train_file_list = os.listdir('../data/train/')
val_file_list = os.listdir('../data/val/')

print('Train file list.')
for i, f in enumerate(train_file_list):
    print(f'\t{i}] {f}')
tf = input('Select train file: ')
tf = int(tf)
train_file = train_path + train_file_list[tf]

print('Validation file list.')
for i, f in enumerate(val_file_list):
    print(f'\t{i}] {f}')
vf = input('Select val file: ')
vf = int(vf)
val_file = val_path + val_file_list[vf]


# In[6]:


train_texts, train_labels = load_hope_data(train_file, label_map, label_type)
val_texts, val_labels = load_hope_data(val_file, label_map, label_type)


# In[7]:


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
		return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': label}


# In[8]:


model_list = ['bert-base-uncased', 'bert-base-cased', 'bert-large-uncased',
              'bert-large-cased',
              'dccuchile/bert-base-spanish-wwm-uncased', 
              'dccuchile/bert-base-spanish-wwm-uncased', 
              'dccuchile/albert-xxlarge-spanish',
              'google/flan-t5-base',
              'google/flan-t5-large',
              'google/flan-t5-xl',
              'google/flan-t5-xxl',
              'google/flan-t5-small'
             ]

print('Model list.')
for i, f in enumerate(model_list):
    print(f'\t{i}] {f}')
m = input('Select model: ')
m = int(m)
model_name = model_list[m]


# In[9]:


tokenizer = AutoTokenizer.from_pretrained(model_name)


# In[10]:


train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, 128)
val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, 128)


# In[11]:


model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)


# In[12]:


metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# In[13]:


output_dir = input('Enter output directory: ')
training_args = TrainingArguments(output_dir=output_dir,
                                  evaluation_strategy="epoch", 
                                  learning_rate=2e-05,
                                  num_train_epochs=4.0,
                                  save_strategy="epoch",
                                  load_best_model_at_end=True,
                                  per_device_train_batch_size=4,
                                  per_device_eval_batch_size=4,
                                  save_only_model=True,
                                  save_total_limit=2
                                  )
#alberta_xxl_spanish_multiclass


# In[14]:


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)


# In[15]:


train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)

metrics = trainer.evaluate(val_dataset)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)


# In[ ]:




