import re
from datasets import load_dataset
import pandas as pd
from collections import Counter
import spacy

nlp = spacy.load('en_core_web_sm')

ds = load_dataset('bookcorpus')

texts = ds['train']['text']

names = []

for text in texts:
	doc = nlp(text)
	for e in doc.ents:
		if e.label_ == 'PERSON':
			names.append(e.text)

name_counts = Counter(names)
most_popular_names = name_counts.most_common(100)

print(most_popular_names)

with open('name.txt', 'w') as fp:
	for name in most_popular_names:
		fp.write(name[0] + '\n')

