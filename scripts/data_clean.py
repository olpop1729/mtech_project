import re
from datasets import load_dataset
import pandas as pd
import random

df = pd.read_csv('../data/val/val_polyhope_spanish.csv')
texts = df['text'].tolist()

lines = []
with open('../data/name.txt', 'r') as fp:
	lines = fp.readlines()
names_list = [i[:-1].title() for i in lines[:20]]

proc_texts = []

def remove_emojis(data, flag):
	if not flag:
		return data
	emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
	return re.sub(emoj, '', data)
    
    
for text in texts:
	users = re.sub(r'@([A-Za-z0-9_]+)', '', text)
	collapsed_text = re.sub(r"(#USER# ?){3,}", "Todas, ", users)
	two_tok = re.sub(r"(#USER# ?){2}", 
					random.choice(names_list) + ' y ' + random.choice(names_list) + ', ',
					collapsed_text)
	one_tok = re.sub(r"#USER#", random.choice(names_list) + ', ', two_tok)
	url_pattern = re.compile(r'https?://\S+|www\.\S+')
	url_free = url_pattern.sub('#URL#', one_tok)
	emoji_free = remove_emojis(url_free, False)
	proc_texts.append(emoji_free.replace('\n', ' '))

df = pd.DataFrame(zip(proc_texts, df['id'].tolist(), df['binary'], df['multiclass']), columns=['text', 'id', 'binary', 'multiclass'])
df.to_csv('../data/val/val_polyhope_spanish_cleaned.csv', index= False)

#with open('train_noemoji.txt', 'w') as fp:
#	for line in proc_texts:
#		fp.write(line+'\n')
