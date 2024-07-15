from codalab_utils.get_data import GetData
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained('bert-base-uncased')
fn = '../data/train/train_polyhope_english.csv'

gt = GetData(fn, tok)

gt.get_text_dataloader()