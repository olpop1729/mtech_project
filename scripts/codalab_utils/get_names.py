import os

data_path = '/home/coep/general/bert/codalab/data/'
train_path = '/home/coep/general/bert/codalab/data/train/'
val_path = '/home/coep/general/bert/codalab/data/val/'
test_path = '/home/coep/general/bert/codalab/data/test/'
embed_path = '/home/coep/general/bert/codalab/data/embeds/'
models_path = '/home/coep/general/bert/codalab/models/'

pretrained_models = '/home/coep/general/bert/codalab/scripts/codalab_utils/pretrained_model_names.txt'

ENG = 'english'
ESP = 'spanish'


class GenNames():
    def __init__(self, **args):
        self.dataset = args['dataset']
        self.lang = args['lang']
        self.clf = args['clf']
        self.tag = args['tag']
        self.short_name = args['short_name']
    
    def base_name(self):
    	return f'{self.dataset}_{self.lang}_{self.clf}_{self.tag}_{self.short_name}'
    
    def hf_model_name(self):
    	return self.base_name()
    	
    def torch_model_name(self):
    	return f'{self.base_name()}.pth'
    	
    def embed_name(self, data, sentiment=False, sentence=False):
    	if sentiment:
    		return f'{data}_{lang}_'
    	return f'{data}_{self.base_name()}.pt'
		

class Names():
	def __init__(self):
		self.val_path = val_path
		self.train_path = train_path
		self.test_path = test_path
		self.models_path = models_path
		
	def select_saved_model(self, param=None):
		if param == 'hf':
			model_list = self.hf_models()
		elif param == 'torch':
			model_list =self.torch_models()
		else:
			model_list = self.saved_models()
		for i, n in enumerate(model_list):
			print(f'\t{i} : {n}')
		index = int(input(f'Select model : '))
		return model_list[index]
		
	def select_hf_model(self):
		model_list = self.hf_models()
		for i, n in enumerate(model_list):
			print(f'\t{i} : {n}')
		index = int(input(f'Select model : '))
		return model_list[index]
		
	def select_bert_model(self):
		model_list = self.pretrained_models()
		for i, n in enumerate(model_list):
			print(f'\t{i} : {n}')
		index = int(input(f'Select model : '))
		return model_list[index]
		
	def select_file(self, param):
		file_list = self.files(param)
		for i, f in enumerate(file_list):
			print(f'\t{i} : {f}')
		index = int(input(f'Select {param} file : '))
		return file_list[index]
			
		
	def files(self, param, lang=None):
		if param == 'test':
			return self.test_files(lang)
		elif param == 'train':
			return self.train_files(lang)
		elif param == 'val':
			return self.val_files(lang)
			
	def pretrained_models(self, lang=None):
		with open(pretrained_models, 'r') as fp:
			lines = fp.readlines()
		res = [i.strip() for i in lines]
		if not lang:
			return res
		else:
			return [i for i in res if lang in i]
			
	def embed_files(self, lang=None):
		res = [i for i in os.listdir(embed_path) if i.endswith('.pt')]
		if not lang:
			return res
		else:
			return [i for i in res if lang in i]
			
	def saved_models(self):
		res = []
		with os.scandir(models_path) as it:
			for e in it:
				if not e.name.startswith('.'):
					res.append(e.name)
		return res
			
	def torch_models(self, lang=None):
		tm = [i for i in os.listdir(models_path) if i.endswith('.pth') or i.endswith('.pt')]
		if not lang:
			return tm
		else:
			return [i for i in tm if lang in i]
			
	def hf_models(self, lang=None):
		res = []
		with os.scandir(models_path) as it:
			for e in it:
				if e.is_dir() and not e.name.startswith('.'):
					res.append(e.name)
		if not lang:
			return res
		else:
			return [i for i in res if lang in i]
			
	def embed_files(self, lang=None):
		if not lang:
			return os.listdir(embed_path)
		else:
			return [i for i in os.listdir(embed_path) if lang in i]
	
	def train_files(self, lang=None):
		if not lang:
			return os.listdir(train_path)
		else:
			return [i for i in os.listdir(train_path) if lang in i]
	
	def val_files(self, lang=None):
		if not lang:
			return os.listdir(val_path)
		else:
			return [i for i in os.listdir(val_path) if lang in i]
			
	def test_files(self, lang=None):
		if not lang:
			return os.listdir(test_path)
		else:
			return [i for i in os.listdir(test_path) if lang in i]
			
#if __name__ == "__main__":
#	test = GetFileNames()
#	print(test.train_files())

