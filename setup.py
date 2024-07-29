import os
import json

print('Running setup ...')

cwd = os.getcwd()

path_dict = {}
path_dict['data_path'] = f'{cwd}/data/'
path_dict['train_path'] = f'{cwd}/data/train/'
path_dict['val_path'] = f'{cwd}/data/val/'
path_dict['test_path'] = f'{cwd}/data/test/'
path_dict['embed_path'] = f'{cwd}/data/embeds/'
path_dict['models_path'] = f'{cwd}/models/'
path_dict['pretrained_models'] = f'{cwd}/scripts/codalab_utils/pretrained_model_names.txt'

with open('path_names.json', 'w') as f:
    json.dump(path_dict, f)

path_check = 0
for k, v in path_dict.items():
    if not os.path.exists(v):
        print(f'\tUnable to locate diretory: {v}')
    else:
        print(f'\tFound: {v}')
        path_check += 1

if path_check == len(path_dict.keys()):
    print('Setup completed successfully.')
else:
    print('Error in setup. One or more files are missing.')