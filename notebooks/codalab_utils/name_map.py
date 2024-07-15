import json

name_map = {'bbu':'bert-base-uncased',
            'bbc':'bert-base-cased',
            'blu':'bert-large-uncased',
            'blc':'bert-large-cased',
            'beto_bc':'dccuchile/bert-base-spanish-wwm-cased',
            'beto_bu':'dccuchile/bert-base-spanish-wwm-uncased',
            'rb':'roberta-base',
            'rl':'roberta-large'
            }
with open('short_name.json', 'w') as fp:
    fp.write(json.dumps(name_map, indent=4))

