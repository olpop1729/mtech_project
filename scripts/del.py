from codalab_utils.get_names import Names

test = Names()
#print(test.test_files('spanish'))
#print(test.models())
#print(test.torch_models())
for i, e in enumerate(test.hf_models()):
    print(f'{i} : {e}')
