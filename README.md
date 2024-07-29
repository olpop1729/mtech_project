# **Optimizing BERT-based Models for Hope Speech Classification**

This project investigates the application of BERT and BERT-like models for classifying hope speech in text data. By employing various fine-tuning techniques, we aim to improve the performance of these models on the PolyHope dataset, contributing to a deeper understanding of hope speech expression.

## **Installation:**

1) Clone or download a zip folder of the project.
2) Navigate inside the project folder. To create a virtual environment, use the command: `python3 -m venv .`
3) Execute the setup.py python script. `python setup.py`


## **Script Details:**
1) `bert_evaluate.py` contains script for evaluating the saved fine-tuned models on validation and test data. Prints the classification report on the console.
2) `bert_classify.py` contains script for fine-tuning models on the dataset. (Requires PyTorch and transformers==4.40.1)
3) `raw_embeds.py` contains script for storing zero-shot embeddings for text data. The file names of the text data have to be hard-coded in the script.
4) `top_names.py` contains script for extracting top 100 names from the book corpus.
5) `one_for_all.py` contains script which uses the trainer class to fine-tune the models. All models with classification head can be finetuned using the script.

 
