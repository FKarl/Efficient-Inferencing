
TASKS = ['SectionClassification', 'CiteWorthiness']
DATASETS = {
    'CiteWorth': 'datasets/CiteWorth_Huggingface.pkl',
    'SciSen': 'datasets/SciSen_Huggingface.pkl',
}

MODELS = {
    'BERT': 'bert-base-uncased',  # SciSen Done
    'mobileBERT': 'google/mobilebert-uncased',  # Overflow error
    'ALBERT': 'albert-base-v2',  # Scisen Done
    'DeBERTa': 'microsoft/deberta-base',  # Scisen Done
    'TinyBERT': 'prajjwal1/bert-tiny',  # Overflow error
    'SciBERT': 'allenai/scibert_scivocab_uncased',  # Overflow error
    'DistilBERT': 'distilbert-base-uncased',  # No token type ids
    'DeBERTa_large': 'microsoft/deberta-v2-xxlarge',
    'Llama-2': 'meta-llama/Llama-2-7b-hf'
}