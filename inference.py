import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
from tokenizing import tokenizer as tk
from util import load_data, MODELS, TASKS, DATASETS
from util import get_random_sentences
from tqdm import tqdm
import numpy as np

class ModelInference:
    def __init__(self, model: str, task: str, optimized_model: str = None, token_length: int = 30):
        self.token_length = token_length
        #tk.tokenizer = AutoTokenizer.from_pretrained(MODELS[model], use_fast=True, padding='max_length', truncation=True, max_length=token_length)
        print(task, token_length)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODELS[model], num_labels=2 if task == 'CiteWorthiness' else 7, problem_type="single_label_classification" if task == 'CiteWorthiness' else "multi_label_classification")
        if optimized_model is not None:
            self.model.load_state_dict(torch.load(f'finetuned_models/{task}/{optimized_model}/pytorch_model.bin', map_location=torch.device('cpu')))
        self.model.eval()
        self.model.to('cpu')
    
    def predict_token(self, tokens: dict):
        batch = {k: v.to('cpu', non_blocking=True) for k, v in tokens.items()}
        with torch.no_grad():
            start_time = time.process_time()
            start_wall_time = time.time()
            outputs = self.model(**batch)
            end_wall_time = time.time()
            end_time = time.process_time()
            cpu_time = end_time - start_time
            wall_time = end_wall_time - start_wall_time
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
        return predicted_class, cpu_time, wall_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=TASKS, default='SectionClassification')
    parser.add_argument('--dataset', choices=DATASETS.keys(), default='CiteWorth')
    parser.add_argument('--model', default='BERT')
    parser.add_argument('--number_sentences', default=1000, type=int)
    parser.add_argument('--optimized_model', type=str)
    parser.add_argument('--token_length', default=30, type=int)
    args = parser.parse_args()

    inference_executor = ModelInference(model=args.model, task=args.task, optimized_model=args.optimized_model, token_length=args.token_length)
    total_cpu_time = 0
    total_wall_time = 0
    cpu_times = []
    wall_times = []

    test_dataloader = load_data(dataset_name=args.dataset, split_name='test', task_type=args.task, batch_size=1, dataset_path=f'datasets/{args.dataset}/{MODELS[args.model]}', max=args.number_sentences, shuffle=False)


    for batch in tqdm(test_dataloader, desc=f'Inference', unit='batch', position=0):
        predicted_class, cpu_time, wall_time = inference_executor.predict_token(batch)
        total_cpu_time += cpu_time
        total_wall_time += wall_time
        cpu_times.append(cpu_time)
        wall_times.append(wall_time)

    cpu_std_dev = np.std(cpu_times)
    wall_std_dev = np.std(wall_times)


    #print(f'Average CPU time: {total_cpu_time / len(test_dataloader)} seconds. Std Dev: {cpu_std_dev}')
    print(f'Average Wall time: {total_wall_time / len(test_dataloader)} seconds. Std Dev: {wall_std_dev}')

if __name__ == '__main__':
    main()



