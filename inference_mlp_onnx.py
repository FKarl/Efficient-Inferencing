import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
from time import time as t_time

from mlp_model import collate_for_mlp_multilabel, collate_for_mlp
from tooling import read_pickle_file
from util import load_data, MODELS, TASKS, DATASETS
from util import get_random_sentences
from tqdm import tqdm
import numpy as np

from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers
from contextlib import contextmanager


def create_model_for_provider(model_path_str: str, provider: str) -> InferenceSession:
    assert provider in get_all_providers(), f"provider {provider} not found, {get_all_providers()}"

    # Few properties that might have an impact on performances (provided by MS)
    options = SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    # Load the model as a graph and prepare the CPU backend
    session = InferenceSession(model_path_str, options, providers=[provider])
    session.disable_fallback()

    return session

class ModelInference:

    def __init__(self, model: str, task: str, optimized_model: str, token_length: int = 30):
        self.token_length = token_length
        #tk.tokenizer = AutoTokenizer.from_pretrained(MODELS[model], use_fast=True, padding='max_length', truncation=True, max_length=token_length)
        print(task, token_length)
        self.model = create_model_for_provider(optimized_model, "CPUExecutionProvider")
        # Get the input names required by the model
        input_names = self.model.get_inputs()
        self.input_names = [input.name for input in input_names]
        print(self.input_names)

    def predict_token(self, tokens: dict):
        batch = {k: v.to('cpu', non_blocking=True) for k, v in tokens.items() if k in self.input_names}
        print(batch)
        with torch.no_grad():
            model_inputs = {k: v.numpy().astype('int64') for k, v in batch.items()}
            print(model_inputs)
            start_time = time.process_time()
            start_wall_time = time.time()
            outputs = self.model.run(None, input_feed=model_inputs)
            end_wall_time = time.time()
            end_time = time.process_time()
            cpu_time = end_time - start_time
            wall_time = end_wall_time - start_wall_time
        return cpu_time, wall_time

    # def predict_token(self, tokens: dict):
    #     batch = tuple(t.to('cpu') for t in tokens)
    #     with torch.no_grad():
    #         batch = {v: v.numpy().astype('int64') for i, v in enumerate(batch)}
    #         model_input_0 = batch[0].numpy().astype('int64')
    #         model_input_1 = batch[1].numpy().astype('int64')
    #         model_input_2 = batch[2].numpy().astype('int64')
    #         start_time = time.process_time()
    #         start_wall_time = time.time()
    #         outputs = self.model.run(None, input_feed=(model_input_0, model_input_1, model_input_2))
    #         end_wall_time = time.time()
    #         end_time = time.process_time()
    #         cpu_time = end_time - start_time
    #         wall_time = end_wall_time - start_wall_time
    #     return cpu_time, wall_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=TASKS, default='SectionClassification')
    parser.add_argument('--dataset', choices=DATASETS.keys(), default='CiteWorth')
    parser.add_argument('--model', choices=MODELS.keys(), default='BERT')
    parser.add_argument('--model_path', type=str, default='./models/bert-base-uncased.onnx')
    parser.add_argument('--number_sentences', default=1000, type=int)
    parser.add_argument('--token_length', default=30, type=int)
    args = parser.parse_args()

    inference_executor = ModelInference(model=args.model, task=args.task, optimized_model=args.model_path, token_length=args.token_length)
    total_cpu_time = 0
    total_wall_time = 0
    cpu_times = []
    wall_times = []

    # load data
    file_path = f'datasets/{args.dataset}/bert-base-uncased/test.pkl'
    dataset_tokenized = read_pickle_file(file_path)

    if args.task == 'SectionClassification':
        task_label = 'section_category'
        collate_fn = collate_for_mlp_multilabel
    else:
        task_label = 'label'
        collate_fn = collate_for_mlp

    input_ids = dataset_tokenized['input_ids']
    label_ids = dataset_tokenized[task_label].tolist()

    test_data = list(zip(input_ids.tolist(), label_ids))
    test_data = test_data[:args.number_sentences]

    test_dataloader = torch.utils.data.DataLoader(test_data, collate_fn=collate_fn,
                                                  batch_size=1,
                                                  pin_memory=True, num_workers=16, shuffle=False)

    for batch in tqdm(test_dataloader, desc=f'Inference', unit='batch', position=0):
        b = {'arg0': batch[0], 'arg1': batch[1], 'arg2': batch[2]}
        print(f'batch0: {len(batch[0])}, batch1: {len(batch[1])}, batch2: {len(batch[2])}')
        cpu_time, wall_time = inference_executor.predict_token(b)
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