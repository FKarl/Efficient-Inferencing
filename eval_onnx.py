import os
import re
import numpy as np

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from util import load_data, MODELS, TASKS, DATASETS
import argparse
import logging
import torch
from tqdm import tqdm
import optimum.onnxruntime as rt

from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import tokenizing.tokenizer as tk

from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers
from contextlib import contextmanager
from time import time
import yaml, os

from typing import Union

def create_model_for_provider(model_path: str, provider: str) -> InferenceSession:
    assert provider in get_all_providers(), f"provider {provider} not found, {get_all_providers()}"

    # Few properties that might have an impact on performances (provided by MS)
    options = SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    # Load the model as a graph and prepare the CPU backend
    session = InferenceSession(model_path, options, providers=[provider])
    session.disable_fallback()

    return session


@contextmanager
def track_infer_time(buffer: [int]):
    start = time()
    yield
    end = time()

    buffer.append(end - start)


def serialize_results(new_results: dict) -> None:
    existing_results = {}

    if os.path.exists('results.yaml'):
        with open('results.yaml', 'r') as f:
            existing_results = yaml.safe_load(f)

    results = {**existing_results, **new_results}

    with open('results.yaml', 'w') as f:
        yaml.dump(results, f)



def predict(model, dataloader, args):
    # Get the input names required by the model
    input_names = model.get_inputs()
    input_names = [input.name for input in input_names]

    print("input_names", input_names)

    predictions = []
    labels = []
    print('Evaluating...')
    for batch in tqdm(dataloader):
        # print(batch.items())
        # Filter the model_inputs dictionary to only include the required inputs
        # print("filter")
        model_inputs = {k: v for k, v in batch.items() if k in input_names}
        # convert to tensor(int64)
        # print("convert")
        model_inputs = {k: v.numpy().astype('int64') for k, v in model_inputs.items()}

        with torch.no_grad():
            # print("run")
            outputs = model.run(None, input_feed=model_inputs)
            if args.task == 'CiteWorthiness':
                predictions.extend(outputs[0].argmax(axis=1).tolist())
                labels.extend(batch['labels'].cpu().numpy().tolist())
            elif args.task == 'SectionClassification':
                # predict class if sigmoid > threshold
                outputs = outputs[0]
                sigmoid = 1 / (1 + np.exp(-outputs))
                predictions.extend((sigmoid > args.threshold).astype(int).tolist())
                labels.extend(batch['labels'].cpu().numpy().tolist())
            else:
                raise ValueError(f'Unknown task {args.task}')
    # calc metrics

    accuracy = metrics.accuracy_score(labels, predictions)
    if args.task == 'SectionClassification':
        f1_score_avg = metrics.f1_score(labels, predictions, average='samples')
    else:
        f1_score_avg = metrics.f1_score(labels, predictions, average='binary')
    f1_score_micro = metrics.f1_score(labels, predictions, average='micro')
    f1_score_macro = metrics.f1_score(labels, predictions, average='macro')

    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Samples/Binary) = {f1_score_avg}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")

    if args.task == 'SectionClassification':
        titels = ['conclusion', 'related work', 'introduction', 'result', 'discussion', 'experiment', 'method']
        classification_report = metrics.classification_report(
            labels,
            predictions,
            output_dict=False,
            target_names=titels,
            digits=4)

        print("--- Classification Report: ---")
        print(classification_report)
        # print("--- LATEX: ---")
        # print(tabulate(classification_report, headers='keys', tablefmt='latex'))
    else:
        cm = metrics.confusion_matrix(labels, predictions)
        print("--- Confusion Matrix: ---")
        print(cm)

        precision_ = metrics.precision_score(labels, predictions, average=None)
        recall_ = metrics.recall_score(labels, predictions, average=None)
        f1_ = metrics.f1_score(labels, predictions, average=None)

        print('precision', precision_)
        print('recall', recall_)
        print('f1', f1_)


def main():
    """Main function."""
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=TASKS, default='SectionClassification')
    parser.add_argument('--dataset', choices=DATASETS.keys(), default='CiteWorth')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_path', type=str, default='./models/bert-base-uncased.onnx')
    # parser.add_argument('--tokenizer_path', type=str, default='./models/CiteWorthiness/BERT/tokenizer')
    parser.add_argument('--model', choices=MODELS.keys(), default='BERT')
    parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')
    parser.add_argument('--evaluation_split', choices=['validation', 'test'], default='test')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    # logging
    logging.basicConfig(level=args.log_level)

    print('Loading data...')

    model_path = args.model_path

    tk.tokenizer = AutoTokenizer.from_pretrained(MODELS[args.model], use_fast=True)
    # get original tokenizer name by removing the path before 'tokenizer' using regex
    #tokenizer_name = re.search(r'(?<=tokenizer/).*(?=.*)', tokenizer_path).group(0)

    data_loader = load_data(args.dataset, args.evaluation_split, args.task, args.batch_size, f'datasets/{args.dataset}/{MODELS[args.model]}')

    cpu_model = create_model_for_provider(model_path, "CPUExecutionProvider")

    predict(cpu_model, data_loader, args)


if __name__ == '__main__':
    main()
