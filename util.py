import logging
import os
import torch
from torch.utils.data import DataLoader, Subset
from datasets import Dataset
from utils.tooling import read_pickle_file, save_tokenized_data_in_folder_structure
from tokenizing import tokenizer as tk
import argparse
import numpy as np
import uuid
import shortuuid
import yaml

import pandas as pd

from torch.utils.data.sampler import Sampler

MODELS = {
    'MLP': 'bert-base-uncased',
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
TASKS = ['SectionClassification', 'CiteWorthiness']
DATASETS = {
    'CiteWorth': 'datasets/CiteWorth_preprocessed.pkl',
    'SciSen': 'datasets/SciSen_preprocessed.pkl',
}


class feature_dataset(torch.utils.data.Dataset):
    def __init__(self, features, generate_guids: bool = False):
        self.generate_guids = generate_guids
        self.features = features
        self.guid = 0

    def __getitem__(self, index):
        if self.generate_guids:
            self.features[index]['guids'] = torch.tensor(self.guid)
            self.guid += 1
        return self.features[index]

    def __len__(self):
        return len(self.features)


class ExcludeIndicesSampler(Sampler):
    def __init__(self, length, excluded_indices):
        #self.data_source = data_source
        self.excluded_indices = set(excluded_indices)
        self.indices = [i for i in range(length) if i not in self.excluded_indices]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
    

def load_data_distillation(dataset_name: str, split_name: str, task_type: str, batch_size: int, dataset_path:str, teacher_dataset_path:str, max: int = None, shuffle: bool = False, generate_guids: bool = False) -> Dataset:
    dataset_path_same_dir = f'{dataset_path}/{split_name}.pkl'
    dataset_path_parent_dir = f'../{dataset_path}/{split_name}.pkl'
    if os.path.exists(dataset_path_same_dir) or os.path.exists(f'{dataset_path_same_dir}.gz'):
        dataset_path = dataset_path_same_dir
    elif os.path.exists(dataset_path_parent_dir) or os.path.exists(f'{dataset_path_parent_dir}.gz'):
        dataset_path = dataset_path_parent_dir
    else:
        logging.info(f'No dataset found under {dataset_path}. Tokenizing...')
        data_preprocessed = read_pickle_file(DATASETS[dataset_name])
        dataset_tokenized = tk.tokenize_df(df_to_tokenize=data_preprocessed)
        save_tokenized_data_in_folder_structure(df=dataset_tokenized, base_directory='datasets',
                                                dataset_name=dataset_name, tokenizer_name=tk.tokenizer.name_or_path)
        return

    logging.info(f'Existing dataset {dataset_path} found. Loading...')
    dataset_tokenized = read_pickle_file(dataset_path)

    teacher_dataset_path_same_dir = f'{teacher_dataset_path}/{split_name}.pkl'
    teacher_dataset_path_parent_dir = f'../{teacher_dataset_path}/{split_name}.pkl'
    if os.path.exists(teacher_dataset_path_same_dir) or os.path.exists(f'{teacher_dataset_path_same_dir}.gz'):
        teacher_dataset_path = teacher_dataset_path_same_dir
    elif os.path.exists(teacher_dataset_path_parent_dir) or os.path.exists(f'{teacher_dataset_path_parent_dir}.gz'):
        teacher_dataset_path = teacher_dataset_path_parent_dir
    else:
        logging.info(f'No teacher dataset found under {teacher_dataset_path}. Tokenizing...')
        data_preprocessed = read_pickle_file(DATASETS[dataset_name])
        dataset_tokenized = tk.tokenize_df(df_to_tokenize=data_preprocessed)
        save_tokenized_data_in_folder_structure(df=dataset_tokenized, base_directory='datasets',
                                                dataset_name=dataset_name, tokenizer_name=tk.tokenizer.name_or_path)
        return
    
    logging.info(f'Existing teacher dataset {teacher_dataset_path} found. Loading...')
    teacher_dataset_tokenized = read_pickle_file(teacher_dataset_path)

    label = {
        'SectionClassification': 'section_category',
        'CiteWorthiness': 'label'
    }

    label_to_delete = [l for k, l in label.items() if k != task_type][0]
    dataset_tokenized.drop(label_to_delete, axis=1, inplace=True)
    teacher_dataset_tokenized.drop(label_to_delete, axis=1, inplace=True)

    last_item = dataset_tokenized.iloc[-1]
    last_item_teacher = teacher_dataset_tokenized.iloc[-1]

    last_index = last_item.name
    last_index_teacher = last_item_teacher.name

    logging.info(f'Length dataset BEFORE: {len(dataset_tokenized)} | Length teacher dataset: {len(teacher_dataset_tokenized)}')
    logging.info(f'Last index: {last_index} | Last index teacher: {last_index_teacher}')

    all_indices = set(range(0, last_index))
    used_indices = set(dataset_tokenized.index)
    open_indices = all_indices - used_indices

    all_indices_teacher = set(range(0, last_index_teacher))
    used_indices_teacher = set(teacher_dataset_tokenized.index)
    open_indices_teacher = all_indices_teacher - used_indices_teacher

    logging.info(f'Open indices: {len(open_indices)} | Open indices teacher: {len(open_indices_teacher)}')

    all_deleted_indices = list(open_indices.union(open_indices_teacher))

    logging.info(f'All deleted indices: {len(all_deleted_indices)}')

    dataset_tokenized.drop(all_deleted_indices, errors='ignore', inplace=True)
    teacher_dataset_tokenized.drop(all_deleted_indices, errors='ignore', inplace=True)

    logging.info(f'Length dataset AFTER: {len(dataset_tokenized)} | Length teacher dataset: {len(teacher_dataset_tokenized)}')

    if max is not None:
        dataset_tokenized = dataset_tokenized.head(max)
        teacher_dataset_tokenized = teacher_dataset_tokenized.head(max)

    dataset_dict = dataset_tokenized.to_dict(orient='records')
    teacher_dataset_dict = teacher_dataset_tokenized.to_dict(orient='records')

    def create_features_cw(datapoint):
        dp = {
            'input_ids': torch.LongTensor(datapoint['input_ids']),
            'attention_mask': torch.LongTensor(datapoint['attention_mask']),
            'labels': torch.LongTensor([datapoint['label']]),
        }
        if datapoint.get('token_type_ids', False):
             dp['token_type_ids'] = torch.LongTensor(datapoint['token_type_ids'])
        return dp

    def create_features_sc(datapoint):
        dp = {
            'input_ids': torch.LongTensor(datapoint['input_ids']),
            'attention_mask': torch.LongTensor(datapoint['attention_mask']),
            'labels': torch.sum(
                    torch.nn.functional.one_hot(
                        torch.tensor([datapoint['section_category']]),
                        num_classes=7),
                    dim=1
                ).to(torch.float).squeeze(0),
        }
        if datapoint.get('token_type_ids', False):
            dp['token_type_ids'] = torch.LongTensor(datapoint['token_type_ids'])
        return dp

    create_features = create_features_sc if task_type == 'SectionClassification' else create_features_cw

    features = [create_features(datapoint) for datapoint in tqdm(dataset_dict, desc='Tensoring', unit='dp')]
    teacher_features = [create_features(datapoint) for datapoint in tqdm(teacher_dataset_dict, desc='Tensoring', unit='dp')]

    student_dataset = feature_dataset(features,generate_guids=generate_guids)
    teacher_dataset = feature_dataset(teacher_features,generate_guids=generate_guids)

    for i in tqdm(range(len(student_dataset)),desc='Aserting labels', unit='dp' ):
        #assert student_dataset[i]['labels'] == teacher_dataset[i]['labels'], f"Labels at index {i} do not match"
        assert torch.equal(student_dataset[i]["labels"], teacher_dataset[i]["labels"]), f"Labels at index {i} do not match. Labels: {student_dataset[i]['labels']} | Teacher labels: {teacher_dataset[i]['labels']}"

    #dataset = torch.utils.data.TensorDataset(student_dataset, teacher_dataset)

    return DataLoader(dataset=student_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=16, pin_memory=True), DataLoader(dataset=teacher_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=16, pin_memory=True)



def load_data(dataset_name: str, split_name: str, task_type: str, batch_size: int, dataset_path:str, max: int = None, shuffle: bool = False, generate_guids: bool = False, primary_model: MODELS = None, secondary_model: MODELS = None) -> Dataset:
    dataset_path_same_dir = f'{dataset_path}/{split_name}.pkl'
    dataset_path_parent_dir = f'../{dataset_path}/{split_name}.pkl'
    # print(dataset_path_same_dir)
    # print(dataset_path_parent_dir)
    #print(os.getcwd())

    if os.path.exists(dataset_path_same_dir) or os.path.exists(f'{dataset_path_same_dir}.gz'):
        dataset_path = dataset_path_same_dir
    elif os.path.exists(dataset_path_parent_dir) or os.path.exists(f'{dataset_path_parent_dir}.gz'):
        dataset_path = dataset_path_parent_dir
    else:
        logging.info(f'No dataset found under {dataset_path}. Tokenizing...')
        data_preprocessed = read_pickle_file(DATASETS[dataset_name])
        dataset_tokenized = tk.tokenize_df(df_to_tokenize=data_preprocessed)
        save_tokenized_data_in_folder_structure(df=dataset_tokenized, base_directory='datasets',
                                                dataset_name=dataset_name, tokenizer_name=tk.tokenizer.name_or_path)
        return

    logging.info(f'Existing dataset {dataset_path} found. Loading...')
    dataset_tokenized = read_pickle_file(dataset_path)


    if max is not None:
        dataset_tokenized = dataset_tokenized.iloc[::10]
        dataset_tokenized = dataset_tokenized.sample(n=max)


    # Adapt label to task
    #       Task                    | Label
    label = {
        'SectionClassification': 'section_category',
        'CiteWorthiness': 'label'
    }

    # Drop the unnecessary label column
    label_to_delete = [l for k, l in label.items() if k != task_type][0]
    dataset_tokenized.drop(label_to_delete, axis=1, inplace=True)



    dataset_dict = dataset_tokenized.to_dict(orient='records')

    def create_features_cw(datapoint):
        dp = {
            'input_ids': torch.LongTensor(datapoint['input_ids']),
            'attention_mask': torch.LongTensor(datapoint['attention_mask']),
            'labels': torch.LongTensor([datapoint['label']]),
        }
        if datapoint.get('token_type_ids', False):
             dp['token_type_ids'] = torch.LongTensor(datapoint['token_type_ids'])
        return dp

    def create_features_sc(datapoint):
        dp = {
            'input_ids': torch.LongTensor(datapoint['input_ids']),
            'attention_mask': torch.LongTensor(datapoint['attention_mask']),
            'labels': torch.sum(
                    torch.nn.functional.one_hot(
                        torch.tensor([datapoint['section_category']]),
                        num_classes=7),
                    dim=1
                ).to(torch.float).squeeze(0),
        }
        if datapoint.get('token_type_ids', False):
            dp['token_type_ids'] = torch.LongTensor(datapoint['token_type_ids'])
        return dp

    create_features = create_features_sc if task_type == 'SectionClassification' else create_features_cw

    features = [create_features(datapoint) for datapoint in tqdm(dataset_dict, desc='Tensoring', unit='dp')]

    # Create and return the DataLoader
    return DataLoader(dataset=feature_dataset(features,generate_guids=generate_guids), batch_size=batch_size, shuffle=shuffle, num_workers=16, pin_memory=True)


## i need a function that returns a number of sentences from my dataframe


import wikipedia
import random
from tqdm import tqdm

def get_random_sentences(num_sentences=10, max_length=512):
    random_sentences = []
    
    # tqdm is wrapped around the loop condition to show progress as sentences are added
    pbar = tqdm(total=num_sentences)
    
    while len(random_sentences) < num_sentences:
        try:
            # Get a random Wikipedia page
            random_page = wikipedia.random()
            #logging.info("Random page:", random_page)
            page_content = wikipedia.page(random_page).content
        except wikipedia.DisambiguationError as e:
            try:
                # Select a random page from the disambiguation options
                random_page = random.choice(e.options)
                #logging.info("Random page (from disambiguation):", random_page)
                page_content = wikipedia.page(random_page).content
            except wikipedia.DisambiguationError:
                continue
            except Exception:
                continue
        except Exception:
            continue

        # Split content into sentences and filter those that are too long
        sentences = [s for s in page_content.split('. ') if len(s) <= max_length]
        num_to_extract = min(len(sentences), num_sentences - len(random_sentences))
        
        # If the filtered sentences are empty, skip to the next iteration
        if not sentences:
            continue
        
        chosen_sentences = random.sample(sentences, num_to_extract)
        random_sentences.extend(chosen_sentences)

        # Update tqdm progress bar by the number of sentences added
        pbar.update(len(chosen_sentences))

    # Close the progress bar when done
    pbar.close()

    return random_sentences[:num_sentences]

def save_metrics(metrics: dict, args: argparse.Namespace, uid: str, split: str):

    args_dict = vars(args)
    metrics_file = f'{args.model_dir}/{args.task}/{args.model}/metrics_{split}.csv'
    if os.path.exists(metrics_file):
        f = open(metrics_file, 'a')
    else:
        if not os.path.exists(f'{args.model_dir}/{args.task}/{args.model}'):
            os.makedirs(f'{args.model_dir}/{args.task}/{args.model}')
        f = open(metrics_file, 'x')
        csv_columns = list(args_dict.keys()) + list(metrics.keys())
        f.write(f'uid,{",".join(csv_columns)}\n')
    with f:
        csv_values = [str(val) for val in list(args_dict.values()) + list(metrics.values())]
        f.write(f'{uid},{",".join(csv_values)}\n')


