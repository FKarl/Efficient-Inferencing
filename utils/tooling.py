"""This script holds general tools."""

import pickle
import pandas as pd
import json
import os
from yaml import safe_load, dump
from typing import Union
from pathlib import Path
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from math import isclose
import numpy as np
import gzip
import psutil
import pynvml
import time
import threading
import torch
import joblib
import logging

class DatasetSplitError(Exception):
    "Splitting the dataset into training, test and validation did not work."

try:
    pynvml.nvmlInit()
    gpu_available = True
except Exception as e:
    gpu_available = False
    logging.info(f"Could not initialize NVML library. GPU stats will not be available. Cuda available: {torch.cuda.is_available()}")
    logging.info(e)


def decompress_with_progressbar(file_name: Union[str, Path]):
    # Get the size of the compressed file for the progress bar maximum value
    file_size = os.path.getsize(f'{file_name}.gz')

    # Create a progress bar
    with gzip.open(f'{file_name}.gz', 'rb') as f_in:
        with open(f'{file_name}', 'wb') as f_out:
            for chunk in tqdm(iter(lambda: f_in.read(1024 * 1024), b''), unit='B', unit_scale=True, desc=f"Decompressing {file_name}.gz"):  # read 1MB at a time
                f_out.write(chunk)

    logging.info("Done decompressing.")
    # After decompressing, read the file into a dataframe
    #pbar = tqdm(total=1, desc="Loading...", position=0)

def load_data(file_name: Union[str, Path]) -> pd.DataFrame:
    stop_thread = False

    def update_memory_stats():
        while not stop_thread:
            sys_mem, gpu_mem = memory_stats()
            tqdm.write(f'{sys_mem} | {gpu_mem}', end='\r')
            time.sleep(1)
        tqdm.write('', end='\n')

    thread = threading.Thread(target=update_memory_stats)
    thread.daemon = True
    thread.start()

    logging.info("Loading pickle as dataframe into memory...")
    content = pd.read_pickle(file_name)
    stop_thread = True
    thread.join()

    logging.info("\nDone Loading...")

    #pbar.update(1)
    #pbar.close()

    time.sleep(2)

    return content

def memory_stats():
    # System Memory
    mem = psutil.virtual_memory()
    system_memory = f"System Memory - Used: {mem.used / (1024**3):.2f} GB, Total: {mem.total / (1024**3):.2f} GB"

    # GPU Memory
    if gpu_available:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 0 for GPU 0. Adjust if you have multiple GPUs and want to monitor a different one.
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory = f"GPU Memory - Used: {meminfo.used / (1024**3):.2f} GB, Total: {meminfo.total / (1024**3):.2f} GB"
    else:
        gpu_memory = "GPU Memory - Not available"

    return system_memory, gpu_memory


def write_pickle_file(file_name: Union[str, Path], data_to_save: Union[pd.DataFrame, Dataset, DatasetDict, torch.FloatTensor, dict]) -> None:
    """Save data to .pkl file.

    :param file_name: Target name of the file.
    :param data_to_save: Content of the file.
    """
    file_name = Path(file_name) if isinstance(file_name, str) else file_name
    with open(file_name, "wb") as file:
        pickle.dump(data_to_save, file)


def read_pickle_file(file_name: Union[str, Path]) -> pd.DataFrame:
    """Read a pickled object from a file and return the object.

    :param file_name: The path to the pickled file.
    :return: The unpickled object.
    """
    if not os.path.exists(f'{file_name}') and os.path.exists(f'{file_name}.gz'):
        logging.info(f"Compressed file found {file_name}.gz. Uncompressing...")
        decompress_with_progressbar(file_name)
    else:
        logging.info(f"Loading standard file {file_name}.")
    content = load_data(file_name)
    return content


def write_yaml_file(data_to_save: dict, file_name: Union[str, Path]) -> None:
    """Saves a dictionary to a YAML file.

    :param data_to_save: The dictionary to be saved.
    :param file_name: The path to save the YAML file.
    """
    with open(file_name, 'w') as file:
        dump(data_to_save, file)


def read_yaml_file(file_name: Union[str, Path]) -> Union[dict, list]:
    """Read a YAML file and return the data as a Python object.

    :param file_name: The path to the YAML file.
    :return: The contents of the YAML file as a Python object.
    """
    with open(file_name, 'r') as file:
        data = safe_load(file)
    return data


def read_json_file(file_path: Union[str, Path]) -> Union[dict, list]:
    """Loads data from a JSON file and returns it as a dictionary.

    :param file_path: The path to the JSON file.
    :return: The data loaded from the JSON file as a dictionary.
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            data.append(json_obj)
    return data


def read_txt_file(file_path: Union[str, Path]) -> Union[dict, list]:
    """Loads data from a text file and returns it as a dictionary.

    :param file_path: The path to the text file.
    :return: The data loaded from the text file as a dictionary.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    return list(lines)


def convert_json_to_yaml(json_file_path: Union[str, Path], yaml_file_path: Union[str, Path]) -> None:
    """Loads a .json file into a dictionary and saves it to a .yaml file

    :param json_file_path: Path to .json file.
    :param yaml_file_path: Path to .yaml file.
    """
    json_data = read_json_file(json_file_path)
    write_yaml_file(json_data, yaml_file_path)  # Save the dictionary to a YAML file
    logging.info("YAML saved to", yaml_file_path)


def convert_txt_to_yaml(txt_file_path: Union[str, Path], yaml_file_path: Union[str, Path]) -> None:
    """Loads a .txt file into a dictionary and saves it to a .yaml file

    :param txt_file_path: Path to .txt file.
    :param yaml_file_path: Path to .yaml file.
    """
    txt_data = read_txt_file(txt_file_path)
    write_yaml_file(txt_data, yaml_file_path)  # Save the dictionary to a YAML file
    logging.info("YAML saved to", yaml_file_path)


def clean_dataset(data_to_clean: pd.DataFrame, cfg_for_cleaning: Union[Path, str]) -> pd.DataFrame:
    """Summarize cleaning steps.

    1. Rename columns.
    2. Delete unnecessary information from the dataset.
    3. Sort the remaining columns by the given order.
    4. Reset the index
    5. Delete Appendix and Abstract from the dataset

    :param data_to_clean: Raw data set.
    :param cfg_for_cleaning: Link to config file that defines the final structure of the dataset.
    :return: Clean data set.
    """
    original_shape = data_to_clean.shape
    cfg = read_yaml_file(file_name=cfg_for_cleaning)
    # 1.
    data_renamed = data_to_clean.rename(columns=cfg.get("rename_columns"))  # Rename based on config
    # 2.
    columns_to_remove = set(data_renamed.columns) - set(cfg.get("columns_to_keep"))
    data_limited = data_renamed.drop(columns_to_remove, axis=1)
    # 3.
    data_sorted = data_limited.reindex(columns=cfg.get("columns_to_keep"))  # Sort columns based on config
    # 4.
    data_clean = data_sorted.reset_index(drop=True)
    final_shape = data_clean.shape
    logging.info(f"Compressed dimensions {original_shape} to {final_shape}")
    # 5.
    from_appendix = data_clean["section_category"].apply(lambda x: "appendix" in x)
    from_abstract = data_clean["section_category"].apply(lambda x: "abstract" in x)
    to_delete = from_appendix | from_abstract
    print(f"Deleted {len(data_clean[to_delete])} datapoints because they are from appendix or abstract")
    return data_clean[~to_delete]


def print_markdown_sample(df: pd.DataFrame) -> None:
    """logging.info the head of a DataFrame in Markdown.

    :param df: DataFrame to convert to Markdown.
    """
    sample = df.head().copy()
    sample["text"] = sample["text"].apply(lambda x: x[:20] + "...")
    logging.info(sample.to_markdown())


def save_dataframe_to_markdown(df, filename):
    """
    Save a pandas DataFrame to a Markdown file.

    :param df: The pandas DataFrame to save.
    :param filename: The name of the Markdown file to save to.
    """
    markdown_string = df.to_markdown()
    with open(filename, 'w') as file:
        file.write(markdown_string)
    print(f"Saved table to Markdown at {filename}")


def convert_to_huggingface_format(df_to_convert: pd.DataFrame, column_name_with_splits: str) -> DatasetDict:
    """Converts a pandas DataFrame to a Huggingface's DatasetDict.

    :param df_to_convert: A pandas DataFrame containing the data.
    :param column_name_with_splits: Name of column that contains split labels.
    :return: A DatasetDict containing the splits of data as specified in df_to_convert[column_name_with_splits].
    """
    splits = df_to_convert[column_name_with_splits].unique()
    datasets = DatasetDict()

    for split in splits:
        split_df = df_to_convert[df_to_convert[column_name_with_splits] == split].drop(columns=[column_name_with_splits])
        datasets[split] = Dataset.from_pandas(split_df)

    return datasets


def save_tokenized_data_in_folder_structure(df: pd.DataFrame, base_directory: Union[Path, str], dataset_name: str, tokenizer_name:str):
    # Create the base directory
    base_dir = base_directory / Path(dataset_name) / tokenizer_name
    base_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over unique splits and save each dataframe split to its appropriate path
    for split_name in df['split'].unique():
        logging.info('Saving split:', split_name)
        split_df = df[df['split'] == split_name]

        # Define the save path for the current split
        save_path = base_dir / f"{split_name}.pkl"

        columns_to_drop = set(split_df.columns) - set(["section_category", "label", "input_ids", "token_type_ids", "attention_mask"])
        # Limit to necessary content
        split_df = split_df.drop(columns_to_drop, axis=1)

        # Save the dataframe to the specified path
        write_pickle_file(file_name=save_path, data_to_save=split_df)


def do_manual_split(full_dataset: pd.DataFrame, train_split_size: float) -> pd.DataFrame:
    """Split 80/10/10. Each can only contain full papers!

    :param full_dataset: A pandas DataFrame containing the data to be split.
    :param train_split_size: The proportion of the full dataset to include in the train split.
                             Should be a float between 0.0 and 1.0. Test and Validation will be split symmetrically.

    :return: A DatasetDict containing the splits of data for 'train', 'test', and 'validation'.
    """
    # Get percentage share of each paper
    number_of_sentences_per_paper = full_dataset.groupby("paper_index")["paper_index"].count()
    percentage_share_of_paper = number_of_sentences_per_paper / number_of_sentences_per_paper.sum()
    cumulated = percentage_share_of_paper.cumsum()

    # Separate splits
    test_split_size = (1 - train_split_size) / 2

    # Specify conditions and corresponding replacements
    mapping = {
        "train": (cumulated <= train_split_size),
        "test": ((cumulated > train_split_size) & (cumulated <= train_split_size + test_split_size)),
        "validation": (cumulated > train_split_size + test_split_size)
    }

    # Replace with different values based on conditions
    split_column = pd.Series(np.select(mapping.values(), mapping.keys()), index=cumulated.index, name="split")
    full_dataset = full_dataset.drop("split", axis=1) if "split" in full_dataset.columns else full_dataset
    full_dataset = full_dataset.join(split_column, on="paper_index")

    if check_test_size(full_dataset, train_split_size):
        raise DatasetSplitError

    return full_dataset


def check_test_size(ds_to_check: pd.DataFrame, train_split_size: float):
    """Do some checks:

    :param ds_to_check: Data set to check.
    :param train_split_size: Planned split size. E.g. if 80/10/10 -> train_split_size = 0.8
    """
    test_split_size = (1 - train_split_size) / 2
    # 1. test if achieved desired split size
    tolerance = 0.01
    num_sentences = ds_to_check["split"].value_counts()
    split_in_percentage = (num_sentences / num_sentences.sum()).to_dict()
    all_close = isclose(split_in_percentage["train"], train_split_size, rel_tol=tolerance) & \
                isclose(split_in_percentage["test"], test_split_size, rel_tol=tolerance) & \
                isclose(split_in_percentage["validation"], test_split_size, rel_tol=tolerance)
    if not all_close:
        return 1

    # 2. double check if the same paper appears in different splits
    paper_train = set(ds_to_check.loc[ds_to_check["split"] == "train", "paper_index"])
    paper_test = set(ds_to_check.loc[ds_to_check["split"] == "test", "paper_index"])
    paper_validation = set(ds_to_check.loc[ds_to_check["split"] == "validation", "paper_index"])
    overlapping_paper = set.intersection(paper_train, paper_test, paper_validation)
    if overlapping_paper:
        return 1

    else:
        logging.info(pd.Series(split_in_percentage, name="share").to_markdown())
        return 0