from typing import Tuple, Dict
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.tooling import read_yaml_file, write_pickle_file, read_pickle_file
from tokenizing.tokenizer import get_context


# Constants
DATASETS_TO_ANALYZE = ["citeworth", "scisen"]
DATASET_PATH = Path.cwd().parent / "datasets"
PATH_TO_MODEL_LIST = Path.cwd().parent / "models.yaml"
CALCULATE_TOKEN_LENGTH_FROM_SCRATCH = True
CALCULATED_TOKEN_LENGTHS_PATH = Path.cwd().parent / "datasets"


def check_token_lengths(
        dataset: pd.DataFrame,
        models: Dict[str, Dict[str, str]],
        dataset_name: str
) -> pd.DataFrame:
    """
    Check if token lengths exist for each model in the dataset, and calculate if not.

    :param dataset: Dataframe containing the dataset to be checked.
    :param models: Dictionary of model names and their corresponding links.
    :param dataset_name: Name of the dataset.
    :return: Updated dataframe with calculated token lengths.
    """
    new_data_added = False
    global tokenizer
    for model_name, model_links in models.items():
        token_length_col_name = f"token_length_{dataset_name}_{model_name}"
        if token_length_col_name not in dataset.columns:
            print(f">> No token lengths exist for {model_name}. Calculating them now.")
            tokenizer = AutoTokenizer.from_pretrained(model_links.get("tokenizer"), use_fast=True)
            token_length_main_sentence, token_length_context = get_token_length_per_sentence(dataset)
            dataset[f"{token_length_main_sentence.name}_{dataset_name}_{model_name}"] = token_length_main_sentence
            dataset[f"{token_length_context.name}_{dataset_name}_{model_name}"] = token_length_context
            new_data_added = True
    if new_data_added:
        write_pickle_file(DATASET_PATH / f"{dataset_name}_preprocessed.pkl", dataset)
    return dataset


def get_token_length_per_sentence(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate token length per sentence and context for the given dataset.

    :param df: Dataframe containing the dataset.
    :return: Tuple of two series containing token lengths for main sentences and contexts.
    """
    tqdm.pandas(desc='>>> Tokenizing')
    df["main_tokenized"] = df["text"].progress_apply(encode)
    df["token_length"] = df["main_tokenized"].apply(len)
    df = get_context(df, column_to_shift="token_length")
    df["token_length_context"] = df["prev_sentence"] + df["token_length"] + df["next_sentence"] + 4
    return df["token_length"], df["token_length_context"]


def encode(text: str) -> List[int]:
    """
    Tokenizes the input text and returns the token IDs.

    :param text: Text to be tokenized.
    :return: List of token IDs.
    """
    return tokenizer.encode_plus(
        text,
        add_special_tokens=False,
        truncation=False,
        return_tensors=None,
        return_attention_mask=False,
        return_token_type_ids=False
    )["input_ids"]


def main() -> None:
    """
    Main function to handle token length checking and calculation for multiple datasets.
    """
    models = read_yaml_file(PATH_TO_MODEL_LIST)
    for dataset_name in DATASETS_TO_ANALYZE:
        print(f"> {dataset_name.upper()}")
        dataset = read_pickle_file(DATASET_PATH / f"{dataset_name}_preprocessed.pkl")
        check_token_lengths(dataset, models, dataset_name)


if __name__ == "__main__":
    main()
