import sys
from pathlib import Path
PROJECT_PATH = str(Path.cwd())
print(PROJECT_PATH)
sys.path.append(PROJECT_PATH)
from transformers import AutoTokenizer
from utils.tooling import read_pickle_file, save_tokenized_data_in_folder_structure, read_yaml_file, check_test_size, DatasetSplitError
import pandas as pd
from tqdm import tqdm
from math import floor, ceil
from typing import Union
import os
import warnings
import yaml

global tokenizer


# Constants
MODEL_TO_USE = "bert-base-uncased"
PATH_TO_MODEL_LIST = f"{PROJECT_PATH}/models.yaml"
DATASETS_TO_ANALYZE = ["citeworth", "scisen"]
DATASET_PATH = "datasets"
OUTPUT_TOKENIZED_DATASETS_PATH = DATASET_PATH

# Additional Constants
OUTPUT_TOKENIZED_DATASETS_TEMPLATE = "{}/{}/{}.pkl"  # "{}" in template is necessary for adding name and info

def get_context(df: pd.DataFrame, column_to_shift: str = "text") -> pd.DataFrame:
    """Generate a DataFrame which includes context for each sentence.

    This function takes as input a DataFrame that contains sentences. It adds two new
    columns to the DataFrame: 'prev_sentence' and 'next_sentence', which contain the
    previous and next sentences for each row. If the sentence is the first or last in
    its section, the 'prev_sentence' or 'next_sentence' is set to an empty string.

    :param df: A DataFrame that contains sentences in a 'text' column.
    :param column_to_shift: Can be used to specify different shift column (e.g. token count)
    :return: The original DataFrame with added 'prev_sentence', 'next_sentence'
    and 'full_context' columns.
    """
    # Get previous and next sentences
    df['prev_sentence'] = df[column_to_shift].shift(1, fill_value="")
    df['next_sentence'] = df[column_to_shift].shift(-1, fill_value="")

    # Remove previous sentence where main sentence is the first of the section and next sentence where it is the last.
    is_first_sentence_of_section = df['text_index'] == 0
    is_last_sentence_of_section = is_first_sentence_of_section.shift(-1, fill_value=True)
    df.loc[is_first_sentence_of_section, 'prev_sentence'] = ""
    df.loc[is_last_sentence_of_section, 'next_sentence'] = ""

    return df


def centered_truncation(oversized_element: pd.Series):
    """Performs centered truncation on a tokenized text element. This function is designed to handle token sequences
    that exceed the maximum token length allowed by the tokenizer. It truncates the tokens from the middle of the
    sequence to fit within the model's maximum token length, ensuring that the most relevant parts of the text
    (typically around the main sentence) are preserved.

    :param oversized_element: A pandas Series representing a single row from a DataFrame, containing tokenized text data.
    :return: A pandas Series with the token sequence truncated to fit within the model's maximum token length.
    """
    resized_element = oversized_element.copy()  # predefine resulting element

    # Tokenize sentences individually to see where the overhead is coming from
    s = encode(oversized_element["context_with_special_tokens"])  # sequence chain
    s_prev = encode(tokenizer.cls_token + oversized_element["prev_sentence"] + tokenizer.sep_token)  # sequence chain previous sentence
    s_main = encode(oversized_element["text"] + tokenizer.sep_token)  # sequence chain main sentence
    s_next = encode(oversized_element["next_sentence"] + tokenizer.sep_token)  # sequence chain next sentence

    # Get the token length n_x of each sentence x from sequence chain s_x
    n_prev, n_main, n_next = (get_token_length(s_x) for s_x in (s_prev, s_main, s_next))

    # Compute centered frame and intersections with previous and next sentence
    f_center = n_prev + floor(n_main / 2)
    i_prev = f_center - floor(tokenizer.model_max_length / 2)
    i_next = f_center + ceil(tokenizer.model_max_length / 2)

    # Introducing token filters
    idx_s = list(range(0,tokenizer.model_max_length))  # Indices of sequence chain S
    idx_special_tokens = [0,  # CLS token
                          n_prev - 1,  # SEP token at the end of previous sentence
                          n_prev + n_main - 1,  # SEP token at the end of main sentence
                          n_prev + n_main + n_next - 1]  # SEP token at the end of next sentence



    # 1. Case: Only previous sentence needs to be cut
    if (i_prev >= 0) and (i_next > tokenizer.model_max_length):
        i_prev -= i_next - tokenizer.model_max_length  # Add margin from the right
        idx_s = idx_s[i_prev:]  # Right alignment on next sentence
        idx_s[:2] = idx_special_tokens[:2]  # Replace first two tokens with CLS and SEP token

    # 2. Case: Only next sentence needs to be cut
    elif (i_next <= tokenizer.model_max_length) and (i_prev < 0):
        i_next += abs(i_prev)  # Add margin from the left
        idx_s = idx_s[:i_next]  # Left alignment on previous sentence
        idx_s[-1] = idx_special_tokens[-1]  # Replace last token with final SEP token

    # 3. Case: Both sentences need to be cut from left and right because they both reach outside the frame.
    elif (i_prev >= 0) and (i_next <= tokenizer.model_max_length):
        idx_s = idx_s[i_prev:i_next]
        idx_s[:2] = idx_special_tokens[:2]
        idx_s[-1] = idx_special_tokens[-1]

    # Filter tokens, type ids and attention mask
    s = {key: filter_by_indices(value, idx_s) for key, value in s.items()}

    resized_element["encoded_dict"] = s
    resized_element["tokens_length_pre_padding"] = get_token_length(s)

    return test_size_of_truncated_element(resized_element)


def filter_by_indices(original_list, indices):
    """Filters elements in a list based on specified indices. This function is used to select specific tokens from a
    tokenized sequence based on their indices, such as when truncating token sequences.

    :param original_list: The original list of elements (e.g., tokens) to be filtered.
    :param indices: A list of indices indicating which elements in the original list should be retained.
    :return: A filtered list containing only the elements at the specified indices.
    """
    return [original_list[i] for i in indices]


def test_size_of_truncated_element(element_to_check: pd.Series):
    """Validates whether a truncated element's token length matches the model's maximum token length. This function is used
    after truncation to ensure that the resulting token sequence is of the correct length.

    :param element_to_check: A pandas Series representing a single row from a DataFrame, containing a truncated token sequence.
    :return: The same pandas Series if the token length matches the model's maximum token length.
    :raises ValueError: If the token sequence length does not match the model's maximum token length.
    """
    if element_to_check["tokens_length_pre_padding"] != tokenizer.model_max_length:
        raise ValueError(f"After truncation, still not {tokenizer.model_max_length} long! Element: {element_to_check}.")
    else:
        return element_to_check


def tokenize_df(df_to_tokenize: pd.DataFrame) -> pd.DataFrame:
    """Tokenizes the text data in a DataFrame. This function processes a DataFrame containing text data, applying tokenization
    to each text element. It handles special cases where token sequences need to be truncated and ensures that the token
    length post-processing matches the model's maximum token length.

    :param df_to_tokenize: A pandas DataFrame containing text data to be tokenized.
    :return: A DataFrame with tokenized data, including additional metadata such as token length and truncation flags.
    """
    tqdm.pandas(desc='>>> Tokenizing')
    # Identify previous and next sentences accounting for fist and last sentence of section
    df = get_context(df_to_tokenize)

    # Combine previous, main and next sentence and add special tokens to the chain of strings
    df["context_with_special_tokens"] = get_sequence_chain(df)

    # Tokenize the chain of strings by using parallel processing to accelerate the process
    df["encoded_dict"] = df["context_with_special_tokens"].progress_apply(encode)

    df["tokens_length_pre_padding"] = df["encoded_dict"].apply(get_token_length)

    tears_token_limit = df["tokens_length_pre_padding"] > tokenizer.model_max_length

    # Truncation
    df["is_truncated"] = False
    if tears_token_limit.any():
        df, tears_token_limit = check_token_length_of_main_sentences(df, tears_token_limit)  # Case 0: Main sentence already exceeds the token limit (not supported)
        tqdm.pandas(desc='>>> Truncation')
        print(f"Truncation to {tokenizer.model_max_length} tokens has to be applied to {tears_token_limit.value_counts()[True]} elements.")
        df.loc[tears_token_limit] = df.loc[tears_token_limit].progress_apply(centered_truncation, axis=1)
        df.loc[tears_token_limit, "is_truncated"] = True

    # # Padding
    # tqdm.pandas(desc='>>> Padding')
    # df["encoded_dict"] = df["encoded_dict"].progress_apply(
    #     lambda x: tokenizer.pad(x, padding='max_length', max_length=tokenizer.model_max_length))

    # Check token length without apply to speed up, but with a tqdm progress bar
    tqdm.pandas(desc='>>> Checking token lengths after padding')
    df["token_length_post_padding"] = df["encoded_dict"].progress_apply(get_token_length)
    if any(df["token_length_post_padding"] != tokenizer.model_max_length):
        raise ValueError(f"A token sequence is not equal to the token limit after padding: "
                         f"{df.loc[df['token_length_post_padding'] != tokenizer.model_max_length]}")

    # Explode encoded dicts onto columns
    tqdm.pandas(desc='>>> Joining')
    df = df.join(df["encoded_dict"].progress_apply(pd.Series))

    # Tokenize label
    mapping = read_yaml_file(f"{PROJECT_PATH}/preprocessing/config/label_mapping.yaml")
    tqdm.pandas(desc='>>> Labeling categories')
    df["section_category"] = df["section_category"].progress_apply(
        lambda x: [mapping["section_category"][label] for label in x])
    df["label"] = df["label"].replace(mapping["label"])

    return df


def check_token_length_of_main_sentences(df, tears_token_limit):
    """Checks and handles cases where the main sentence in the DataFrame exceeds the model's maximum token length.
    If a main sentence is too long, this function drops it from the DataFrame to ensure that all token sequences
    can be properly processed.

    :param df: A pandas DataFrame containing the data with tokenized main sentences.
    :param tears_token_limit: A pandas Series indicating which rows in the DataFrame have main sentences exceeding the token limit.
    :return: A tuple containing the updated DataFrame and a Series marking rows where token sequences were within the token limit.
    """
    # Main sentence has more tokens than the limit allows (no margin, frame <= main sentence)
    tqdm.pandas(desc='>>> Checking main sentence >= model max length')
    len_main_sentence_tokenized = df.loc[tears_token_limit, "text"].progress_apply(
        lambda x: get_token_length(encode(x))) + 4
    is_too_long = len_main_sentence_tokenized >= tokenizer.model_max_length
    num_too_long = is_too_long.value_counts().get(True, 0)
    sentence_identifier = pd.DataFrame(is_too_long.loc[is_too_long]).index
    if any(is_too_long):
        warnings.warn(
            f"{num_too_long} sentences main sentence (plus 4 special tokens) longer or equal to the maximum token length ({tokenizer.model_max_length}). "
            f"-> Will be dropped. Resulting dataset is {(len(df) - num_too_long) / len(df)}% of original. Don't worry, split distribution will be checked.")

        df = df.drop(sentence_identifier)
        tears_token_limit = tears_token_limit.drop(sentence_identifier)
        with open('sentence_identifier.yaml', 'w') as file:
            yaml.dump(list(sentence_identifier), file)

        if check_test_size(df, 0.8):
            raise DatasetSplitError("The split has changed significantly because too many main sentences are below "
                                    "the model max length of tokens! You need to increase the maximum token length "
                                    "of the model.")
                    
        return df, tears_token_limit


def get_sequence_chain(df_element: Union[pd.Series, pd.DataFrame]) -> Union[dict, pd.Series]:
    """Constructs a sequence chain of text for tokenization, which includes the previous sentence, the main sentence,
    and the next sentence, separated by special tokens. This function prepares the text data for tokenization by
    the transformer model.

    :param df_element: A pandas Series or DataFrame representing the text data.
    :return: A string or pandas Series containing the sequence chain ready for tokenization.
    """
    return tokenizer.cls_token + \
        df_element["prev_sentence"] + \
        tokenizer.sep_token + \
        df_element["text"] + \
        tokenizer.sep_token + \
        df_element["next_sentence"] + \
        tokenizer.sep_token


def get_model_max_length(model_name: str, dataset_name: str) -> int:
    """Retrieves the maximum token length for a given model and dataset combination. This function reads from a YAML file
    containing the maximum token lengths and returns the appropriate length for the specified model and dataset.

    :param model_name: The name of the model.
    :param dataset_name: The name of the dataset.
    :return: The maximum token length for the specified model and dataset.
    """
    all_max_lengths = read_yaml_file(Path(f"{PROJECT_PATH}/tokenizing/model_max_token_lengths.yaml"))
    return all_max_lengths[dataset_name][model_name]


def encode(x):
    """Own encoding method just to define arguments one time, since it is used multiple times."""
    return tokenizer.encode_plus(x, add_special_tokens=False)


def get_token_length(x):
    """Get token length when given a encoded dict."""
    return len(x["input_ids"])


def main(data2process: pd.DataFrame, dataset_name: str):
    ds = tokenize_df(df_to_tokenize=data2process)

    # Save to folder structure
    save_tokenized_data_in_folder_structure(df=ds,
                                            base_directory=PROJECT_PATH,
                                            dataset_name=dataset_name,
                                            tokenizer_name=tokenizer.name_or_path)


if __name__ == "__main__":
    model = AutoTokenizer.from_pretrained(MODEL_TO_USE, use_fast=True)
    models = read_yaml_file(PATH_TO_MODEL_LIST)
    for dataset_name in DATASETS_TO_ANALYZE:
        print(f"> {dataset_name.upper()}")
        dataset = read_pickle_file(Path(f"{PROJECT_PATH}/{DATASET_PATH}/{dataset_name}_preprocessed.pkl"))

        for model_name, model_links in models.items():
            print(f">> {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_links.get("tokenizer"), use_fast=True)
            tokenizer.model_max_length = get_model_max_length(model_name=model_name, dataset_name=dataset_name)  # Setting for truncation

            if os.path.exists(f"{PROJECT_PATH}/{OUTPUT_TOKENIZED_DATASETS_PATH}/{dataset_name}/{tokenizer.name_or_path}"):
                print(f">> {model_name} already tokenized.")
            else:
                main(data2process=dataset, dataset_name=dataset_name)
