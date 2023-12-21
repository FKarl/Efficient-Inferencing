import pandas as pd
from datasets import load_dataset_builder, load_dataset, Dataset
from typing import Union
from utils.tooling import write_pickle_file, \
    read_yaml_file, \
    read_pickle_file, \
    clean_dataset, \
    print_markdown_sample, \
    check_test_size
from pathlib import Path


config_path = Path.cwd().parent / "config"


def get_data_from_huggingface(hugging_face_link: str, split: Union[None, str] = None, is_test: bool = False) -> Dataset:
    """Sources data from Hugging Face.

    :param hugging_face_link: Link on Hugging Face to dataset.
    :param split: Nome of split to filter by.
    :param is_test: Can be set to True, if the smallest split should be used. Good for testing and debugging.
    :return: Hugging Face Dataset.
    """
    if is_test and split is None:
        split = get_smallest_split(hugging_face_link)

    data = load_dataset(hugging_face_link, split=split)
    return data


def get_smallest_split(hugging_face_link: str) -> str:
    """Find the smallest split in terms of byte size.

    :param hugging_face_link: Link on Hugging Face to dataset yet to be analyzed.
    :return: Name of the smallest dataset split (usually test or validation)
    """
    ds_builder = load_dataset_builder(hugging_face_link)  # connect without loading data
    num_bytes_per_split = {split_name: split_info.num_bytes  # limit info to number of bytes
                           for split_name, split_info in ds_builder.info.splits.items()}
    return min(num_bytes_per_split, key=num_bytes_per_split.get)  # find name of the smallest split


def remove_nested_structure(nested_dataset: Dataset, nested_column: str) -> pd.DataFrame:
    """Break up nested DataFrame.

    :param nested_dataset: DataFrame where one column contains another table-like structure
    :param nested_column: Name of the column that contains table-like elements.
    :return: Single, 2-dimensional (one layer) DataFrame with the same information density.
    """
    if isinstance(nested_dataset.shape, dict):  # check if multiple splits exist
        nested_df = pd.DataFrame()
        for split in nested_dataset.shape:
            split_df = pd.DataFrame(nested_dataset[split])
            split_df["split"] = split
            nested_df = pd.concat([nested_df, split_df])
    else:
        nested_df = pd.DataFrame(nested_dataset)
    nested_df = nested_df.reset_index(drop=True)
    exploded = nested_df.explode('samples').reset_index(drop=True)

    # Create a separate DataFrame from the dictionary column and concatenate it with the original data
    samples = pd.DataFrame(exploded['samples'].tolist())
    exploded_df = pd.concat([exploded.drop(columns='samples'), samples], axis=1)

    return exploded_df




def label_categories(column_to_label: pd.Series) -> pd.Series:
    """Translate the section titles to section labels, predefined by SciSen.

    :param column_to_label: Column with section titles.
    :return: Column with section labels.
    """
    section_title_translator = read_yaml_file(config_path / "synonyms_extended.yaml")
    column_labeled = column_to_label.apply(  # Translates all entries of the given column based on a dict holding labels
        lambda column_entry: section_title_translator.get(column_entry.lower()))
    return column_labeled


def main(recompile_dataset: bool, is_test: bool):
    """Main function to format CiteWorth dataset.

    :param recompile_dataset: Can be set to True in order to run the whole pipeline again.
    :param is_test: Can be set to True in order to only load a small fraction of the dataset from Hugging Face.
    """
    """RESTRUCTURING"""
    if recompile_dataset:
        citeworth_link = "copenlu/citeworth"
        citeworth_data = get_data_from_huggingface(hugging_face_link=citeworth_link, is_test=is_test)
        single_layer_df = remove_nested_structure(nested_dataset=citeworth_data, nested_column="samples")
        write_pickle_file(file_name=Path.cwd().parent.parent / "datasets" / "CiteWorth_restructured.pkl",
                          data_to_save=single_layer_df)
    else:
        data_path = Path.cwd().parent.parent / "datasets" / "CiteWorth_restructured.pkl"
        single_layer_df = read_pickle_file(file_name=data_path)

    """LABELING"""
    single_layer_df["section_category"] = label_categories(single_layer_df["section_title"])
    ds = single_layer_df.dropna(subset="section_category")

    """CLEANING"""
    ds = clean_dataset(data_to_clean=ds, cfg_for_cleaning=config_path / "cite_worth_structure.yaml")
    ds['text_index'] = ds.groupby(['paper_index', 'section_index']).cumcount()  # Add text index

    """SPLIT"""
    if check_test_size(ds, 0.8):
        raise ValueError

    write_pickle_file(file_name=Path.cwd().parent.parent / "datasets" / "CiteWorth_preprocessed.pkl",
                      data_to_save=ds)
    print_markdown_sample(ds)


if __name__ == "__main__":
    main(recompile_dataset=True, is_test=False)
