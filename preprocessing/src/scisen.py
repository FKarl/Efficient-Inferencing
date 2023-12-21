from pathlib import Path
from utils.tooling import read_json_file, \
    clean_dataset, \
    print_markdown_sample, \
    write_pickle_file, \
    read_pickle_file, \
    do_manual_split
import pandas as pd

config_path = Path.cwd().parent / "config"
datasets_path = Path.cwd().parent.parent / "datasets"


def remove_invalid_sentences(list_of_sentences: list) -> list:
    list_of_sentences = list(filter(lambda x: x != "[removed]", list_of_sentences))  # Remove "[removed]" entries.
    list_of_sentences = [sentence for sentence in list_of_sentences if isinstance(sentence, str)]  # Remove non Strings.
    return list_of_sentences


def break_sentences(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Breaks the sentences in a DataFrame column containing string elements into individual sentences.

    :param df: The DataFrame containing the column with string elements.
    :param column_name: The name of the column containing the string elements.
    :return: A new DataFrame with each sentence in a separate row, along with an increasing
             sentence index and the same information in the other columns.
    """
    df["text_index"] = df[column_name].apply(lambda x: list(range(len(x))))  # Add a text index
    exploded_df = df.explode(column=["sentences", "text_index"]).reset_index(
        drop=True)  # Explode the DataFrame by the sentences
    return exploded_df


def label_sentences(column_with_text: pd.Series) -> pd.Series:
    """Checks a columns String entries and labels them based on a token.

    The SciSen group replaced citation markers with random names from Facebook. We adapted SciSen's preprocessing that,
    instead of random names, a token "<citation>" is used. This function checks the sentence for such a token and uses
    it to label the data.

    Example:
    column_with_text
    | 0| "As shown in <citation>."|
    | 1| "This paper is awesome."  |

    column_with_labels
    | 0| "check-worthy"      |
    | 1| "non-check-worthy"  |

    :param column_with_text: Series that contains String elements, where some contain names as a token.
    :return: Series of same size with 'check-worthy' and 'non-check-worthy' labels.
    """
    token = "<citation>"
    check_mask = column_with_text.str.contains(token, na=False)  # Check if entry contains token

    # Replace values based on the mask
    column_with_labels = column_with_text.copy()
    column_with_labels[check_mask] = "check-worthy"
    column_with_labels[~check_mask] = "non-check-worthy"
    return column_with_labels


def remove_citation_token(column_with_tokens: pd.Series) -> pd.Series:
    """Remove the "<citation>" tokens from each string in the column.

    The idea behind this is, that the model does not learn to detect the token, but rather grammar characteristics.

    :param column_with_tokens: Column that contains Strings with potentially "<citation>"-tokens.
    :return: Column of Strings without token in them.
    """
    updated_series = column_with_tokens.str.replace("<citation>", "")
    return updated_series


def main(recompile_dataset: bool):
    if recompile_dataset:
        dataset_path = datasets_path / "SciSections_sentences.jsonl"
        scisen_raw = read_json_file(dataset_path)
        ds = pd.DataFrame(scisen_raw)
        ds["section_index"] = (ds["paper_id"] != ds["paper_id"].shift()).cumsum() - 1
        ds["section_index"] = ds.groupby("section_index").cumcount()
        ds["sentences"] = ds["sentences"].apply(remove_invalid_sentences)
        ds = ds[ds["sentences"].map(len) > 0]  # Remove rows where the list of sentences is empty.
        ds = break_sentences(ds, "sentences")
        ds = clean_dataset(data_to_clean=ds, cfg_for_cleaning=config_path / "sci_sen_structure.yaml")
        ds = do_manual_split(full_dataset=ds, train_split_size=0.8)
        ds["label"] = label_sentences(ds["text"])
        ds["text"] = remove_citation_token(ds["text"])
        scisen = ds.copy()
        write_pickle_file(file_name=Path.cwd().parent.parent / "datasets" / "SciSen_preprocessed.pkl",
                          data_to_save=ds)
    else:
        scisen = read_pickle_file(file_name=Path.cwd().parent.parent / "datasets" / "SciSen_preprocessed.pkl")

    print_markdown_sample(scisen)


if __name__ == "__main__":
    main(recompile_dataset=True)

