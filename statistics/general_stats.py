"""This script generates general statistics about OUR preprocessed versions of CiteWorth and SciSen."""

from typing import Tuple, List
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from utils.tooling import read_pickle_file, save_dataframe_to_markdown

import pandas as pd

data_source = Path.cwd().parent / "datasets"

# Constants - object to change
OUTPUT_PATH_TABLE = Path.cwd().parent / "output/tables"
OUTPUT_PATH_PLOTS = Path.cwd().parent / "output/graphs"


OUTPUT_NAME_TABLE_TEMPLATE = "general_stats_{}.md"  # "{}" in template is necessary for adding name and info
OUTPUT_NAME_PLOT = "general_stats_plot_{}.png"  # "{}" in template is necessary for adding dataset name


def get_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Retrieves the preprocessed CiteWorth and SciSen datasets from Pickle files.

    :return: A tuple containing the CiteWorth and SciSen datasets, respectively.
    """
    citeworth = read_pickle_file(data_source / "CiteWorth_preprocessed.pkl")
    scisen = read_pickle_file(data_source / "SciSen_preprocessed.pkl")
    return citeworth, scisen


def get_comparative_dataset_statistic(datasets: List[pd.DataFrame]):
    """Generates and returns general statistics comparing the provided datasets. This function calculates various
    statistics like total sentences, average sentence length, and check-worthiness percentages.

    :param datasets: A list of datasets (expected to be CiteWorth and SciSen) for which to calculate statistics.
    :return: A DataFrame containing the computed statistics for comparison.
    """
    datasets = {"CiteWorth": datasets[0],
                "SciSen": datasets[1]}

    table = pd.DataFrame(columns=datasets.keys())

    def calc_stat(function2use):
        """Applies a given function to each DataFrame in the datasets dictionary and returns the results.

        :param function2use: A function that will be applied to each DataFrame.
        :return: A dictionary with the same keys as the datasets and values computed by the function.
        """
        return {ds_name: function2use(df) for ds_name, df in datasets.items()}

    def add_dict_to_table(column_name, data2add):
        """Adds a new row to the global 'table' DataFrame using data from a dictionary.

        :param column_name: The name of the new row (column header in the DataFrame).
        :param data2add: A dictionary containing the data to be added as a new row in the table.
        """
        table.loc[column_name] = data2add.values()

    def get_percentile(dict_with_values):
        """
        Calculates the percentile values for each key-value pair in the provided dictionary.

        :param dict_with_values: A dictionary containing numerical values.
        :return: A dictionary with the same keys and their corresponding percentile values.
        """
        return {k: (value / sum(dict_with_values.values())) * 100 for k, value in dict_with_values.items()}

    size = calc_stat(len)
    avg_sentence_length_char = calc_stat(lambda df: df["text"].apply(len).mean()) # average_sentence_length [characters]
    avg_sentence_length_word = calc_stat(lambda df: df["text"].apply(lambda x: len(x.split())).mean())  # average_sentence_length [words]
    check_worthy = calc_stat(lambda df: df.groupby("label").count().loc["check-worthy", df.columns[0]]) # label: check-worthy [#]
    non_check_worthy = calc_stat(lambda df: df.groupby("label").count().loc["non-check-worthy", df.columns[0]])  # label: non-check-worthy [#]

    # Fill table
    add_dict_to_table(column_name="Total sentences", data2add=size)
    add_dict_to_table(column_name="Train sentences", data2add=calc_stat(lambda x: len(x.loc[x["split"] == "train"])))
    add_dict_to_table(column_name="Dev sentences", data2add=calc_stat(lambda x: len(x.loc[x["split"] == "validation"])))
    add_dict_to_table(column_name="Test sentences", data2add=calc_stat(lambda x: len(x.loc[x["split"] == "test"])))
    add_dict_to_table(column_name="average_sentence_length [characters]", data2add=avg_sentence_length_char)
    add_dict_to_table(column_name="average_sentence_length [words]", data2add=avg_sentence_length_word)
    add_dict_to_table(column_name="label: check-worthy [#]", data2add=check_worthy)
    add_dict_to_table(column_name="label: non-check-worthy [#]", data2add=non_check_worthy)
    add_dict_to_table(column_name="label: check-worthy [%]", data2add=get_percentile(check_worthy))
    add_dict_to_table(column_name="label: non-check-worthy [%]", data2add=get_percentile(non_check_worthy))

    # Output
    df_to_save = format_table_to_printable_output(table)
    save_dataframe_to_markdown(df_to_save, OUTPUT_PATH_TABLE / OUTPUT_NAME_TABLE_TEMPLATE.format("comparative_analysis"))
    return table


def format_table_to_printable_output(df):
    """Formats the data in a DataFrame for printable output. This function specifically formats numerical values
    for consistent presentation.

    :param df: The DataFrame to format.
    :return: A formatted DataFrame with adjusted numerical values for printing.
    """
    def converting_matrix(x):
        if isinstance(x, float):
            f"{x:.4f}"
        elif isinstance(x, int):
            f"{x:.0f}"
        else:
            str(x)
        return x
    # Format the float numbers to two decimal places
    formatted_df = df.applymap(converting_matrix)
    return formatted_df


def comparative_analysis(dataset1: pd.DataFrame, dataset2: pd.DataFrame) -> None:
    """Performs and visualizes a comparative analysis between two datasets. This includes creating plots for the
    distribution of section categories, labels, and the number of texts per section.

    :param dataset1: The first dataset for comparison (e.g., CiteWorth).
    :param dataset2: The second dataset for comparison (e.g., SciSen).
    """
    # Adding 'dataset' column to distinguish datasets in plots
    dataset1 = dataset1.copy()
    dataset2 = dataset2.copy()
    dataset1['dataset'] = 'CiteWorth'
    dataset2['dataset'] = 'SciSen'

    # Convert lists in 'section_category' to strings
    dataset1['section_category'] = dataset1['section_category'].apply(str)
    dataset2['section_category'] = dataset2['section_category'].apply(str)

    # Concatenating datasets
    combined_data = pd.concat([dataset1, dataset2])

    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")

    # Compare the distribution of section categories
    plt.figure(figsize=[10, 8])
    sns.countplot(x='section_category', data=combined_data, hue='dataset', stat="percent")
    plt.title('Distribution of Section Categories')
    plt.xticks(rotation=45)
    output_path = OUTPUT_PATH_PLOTS / OUTPUT_NAME_PLOT.format("dist_of_section_categories")
    plt.savefig(output_path)
    print(f"Saved figure to PNG at {output_path}")
    plt.show()

    # Compare the distribution of labels
    plt.figure(figsize=[10,6])
    sns.countplot(x='label', data=combined_data, hue='dataset', stat="percent")
    plt.title('Distribution of Labels')
    output_path = OUTPUT_PATH_PLOTS / OUTPUT_NAME_PLOT.format("dist_of_citeworthiness_labels")
    plt.savefig(output_path)
    print(f"Saved figure to PNG at {output_path}")
    plt.show()

    # Compare the number of texts per section
    texts_per_section_1 = dataset1.groupby(['paper_index', 'section_index']).size()
    texts_per_section_2 = dataset2.groupby(['paper_index', 'section_index']).size()

    texts_per_section_1 = texts_per_section_1 / sum(texts_per_section_1)
    texts_per_section_2 = texts_per_section_2 / sum(texts_per_section_2)

    plt.figure(figsize=(10, 6))
    sns.histplot(texts_per_section_1, label='CiteWorth', color='blue')
    sns.histplot(texts_per_section_2, label='SciSen', color='orange')
    plt.title('Number of Sentences per Section')
    plt.xlabel('Number of Sentences')
    plt.ylabel('Percentage')
    plt.legend()
    output_path = OUTPUT_PATH_PLOTS / OUTPUT_NAME_PLOT.format("dist_of_section_categories")
    plt.savefig(output_path)
    print(f"Saved figure to PNG at {output_path}")
    plt.show()


if __name__ == "__main__":
    ds_citeworth, ds_scisen = get_datasets()
    overall_stats = get_comparative_dataset_statistic(datasets=[ds_citeworth, ds_scisen])
    # comparative_analysis(ds_citeworth, ds_scisen)


