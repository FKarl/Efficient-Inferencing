"""
This module provides utilities for generating an upset plot and a label table for given datasets.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from upsetplot import plot
from utils.tooling import read_pickle_file, read_yaml_file, save_dataframe_to_markdown

# Constants - object to change
DATASET_TO_USE = "citeworth"  # or "scisen"
DATASET_PATH = Path.cwd().parent / "datasets"  # Change this to the location you saved the preprocessed datasets
CONFIG_PATH = Path.cwd().parent / "preprocessing" / "config"
OUTPUT_PATH_TABLE = Path.cwd().parent / "output/tables"
OUTPUT_PATH_UPSET_PLOT = Path.cwd().parent / "output/graphs"

# Additional Constants
OUTPUT_NAME_TABLE_TEMPLATE = "section_label_distribution_{}{}"  # "{}" in template is necessary for adding name and info
OUTPUT_NAME_UPSET_PLOT = "upset_plot_{}.pdf"  # "{}" in template is necessary for adding dataset name
NORMALIZED_SUFFIX = "_normalized.md"
NON_NORMALIZED_SUFFIX = ".md"


def upset_plot(df: pd.DataFrame, dataset_name: str):
    """
    Generate and display an upset plot for a given dataset.

    :param df: DataFrame containing the dataset.
    :param dataset_name: Name of the dataset to use for the title.
    """
    fig = plt.figure(figsize=(30, 5))  # Adjust the figure size here

    df = df["section_category"].apply(','.join).str.get_dummies(sep=',')
    all_columns = list(df.columns)
    df = df.astype(bool)
    df["value"] = 0
    cnt = df.groupby(all_columns).count().squeeze()
    cnt = cnt / cnt.sum()  # normalize
    plot(cnt,
         fig=fig,
         sort_by="cardinality",
         show_percentages='{:.2%}',
         intersection_plot_elements=8,  # height of bar chart
         totals_plot_elements=3,  # width of class count
         element_size=42)  # width

    max_percentage = 0.8
    fig.axes[3].set_ylim(0, max_percentage)  # top bar chart y lim
    fig.axes[2].set_xlim(max_percentage, 0)  # top bar chart y lim

    # fig.suptitle(dataset_name)

    output_path = OUTPUT_PATH_UPSET_PLOT / OUTPUT_NAME_UPSET_PLOT.format(dataset_name)
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved figure to {OUTPUT_NAME_UPSET_PLOT.format(dataset_name)} at {output_path}")
    plt.show()


def _2d_label_table(df: pd.DataFrame, dataset_name: str, normalize: bool = False) -> pd.DataFrame:
    """
    Generate a 2D table of label distributions and save it to a markdown file.

    :param df: DataFrame containing the dataset.
    :param dataset_name: Name of the dataset.
    :param normalize: Whether to normalize the distributions.

    :return table: DataFrame containing the 2D label distributions.
    """
    unique_labels = read_yaml_file(CONFIG_PATH / "label_mapping.yaml")["section_category"]
    unique_labels = list(unique_labels.keys())

    table = pd.DataFrame(columns=unique_labels, index=unique_labels)  # Create empty table

    # Calculate label distributions
    label_distribution = df.section_category.value_counts(normalize=normalize)
    label_count = label_distribution.reset_index().section_category.apply(len)
    single_labels = label_distribution.loc[pd.Index(label_count == 1)]
    multi_labels = label_distribution.loc[pd.Index(label_count > 1)]

    # Fill table
    for l, val in single_labels.items():
        table.loc[l[0], l[0]] = val
    for l, val in multi_labels.items():
        table.loc[l[0], l[1]] = val

    output_suffix = NORMALIZED_SUFFIX if normalize else NON_NORMALIZED_SUFFIX
    save_dataframe_to_markdown(table, OUTPUT_PATH_TABLE / OUTPUT_NAME_TABLE_TEMPLATE.format(dataset_name, output_suffix))

    return table


def main():
    """
    Main execution logic: Reads a dataset, generates a 2D label table, and an upset plot.
    """
    dataset_path = {
        "citeworth": "CiteWorth_preprocessed.pkl",
        "scisen": "SciSen_preprocessed_fixed.pkl"
    }

    ds = read_pickle_file(DATASET_PATH / dataset_path[DATASET_TO_USE])
    # _2d_label_table(ds, dataset_name=DATASET_TO_USE, normalize=True)
    upset_plot(ds, dataset_name=DATASET_TO_USE)


if __name__ == "__main__":
    main()
