import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_hosting_capacity_histograms(datasets, output_dir="output/plots"):
    """
    Generate histograms for Hosting Capacity across different datasets and save as PDFs.

    Args:
        datasets (dict): Dictionary of dataset names and corresponding GeoDataFrames.
        output_dir (str): Directory where plots will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    num_datasets = len(datasets)
    rows = (num_datasets // 3) + (num_datasets % 3 > 0)
    cols = min(3, num_datasets)

    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()

    for i, (name, df) in enumerate(datasets.items()):
        if 'HostCap_MW' in df.columns:
            ax = axes[i]
            sns.histplot(df['HostCap_MW'].dropna(), bins=30, kde=True, color='#4472C4', edgecolor='black', stat='count', ax=ax)
            ax.set_title(f'Hosting Capacity in {name}')
            ax.set_xlabel('Hosting Capacity (MW)')
            ax.set_ylabel('Frequency')
            ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    pdf_path = os.path.join(output_dir, "hosting_capacity_histograms.pdf")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Histogram plots saved to {pdf_path}")


def plot_voltage_distribution(datasets, output_dir="output/plots"):
    """
    Generate histograms for Voltage (kV) across different datasets and save as PDFs.

    Args:
        datasets (dict): Dictionary of dataset names and corresponding GeoDataFrames.
        output_dir (str): Directory where plots will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    num_datasets = len(datasets)
    rows = (num_datasets // 3) + (num_datasets % 3 > 0)
    cols = min(3, num_datasets)

    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()

    for i, (name, df) in enumerate(datasets.items()):
        if 'Voltage_kV' in df.columns:
            ax = axes[i]
            sns.histplot(df['Voltage_kV'].dropna(), bins=30, kde=True, color='#4472C4', edgecolor='black', stat='count', ax=ax)
            ax.set_title(f'Voltage in {name}')
            ax.set_xlabel('Voltage (kV)')
            ax.set_ylabel('Frequency')
            ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    pdf_path = os.path.join(output_dir, "voltage_histograms.pdf")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Histogram plots saved to {pdf_path}")


def plot_circuit_rating_distribution(datasets, output_dir="output/plots"):
    """
    Generate histograms for Circuit Rating (A) across different datasets and save as PDFs.

    Args:
        datasets (dict): Dictionary of dataset names and corresponding GeoDataFrames.
        output_dir (str): Directory where plots will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    num_datasets = len(datasets)
    rows = (num_datasets // 3) + (num_datasets % 3 > 0)
    cols = min(3, num_datasets)

    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()

    for i, (name, df) in enumerate(datasets.items()):
        if 'CircRat_A' in df.columns:
            ax = axes[i]
            sns.histplot(df['CircRat_A'].dropna(), bins=30, kde=True, color='#4472C4', edgecolor='black', stat='count', ax=ax)
            ax.set_title(f'Circuit Rating in {name}')
            ax.set_xlabel('Circuit Rating (A)')
            ax.set_ylabel('Frequency')
            ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    pdf_path = os.path.join(output_dir, "circuit_rating_histograms.pdf")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Histogram plots saved to {pdf_path}")


def plot_utility_lines(datasets, output_dir="output/plots", color_map=None, linewidth=0.25, alpha=0.1, figsize=(10, 5), title='US Northeast'):
    """
    Plots utility line datasets on a single matplotlib axis.

    Parameters:
        datasets (dict): Dictionary of GeoDataFrames to plot.
        color_map (dict): Optional dictionary mapping utility name prefixes to colors.
        alpha (float): Transparency level for the lines.
        figsize (tuple): Figure size.
        title (str): Title of the plot.
    """
    if color_map is None:
        color_map = {
            'lines_NY_CenHud': 'green',
            'lines_NY_ConEd': 'red',
            'lines_NY_AvGri': 'blue',
            'lines_NY_NatGrid': 'orange',
            'lines_NY_ORU': 'grey',
            'lines_ME_CMP': 'green',
            'lines_NH_Liberty': 'red',
            'lines_NH_Eversource': 'blue',
            'lines_MA_Unitil': 'green',
            'lines_MA_NatGrid': 'red',
            'lines_MA_Eversource': 'blue',
            'lines_RI': 'orange',
            'lines_CT_United': 'yellow',
            'lines_CT_Eversource': 'grey',
            'lines_VT': 'black'
        }

    fig, ax = plt.subplots(figsize=figsize)

    for name, gdf in datasets.items():
        color = None
        for prefix, c in color_map.items():
            if name.startswith(prefix):
                color = c
                break
        gdf.plot(ax=ax, alpha=alpha, color=color)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)

    pdf_path = os.path.join(output_dir, "plot_utility_lines.pdf")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Plots utility line datasets saved to {pdf_path}")


def plot_merged_utility_lines(dataset, output_dir="output/plots", color='#4472C4', linewidth=0.25, alpha=0.1, figsize=(10, 5), title='US Northeast'):
    """
    Plots a single merged utility line dataset on a matplotlib axis and saves it as a PDF.

    Parameters:
        dataset (GeoDataFrame): The merged GeoDataFrame to plot.
        output_dir (str): Directory to save the output plot.
        color (str): Line color.
        alpha (float): Transparency level for the lines.
        figsize (tuple): Figure size (width, height).
        title (str): Title of the plot.
    """
    
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    dataset.plot(ax=ax, alpha=alpha, color=color)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)

    pdf_path = os.path.join(output_dir, "plot_merged_utility_lines.pdf")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"Utility line plot saved to: {pdf_path}")
