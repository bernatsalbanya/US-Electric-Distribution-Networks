import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def combine_metrics(unique_geoids, output_dir="ResultsNortheast/", save=True):
    """
    Combine node metrics and global metrics from CSV files for all GEOIDs.

    Parameters:
    - unique_geoids (list): List of GEOID strings.
    - output_dir (str): Directory where the CSV files are stored and results will be saved.
    - save (bool): Whether to save the combined DataFrames to CSV.

    Returns:
    - nodes_combined (DataFrame): Combined DataFrame of all node metrics.
    - global_metrics_df (DataFrame): Combined DataFrame of all global metrics.
    """
    nodes_dfs = []
    global_dfs = []

    for geoid in unique_geoids:
        nodes_file = os.path.join(output_dir, f"{geoid}_nodes_metrics.csv")
        global_file = os.path.join(output_dir, f"{geoid}_global_metrics.csv")

        if os.path.exists(nodes_file):
            nodes_dfs.append(pd.read_csv(nodes_file))
        else:
            print(f"File not found: {nodes_file}")
        
        if os.path.exists(global_file):
            global_dfs.append(pd.read_csv(global_file))
        else:
            print(f"File not found: {global_file}")

    nodes_combined = pd.concat(nodes_dfs, ignore_index=True) if nodes_dfs else pd.DataFrame()
    global_metrics_df = pd.concat(global_dfs, ignore_index=True) if global_dfs else pd.DataFrame()

    if save:
        nodes_combined.to_csv(os.path.join(output_dir, 'node_metrics_no_false.csv'), index=False)
        global_metrics_df.to_csv(os.path.join(output_dir, 'global_metrics_no_false.csv'), index=False)

    return nodes_combined, global_metrics_df


def enrich_geoid_info(global_metrics_df):
    """
    Enrich global metrics DataFrame with geoid prefix metadata.
    """
    global_metrics_df['geoid'] = global_metrics_df['geoid'].astype(str)
    global_metrics_df['geoid_prefix'] = global_metrics_df['geoid'].str[:2]
    # print("Unique prefixes:", global_metrics_df['geoid_prefix'].unique())
    return global_metrics_df


def summarize_best_fit_distributions(global_metrics_df, output_csv='goodness_of_fit_summary.csv'):
    """
    Perform goodness-of-fit tests on numeric columns, generate summary table.
    """
    distributions = ['norm', 'expon', 'lognorm', 'gamma', 'beta', 'chi2',
                     'weibull_min', 'uniform', 't', 'pareto']

    numeric_columns = global_metrics_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    results = []
    summary_table = []

    for column in numeric_columns:
        data = global_metrics_df[column].dropna()
        for dist_name in distributions:
            dist = getattr(stats, dist_name)
            try:
                params = dist.fit(data)
                D, p_value = stats.kstest(data, dist_name, args=params)
                results.append({
                    'Variable': column,
                    'Distribution': dist_name,
                    'D-statistic': D,
                    'p-value': p_value,
                    'Parameters': params
                })
            except Exception:
                continue

        best_fit = max([r for r in results if r['Variable'] == column], key=lambda x: x['p-value'], default=None)
        if best_fit:
            equation = f"f(x) = {best_fit['Distribution']}.pdf(x, *{best_fit['Parameters'][:-2]}, loc={best_fit['Parameters'][-2]}, scale={best_fit['Parameters'][-1]})"
            summary_table.append({
                'Variable': column,
                'Best Fit Distribution': best_fit['Distribution'],
                'Parameters': best_fit['Parameters'],
                'p-value': best_fit['p-value'],
                'Equation': equation
            })

    summary_df = pd.DataFrame(summary_table)
    summary_df.to_csv(output_csv, index=False)
    return pd.DataFrame(results), summary_df


def plot_distributions(results_df, global_metrics_df, output_path='goodfit_plots.png'):
    """
    Plot histograms and best-fit PDFs for each numeric column.
    """
    numeric_columns = global_metrics_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    num_vars = len(numeric_columns)
    num_cols = 3
    num_rows = (num_vars + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    for i, column in enumerate(numeric_columns):
        data = global_metrics_df[column].dropna()
        best_fit = results_df[results_df['Variable'] == column].sort_values(by='p-value', ascending=False).iloc[0]
        best_dist = getattr(stats, best_fit['Distribution'])
        params = best_fit['Parameters']
        x = np.linspace(min(data), max(data), 100)
        pdf = best_dist.pdf(x, *params[:-2], loc=params[-2], scale=params[-1])

        ax = axes[i]
        n, bins, _ = ax.hist(data, bins=30, density=True, color='#4472C4', edgecolor='black', label='Data')
        max_hist_height = np.max(n)
        ax.plot(x, np.clip(pdf, 0, max_hist_height), '#ED7D31', lw=2, label='Best Fit Distribution')
        ax.set_ylim(0, max_hist_height * 1.1)
        ax.set_title(f'{column} - Best Fit: {best_fit["Distribution"]}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()

    for j in range(num_vars, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
