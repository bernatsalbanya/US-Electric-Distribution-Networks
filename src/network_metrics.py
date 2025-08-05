import numpy as np
import pandas as pd
import networkx as nx
import math
from shapely.geometry import LineString
import geopandas as gpd
import momepy
import os
from joblib import Parallel, delayed

import warnings

warnings.filterwarnings(
    "ignore",
    message="Geometry is in a geographic CRS.*",
    category=UserWarning
)




def boxcount(Z, k):
    """Count non-empty and non-full boxes in a grid of size k x k."""
    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
        np.arange(0, Z.shape[1], k),
        axis=1,
    )
    return len(np.where((S > 0) & (S < k * k))[0])


def fractal_dimension(Z, threshold=0.8):
    """
    Estimate the fractal dimension of a 2D array using box-counting.
    """
    assert len(Z.shape) == 2
    Z = Z < threshold
    p = min(Z.shape)
    if p < 2:
        return None
    n = 2 ** np.floor(np.log(p) / np.log(2))
    n = int(np.log(n) / np.log(2))
    sizes = 2 ** np.arange(n, 1, -1)
    counts = [boxcount(Z, size) for size in sizes]
    if len(sizes) == 0 or len(counts) == 0 or np.any(np.array(counts) <= 0):
        return None
    log_sizes = np.log(sizes)
    log_counts = np.log(counts)
    if np.any(np.isinf(log_counts)) or np.any(np.isnan(log_counts)):
        return None
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    return -coeffs[0]


def calculate_bearing(x1, y1, x2, y2):
    """Calculate compass bearing between two points."""
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return angle % 360


def get_bearings(G):
    """Extract edge bearings from a graph."""
    Gu = G.to_undirected()
    bearings = []
    for u, v, data in Gu.edges(data=True):
        x1, y1 = u[0], u[1]
        x2, y2 = v[0], v[1]
        bearing = calculate_bearing(x1, y1, x2, y2)
        bearings.append(bearing)
    return pd.Series(bearings)


def count_and_merge(n: int, bearings: pd.Series) -> np.ndarray:
    """Bin and merge bearings to calculate orientation entropy."""
    n = n * 2
    bins = np.arange(n + 1) * 360 / n
    count, _ = np.histogram(bearings, bins=bins)
    count = np.roll(count, 1)
    return count[::2] + count[1::2]


def get_orientation_order(count: np.ndarray) -> float:
    """Calculate entropy of street orientation order."""
    H0 = 0
    for c in count:
        Pi = c / sum(count)
        if Pi != 0:
            H0 += Pi * np.log(Pi)
    H0 = -1 * H0
    Hmax = np.log(len(count))
    Hg = np.log(2)
    return 1 - (((H0 - Hg) / (Hmax - Hg)) ** 2)


def get_entropy(graph: nx.Graph) -> float:
    """Compute entropy of edge orientations in a graph."""
    bearings = get_bearings(graph)
    if bearings.empty:
        return np.nan
    count = count_and_merge(36, bearings)
    return get_orientation_order(count)


def pp_compactness(geom):
    """Calculate Polsby-Popper compactness score."""
    p = geom.length
    a = geom.area
    return (4 * math.pi * a) / (p * p)


def calculate_compactness(edges_gdf):
    # Ensure the geometry column is set
    if 'geometry' not in edges_gdf.columns:
        raise ValueError("No geometry column found in the GeoDataFrame.")
    edges_gdf = edges_gdf.set_geometry('geometry')

    # Ensure CRS is set
    if edges_gdf.crs is None:
        edges_gdf.set_crs("EPSG:4326", inplace=True)

    # Reproject to a projected CRS for accurate calculations
    edges_gdf = edges_gdf.to_crs("EPSG:3857")

    # Calculate the convex hull
    convex_hull = edges_gdf.unary_union.convex_hull
    return convex_hull


def process_one_geoid(geoid, sub_lines, output_dir):
    """
    Process one GEOID to compute node, edge, and global metrics.
    """
    sub_lines = sub_lines[sub_lines['HostCap_MW'] >= 0]
    sub_lines = sub_lines.explode(index_parts=False)
    sub_lines = sub_lines[sub_lines['geometry'].apply(lambda geom: isinstance(geom, LineString))]

    if sub_lines.empty:
        print(f"Skipping GEOID {geoid}: no valid geometries.")
        return

    primal_graph = momepy.gdf_to_nx(sub_lines, approach="primal")
    nodes_gdf, edges_gdf = momepy.nx_to_gdf(primal_graph)

    node_metrics = [
        ("nodes_betweenness", momepy.betweenness_centrality, {"mode": "nodes", "weight": "HostCap_MW"}),
        ("clustering_coefficient", momepy.clustering, {}),
        ("degree", momepy.node_degree, {}),
        ("closeness400", momepy.closeness_centrality, {"radius": 50, "distance": "mm_len", "weight": "HostCap_MW"}),
        ("closeness_global", momepy.closeness_centrality, {"weight": "HostCap_MW"}),
        ("straightness_centrality", momepy.straightness_centrality, {"weight": "mm_len"}),
    ]

    for metric_name, metric_func, metric_args in node_metrics:
        primal_graph = metric_func(primal_graph, name=metric_name, **metric_args)
        metric_values = momepy.nx_to_gdf(primal_graph, lines=False)
        nodes_gdf = nodes_gdf.merge(metric_values[['nodeID', metric_name]], on='nodeID', how='left')

    edge_graph = momepy.gdf_to_nx(edges_gdf, approach="primal")
    edge_graph2 = momepy.betweenness_centrality(edge_graph, name="edges_betweenness", mode="edges", weight="HostCap_MW")
    edges_metrics = momepy.nx_to_gdf(edge_graph2, points=False)
    edges_gdf = edges_gdf.merge(edges_metrics[['geometry', 'edges_betweenness']], on='geometry', how='left')

    if isinstance(edge_graph, (nx.MultiGraph, nx.MultiDiGraph)):
        edge_graph = nx.Graph(edge_graph)

    if nx.is_connected(edge_graph):
        avg_shortest_path = nx.average_shortest_path_length(edge_graph)
    else:
        component_lengths = [
            nx.average_shortest_path_length(edge_graph.subgraph(c)) for c in nx.connected_components(edge_graph)
        ]
        avg_shortest_path = sum(component_lengths) / len(component_lengths) if component_lengths else None

    adj_matrix = nx.to_numpy_array(edge_graph)
    global_metrics = {
        'geoid': geoid,
        'avg_shortest_path': avg_shortest_path,
        'connectivity': nx.number_connected_components(edge_graph),
        'assortativity': nx.degree_assortativity_coefficient(edge_graph),
        'density': nx.density(edge_graph),
        'fractality': fractal_dimension(adj_matrix),
        'entropy': get_entropy(edge_graph),
        #'compactness': calculate_compactness(edge_graph),
        'average_segment_length': edges_gdf.geometry.length.mean(),
        'density_of_intersections': len(nodes_gdf) / sub_lines.total_bounds[-2],
        'density_of_segments': len(edges_gdf) / sub_lines.total_bounds[-2],
        'avg_intersection_connectivity': len(nodes_gdf) / len(edges_gdf),
        'avg_nodes_betweenness_centrality': nodes_gdf['nodes_betweenness'].mean(),
        'avg_edges_betweenness_centrality': edges_gdf['edges_betweenness'].mean(),
        'avg_local_closeness': nodes_gdf['closeness400'].mean(),
        'avg_global_closeness': nodes_gdf['closeness_global'].mean(),
        'avg_straightness_centrality': nodes_gdf['straightness_centrality'].mean(),
        'avg_node_degree': nodes_gdf['degree'].mean(),
        'avg_node_strength': np.nanmean(list(dict(nx.degree(primal_graph, weight='HostCap_MW')).values())),
        'total_node_degree': nodes_gdf['degree'].sum(),
        'total_node_strength': np.nansum(list(dict(nx.degree(primal_graph, weight='HostCap_MW')).values())),
        'number_edges': len(edges_gdf),
        'number_nodes': len(nodes_gdf)
    }

    nodes_csv_path = os.path.join(output_dir, f"{geoid}_nodes_metrics.csv")
    global_metrics_csv_path = os.path.join(output_dir, f"{geoid}_global_metrics.csv")

    nodes_gdf.to_csv(nodes_csv_path, index=False)
    pd.DataFrame([global_metrics]).to_csv(global_metrics_csv_path, index=False)

def split_by_geoid(gdf: gpd.GeoDataFrame, geoid_column: str = "GEOID"):
    """
    Splits a GeoDataFrame into a dictionary of sub-GeoDataFrames based on unique GEOID values.

    Parameters:
        gdf (GeoDataFrame): The input GeoDataFrame with a GEOID column.
        geoid_column (str): The column name to split on. Defaults to 'GEOID'.

    Returns:
        sub_dfs (dict): Dictionary where keys are GEOIDs and values are corresponding sub-GeoDataFrames.
        unique_geoids (List[int]): Sorted list of unique GEOID values.
    """
    # Drop rows where GEOID is null
    gdf = gdf[gdf[geoid_column].notnull()].copy()

    # Convert GEOID to int (assuming all are valid numeric strings or floats)
    gdf[geoid_column] = gdf[geoid_column].astype(int)

    # Get sorted unique GEOIDs
    unique_geoids = sorted(gdf[geoid_column].unique())

    # Create dictionary of sub-GeoDataFrames
    sub_dfs = {
        geoid: gdf[gdf[geoid_column] == geoid]
        for geoid in unique_geoids
    }

    return sub_dfs, unique_geoids

def run_pipeline(input_path, output_dir, n_jobs):
    """ 
    Run the entire pipeline to process the input GeoDataFrame and compute metrics for each GEOID.
    parameters:
        input_path (str): Path to the input GeoDataFrame file.
        output_dir (str): Directory to save the output CSV files.
        n_jobs (int): Number of parallel jobs to run.
    """
    lines = gpd.read_file(input_path)
    sub_dfs, unique_geoids = split_by_geoid(lines)

    print(unique_geoids)

    Parallel(n_jobs=n_jobs)(
        delayed(process_one_geoid)(geoid, sub_dfs[geoid], output_dir)
        for geoid in unique_geoids
    )

    # Combine global metrics
    global_dfs = []
    for geoid in unique_geoids:
        global_file = os.path.join(output_dir, f"{geoid}_global_metrics.csv")
        if os.path.exists(global_file):
            global_dfs.append(pd.read_csv(global_file))

    if global_dfs:
        global_metrics_df = pd.concat(global_dfs, ignore_index=True)
        global_metrics_df.to_csv(os.path.join(output_dir, "global_metrics.csv"), index=False)
        print("Saved combined global metrics.")
    else:
        print("No global metrics files found.")