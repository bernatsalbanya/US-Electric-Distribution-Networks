import geopandas as gpd
import os
import requests
import zipfile
import yaml
import pandas as pd
from pathlib import Path
from shapely.geometry import shape, mapping, LineString, Point, MultiLineString

# Load configuration file
config_path = Path(__file__).resolve().parent.parent / "config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Define base directories
CENSUS_DIR = "data/census_tracts/"
CIRCUITS_DIR = "data/circuits/"

# Dynamically load file paths from config.yaml
CENSUS_FILES = {key: os.path.join(CENSUS_DIR, value) for key, value in config['data_paths']['census_tracts'].items()}
CIRCUIT_FILES = {key: os.path.join(CIRCUITS_DIR, value) for key, value in config['data_paths']['circuits'].items()}

def load_census_tracts():
    """
    Load and concatenate census tract data for Northeastern states.
    
    Returns:
        GeoDataFrame: Combined census tract dataset.
    """
    census_gdfs = []
    
    for state, path in CENSUS_FILES.items():
        if os.path.exists(path):
            gdf = gpd.read_file(path)
            gdf["State"] = state  # Add a state column for reference
            census_gdfs.append(gdf)
        else:
            print(f"Warning: Census tract file missing for {state} ({path})")

    if not census_gdfs:
        raise FileNotFoundError("No valid census tract files found.")

    # Merge all census tract data
    northeast = gpd.GeoDataFrame(pd.concat(census_gdfs, ignore_index=True))

    # Remove waterland (ALAND = 0)
    northeast = northeast[northeast['ALAND'] > 0]

    return northeast


def load_circuit_data():
    """
    Load circuit datasets dynamically based on available files.

    Returns:
        dict: Dictionary containing circuit dataframes with their respective keys.
    """
    circuits = {}

    for key, path in CIRCUIT_FILES.items():
        if os.path.exists(path):
            circuits[key] = gpd.read_file(path)
        else:
            print(f"Warning: Circuit file missing for {key} ({path})")

    if not circuits:
        raise FileNotFoundError("No valid circuit datasets found.")

    return circuits

def list_files_in_dataverse_dataset(url):
    """Get a list of all files in a Dataverse dataset given a DOI."""
    response = requests.get(f"{url}/versions/latest/files")

    if response.status_code != 200:
        raise Exception(f"Could not access dataset: {response.status_code}")

    return response.json()["data"]

def download_all_files_from_dataverse(doi: str, output_dir: str = "data/raw"):
    """Download all files from a Dataverse dataset to a local folder."""
    os.makedirs(output_dir, exist_ok=True)
    files = list_files_in_dataverse_dataset(doi)

    for f in files:
        file_id = f["dataFile"]["id"]
        filename = f["label"]
        download_url = f"https://dataverse.harvard.edu/api/access/datafile/{file_id}"
        output_path = os.path.join(output_dir, filename)

        if os.path.exists(output_path):
            print(f"{filename} already exists, skipping.")
            continue

        print(f"Downloading {filename}...")
        with requests.get(download_url, stream=True) as r:
            if r.status_code != 200:
                print(f"Failed to download {filename}")
                continue

            with open(output_path, "wb") as f_out:
                for chunk in r.iter_content(chunk_size=8192):
                    f_out.write(chunk)

        print(f"Downloaded: {filename}")

def unzip_file(zip_path: str, extract_to: str):
    print(f"Unzipping {zip_path} to {extract_to}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Unzip complete.")


if __name__ == "__main__":
    print("Loading census tracts...")
    census_tracts = load_census_tracts()
    print(f"Loaded {len(census_tracts)} census tracts.")

    print("Loading circuit data...")
    circuit_data = load_circuit_data()
    print(f"Loaded {len(circuit_data)} circuit datasets.")
