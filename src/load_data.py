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

def extract_doi_from_url(url: str) -> str:
    """Extract DOI from a Dataverse dataset URL."""
    match = re.search(r"doi:10\.\d{4,9}/[-._;()/:A-Z0-9]+", url, re.IGNORECASE)
    if not match:
        raise ValueError("DOI not found in the provided URL.")
    return match.group(0)

def list_files_in_dataverse_dataset(dataset_url: str):
    """List all files in a Dataverse dataset given its HTML page URL."""
    doi = extract_doi_from_url(dataset_url)
    api_url = "https://dataverse.harvard.edu/api/datasets/:persistentId/versions/latest/files"
    params = {"persistentId": doi}
    response = requests.get(api_url, params=params)

    if response.status_code != 200:
        raise Exception(f"Could not access dataset: {response.status_code}")

    return response.json()["data"]

def download_all_files_from_dataverse(dataset_url: str, output_dir: str = "data/raw"):
    """Download all files from a Dataverse dataset URL to a local folder.
       Automatically unzip ZIP files and delete them afterward."""
    os.makedirs(output_dir, exist_ok=True)
    files = list_files_in_dataverse_dataset(dataset_url)

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

        # If it's a ZIP, unzip and delete it
        if filename.lower().endswith(".zip"):
            unzip_file(output_path, output_dir)

def unzip_file(zip_path: str, extract_to: str):
    """Unzip a ZIP archive into a given directory and delete the original ZIP."""
    print(f"Unzipping {zip_path} to {extract_to}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Unzip complete.")

    try:
        os.remove(zip_path)
        print(f"Deleted original ZIP file: {zip_path}")
    except OSError as e:
        print(f"Error deleting {zip_path}: {e}")


if __name__ == "__main__":
    print("Loading census tracts...")
    census_tracts = load_census_tracts()
    print(f"Loaded {len(census_tracts)} census tracts.")

    print("Loading circuit data...")
    circuit_data = load_circuit_data()
    print(f"Loaded {len(circuit_data)} circuit datasets.")
