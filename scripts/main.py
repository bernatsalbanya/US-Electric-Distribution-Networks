import geopandas as gpd
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import argparse

# Ensure Python can find the `src/` module
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from load_data import *
from process_data import *
from visualization import *
from network_metrics import *
from network_analysis import *

# Change the working directory back to the main folder
os.chdir(Path(__file__).resolve().parent.parent)

# Command-line argument parsing
# This allows the script to be run with command-line options 
parser = argparse.ArgumentParser(description="Process and visualize electric circuit data for the US Northeast.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-p", "--preprocess", action="store_true", help="preprocess data")
parser.add_argument("-d", "--download", action="store_true", help="Download data from Dataverse")
args = parser.parse_args()
config = vars(args)

if __name__ == "__main__":
  if args.download:
    print("Downloading datasets...")
    download_all_files_from_dataverse(dataset_url="https://dataverse.harvard.edu/api/datasets/:persistentId/versions/:latest/files?persistentId=doi:10.7910/DVN/XZBDDR", output_dir = "data/census_tracts") # Download Census Tract Data from Dataverse
    download_all_files_from_dataverse(dataset_url="https://dataverse.harvard.edu/api/datasets/:persistentId/versions/:latest/files?persistentId=doi:10.7910/DVN/HSHLLT", output_dir = "data/circuits") # Download Circuit Data from Dataverse
    download_all_files_from_dataverse(dataset_url="https://dataverse.harvard.edu/api/datasets/:persistentId/versions/:latest/files?persistentId=doi:10.7910/DVN/8Z4JRD", output_dir = "data/cleaned") # Download Cleaned Data from Dataverse
    download_all_files_from_dataverse(dataset_url="https://dataverse.harvard.edu/api/datasets/:persistentId/versions/:latest/files?persistentId=doi:10.7910/DVN/PPL4JZ", output_dir = "results/ResultsNortheast") # Download Results Data from Dataverse
    print("Datasets downloaded.")

    REPROCESS_DATA = config["preprocess"]

    if REPROCESS_DATA:
        print("Loading census tracts...")
        census_tracts = load_census_tracts()
        print(f"Loaded {len(census_tracts)} census tracts.")

    ##############################################

        print("Loading circuit data...")
        circuit_data = load_circuit_data()
        print(f"Loaded {len(circuit_data)} circuit datasets.")

    ##############################################

        print("Cleaning Vermont circuit dataset...")
        # Example Vermont bounds (adjust if needed)
        VERMONT_BOUNDS = (-73.415, 42.726933, -71.61, 44.95)  # Approximate Vermont bounds

        # Process Vermont dataset
        lines_VT = circuit_data.get("VT_GMP")
        if lines_VT is not None:
            # Set the CRS if not already set
            if lines_VT.crs is None:
                lines_VT.set_crs("EPSG:4326", inplace=True)
            lines_VT = adjust_geometry(lines_VT, VERMONT_BOUNDS)
            lines_VT = clean_vt_circuit_data(lines_VT, company_name="Green Mountain Power", phase=1)
            lines_VT = calculate_voltage(lines_VT)
            lines_VT_joined = spatial_join_with_census({"lines_VT": lines_VT}, census_tracts, state_name="Vermont")
            print("Vermont circuit datasets cleaned.")
        else:
            print("Vermont circuit datasets not found.")

    ##############################################

        print("Cleaning Rhode Island circuit dataset...")
        # Process Rhode Island dataset
        lines_RI = circuit_data.get("RI")
        if lines_RI is not None:
            # Set the CRS if not already set
            if lines_RI.crs is None:
                lines_RI.set_crs("EPSG:4326", inplace=True)
            lines_RI = clean_ri_circuit_data(lines_RI) # Column rename mapping
            lines_RI = calculate_circuit_rating(lines_RI) # Calculate circuit rating
            lines_RI_joined = spatial_join_with_census({"lines_RI": lines_RI}, census_tracts, state_name="Rhode Island") # Perform spatial join
            print("Rhode Island circuit datasets cleaned.")
        else:
            print("Rhode Island circuit datasets not found.")

    ##############################################

        print("Cleaning New Hampshire circuit datasets...")
        # Process New Hampshire datasets for Liberty Utilities and Eversource
        # Process Liberty Utilities dataset
        lines_NH_Liberty1 = circuit_data.get("NH_Liberty1")
        lines_NH_Liberty3 = circuit_data.get("NH_Liberty3")
        lines_NH_Eversource = circuit_data.get("NH_Eversource")

        # Initialize variables
        lines_NH_Liberty = None
        lines_NH_Liberty_joined = None
        lines_NH_Eversource_joined = None

        if lines_NH_Liberty1 is not None and lines_NH_Liberty3 is not None:
            # Reproject if necessary
            if lines_NH_Liberty3.crs != census_tracts.crs:
                lines_NH_Liberty3 = lines_NH_Liberty3.to_crs(census_tracts.crs)
            if lines_NH_Liberty1.crs != census_tracts.crs:
                lines_NH_Liberty1 = lines_NH_Liberty1.to_crs(census_tracts.crs)
            
            lines_NH_Liberty1 = clean_nh_liberty(lines_NH_Liberty1, phase = 1)
            lines_NH_Liberty3 = clean_nh_liberty(lines_NH_Liberty3, phase = 3)
            lines_NH_Liberty = pd.concat([lines_NH_Liberty1, lines_NH_Liberty3], ignore_index=True)

        if lines_NH_Eversource is not None:
            if lines_NH_Eversource.crs != census_tracts.crs:
                lines_NH_Eversource = lines_NH_Eversource.to_crs(census_tracts.crs)
            lines_NH_Eversource = clean_nh_eversource(lines_NH_Eversource)

        if lines_NH_Liberty is not None:
            lines_NH_Liberty_joined = spatial_join_with_census({"lines_NH_Liberty": lines_NH_Liberty}, census_tracts, "New Hampshire")

        if lines_NH_Eversource is not None:
            lines_NH_Eversource_joined = spatial_join_with_census({"lines_NH_Eversource": lines_NH_Eversource}, census_tracts, "New Hampshire")

        # Save cleaned datasets
        lines_NH_joined = gpd.GeoDataFrame(pd.concat([df for df in [lines_NH_Liberty_joined, lines_NH_Eversource_joined] if df is not None], ignore_index=True), crs=lines_NH_Liberty_joined.crs)


        if not lines_NH_joined.empty:
            print("New Hampshire circuit datasets cleaned.")
        else:
            print("New Hampshire circuit datasets not found.")
        
    ##############################################

        print("Cleaning Maine circuit datasets...")
        # Load Maine datasets
        lines_ME_CMP3 = circuit_data.get("ME_CMP3")
        lines_ME_CMP1 = circuit_data.get("ME_CMP1")

        if lines_ME_CMP3 is not None and lines_ME_CMP1 is not None:
            # Set the CRS if not already set
            lines_ME_CMP3.set_crs('EPSG:4326', inplace = True, allow_override=True)
            lines_ME_CMP1.set_crs('EPSG:4326', inplace = True, allow_override=True)

            # Clean and merge circuit data
            lines_ME = clean_me_cmp(lines_ME_CMP3, lines_ME_CMP1)

            # Perform spatial join
            lines_ME_joined = spatial_join_with_census({"lines_ME": lines_ME}, census_tracts, state_name="Maine")

            # Save cleaned dataset
            print("Maine circuit datasets cleaned.")
        else:
            print("Maine circuit datasets not found.")

    ##############################################

        print("Cleaning Connecticut circuit datasets...")
        # Load Connecticut datasets
        lines_CT_United = circuit_data.get("CT_United")
        lines_CT_Eversource = circuit_data.get("CT_Eversource")

        lines_CT_United_joined = None
        lines_CT_Eversource_joined = None

        if lines_CT_United is not None:
            # Reproject if necessary
            if lines_CT_Eversource.crs != census_tracts.crs:
                lines_CT_Eversource = lines_CT_Eversource.to_crs(census_tracts.crs)
            # Clean circuit data
            lines_CT_United = clean_ct_united(lines_CT_United)
            # Perform spatial join with census tracts    
            lines_CT_United_joined = spatial_join_with_census({"lines_CT_United": lines_CT_United}, census_tracts, state_name="Connecticut") 

        if lines_CT_Eversource is not None:
            # Reproject if necessary
            if lines_CT_Eversource.crs != census_tracts.crs:
                lines_CT_Eversource = lines_CT_Eversource.to_crs(census_tracts.crs)
            # Clean circuit data
            lines_CT_Eversource = clean_ct_eversource(lines_CT_Eversource)
            # Perform spatial join with census tracts
            lines_CT_Eversource_joined = spatial_join_with_census({"lines_CT_Eversource": lines_CT_Eversource}, census_tracts, state_name="Connecticut")

        # Save cleaned datasets
        lines_CT_joined = gpd.GeoDataFrame(pd.concat([df for df in [lines_CT_United_joined, lines_CT_Eversource_joined] if df is not None], ignore_index=True), crs=lines_CT_United_joined.crs)

        if not lines_CT_joined.empty:
            print("Connecticut circuit datasets cleaned.")
        else:
            print("Connecticut circuit datasets not found.")
            
    ##############################################

        print("Cleaning Massachusetts circuit datasets...")
        # Load Massachusetts datasets
        lines_MA_UnitMaOH = circuit_data.get("MA_UnitMaOH")
        lines_MA_UnitMaUG = circuit_data.get("MA_UnitMaUG")
        lines_MA_NatGrid3 = circuit_data.get("MA_NatGrid3")
        lines_MA_WMA = circuit_data.get("MA_WMA")
        lines_MA_EMA = circuit_data.get("MA_EMA")

        lines_MA_Unitil_joined = None
        if lines_MA_UnitMaOH is not None and lines_MA_UnitMaUG is not None:
            # Reproject if necessary
            if lines_MA_UnitMaOH.crs != census_tracts.crs:
                lines_MA_UnitMaOH = lines_MA_UnitMaOH.to_crs(census_tracts.crs)
            if lines_MA_UnitMaUG.crs != census_tracts.crs:
                lines_MA_UnitMaUG = lines_MA_UnitMaUG.to_crs(census_tracts.crs)
            # Clean and merge circuit data
            lines_MA_Unitil = clean_ma_unitil(lines_MA_UnitMaOH, lines_MA_UnitMaUG)
            # Perform spatial join with census tracts
            lines_MA_Unitil_joined = spatial_join_with_census({"lines_MA_Unitil": lines_MA_Unitil}, census_tracts, state_name="Massachusetts")

        lines_MA_NatGrid_joined = None
        if lines_MA_NatGrid3 is not None:
            # Reproject if necessary
            if lines_MA_NatGrid3.crs != census_tracts.crs:
                lines_MA_NatGrid3 = lines_MA_NatGrid3.to_crs(census_tracts.crs)
            # Clean circuit data
            lines_MA_NatGrid = clean_ma_national_grid(lines_MA_NatGrid3)
            # Perform spatial join with census tracts
            lines_MA_NatGrid_joined = spatial_join_with_census({"lines_MA_NatGrid": lines_MA_NatGrid}, census_tracts, state_name="Massachusetts")

        lines_MA_Eversource_joined = None
        if lines_MA_WMA is not None and lines_MA_EMA is not None:
            # Reproject if necessary
            if lines_MA_WMA.crs != census_tracts.crs:
                lines_MA_WMA = lines_MA_WMA.to_crs(census_tracts.crs)
            if lines_MA_EMA.crs != census_tracts.crs:
                lines_MA_EMA = lines_MA_EMA.to_crs(census_tracts.crs)
            # Clean and merge circuit data
            lines_MA_Eversource = clean_ma_eversource(lines_MA_WMA, lines_MA_EMA)
            # Perform spatial join with census tracts
            lines_MA_Eversource_joined = spatial_join_with_census({"lines_MA_Eversource": lines_MA_Eversource}, census_tracts, state_name="Massachusetts")
        
        # Save cleaned datasets
        lines_MA_joined = gpd.GeoDataFrame(pd.concat([df for df in [lines_MA_Unitil_joined, lines_MA_NatGrid_joined, lines_MA_Eversource_joined] if df is not None], ignore_index=True), crs=lines_MA_Unitil_joined.crs)

        if not lines_MA_joined.empty:
        # merge_and_save(lines_MA_joined, "data/cleaned/cleaned_lines_MA.shp")
            print("Massachusetts circuit datasets cleaned.")
        else:
            print("Massachusetts circuit datasets not found.")

    ##############################################

        print("Cleaning New York circuit datasets...")
        # Load Massachusetts datasets
        lines_NY_CenHud12 = circuit_data.get("NY_CenHud12")
        lines_NY_CenHud3 = circuit_data.get("NY_CenHud3")
        lines_NY_ConEd12 = circuit_data.get("NY_ConEd12")
        lines_NY_ConEd3 = circuit_data.get("NY_ConEd3")
        lines_NY_AvGriH = circuit_data.get("NY_AvGriH")
        lines_NY_AvGriNoH = circuit_data.get("NY_AvGriNoH")
        lines_NY_NatGridHC = circuit_data.get("NY_NatGridHC")
        lines_NY_NatGridprimary = circuit_data.get("NY_NatGridprimary")
        lines_NY_ORU1 = circuit_data.get("NY_ORU1")
        lines_NY_ORU3 = circuit_data.get("NY_ORU3")

        lines_NY_CenHud_joined = None
        lines_NY_ConEd_joined = None
        lines_NY_AvGri_joined = None
        lines_NY_NatGrid_joined = None 
        lines_NY_ORU_joined = None

        cen_hud = None
        if lines_NY_CenHud12 is not None and lines_NY_CenHud3 is not None:
            # Set the CRS if not already set
            lines_NY_CenHud12.set_crs('EPSG:4326', inplace = True, allow_override=True)
            lines_NY_CenHud3.set_crs('EPSG:4326', inplace = True, allow_override=True)
            # Merge and Clean CenHud data
            lines_NY_CenHud = clean_ny_cenhudson(lines_NY_CenHud12, lines_NY_CenHud3)
            # Perform spatial join
            lines_NY_CenHud_joined = spatial_join_with_census({"lines_NY_CenHud": lines_NY_CenHud}, census_tracts, state_name="New York")

        con_ed = None
        if lines_NY_ConEd12 is not None and lines_NY_ConEd3 is not None:
            # Set the CRS if not already set
            lines_NY_ConEd12.set_crs('EPSG:4326', inplace = True, allow_override=True)
            lines_NY_ConEd3.set_crs('EPSG:4326', inplace = True, allow_override=True)
            # Merge and Clean ConEd data
            lines_NY_ConEd = clean_ny_conedison(lines_NY_ConEd12, lines_NY_ConEd3)
            # Perform spatial join
            lines_NY_ConEd_joined = spatial_join_with_census({"lines_NY_ConEd": lines_NY_ConEd}, census_tracts, state_name="New York")
        
        avangrid = None
        if lines_NY_AvGriH is not None and lines_NY_AvGriNoH is not None:
            # Set the CRS if not already set
            lines_NY_AvGriH.set_crs('EPSG:4326', inplace = True, allow_override=True)
            lines_NY_AvGriNoH.set_crs('EPSG:4326', inplace = True, allow_override=True)
            # Merge and Clean Avangrid data
            lines_NY_AvGri = clean_ny_avangrid(lines_NY_AvGriH, lines_NY_AvGriNoH)
            # Perform spatial join
            lines_NY_AvGri_joined = spatial_join_with_census({"lines_NY_AvGri": lines_NY_AvGri}, census_tracts, state_name="New York")
        
        natgrid = None
        if lines_NY_NatGridprimary is not None and lines_NY_NatGridHC is not None:
            # Reproject if necessary
            if lines_NY_NatGridprimary.crs != census_tracts.crs:
                lines_NY_NatGridprimary = lines_NY_NatGridprimary.to_crs(census_tracts.crs)

            # Reproject if necessary
            if lines_NY_NatGridHC.crs != census_tracts.crs:
                lines_NY_NatGridHC = lines_NY_NatGridHC.to_crs(census_tracts.crs)
            # Merge and Clean National Grid data
            lines_NY_NatGrid = clean_ny_national_grid(lines_NY_NatGridprimary, lines_NY_NatGridHC)
            # Perform spatial join
            lines_NY_NatGrid_joined = spatial_join_with_census({"lines_NY_NatGrid": lines_NY_NatGrid}, census_tracts, state_name="New York")

        oru = None
        if lines_NY_ORU1 is not None and lines_NY_ORU3 is not None:
            # Set the CRS if not already set
            lines_NY_ORU1.set_crs('EPSG:4326', inplace = True, allow_override=True)
            lines_NY_ORU3.set_crs('EPSG:4326', inplace = True, allow_override=True)
            # Merge and Clean CenHud data
            lines_NY_ORU = clean_ny_oru(lines_NY_ORU1, lines_NY_ORU3)
            # Perform spatial join
            lines_NY_ORU_joined = spatial_join_with_census({"lines_NY_ORU": lines_NY_ORU}, census_tracts, state_name="New York")

        # Save cleaned datasets
        lines_NY_joined = gpd.GeoDataFrame(pd.concat([df for df in [lines_NY_CenHud_joined, lines_NY_ConEd_joined, lines_NY_AvGri_joined, lines_NY_NatGrid_joined, lines_NY_ORU_joined] if df is not None], ignore_index=True), crs=lines_NY_CenHud_joined.crs)

        if not lines_NY_joined.empty:
            print("New York circuit datasets cleaned.")
        else:
            print("New York circuit datasets not found.")

    ##############################################

        print("Merging US Northeast circuits datasets...")
        # List of all datasets
        datasets = [lines_CT_joined, lines_MA_joined, lines_ME_joined, lines_NH_joined, lines_NY_joined, lines_RI_joined, lines_VT_joined]

        # Check for None or empty
        datasets = [gdf for gdf in datasets if isinstance(gdf, gpd.GeoDataFrame) and not gdf.empty]

        # Merge all GeoDataFrames
        merged_lines = gpd.GeoDataFrame(pd.concat(datasets, ignore_index=True), crs=datasets[0].crs)

        if merged_lines is not None:
            save_cleaned_data(merged_lines, "data/cleaned/US_Northeast_Electric_Network.shp")
        else:
            print("US Northeast circuit datasets not found.")

##############################################

        print("Saving data visualization...")

        # Define datasets (replace these placeholders with your actual datasets)
        datasets = {
            'lines_NY_CenHud': lines_NY_CenHud_joined,
            'lines_NY_ConEd': lines_NY_ConEd_joined,
            'lines_NY_AvGri': lines_NY_AvGri_joined,
            'lines_NY_NatGrid': lines_NY_NatGrid_joined,
            'lines_NY_ORU': lines_NY_ORU_joined,
            'lines_ME_CMP': lines_ME_joined,
            'lines_NH_Liberty': lines_NH_Liberty_joined,
            'lines_NH_Eversource': lines_NH_Eversource_joined,
            'lines_MA_Unitil': lines_MA_Unitil_joined,
            'lines_MA_NatGrid': lines_MA_NatGrid_joined,
            'lines_MA_Eversource': lines_MA_Eversource_joined,
            'lines_RI': lines_RI_joined,
            'lines_CT_United': lines_CT_United_joined,
            'lines_CT_Eversource': lines_CT_Eversource_joined,
            'lines_VT': lines_VT_joined
        }

        print("Visualizing hosting capacity histograms...")
        plot_hosting_capacity_histograms(datasets, output_dir="data/plots")

        print("Visualizing voltage distribution...")
        plot_voltage_distribution(datasets, output_dir="data/plots")

        print("Visualizing circuit rating distribution...")
        plot_circuit_rating_distribution(datasets, output_dir="data/plots")

        print("Visualizing separated utility lines...")
        plot_utility_lines(datasets, output_dir="data/plots")

        print("Visualizing merged utility lines...")
        plot_merged_utility_lines(merged_lines, output_dir="data/plots")

        print("Data processing and visualization complete.")

    else:
        print("Skipping data cleaning. Loading existing cleaned dataset...")
        merged_lines = gpd.read_file("data/cleaned/US_Northeast_Electric_Network.shp")

##############################################
    # Calculate the Topological Metrics by Census Tract 
    print("Calculating Topological Network Metrics...")
    
    sub_dfs, unique_geoids = split_by_geoid(merged_lines)

    # Modify how you pass data in the parallel loop
    Parallel(n_jobs=-1)(delayed(process_one_geoid)(geoid, sub_dfs[geoid], output_dir="results/ResultsNortheast/") for geoid in unique_geoids if geoid in sub_dfs)

    print("Topological Network Metrics calculated.")

##############################################
    # Perform the Goodness of Fit Analysis
    print("Performing Goodness of Fit Analysis...")
    # Step 1: Reconstruct global_metrics_df
    node_metrics_df, global_metrics_df = combine_metrics(unique_geoids, "results/ResultsNortheast/")

    # Step 2: Add geoid prefix info
    global_metrics_df = enrich_geoid_info(global_metrics_df)

    # Step 3: Fit distributions
    results_df, summary_df = summarize_best_fit_distributions(global_metrics_df, output_csv="results/goodfit_summary.csv")

    # Step 4: Plot distributions
    plot_distributions(results_df, global_metrics_df, output_path="results/goodfit_plots.png")

    print("Goodness of Fit Analyisis Performed.")
