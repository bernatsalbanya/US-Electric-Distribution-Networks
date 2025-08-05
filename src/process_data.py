import geopandas as gpd
import pandas as pd
import numpy as np
import os

def adjust_geometry(lines, state_bounds):
    """
    Scale and translate a GeoDataFrame to fit within the specified Vermont bounds.

    Args:
        gdf (GeoDataFrame): The input GeoDataFrame with geometries to be adjusted.
        vermont_bounds (tuple): The desired Vermont bounds (minx, miny, maxx, maxy).

    Returns:
        GeoDataFrame: The adjusted GeoDataFrame.
    """
    from shapely.affinity import scale, translate

    if not isinstance(lines, gpd.GeoDataFrame):
        raise TypeError("Input must be a GeoDataFrame.")

    # Ensure CRS consistency
    if lines.crs != "EPSG:4326":
        lines = lines.to_crs("EPSG:4326")
        
    # Get original bounds
    current_bounds = lines.total_bounds
    minx, miny, maxx, maxy = current_bounds

    # Extract Vermont bounds
    vermont_minx, vermont_miny, vermont_maxx, vermont_maxy = state_bounds

    # Compute scaling factors
    current_width = maxx - minx
    current_height = maxy - miny
    new_width = vermont_maxx - vermont_minx
    new_height = vermont_maxy - vermont_miny

    scale_x = new_width / current_width
    scale_y = new_height / current_height

    # Apply scaling to geometries
    lines["geometry"] = lines["geometry"].apply(lambda geom: scale(geom, xfact=scale_x, yfact=scale_y, origin=(minx, miny)))

    # Recalculate bounds after scaling
    scaled_bounds = lines.total_bounds
    scaled_minx, scaled_miny, _, _ = scaled_bounds

    # Compute translation shift
    shift_x = vermont_minx - scaled_minx
    shift_y = vermont_miny - scaled_miny

    # Apply translation
    lines["geometry"] = lines["geometry"].apply(lambda geom: translate(geom, xoff=shift_x, yoff=shift_y))

    return lines

def amps_to_voltage(mw, amps, power_factor=0.95, phase=1):
    """
    Calculate Voltage (kV) from Power (MW) and Current (A).
    parameters:
        mw (float): Power in MW.
        amps (float): Current in A.
        power_factor (float): Power factor for conversion. Default is 0.95.
        phase (int): Phase type (1 for single-phase, 3 for three-phase).
    """
    if phase == 1:
        return (mw * 1e6) / (amps * power_factor) * 1e-3
    elif phase == 3:
        return (mw * 1e6) / (np.sqrt(3) * amps * power_factor) * 1e-3
    else:
        raise ValueError("Unsupported phase configuration")

def calculate_voltage(lines):
    """
    Estimate voltage levels based on Hosting Capacity and Circuit Rating.

    Args:
        lines (GeoDataFrame): Circuit dataset with Hosting Capacity and Circuit Rating.

    Returns:
        GeoDataFrame: Dataset with estimated Voltage (kV).
    """

    # Calculate voltage
    lines["Voltage_kV"] = lines.apply(
        lambda row: amps_to_voltage(row["HostCap_MW"], row["CircRat_A"], phase=row["Phase"]),
        axis=1,
    )

    # Adjust Voltage Scale
    lines["Voltage_kV"] /= 10

    # Define voltage bins and round to nearest value
    voltage_bins = [0.48, 4.00, 4.16, 4.80, 7.20, 8.20, 12.00, 12.47, 13.00, 13.20, 13.80, 22.80, 23.00, 27.00, 34.50, 38.00]
    lines["Voltage_kV"] = lines["Voltage_kV"].apply(lambda x: min(voltage_bins, key=lambda v: abs(v - x)))

    return lines

def clean_voltage_column(lines):
    """
    Clean and standardize the voltage column in the dataset.
    
    Parameters:
        df (DataFrame): The dataset containing the voltage column.
        voltage_column (str): Name of the column containing voltage values.
    
    Returns:
        DataFrame: Updated dataset with standardized voltage values.
    """
    voltage_mapping = {
        '<Null>': '0.24 kV',
        '': '0.24 kV',
        '2.4KV': '2.4 kV',
        '240/120 Volts': '0.24 kV',
        '120/240 Volts': '0.24 kV',
        '0 kV': '0.24 kV',
        '0' : '0.24'
    }
    
    lines['Voltage_kV'] = lines['Voltage_kV'].replace(voltage_mapping)
    lines['Voltage_kV'] = pd.to_numeric(lines['Voltage_kV'].replace(' kV', '', regex=True), errors='coerce')
    lines['Voltage_kV'] = lines['Voltage_kV'].fillna(0.24)
            
    return lines

def map_phase_values(lines, phase_column = 'Phase'):
    """
    Map phase values to standardized numeric representations.
    
    Parameters:
        df (DataFrame): The dataset containing the phase column.
        phase_column (str): Name of the column containing phase values.
    
    Returns:
        DataFrame: Updated dataset with standardized phase values.
    """
    phase_mapping = {
        # 1-Phase
        'Phase 1 and 2': 1, 'Phase 1': 1, 'Phase 2': 2, 'Phase 3': 3,
        'A': 1, 'B': 1, 'C': 1,
        # 2-Phase
        'AB': 2, 'BC': 2, 'AC': 2, 'BN': 2, 'AN': 2, 'CN': 2, '[CN]': 2,
        '[AN]': 2, '[BN]': 2, 'CA': 2, '[BN][CN]': 2, '[AN][BN][CN]': 2,
        '[AB]': 2, '[BC]': 2, '[AC]': 2, '[AN][BN]': 2, '[AN][CN]': 2,
        # 3-Phase
        'ABC': 3, 'ACN': 3, 'ABN': 3, 'ABCN': 3, 'BCN': 3, '[ABCN]': 3,
        '[ABC]': 3, '[ACN]': 3, '[BCN]': 3, '[ABC]N': 3, 'CAB': 3,
        'BAC': 3, 'CBA': 3, 'BCA': 3, 'ACB': 3, 'NCBA': 3, '[ABCN]': 3,
        'BCN': 3, '[BCN]': 3, '[ABC]N': 3, '[ABC]': 3, '[ABN]': 3,
        '[ACN]': 3, '[BC]N': 3, '[ABCN]': 3, 2.: 2, 3.: 3, 1.: 1
    }
    pd.set_option('future.no_silent_downcasting', True)
    if phase_column in lines.columns:
        lines[phase_column] = lines[phase_column].replace(phase_mapping)
        lines[phase_column] = pd.to_numeric(lines[phase_column], errors='coerce').fillna(0).astype(int)
    
    return lines

def spatial_join_with_census(datasets, census_tracts, state_name):
    """
    Perform spatial join between circuit datasets and census tracts.

    Args:
        datasets (dict of GeoDataFrames): Circuit datasets.
        census_tracts (GeoDataFrame): Census tract dataset.
        state_name (str): Name of the state to assign.

    Returns:
        GeoDataFrame: Dataset joined with census tract information.
    """

    joined_dfs = []

    for name, gdf in datasets.items():
        # Fix CRS check on the actual GeoDataFrame
        if gdf.crs != census_tracts.crs:
            gdf = gdf.to_crs(census_tracts.crs)

        # Correctly join the individual gdf, not the whole dictionary
        joined_gdf = gpd.sjoin(gdf, census_tracts, how="left", predicate="intersects")

        # Optional: track source dataset
        joined_gdf["State"] = state_name
        joined_dfs.append(joined_gdf)

    # Return a single combined GeoDataFrame
    return gpd.GeoDataFrame(pd.concat(joined_dfs, ignore_index=True), crs=census_tracts.crs)

def save_cleaned_data(lines, output_path):
    """
    Save cleaned dataset to file.

    Args:
        lines (GeoDataFrame): Cleaned dataset.
        output_path (str): File path to save the cleaned data.
    """
    lines.columns = [col[:10] for col in lines.columns]
    lines["ALAND"] = lines["ALAND"].astype(str)
    lines["AWATER"] = lines["AWATER"].astype(str)
    lines.to_file(output_path)
    print(f"Cleaned data saved to {output_path}")

def calculate_circuit_rating(lines):
    """
    Calculate Circuit Rating (A) from Hosting Capacity (MW) and Voltage (kV).

    Args:
        lines (GeoDataFrame): Circuit dataset.

    Returns:
        GeoDataFrame: Updated dataset with Circuit Rating (A).
    """
    if "HostCap_MW" not in lines.columns or "Voltage_kV" not in lines.columns:
        raise KeyError("Dataset must contain 'HostCap_MW' and 'Voltage_kV' columns.")

    # Calculate circuit rating using the `mw_to_amps` function
    lines["CircRat_A"] = lines.apply(
        lambda row: mw_to_amps(row["HostCap_MW"], row["Voltage_kV"], phase=row["Phase"]),
        axis=1
    )

    # Cap the values at the 99th percentile to prevent extreme outliers
    circuit_rating_cap = lines["CircRat_A"].quantile(0.99)
    lines["CircRat_A"] = lines["CircRat_A"].clip(upper=circuit_rating_cap)

    return lines

def mw_to_amps(mw, voltage_kv, power_factor=0.95, phase=None):
    """
    Convert Hosting Capacity (MW) to Circuit Rating (A) based on phase.

    Args:
        mw (float): Power in MW.
        voltage_kv (float): Voltage in kV.
        power_factor (float, optional): Power factor for conversion. Default is 0.95.
        phase (int, optional): Phase type (1 for single-phase, 3 for three-phase).

    Returns:
        float: Circuit Rating (A).
    """
    if phase == 1:  # Single-phase
        return (mw * 1e6) / (voltage_kv * 1e3 * power_factor)
    elif phase == 3:  # Three-phase
        return (mw * 1e6) / (np.sqrt(3) * voltage_kv * 1e3 * power_factor)
    else:
        raise ValueError("Unsupported phase configuration")

def kva_to_mw(kva, power_factor=0.95):
    """Convert Apparent Power (kVA) to Active Power (MW)"""
    return kva * power_factor * 1e-3

def kva_to_amps(kva, voltage_kv, phase):
    """Convert Apparent Power (kVA) to Current (A) based on phase"""
    if phase == 1:  # Single-phase
        return (kva * 1e3) / (voltage_kv * 1e3)
    elif phase == 2:  # Two-phase
        return (kva * 1e3) / (2 * voltage_kv * 1e3)
    elif phase == 3:  # Three-phase
        return (kva * 1e3) / (np.sqrt(3) * voltage_kv * 1e3)
    else:
        raise ValueError("Unsupported phase configuration")

def merge_and_save(datasets, output_path):
    """
    Merge multiple datasets and save as a single cleaned file.

    Args:
        datasets (list of GeoDataFrames): List of datasets to merge.
        output_path (str): File path to save the cleaned data.
    """
    
    # Merge datasets
    merged_lines = gpd.GeoDataFrame(pd.concat(datasets, ignore_index=True, join="outer"))

    # Save to file
    merged_lines.columns = [col[:10] for col in merged_lines.columns]
    merged_lines.to_file(output_path)
    print(f"Cleaned data saved to {output_path}")

def clean_circuit_data(lines, company_name, rename_map=None, drop_columns=None, phase=None, circuit_type=None):
    """
    General function to clean and process a circuit dataset.

    Args:
        lines (GeoDataFrame): Circuit dataset.
        company_name (str): Name of the company operating the circuits.
        rename_map (dict, optional): Mapping of column names for standardization.
        drop_columns (list, optional): List of columns to drop.
        phase (str, optional): Phase type to assign (if applicable).
        circuit_type (str, optional): Overhead or Underground circuit classification.

    Returns:
        GeoDataFrame: Cleaned dataset.
    """
    if not isinstance(lines, gpd.GeoDataFrame):
        raise TypeError("Input must be a GeoDataFrame.")

    # Assign company, phase, and type
    lines["Company"] = company_name
    if phase is not None:
        lines["Phase"] = phase
    if circuit_type is not None:
        lines["Type"] = circuit_type

    # Drop unnecessary columns
    if drop_columns:
        lines.drop(columns=drop_columns, inplace=True, errors="ignore")

    # Rename columns for clarity
    if rename_map:
        lines.rename(columns=rename_map, inplace=True)

    # Create Hosting Capacity column if it doesn't exist
    if "HostCap_MW" not in lines.columns and "Maximum Hosting Capacity (MW)" in lines.columns:
        lines["HostCap_MW"] = lines["Maximum Hosting Capacity (MW)"]

    return lines

def clean_ct_eversource(lines):
    """
    Clean and process Connecticut Eversource dataset.

    Args:
        lines_CT_Eversource (GeoDataFrame): Connecticut Eversource dataset.

    Returns:
        GeoDataFrame: Cleaned dataset.
    """

    rename_columns = {
        'PHASE': 'Phase',
        'HOSTING_CA': 'HostCap_MW',
        'OPERATING_': 'Voltage_kV',
        'CIRCUIT_RA': 'CircRat_A',
        'SUBSTATION': 'Substation',
    }

    drop_columns = [
        'OBJECTID_1', 'ID', 'SECTION_OP', 'CIRCUIT_NA', 'DIST_SUB_N', 
        'DATE_UPDAT', 'GIS_SECTIO', 'BULK_CIRCU', 'BULK_SUB_N', 'BULK_SUB_H', 'BULK_SUB_L',
        'BULK_SUB_1', 'ASO_STUDIE', 'ONLINE_DG_', 'IN_QUEUE_D', 'BULK_SUB_R', 'DIST_SUB_R',
        'FEEDS_SEC_', 'SCHEME_3VO', 'CIP_FEES', 'DIST_SUB_H', 'DIST_SUB_L',
    ]

    lines = clean_circuit_data(
        lines, company_name="Eversource", 
        rename_map=rename_columns, drop_columns=drop_columns)
    
    return lines

def clean_ct_united(lines):
    """
    Clean and process Connecticut United Illuminating dataset.

    Args:
        lines_CT_United (GeoDataFrame): Connecticut United Illuminating dataset.

    Returns:
        GeoDataFrame: Cleaned dataset.
    """

    rename_columns = {
        'HostingCap': 'HostCap_MW',
        'Operating_': 'Voltage_kV',
        'Phase': 'Phase',
        'Rating__A_': 'CircRat_A',
        'Substation': 'Substation',
    }

    drop_columns = [
        'Available_', 'Circuit', 'Circuit_1', 'DER_in_Que', 'DER_in_Q_1', 'DER_in_Q_2', 'DER_in_Q_3',
        'Existing_G', 'Existing_1', 'Existing_2', 'Existing_3', 'FolderPath', 'LAST_AltMo', 'LAST_Base',
        'LAST_Clamp', 'LAST_Extru', 'LAST_Popup', 'LAST_Symbo', 'Name', 'OBJECTID', 'Peak_Load_',
        'Peak_Loa_1', 'Peak_Loa_2', 'Remaining_', 'Remainin_1', 'Shape__Len', 'Substati_1', 'Substati_2',
        'SymbolRang', 'SegementID', 'Rating__MV','Rating__MW', 
    ]

    lines = clean_circuit_data(
        lines, company_name="United Illuminating (Avangrid)", 
        rename_map=rename_columns, drop_columns=drop_columns)

    # Convert Hosting Capacity to MW
    lines["HostCap_MW"] /= 1000

    return lines

def clean_ma_national_grid(lines):
    """
    Clean National Grid dataset and standardize column names.

    Args:
        lines_MA_NatGrid3 (GeoDataFrame): National Grid dataset.

    Returns:
        GeoDataFrame: Cleaned dataset.
    """

    rename_columns = {
        'HC_MAX': 'HostCap_MW',
        'Feeder_CDF': 'Feeder',
        'Normal_Rat': 'CircRat_A'
    }

    drop_columns = [
        'OBJECTID', 'Master_CDF', 'Sub_Feeder', 'XFMR____al', 'Feeder_Ope', 'F3V0_Appli', 'Emergency_',
        'F2018_Peak', 'Min_Load__', 'BUS_CONFIG', 'NC_NO', 'CDST', 'B', 'GIS_Match2', 'Column1', 'f_c', 'f_p',
        'b_c', 'b_p', 'Pending_1M', 'Ongoing_AS', 'Potential_', 'DG_Penetra', 'MAP_Color', 'Day_Update',
        'HC_AVAIL_I', 'FERC_FEEDE', 'feeder_con', 'feeder_c_1', 'feeder_c_2', 'feeder_c_3', 'feeder_pen',
        'feeder_p_1', 'feeder_p_2', 'bank_conne', 'bank_con_1', 'bank_con_2', 'bank_con_3', 'bank_pendi',
        'bank_pen_1', 'bank_pen_2', 'Shape_Leng', 'HC_MIN', 'HC_INCLUDI',
    ]

    lines = clean_circuit_data(lines, company_name="National Grid", rename_map=rename_columns, drop_columns=drop_columns, phase=3)

    return lines

def clean_ma_eversource(lines_MA_WMA, lines_MA_EMA):
    """
    Clean and merge Eversource datasets for WMA and EMA regions.

    Args:
        lines_MA_WMA (GeoDataFrame): Western Massachusetts dataset.
        lines_MA_EMA (GeoDataFrame): Eastern Massachusetts dataset.

    Returns:
        GeoDataFrame: Merged dataset.
    """

    rename_columns = {
        'HOSTING_CA': 'HostCap_MW',
        'OPERATING_': 'Voltage_kV',
        'PHASE': 'Phase',
        'BULK_SUB_N': 'Substation',
        'CIRCUIT_RA': 'CircRat_A'
    }

    drop_columns = [
        'OBJECTID', 'SECTION_OP', 'CIRCUIT_NA', 'DIST_SUB_N', 'SUBSTATION', 'DIST_SUB_H',
        'DIST_SUB_L', 'DATE_UPDAT', 'GIS_SECTIO', 'BULK_CIRCU', 'BULK_SUB_H', 'BULK_SUB_L',
        'BULK_SUB_1', 'CIP_FEES', 'ASO_STUDIE', 'ONLINE_DG_', 'IN_QUEUE_D', 'BULK_SUB_R',
        'DIST_SUB_R', 'FEEDS_SEC_', 'SCHEME_3VO', 'SUBSTATI_1', 'PHASEDESIG', 'ESRI_OID',
    ]

    # Clean both datasets separately
    lines_MA_WMA = clean_circuit_data(lines_MA_WMA, company_name="Eversource", rename_map=rename_columns, drop_columns=drop_columns)
    lines_MA_EMA = clean_circuit_data(lines_MA_EMA, company_name="Eversource", rename_map=rename_columns, drop_columns=drop_columns)

    # Merge WMA and EMA datasets
    lines_MA_Eversource = pd.concat([lines_MA_WMA, lines_MA_EMA], ignore_index=True)

    # Map Phases
    lines_MA_Eversource = map_phase_values(lines_MA_Eversource)

    return lines_MA_Eversource

def clean_ma_unitil(lines_MA_UnitMaOH, lines_MA_UnitMaUG):
    """
    Clean and merge Unitil datasets for Overhead and Underground conductors.

    Args:
        lines_MA_UnitMaOH (GeoDataFrame): Overhead conductor dataset.
        lines_MA_UnitMaUG (GeoDataFrame): Underground conductor dataset.

    Returns:
        GeoDataFrame: Merged dataset.
    """

    rename_columns = {
        'PhaseDesig': 'HostCap_MW',
        'FeederID': 'Feeder',
        'OperatingV': 'Voltage_kV',
    }

    drop_columns = [
        'OBJECTID', 'Enabled', 'CreationUs', 'DateCreate', 'DateModifi',
        'LastUser', 'SubtypeCod', 'FeederID2', 'NeutralMat', 'NeutralSiz',
        'ElectricTr', 'FeederInfo', 'ConductorC', 'MainLineTi', 'DesignID',
        'WorkLocati', 'WorkFlowSt', 'WorkFuncti', 'Favorite', 'ConductorT',
        'CustomerOw', 'Shape.len', 'MeasuredLe', 'LengthSour', 'NominalVol', 
        'Comments', 'FdrMgrNonT', 'LabelText', 'PhaseOrien', 'WorkReques', 'InConduitI',
    ]

    # Clean both datasets separately
    lines_MA_UnitMaOH = clean_circuit_data(
        lines_MA_UnitMaOH, company_name="Unitil", rename_map=rename_columns, drop_columns=drop_columns, circuit_type="Overhead"
    )
    lines_MA_UnitMaUG = clean_circuit_data(
        lines_MA_UnitMaUG, company_name="Unitil", rename_map=rename_columns, drop_columns=drop_columns, circuit_type="Underground"
    )

    # Merge overhead and underground conductors
    lines_MA_Unitil = pd.concat([lines_MA_UnitMaOH, lines_MA_UnitMaUG], ignore_index=True)
    
    lines_MA_Unitil['Phase'] = 3
    lines_MA_Unitil['Voltage_kV'] = lines_MA_Unitil['Voltage_kV'].replace(0, 0.24)
    lines_MA_Unitil['Voltage_kV'] = lines_MA_Unitil['Voltage_kV']/10
    
    # Calculate Circuit Rating (A) from Hosting Capacity (MW) and Voltage (kV) considering phase
    lines_MA_Unitil['CircRat_A'] = lines_MA_Unitil.apply(
        lambda row: mw_to_amps(row['HostCap_MW'], row['Voltage_kV'], phase=row['Phase']),
        axis=1
    )
    
    # Calculate the 99th percentile of 'Circuit Rating'
    circuit_rating_cap = lines_MA_Unitil['CircRat_A'].quantile(0.99)

    # Cap the values at the 99th percentile
    lines_MA_Unitil['CircRat_A'] = lines_MA_Unitil['CircRat_A'].clip(upper=circuit_rating_cap)
      
   # lines_MA_Unitil['HostCap_MW'] = lines_MA_Unitil['Voltage_kV'] * lines_MA_Unitil['CircRat_A'] * np.sqrt(3) * 0.95 / 1000

    return lines_MA_Unitil

def clean_me_cmp(lines_ME_CMP3, lines_ME_CMP1):
    """
    Merge and clean circuit datasets for Maine (CMP Phase 3 and CMP Phase 1 & 2).

    Args:
        lines_ME_CMP3 (GeoDataFrame): CMP Phase 3 dataset.
        lines_ME_CMP1 (GeoDataFrame): CMP Phase 1 & 2 dataset.

    Returns:
        GeoDataFrame: Merged and cleaned dataset.
    """

    rename_columns = {
        'Operating_': 'Voltage_kV',
        'HostingCap': 'HostCap_MW',
        'Circuit_Ra': 'CircRat_A',
        'CYME_Trans': 'Substation',
    }

    drop_columns = [
        'OBJECTID', 'Name', 'SymbolID', 'AltMode', 'Base', 'Clamped', 'Extruded', 'Shape__Len',
        'Circuit__J', 'Circuit_1', 'Division', 'Substation', 'Circuit',
        'Substati_1', 'Substati_2', 'Substati_3', 'Substati_4', 'Substati_5',
        'Circuit_Mi', 'Circuit_Al', 'Circuit_Le', 'SegementID', 'Remaining_', 'SymbolRang'
    ]

    # Clean both datasets separately
    lines_ME_CMP3 = clean_circuit_data(
        lines_ME_CMP3, company_name="Avangrid (CMP)", rename_map=rename_columns, drop_columns=drop_columns, phase="Phase 3"
    )
    lines_ME_CMP1 = clean_circuit_data(
        lines_ME_CMP1, company_name="Avangrid (CMP)", rename_map=rename_columns, drop_columns=drop_columns, phase="Phase 1 and 2"
    )

    # Merge both datasets
    lines_ME_CMP = gpd.GeoDataFrame(pd.concat([lines_ME_CMP3, lines_ME_CMP1], ignore_index=True))

    # Drop extra columns
    lines_ME_CMP.drop(columns=['Symbol Range'], inplace=True, errors='ignore')

    # Fill missing values within the same substation group
    for col in ['Remaining Load Capacity (MW)', 'Hosting Capacity Number (Temporary)']:
        if col in lines_ME_CMP.columns:
            lines_ME_CMP[col] = lines_ME_CMP.groupby('Substation')[col].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))

    # Convert numerical columns for consistency
    lines_ME_CMP['HostCap_MW'] = lines_ME_CMP['HostCap_MW'] / 1000  # Convert to MW
    lines_ME_CMP['CircRat_A'] = lines_ME_CMP['CircRat_A'] * 100  # Convert to kVA

    # Cap circuit rating at 1500 kVA
    lines_ME_CMP['CircRat_A'] = lines_ME_CMP['CircRat_A'].clip(upper=1500)

    return lines_ME_CMP

def clean_nh_liberty(lines, phase = None):
    """
    Process Liberty Utilities circuit data, clean datasets, and merge them.
    
    Parameters:
        lines_NH_Liberty1 (DataFrame): Liberty phase 1 dataset.
        lines_NH_Liberty3 (DataFrame): Liberty phase 3 dataset.
    
    Returns:
        DataFrame: Merged and cleaned Liberty dataset.
    """
    rename_columns = {
        "Load_Max_M": "HostCap_MW",
        "FeederID": "Feeder",
        "Substation": "Substation",
        "Nominal_Vo": "Voltage_kV",
    }

    drop_columns = [
        "SourceID", "Operating", "Gen_Max_MW",'OBJECTID', 
        'OH_UG_Merg', 'OH_UG_Me_1', 'OH_UG_Me_2', 'OH_UG_Me_3',
       'OH_UG_Me_4', 'OH_UG_Me_5', 'OH_UG_Me_6', 'OH_UG_Me_7', 'OH_UG_Me_8',
       'OH_UG_Me_9', 'OH_UG_Me10', 'Gen_Limite', 'Load_Limit', 'Shape__Len',
       'Gen_Max__1', 'Gen_Max__2', 'Gen_Max__3', 'Gen_Max__4', 'Load_Max_1',
       'Load_Max_2', 'Load_Max_3', 'Last_Updat', 'Analysis_Y', 'SectionID',
       'Feeder', 'Phase_Coun', 'NameDesc', 'PlanArea', 'Feeder_Que', 'Feeder_Act', 
        'ObjectID_1', 'Source_ID', 'Type', 'SourceID_T', 'Type_1', 'Source_Typ', 
    ]
     
    # Map Phases
    lines = map_phase_values(lines)
    
    # Clean Liberty datasets separately
    lines = clean_circuit_data(
        lines, company_name="Liberty Utilities", 
        rename_map=rename_columns, drop_columns=drop_columns, phase=phase
    )
    
    lines = clean_voltage_column(lines)
    lines['Voltage_kV'] = lines['Voltage_kV'].replace(0, 0.24)
        
    # Calculate Circuit Rating (A) from Hosting Capacity (MW) and Voltage (kV) considering phase
    if "CircRat_A" not in lines.columns:
        lines['CircRat_A'] = lines.apply(
            lambda row: mw_to_amps(row['HostCap_MW'], row['Voltage_kV'], phase=row['Phase']), axis=1)
            
        # Calculate the 75th percentile of 'Circuit Rating'
        circuit_rating_cap = lines['CircRat_A'].quantile(0.987)

        # Cap the values at the 75th percentile
        lines['CircRat_A'] = lines['CircRat_A'].clip(upper=circuit_rating_cap)
        
    return lines

def clean_nh_eversource(lines):
    """
    Process Eversource circuit data, clean dataset.
    
    Parameters:
        lines_NH_Eversource (DataFrame): Eversource dataset.
    
    Returns:
        DataFrame: Cleaned Eversource dataset.
    """
    rename_columns = {
        'PHASE': 'Phase',
        'HOSTING_CA': 'HostCap_MW',
        'OPERATING_': 'Voltage_kV',
        'CIRCUIT_RA': 'CircRat_A',
    }

    drop_columns = [
        'OBJECTID_1', 'ID', 'SECTION_OP', 'CIRCUIT_NA', 'DIST_SUB_N',
        'SUBSTATION', 'DATE_UPDAT', 'GIS_SECTIO', 'BULK_CIRCU', 'BULK_SUB_N', 
        'BULK_SUB_H', 'BULK_SUB_L', 'BULK_SUB_1', 'ASO_STUDIE', 'ONLINE_DG_',
        'IN_QUEUE_D', 'BULK_SUB_R', 'DIST_SUB_R', 'FEEDS_SEC_', 'SCHEME_3VO',
        'DIST_SUB_H', 'DIST_SUB_L'
    ]

    lines = clean_circuit_data(
        lines, company_name="Eversource", 
        rename_map=rename_columns, drop_columns=drop_columns
    )
    
    # Map Phases
    lines = map_phase_values(lines)
    
    # Calculate Circuit Rating (A) from Hosting Capacity (MW) and Voltage (kV) considering phase
    if "CircRat_A" not in lines.columns:
        lines['CircRat_A'] = lines.apply(
            lambda row: mw_to_amps(row['HostCap_MW'], row['Voltage_kV'], phase=row['Phase']), axis=1)
    
    return lines

def clean_ny_cenhudson(cen_hud12, cen_hud3):
    """
    Clean and merge Central Hudson data.
    parameters:
        cen_hud12 (GeoDataFrame): Central Hudson Phase 1 & 2 dataset
        cen_hud3 (GeoDataFrame): Central Hudson Phase 3 dataset
    """
    cen_hud12 = clean_circuit_data(
        cen_hud12,
        company_name="Central Hudson",
        phase=1,
        rename_map={
            'Voltage_kV': 'Voltage_kV',
            'Winter_Hea': 'HostCap_MW',
            'CircuitRat': 'CircRat_A',
        },
        drop_columns=['OBJECTID_1', 'Name', 'FolderPath', 'SymbolID', 'AltMode', 'Base',
                   'Clamped', 'Extruded', 'PopupInfo', 'OBJECTID', 'Name_1', 'Summer_Hea', 
                     'WinterPeak', 'date', 'Shape_Leng', 'SubBank_Ca', 'SummerPeak'],
    )

    cen_hud3 = clean_circuit_data(
        cen_hud3,
        company_name="Central Hudson",
        phase=3,
        rename_map={
            'Voltage_kV': 'Voltage_kV',
            'Winter_Hea': 'HostCap_MW',
            'CircuitRat': 'CircRat_A',
        },
        drop_columns=['OBJECTID_1', 'Name', 'FolderPath', 'SymbolID', 'AltMode', 'Base',
                   'Clamped', 'Extruded', 'PopupInfo', 'OBJECTID', 'Name_1', 'Summer_Hea', 
                     'WinterPeak', 'date', 'Shape_Leng', 'SubBank_Ca', 'SummerPeak'],
    )

    # Merge datasets
    cen_hud = pd.concat([cen_hud12, cen_hud3], ignore_index=True)

    # Convert Circuit Rating to amps and cap at 1500 A
    cen_hud["CircRat_A"] *= 100
    cen_hud["CircRat_A"] = cen_hud["CircRat_A"].clip(upper=1500)

    return cen_hud

def clean_ny_conedison(con_ed12, con_ed3):
    """
    Clean and merge Con Edison data.

    parameters:
        con_ed12 (GeoDataFrame): Con Edison Phase 1 & 2 dataset
        con_ed3 (GeoDataFrame): Con Edison Phase 3 dataset
    returns:
        GeoDataFrame: Merged and cleaned Con Edison dataset.
    """

    con_ed12 = clean_circuit_data(
        con_ed12,
        company_name="Con Edison",
        phase=1,
        rename_map={
            "SUBSTATION": "Substation",
            "VOLTAGE_KV": "Voltage_kV",
            "FEEDER_ID": "Feeder",
            'SUBST_PEAK': 'SubstPeak_MW',
            'TOTAL_XFMR': 'SubstCap_kVA',
        },
        drop_columns=["Shape__Len", "OBJECTID"]
    )

    con_ed3 = clean_circuit_data(
        con_ed3,
        company_name="Con Edison",
        phase=3,
        rename_map={          
            "SUBSTATION": "Substation",
            "VOLTAGE_KV": "Voltage_kV",
            "HC_VALUE": "HostCap_MW"
        },
        drop_columns=["OPERATINGC", "CIRCUIT_NA", "HC_UPDATED", "DERS_UPDAT", "CONNECTED_", 
                      "QUEUED_GEN","MIN_FDR_HC", "MAX_FDR_HC", "Shape__Len", "OBJECTID"],
    )

    # Merge datasets
    con_ed = gpd.GeoDataFrame(pd.merge(con_ed12, con_ed3, on=["Substation", "Voltage_kV", "Phase", "Company", "geometry"], how="outer"))

    # Fix spaces between 'NO.' and numbers in the Substation column
    con_ed['Substation'] = con_ed['Substation'].str.replace(r'\bNO\. (\d+)', r'NO.\1', regex=True)
    
    # Fill NaN values in Hosting Capacity with the respective value for the same Substation
    con_ed['HostCap_MW'] = pd.to_numeric(con_ed['HostCap_MW'], errors='coerce')
    con_ed['HostCap_MW'] = con_ed.groupby('Substation')['HostCap_MW'].transform(lambda x: x.ffill().bfill())
    
    # Fill NaN values in Substation_Peak (MW) with the respective value for the same Substation
    con_ed['SubstPeak_MW'] = con_ed.groupby('Substation')['SubstPeak_MW'].transform(lambda x: x.ffill().bfill())
    
    # Fill NaN values in Total_Transformer_Capacity (kVA) with the respective value for the same Substation
    con_ed['SubstCap_kVA'] = con_ed.groupby('Substation')['SubstCap_kVA'].transform(lambda x: x.ffill().bfill())
    
    # Calculate Circuit Rating (A) from Substation Capacity (kVA) and Voltage (kV) considering phase
    con_ed['CircRat_A'] = con_ed.apply(lambda row: kva_to_amps(row['SubstCap_kVA'], row['Voltage_kV'], row['Phase']), axis=1)
    con_ed['CircRat_A'] = con_ed['CircRat_A']*10
    
    con_ed.drop(columns=['SubstPeak_MW', 'SubstCap_kVA'], inplace = True)
    
    return con_ed

def clean_ny_avangrid(avangrid_h, avangrid_noh):
    """Clean and merge Avangrid data."""
    
    avangrid_h = clean_circuit_data(
        avangrid_h,
        company_name="Avangrid",
        phase=3,
        rename_map={          
            "SUBSTATION": "Substation",
            "VOLTAGE": "Voltage_kV",
            "MAX_hostin": "HostCap_MW"
        },
        drop_columns=['OBJECTID', 'circuit_1', 'stage3host', 'Name',
                       'MIN_hostin', 'SUM_NamPla', 'SUM_NamP_1', 'HCA_Date',
                       'DG_Date', 'Zone', 'MIN_unin', 'Notes', 'COcode', 'Shape__Len'],
    )
    
    avangrid_noh = clean_circuit_data(
        avangrid_noh,
        company_name="Avangrid",
        phase=1,
        rename_map={          
            "SUBSTATION": "Substation",
            "VOLTAGE": "Voltage_kV",
            "MAX_hostin": "HostCap_MW"
        },
        drop_columns=['OBJECTID', 'circuit_1', 'stage3host', 'Name',
                       'MIN_hostin', 'MIN_uninte', 'SUM_NamPla', 'SUM_NamP_1',
                       'Zone', 'HCA_Date', 'DG_Date', 'Shape__Len'],
    )

    avangrid = gpd.GeoDataFrame(pd.merge(avangrid_h, avangrid_noh, on=["Voltage_kV", "HostCap_MW", "Substation", "Phase", "Company", "geometry"], how="outer"))
    
    # Adjust Voltage (kV)
    avangrid['Voltage_kV'] = avangrid['Voltage_kV']/10000
    avangrid['Voltage_kV'] = avangrid['Voltage_kV'].replace(0, 0.24)
    
    # Calculate Circuit Rating (A) from Hosting Capacity (MW) and Voltage (kV) considering phase and cap the values
    avangrid['CircRat_A'] = avangrid.apply(lambda row: mw_to_amps(row['HostCap_MW'], row['Voltage_kV'], phase=row['Phase']), axis=1)
    avangrid['CircRat_A'] = avangrid['CircRat_A'].clip(upper=1500)
       
    return avangrid

def clean_ny_national_grid(nat_grid_primary, nat_grid_hc):
    """
    Clean and merge National Grid data.
    parameters:
        nat_grid_primary (GeoDataFrame): National Grid Phase 1 dataset
        nat_grid_hc (GeoDataFrame): National Grid Phase 3 dataset
    returns:
        GeoDataFrame: Merged and cleaned National Grid dataset.
    """

    nat_grid_hc = clean_circuit_data(
        nat_grid_hc,
        company_name="National Grid",
        phase=3,
        rename_map={          
            "feeder_dg_": "Feeder",
            "substati_1": "Substation",
            "feeder_vol": "Voltage_kV",
            "feeder_max": "HostCap_MW"
        },
        drop_columns=['OBJECTID', 'Master_CDF', 'operating_', 'nyiso_load', 'notes', 'hca_refres', 
                      'feeder_ant', 'feeder_d_1', 'feeder_ins', 'feeder_min', 'feeder_que',
                       'color', 'Shape_Leng', 'gridforce_', 'gridforc_1', 'substation'],
    )
    
    nat_grid_primary = clean_circuit_data(
        nat_grid_primary,
        company_name="National Grid",
        phase=1,
        rename_map={          
            "feeder_cdf": "Feeder",
            "substation": "Substation",
            "primary_vo": "Voltage_kV",
            "primary_hc": "HostCap_MW"
        },
        drop_columns=['distance_t', 'OBJECTID', 'ID', 'primary__1', 
                      'primary__2', 'primary__3', 'primary__4', 'primary__5', 'primary__6', 
                      'feeder_rat', 'color', 'Shape_Leng', 'gridforce_'],
    )
    
    nat_grid_primary['Substation'] = nat_grid_primary['Substation'].astype(str)
    nat_grid_hc['Substation'] = nat_grid_hc['Substation'].astype(str)
    nat_grid_primary['Feeder'] = nat_grid_primary['Feeder'].astype(str)
    nat_grid_hc['Feeder'] = nat_grid_hc['Feeder'].astype(str)
    
    nat_grid = gpd.GeoDataFrame(pd.merge(nat_grid_primary, nat_grid_hc, on=["Substation", "Feeder", "Voltage_kV", "HostCap_MW", "Phase", "Company", "geometry"], how="outer"))

    # Calculate Circuit Rating (A) from Hosting Capacity (MW) and Voltage (kV) considering phase
    nat_grid['CircRat_A'] = nat_grid.apply(lambda row: mw_to_amps(row['HostCap_MW'], row['Voltage_kV'], phase=row['Phase']), axis=1)
    nat_grid['CircRat_A'] = nat_grid['CircRat_A'].clip(upper=800)
    
    # Fill NaN values in Hosting Capacity with the respective value for the same Substation
    nat_grid['HostCap_MW'] = (nat_grid.groupby('Feeder')['HostCap_MW'].transform(lambda x: x.ffill().bfill()))
    
    return nat_grid

def clean_ny_oru(oru1, oru3):
    """
    Clean and merge National Grid data.
    parameters:
        oru1 (GeoDataFrame): Orange and Rockland Phase 1 dataset
        oru3 (GeoDataFrame): Orange and Rockland Phase 3 dataset
    returns:
        GeoDataFrame: Merged and cleaned Orange and Rockland dataset.
    """

    oru1 = clean_circuit_data(
        oru1,
        company_name="Orange And Rockland",
        phase=1,
        rename_map={          
            'VOLT': 'Voltage_kV',
            'SUBSTATION': 'Substation'
        },
        drop_columns=['OBJECTID', 'CIRCUIT','OPERATINGC','LoadCurve','SystemData', 'Shape__Len'],
    )
    
    oru3 = clean_circuit_data(
        oru3,
        company_name="Orange And Rockland",
        phase=1,
        rename_map={          
            'LOCAL_VOLT': 'Voltage_kV',
            'LOCAL_MAX': 'HostCap_MW',
            'SUBSTATION': 'Substation'
        },
        drop_columns=['OBJECTID', 'CIRCUIT', 'OPERATINGC','LoadCurve', 'HC_REFESH_', 
                      'DER_REFESH', 'NOTES', 'SystemData', 'Shape__Len', 'LOCAL_MIN', 
                     'ANTI_ISLAN', 'NYISO_LOAD','SUBSTATI_1', 'CONNECTED_', 
                      'QUEUED_DER', 'DG_ANALYSI'],
    )
      
    oru = gpd.GeoDataFrame(pd.merge(oru1, oru3, on=['Substation', 'Voltage_kV', 'geometry', 'Phase', 'Company'], how="outer"))

    # Fill NaN values in Maximum Local Hosting Capacity (MW) with the respective value for the same Substation
    oru['HostCap_MW'] = (oru.groupby('Substation')['HostCap_MW'].transform(lambda x: x.ffill().bfill()))

    # Calculate Circuit Rating (A) from Hosting Capacity (MW) and Voltage (kV) considering phase
    oru['CircRat_A'] = oru.apply(lambda row: mw_to_amps(row['HostCap_MW'], row['Voltage_kV'], phase=row['Phase']), axis=1)
    oru['CircRat_A'] = oru['CircRat_A'].clip(upper=1500)
    
    return oru

def clean_ri_circuit_data(lines):
    """
    Clean and process circuit dataset by renaming columns, dropping unnecessary fields, 
    and filling missing values.

    Args:
        lines (GeoDataFrame): Circuit dataset.
        company_name (str): Name of the company operating the circuits.
        phase (int or str): Default phase for the dataset.

    Returns:
        GeoDataFrame: Cleaned dataset.
    """
    # Rename columns for consistency    
    rename_columns = {
            "Voltage": "Voltage_kV",
            "MaxHC": "HostCap_MW",
            "Phases": "Phase",
        }

    # Unnecessary columns to drop
    drop_columns = [
        'OBJECTID', 'Master_CDF', 'Planning_A', 'Transforme', 'APRating', 'DG_Connect', 'DG_Pending', 'MaxHC_Conn',
        'COLOR', 'HC_Color_C', 'HC_Refresh', 'Challengin', 'Shape_Leng', 'RANGE', 'LOC_VOLT', 'Substati_2', "MinHC",
        'DG_in_HCA', 'DG_at_Map_', 'Unaccounte', 'Map_Refres', 'Unaccoun_1', 'Unaccoun_2', 'DG_Conne_1',
        'DG_Refresh', 'Substati_3', 'Substati_4', 'Substati_5', 'Substati_6', 'Substati_7', 'Substati_8',
        'Substati_9', 'Substati10', 'OBJECTID_1', 'MinHC','LOC_MAX', 'LOC_MIN', 'Substati_1', 'Transfor_1'
    ]

    # Clean data
    lines = clean_circuit_data(
        lines,
        company_name="National Grid",
        rename_map=rename_columns,
        drop_columns=drop_columns
    )

    return lines

def clean_vt_circuit_data(lines, company_name, phase=1):
    """
    Clean and process circuit dataset by renaming columns, dropping unnecessary fields, 
    and filling missing values.

    Args:
        lines (GeoDataFrame): Circuit dataset.
        company_name (str): Name of the company operating the circuits.
        phase (int or str): Default phase for the dataset.

    Returns:
        GeoDataFrame: Cleaned dataset.
    """
    # Assign company and phase
    lines["Company"] = company_name
    lines["Phase"] = phase

    # Drop unnecessary columns
    drop_columns = ["FEEDER_ID", "RATING", "REMAINING_", "REMAININ_1", "ESRI_OID", "DG_KW_FEED"]
    lines.drop(columns=drop_columns, inplace=True, errors="ignore")

    # Rename columns for consistency
    rename_columns = {
        "LOCATION": "Substation",
        "DG_KW_TRAN": "HostCap_MW",
        "MAX_KVA_RA": "CircRat_A",
    }
    lines.rename(columns=rename_columns, inplace=True)

    # Fill missing Circuit Rating values within the same substation
    lines["CircRat_A"] = (
        lines.groupby("Substation")["CircRat_A"]
        .transform(lambda x: x.ffill().bfill())
    )

    # Convert values to correct units
    lines["HostCap_MW"] /= 1000  # Convert kW to MW
    lines["CircRat_A"] /= 100  # Normalize circuit rating

    return lines