import pandas as pd
import re
import numpy as np
import math
import sys

# Constants
RETICLE_X = 26  # mm
RETICLE_Y = 33  # mm
RETICLE_A = RETICLE_X * RETICLE_Y
DIAMETER_OF_WAFER = 300  # mm

# Function to calculate Die Count per Reticle
def die_count_per_reticle(x_die, y_die, scribe_use_flag=0, scribe_x_width=0, scribe_y_width=0):
    if scribe_use_flag:
        x_die += 0.001 * scribe_x_width
        y_die += 0.001 * scribe_y_width
    no_of_dies_in_reticle = RETICLE_A // (x_die * y_die)
    return x_die, y_die, no_of_dies_in_reticle

# Function to calculate MFU
def mfu(x_die, y_die, scribe_use_flag=0, scribe_x_width=0, scribe_y_width=0):
    x_die, y_die, die_count = die_count_per_reticle(x_die, y_die, scribe_use_flag, scribe_x_width, scribe_y_width)
    adie = x_die * y_die
    mfu_percentage = (adie * die_count / RETICLE_A) * 100
    return mfu_percentage


# Function to calculate PDPW using the provided Die Area (A)
def calculate_pdpw(a, d):
    if a > 0:
        return math.floor((math.pi * d**2 / (4 * a)) * math.exp(-2 * math.sqrt(a) / d))
    else:
        return math.nan

# Function to calculate GDPW
def calculate_gdpw(yield_value, pdpw):
    return math.floor(yield_value * pdpw)

# Function to calculate Yield
def calculate_yield(die_area, defect_density, n=1, model_name="seeds_model"):
    if model_name == "seeds_model":
        return (1 / (1 + die_area * defect_density)) ** n
    else:
        raise ValueError(f"Model '{model_name}' not supported yet.")

# File paths
die_construction_file = "die_construction.csv"
technology_defect_density_file = "technology_defect_density_v2.csv"

try:
    # Read the die_construction and technology_defect_density files
    die_construction_df = pd.read_csv(die_construction_file)
    technology_defect_density_df = pd.read_csv(technology_defect_density_file)
    die_defect_density_df = pd.DataFrame()

    # Remove the '%' sign, convert 'Area %' to floats, and divide by 100
    die_construction_df['Area %'] = die_construction_df['Area %'].str.rstrip('%').astype(float) / 100
    
    die_defect_density_df["time"] = technology_defect_density_df["time"]

    # User input for A (Area of a Die)
    die_width = float(input("Enter the width of Die in mm: "))
    die_height = float(input("Enter the height of Die in mm: "))
    if die_width <= 0 or die_height <= 0:
        raise ValueError("Die dimensions must be positive values.")
    die_area = die_height * die_width

    # User input for Edge Exclusion Factor (in mm)
    edge_exclusion_factor = float(input("Enter Edge Exclusion Factor in mm: "))
    d = DIAMETER_OF_WAFER - edge_exclusion_factor

    # Add user inputs to the DataFrame
    die_defect_density_df['Die Width'] = f"{die_width} mm"
    die_defect_density_df['Die Height'] = f"{die_height} mm"
    die_defect_density_df['Edge Exclusion Factor'] = f"{edge_exclusion_factor} mm"
    die_defect_density_df['Die Area'] = f"{die_area} mm²"
    # Initialize an empty dictionary for the final area_dict
    area_dict = {}

    # Loop through Defectivity Labels, Area %, and Utilization/ Efficiency [%] in the DataFrame
    for label, area, utilization in zip(die_construction_df['Defectivity Labels'], die_construction_df['Area %'], die_construction_df['Utilization/ Efficiency [%]']):
        if pd.notna(label) and pd.notna(area):
            if label not in area_dict:
                area_dict[label] = [{'Area': area, 'Utilization': utilization if pd.notna(utilization) else None}]
            else:
                found = False
                for entry in area_dict[label]:
                    if entry['Utilization'] == utilization:
                        entry['Area'] += area
                        found = True
                        break
                if not found:
                    area_dict[label].append({'Area': area, 'Utilization': utilization})

    # Ensure the column is of type float64
    die_defect_density_df['Die Aggregate DD'] = 0.0

    # Extract the content inside the brackets from all columns
    brackets_pattern = r"\[(.*?)\]"
    percentage_pattern = r"(\d+)%"
    column_bracket_dict = {}
    for column in technology_defect_density_df.columns:
        label_match = re.search(brackets_pattern, column)
        percentage_match = re.search(percentage_pattern, column)
        if label_match:
            column_bracket_dict[label_match.group(1)] = {"column_name": column}
            if percentage_match:
                column_bracket_dict[label_match.group(1)].update({"Utilization": percentage_match.group(1)})



    # Calculate 'Die Aggregate DD' using a vectorized approach
    def calculate_aggregate_dd(row, area_dict, column_bracket_dict):
        aggregate_value = 0
        for label, area_list in area_dict.items():
            if label in column_bracket_dict:
                column_name = column_bracket_dict[label]['column_name']
                for area_entry in area_list:
                    area = area_entry['Area']
                    utilization = area_entry['Utilization']
                    if 'Utilization' in column_bracket_dict[label] and utilization:
                        current_utilization = float(utilization.rstrip('%'))
                        expected_utilization = float(column_bracket_dict[label]['Utilization'])
                        if current_utilization == expected_utilization:
                            aggregate_value += area * row[column_name]
                        else:
                            adjusted_value = row[column_name] * (current_utilization / expected_utilization)
                            aggregate_value += area * adjusted_value
                    else:
                        aggregate_value += area * row[column_name]
        return round(float(aggregate_value), 2)

    technology_defect_density_df['Die Aggregate DD'] = technology_defect_density_df.apply(
        lambda row: calculate_aggregate_dd(row, area_dict, column_bracket_dict), axis=1
    )

    # Calculate yield for each row and add a new 'Yield' column
    die_defect_density_df['Yield'] = technology_defect_density_df['Die Aggregate DD'].apply(
        lambda defect_density: calculate_yield(die_area, defect_density)
    )

    pdpw = calculate_pdpw(die_area, d)

    # Apply the function to calculate GDPW
    die_defect_density_df['GDPW'] = die_defect_density_df['Yield'].apply(lambda yield_value: calculate_gdpw(yield_value, pdpw))

    die_defect_density_df['Yield'] = die_defect_density_df['Yield'].apply(lambda y: f"{y * 100:.2f}%")
    die_defect_density_df['Die Aggregate DD'] = technology_defect_density_df['Die Aggregate DD'].astype(float).astype(str) + " (def/cm²)"
    die_defect_density_df['MFU'] = round(mfu(die_width, die_height), 2)
    die_defect_density_df['MFU'] = die_defect_density_df['MFU'].apply(lambda x: f"{round(x, 2)}%")
    # Save the updated DataFrame back to a new CSV file
    output_file = "die_defect_density_updated.csv"
    die_defect_density_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\nAggregated Die Defect Density, Yield, GDPW, and MFU columns added and saved to {output_file}")

except PermissionError:
    print("\nPermission denied: Make sure the file is not opened.")
except ValueError as ve:
    print(f"Value error: {ve}")
except FileNotFoundError:
    print("File not found: Please check the file paths.")
except Exception as e:
    print(f"An error occurred: {e}")