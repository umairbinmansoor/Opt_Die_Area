
import math
import pandas as pd

# Define constants
RETICLE_X = 26  # mm
RETICLE_Y = 33  # mm
RETICLE_A = RETICLE_X * RETICLE_Y  # Precomputed area of the reticle
DIAMETER_OF_WAFER = 300  # mm

dc_data = [
    ["Logic", "short Ht", "SDD", None, None, None, None],
    ["", "Med Ht", "MDD", None, None, None, None],
    ["", "Recovery", "", None, None, None, None],
    ["Memory", "HDC", "HDCDD", None, None, None, None],
    ["", "HCC", "HCDD", None, None, None, None],
    ["", "RF", "RFDD", None, None, None, None],
    ["", "Recovery", "", None, None, None, None],
    ["Mesh", "", "MDD", None, None, None, None],
    ["White\nSpace", "", "MDD", None, None, None, None],
    ["Analog", "", "ADD", None, None, None, None]
]

# Multi-line string representation of the technology defect density table
tdd_data = """
time	Short Ht DD [SDD] at 45%	Mid Ht DD [MDD] at 50%	HDC DD [HDDD]	HCD DD w/R [HDDDR]	HCC DD [HCDD]	HCC DD w/R [HCDDR]	RF DD [RFDD]	RF DD w/R [RFDDR]	Analog DD [ADD]	Used For Calc [junk]
12/30/2024	4	3.8	6.7	5.7	5.6	4.5	4.7	3.6	4.2	
3/30/2025	3.9	3.7	6.6	5.6	5.5	4.4	4.6	3.5	4.1	0.1
6/30/2025	3.79375	3.59375	6.49375	5.49375	5.39375	4.29375	4.49375	3.39375	3.99375	0.10625
9/30/2025	3.68125	3.48125	6.38125	5.38125	5.28125	4.18125	4.38125	3.28125	3.88125	0.1125
12/30/2025	3.5625	3.3625	6.2625	5.2625	5.1625	4.0625	4.2625	3.1625	3.7625	0.11875
3/30/2026	3.4375	3.2375	6.1375	5.1375	5.0375	3.9375	4.1375	3.0375	3.6375	0.125
6/28/2026	3.30625	3.10625	6.00625	5.00625	4.90625	3.80625	4.00625	2.90625	3.50625	0.13125
9/26/2026	3.16875	2.96875	5.86875	4.86875	4.76875	3.66875	3.86875	2.76875	3.36875	0.1375
12/25/2026	3.025	2.825	5.725	4.725	4.625	3.525	3.725	2.625	3.225	0.14375
3/25/2027	2.875	2.675	5.575	4.575	4.475	3.375	3.575	2.475	3.075	0.15
6/23/2027	2.71875	2.51875	5.41875	4.41875	4.31875	3.21875	3.41875	2.31875	2.91875	0.15625
9/21/2027	2.55625	2.35625	5.25625	4.25625	4.15625	3.05625	3.25625	2.15625	2.75625	0.1625
12/20/2027	2.3875	2.1875	5.0875	4.0875	3.9875	2.8875	3.0875	1.9875	2.5875	0.16875
3/19/2028	2.2125	2.0125	4.9125	3.9125	3.8125	2.7125	2.9125	1.8125	2.4125	0.175
6/17/2028	2.03125	1.83125	4.73125	3.73125	3.63125	2.53125	2.73125	1.63125	2.23125	0.18125
9/15/2028	1.84375	1.64375	4.54375	3.54375	3.44375	2.34375	2.54375	1.44375	2.04375	0.1875
12/14/2028	1.65	1.45	4.35	3.35	3.25	2.15	2.35	1.25	1.85	0.19375
3/14/2029	1.45	1.25	4.15	3.15	3.05	1.95	2.15	1.05	1.65	0.2
6/12/2029	1.24375	1.04375	3.94375	2.94375	2.84375	1.74375	1.94375	0.84375	1.44375	0.20625
9/10/2029	1.03125	0.83125	3.73125	2.73125	2.63125	1.53125	1.73125	0.63125	1.23125	0.2125
"""

def die_count_per_reticle(x_die, y_die, scribe_use_flag=0, scribe_x_width=0, scribe_y_width=0):
    """
    Calculate the die count per reticle area.
    """
    # Validate inputs
    if x_die <= 0 or y_die <= 0:
        raise ValueError("Xdie and Ydie must be positive values.")

    # Adjust dimensions for scribe width
    if scribe_use_flag:
        x_die += 0.001 * scribe_x_width
        y_die += 0.001 * scribe_y_width

    # Calculate number of dies and return
    die_area = x_die * y_die
    no_of_dies_in_reticle = RETICLE_A // die_area  # Integer division
    return {
        "Xdie": x_die,
        "Ydie": y_die,
        "Die Area": die_area,
        "No of Dies in Reticle": no_of_dies_in_reticle,
    }

def mfu(x_die, y_die, scribe_use_flag=0, scribe_x_width=0, scribe_y_width=0):
    """
    Calculate Manufacturing Utilization (MFU) percentage.
    """
    # Get die count and related information
    reticle_info = die_count_per_reticle(x_die, y_die, scribe_use_flag, scribe_x_width, scribe_y_width)
    die_area = reticle_info["Die Area"]
    no_of_dies = reticle_info["No of Dies in Reticle"]

    # Calculate MFU percentage
    mfu_percentage = (die_area * no_of_dies / RETICLE_A) * 100

    # Add to dictionary and return
    reticle_info["MFU%"] = mfu_percentage
    return reticle_info

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
