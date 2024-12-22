import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from die_helper import *
from io import StringIO
import re

# # Define functions for calculations
# def Die_count_per_reticle(Xdie, Ydie, Scribe_use_flag=0, Scribe_x_width=0, Scribe_y_width=0):
#     Reticle_X = 26  # mm
#     Reticle_Y = 33  # mm
#     Reticle_A = Reticle_X * Reticle_Y

#     if Scribe_use_flag:
#         Xdie += 0.001 * Scribe_x_width
#         Ydie += 0.001 * Scribe_y_width

#     No_of_dies_in_Reticle = Reticle_A // (Xdie * Ydie)
#     return Xdie, Ydie, Reticle_A, No_of_dies_in_Reticle

# def MFU(Xdie, Ydie, Scribe_use_flag=0, Scribe_x_width=0, Scribe_y_width=0):
#     MFU_dict = {}
#     Xdie, Ydie, MFU_dict['Reticle_A'], MFU_dict['Die_count'] = Die_count_per_reticle(
#         Xdie, Ydie, Scribe_use_flag, Scribe_x_width, Scribe_y_width)
#     MFU_dict['Adie'] = Xdie * Ydie
#     MFU_dict['MFU%'] = (MFU_dict['Adie'] * MFU_dict['Die_count'] / MFU_dict['Reticle_A']) * 100
#     return MFU_dict

# Streamlit app
st.title("DIE YIELD AND MFU OPTIMIZATION DASHBOARD")
st.markdown("""
This dashboard allows you to input die dimensions (Xdie and Ydie) and view:
- A table of possible Xdie, Ydie, Adie, MFU, and Aspect Ratio values.
- A contour plot of MFU with Xdie and Ydie axes.
""")

# Custom CSS to adjust input field width
st.markdown("""
    <style>
        div[data-testid="stNumberInput"] > div > div > input {
            width: 120px !important;
        }
        div[data-testid="stSelectbox"] > div > div > select {
            width: 120px !important;
        }
    </style>
""", unsafe_allow_html=True)

# Input fields in columns for better layout
col1, col2 = st.columns([1, 2])  # Two columns layout

with col1:
    Xdie = st.number_input("Enter Xdie (mm):", min_value=1.0, value=10.0, step=0.1)
    Ydie = st.number_input("Enter Ydie (mm):", min_value=1.0, value=8.0, step=0.1)
    edge_exclusion_factor = st.number_input("Edge Exclusion Factor (mm):", min_value=0.0, value=1.0, step=0.1)
    Scribe_use_flag = st.selectbox("Use Scribe Width?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    Scribe_x_width = st.number_input("Enter Scribe X Width (\u03bcm):", min_value=0.0, value=0.0, step=0.1)
    Scribe_y_width = st.number_input("Enter Scribe Y Width (\u03bcm):", min_value=0.0, value=0.0, step=0.1)

with col2:
    #st.subheader("Die Representation")
    fig_width = Xdie / max(Xdie, Ydie) * 4  # Scale width relative to a base size
    fig_height = Ydie / max(Xdie, Ydie) * 4  # Scale height relative to a base size

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))  # Set dynamic figure size
    ax.add_patch(plt.Rectangle((0, 0), Xdie, Ydie, facecolor="royalblue", edgecolor=None, lw=2))
    ax.set_xlim(0, Xdie)
    ax.set_ylim(0, Ydie)
    ax.set_aspect('equal', adjustable='box')  # Keep the aspect ratio equal
    #ax.set_title(f"Width={Xdie} mm, Height={Ydie} mm", fontsize=6)
    ax.tick_params(axis='both', labelsize=8)  # Reduce X-tick and Y-tick font size
    ax.set_xlabel("Xdie (mm)", fontsize=8)
    ax.set_ylabel("Ydie (mm)", fontsize=8)
    st.pyplot(fig)

# Generate random values
np.random.seed(42)
Xdie_values = np.random.normal(Xdie, 0.5, 1000)
Ydie_values = np.random.normal(Ydie, 0.5, 1000)
aspect_ratio_range = (0.5, 1.5)

# Calculate table data
data = []
target_Adie = Xdie * Ydie

for xd, yd in zip(Xdie_values, Ydie_values):
    Adie = xd * yd
    aspect_ratio = yd / xd
    if aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1] and np.isclose(Adie, target_Adie, atol=0.1):
        mfu_data = mfu(xd, yd, Scribe_use_flag, Scribe_x_width, Scribe_y_width)
        data.append({
            "Xdie (mm)": round(xd, 2),
            "Ydie (mm)": round(yd, 2),
            "Adie (mm^2)": round(mfu_data['Die Area'], 2),
            "MFU (%)": round(mfu_data['MFU%'], 2),
            "Aspect Ratio": round(yd / xd, 2)
        })

mfu_data = mfu(Xdie, Ydie, Scribe_use_flag, Scribe_x_width, Scribe_y_width)
data.append({
            "Xdie (mm)": round(Xdie, 2),
            "Ydie (mm)": round(Ydie, 2),
            "Adie (mm^2)": round(mfu_data['Die Area'], 2),
            "MFU (%)": round(mfu_data['MFU%'], 2),
            "Aspect Ratio": round(Ydie / Xdie, 2)
        })

df = pd.DataFrame(data)

# Display table
st.subheader("OPTIMAL MFU TABLE")
if not df.empty:
    # # Reset the index and add it as a column to preserve original indexing
    # df_with_index = df.reset_index()
    # df_with_index.rename(columns={'index': 'Original Index'}, inplace=True)
    
    # Styling the table for better impact
    styled_df = df.style.format(
        {"Xdie (mm)": "{:.2f}", 
         "Ydie (mm)": "{:.2f}", 
         "Adie (mm^2)": "{:.2f}", 
         "MFU (%)": "{:.2f}", 
         "Aspect Ratio": "{:.2f}"}
    ).set_table_styles([{'selector': 'td', 'props': [('text-align', 'center')]}]).background_gradient(subset="MFU (%)", cmap="viridis")

    # Display as a dataframe to enable sorting
    st.dataframe(styled_df, use_container_width=True)


    # Allow downloading the table as CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Table as CSV",
        data=csv,
        file_name="optimal_die_area.csv",
        mime="text/csv",
    )
else:
    st.warning("No valid data found for the given input values.")

# Contour plot of MFU
if not df.empty:
    st.subheader("Contour Plot of MFU")
    X = df["Xdie (mm)"]
    Y = df["Ydie (mm)"]
    Z = df["MFU (%)"]

    # Create a grid for contour
    X_grid, Y_grid = np.meshgrid(
        np.linspace(X.min(), X.max(), 100),
        np.linspace(Y.min(), Y.max(), 100)
    )
    Z_grid = np.zeros_like(X_grid)
    for i in range(X_grid.shape[0]):
        for j in range(X_grid.shape[1]):
            Z_grid[i, j] = mfu(X_grid[i, j], Y_grid[i, j], Scribe_use_flag, Scribe_x_width, Scribe_y_width)["MFU%"]

    # Plot contour with scatter points
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X_grid, Y_grid, Z_grid, cmap='viridis', levels=20)
    plt.colorbar(contour, label="MFU (%)")
    ax.scatter(X, Y, color="white", s=10, label="ISO AREA MFU Contour")
    ax.scatter(Xdie, Ydie, color="red", marker='x', s=60, label=f"Xdie={Xdie:.1f}, Ydie={Ydie:.1f}")
    ax.set_xlabel("Xdie (mm)")
    ax.set_ylabel("Ydie (mm)")
    ax.set_title("Contour Plot of MFU with Data Points")
    ax.legend()
    st.pyplot(fig)

# Table placeholder
st.subheader("DIE CONSTRUCTION/COMPOSITION")

# Create a DataFrame with the specified column names and empty cells
columns = ["Category", "Subcategory", "Defectivity Labels", "Area%", "Utilization/Efficiency[%]", "Must Work", "Redundancy"]
placeholder_df = pd.DataFrame(dc_data, columns=columns)

# Display the interactive table
edited_df = st.data_editor(
    placeholder_df,
    disabled=("Category", "Subcategory", "Defectivity Labels"),
    use_container_width=False
)

# Add a button to trigger YIELD CALCULATION
if st.button("Calculate Yield and Display Table"):
    
    # YIELD CALCULATION
    die_construction_df = edited_df.copy()
    # Read the data into a pandas DataFrame
    technology_defect_density_df = pd.read_csv(StringIO(tdd_data), sep="\t")
    
    try:
        # Read the die_construction and technology_defect_density files
        #die_construction_df = pd.read_csv(die_construction_file)
        #technology_defect_density_df = pd.read_csv(technology_defect_density_file)
        die_defect_density_df = pd.DataFrame()
        
        # Remove the '%' sign, convert 'Area %' to floats, and divide by 100
        die_construction_df['Area %'] = die_construction_df['Area %'].str.rstrip('%').astype(float) / 100
        
        die_defect_density_df["time"] = technology_defect_density_df["time"]

        # User input for A (Area of a Die)
        # die_width = float(input("Enter the width of Die in mm: "))
        # die_height = float(input("Enter the height of Die in mm: "))
        die_width = Xdie
        die_height = Ydie

        st.write("Hello1")
        if die_width <= 0 or die_height <= 0:
            raise ValueError("Die dimensions must be positive values.")
        die_area = die_height * die_width

        # User input for Edge Exclusion Factor (in mm)
        # edge_exclusion_factor = float(input("Enter Edge Exclusion Factor in mm: "))
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
        # output_file = "die_defect_density_updated.csv"
        # die_defect_density_df.to_csv(output_file, index=False, encoding='utf-8')
        # print(f"\nAggregated Die Defect Density, Yield, GDPW, and MFU columns added and saved to {output_file}")
        st.write("Hello2")
        # Display the Die Defect Density Table
        st.subheader("YIELD TABLE")

        # Check if the dataframe is not empty
        if not die_defect_density_df.empty:
            # Select only the desired columns
            display_df = die_defect_density_df[["time", "Die Aggregate DD", "Yield", "GDPW"]]

            # Format and style the table for better readability
            styled_display_df = display_df.style.format({
                "Die Aggregate DD": "{:.2f}",  # Format DD values to 2 decimals
                "Yield": "{}",
                "GDPW": "{:.2f}"  # Format GDPW values to 2 decimals
            }).set_table_styles([
                {'selector': 'td', 'props': [('text-align', 'center')]},
                {'selector': 'th', 'props': [('text-align', 'center'), ('font-weight', 'bold')]}
            ])

            # Render the table with Streamlit's interactive dataframe
            st.dataframe(styled_display_df, use_container_width=True)

            # Allow downloading the table as CSV
            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Die Defect Density Table as CSV",
                data=csv,
                file_name="die_defect_density_table.csv",
                mime="text/csv",
            )
        else:
            st.warning("No data available to display in the Die Defect Density Table.")
    
    except ValueError as ve:
        print(f"Value error: {ve}")
    except Exception as e:
        print(f"An error occurred: {e}")