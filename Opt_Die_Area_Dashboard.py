import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.optimize import curve_fit
import matplotlib.dates as mdates
from die_helper import *
from io import StringIO
import re

# Streamlit app
st.title("DIE YIELD AND MFU OPTIMIZATION DASHBOARD")
st.markdown("""
This dashboard allows you to input die dimensions (Xdie and Ydie) and view:
- A table of possible Xdie, Ydie, Adie, MFU, and Aspect Ratio values.
- A contour plot of MFU with Xdie and Ydie axes.
""")

col1, col2 = st.columns([1, 2])  # Two columns layout

with col1:
    Xdie = st.number_input("Enter Xdie (mm):", min_value=1.0, value=10.0, step=0.1, disabled=True)
    Ydie = st.number_input("Enter Ydie (mm):", min_value=1.0, value=8.0, step=0.1, disabled=True)
    edge_exclusion_factor = st.number_input("Edge Exclusion Factor (mm):", min_value=0.0, value=1.0, step=0.1, disabled=True)
    Scribe_use_flag = st.selectbox("Use Scribe Width?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", disabled=True)
    Scribe_x_width = st.number_input("Enter Scribe X Width (\u03bcm):", min_value=0.0, value=0.0, step=0.1, disabled=True)
    Scribe_y_width = st.number_input("Enter Scribe Y Width (\u03bcm):", min_value=0.0, value=0.0, step=0.1, disabled=True)
    Disagg_die = st.selectbox("**Disaggregate Die into Chiplet?**", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", disabled=True)

with col2:
    #st.subheader("Die Representation")
    # fig_width = Xdie / max(Xdie, Ydie) * 4  # Scale width relative to a base size
    # fig_height = Ydie / max(Xdie, Ydie) * 4  # Scale height relative to a base size

    # fig, ax = plt.subplots(figsize=(fig_width, fig_height))  # Set dynamic figure size
    # ax.add_patch(plt.Rectangle((0, 0), Xdie, Ydie, facecolor="royalblue", edgecolor=None, lw=2))
    # ax.set_xlim(0, Xdie)
    # ax.set_ylim(0, Ydie)
    # ax.set_aspect('equal', adjustable='box')  # Keep the aspect ratio equal
    # #ax.set_title(f"Width={Xdie} mm, Height={Ydie} mm", fontsize=6)
    # ax.tick_params(axis='both', labelsize=8)  # Reduce X-tick and Y-tick font size
    # ax.set_xlabel("Xdie (mm)", fontsize=8)
    # ax.set_ylabel("Ydie (mm)", fontsize=8)
    # st.pyplot(fig)
    
    # Define die dimensions
    die_area = Xdie * Ydie
    
    if die_area > 0:
        num_dies = RETICLE_A // die_area  # Integer division to fit dies
        num_dies_horizontally = RETICLE_X // Xdie
        num_dies_vertically = RETICLE_Y // Ydie
    else:
        num_dies = 0
        num_dies_horizontally = 0
        num_dies_vertically = 0

    # Visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    # Draw reticle area
    ax.add_patch(plt.Rectangle((0, 0), RETICLE_X, RETICLE_Y, 
                                edgecolor="black", facecolor="lightgray", lw=2, label="Reticle Area"))

    # Draw dies within the reticle
    for i in range(int(num_dies_vertically)):
        for j in range(int(num_dies_horizontally)):
            x = j * Xdie
            y = i * Ydie
            if x + Xdie <= RETICLE_X and y + Ydie <= RETICLE_Y:  # Ensure dies fit
                ax.add_patch(plt.Rectangle((x, y), Xdie, Ydie, 
                                            edgecolor="blue", facecolor="royalblue", lw=1))

    # Formatting
    ax.set_xlim(0, RETICLE_X)
    ax.set_ylim(0, RETICLE_Y)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Reticle Area with {num_dies} Dies (Die Area: {die_area:.2f} mm²)", fontsize=10)
    ax.set_xlabel("Width (mm)")
    ax.set_ylabel("Height (mm)")
    # ax.legend()

    # Display in Streamlit
    st.pyplot(fig)

# Add download and upload buttons side by side
col1, col2 = st.columns([1, 1])  # Two columns for alignment

with col1:
    # Table placeholder
    st.subheader("Die Construction / Composition Table Template")

    # Create a DataFrame with the specified column names and empty cells
    columns = ["Category", "Subcategory", "Defectivity Labels", "Area %", "Utilization/Efficiency[%]", "Must Work", "Redundancy"]
    columns1 = ["Category", "Subcategory", "Area %", "Utilization/Efficiency[%]", "Must Work", "Redundancy"]
    placeholder_df = pd.DataFrame(dc_data1, columns=columns1)

    # Display the interactive table
    edited_df = st.data_editor(
        placeholder_df,
        disabled=("Category", "Subcategory", "Area %", "Utilization/Efficiency[%]", "Must Work", "Redundancy"),
        use_container_width=False
    )
with col2:
    st.subheader("Area % Distribution")

    placeholder_df.loc[placeholder_df['Category'] == 'Mesh', 'Subcategory'] = 'Mesh'
    placeholder_df.loc[placeholder_df['Category'] == 'White Space', 'Subcategory'] = 'White Space'
    placeholder_df.loc[placeholder_df['Category'] == 'Analog', 'Subcategory'] = 'Analog'

    # Ensure Area % column is numeric
    placeholder_df["Area %"] = placeholder_df["Area %"].str.rstrip('%').astype(float, errors='ignore')

    # Create the pie chart
    fig1 = px.sunburst(placeholder_df, path=['Category', 'Subcategory'], values='Area %')
    # fig = px.pie(edited_df, names='Category', values='Area %', hole=0.3)

    # Customize the chart (optional)
    fig1.update_traces(textinfo='label+value', textfont_size=12)
    fig1.update_layout(margin=dict(t=50, l=25, r=25, b=25))

    # Display the chart
    st.plotly_chart(fig1, use_container_width=True, key="chart1")

    placeholder_df.loc[placeholder_df['Category'] == 'Mesh', 'Subcategory'] = ''
    placeholder_df.loc[placeholder_df['Category'] == 'White Space', 'Subcategory'] = ''
    placeholder_df.loc[placeholder_df['Category'] == 'Analog', 'Subcategory'] = ''

# Convert the DataFrame to CSV
csv_template = placeholder_df.to_csv(index=False).encode('utf-8')

# Add download and upload buttons side by side
col1, col2 = st.columns([1, 1])  # Two columns for alignment

with col1:
    # Download button for template CSV
    st.download_button(
        label="Download Template as CSV",
        data=csv_template,
        file_name="die_construction_template.csv",
        mime="text/csv",
        key="download_template_csv"  # Unique key for the download button
    )

with col2:
    # File uploader for filled template
    uploaded_file = st.file_uploader(
        "Upload Filled Template",
        type=["csv"],
        key="upload_template_csv"  # Unique key for the file uploader
    )

# If a file is uploaded, update the placeholder DataFrame and re-render the data editor
if uploaded_file is not None:
    try:
        # Read the uploaded file into a DataFrame
        uploaded_df = pd.read_csv(uploaded_file)
        uploaded_df["Defectivity Labels"] = ["SDD", "MDD", "", "HDCDD", "HCDD", "RFDD", "", "MDD", "MDD", "ADD"]

        # Validate the uploaded DataFrame
        if set(columns) == set(uploaded_df.columns):
            st.success("Template uploaded successfully! Displaying updated table and pie charts below:")
            
            # Update the placeholder DataFrame with uploaded data
            edited_df = uploaded_df.copy()

            # Layout for table and pie charts
            table_col, pie_col = st.columns([1, 1])  # Two columns for alignment
            
            with table_col:
                st.subheader("User Die Construction / Composition Table")
                st.data_editor(
                    uploaded_df.drop("Defectivity Labels", axis='columns'),
                    disabled=("Category", "Subcategory", "Utilization/Efficiency[%]", "Must Work", "Redundancy"),
                    use_container_width=False,
                    height=400  # Adjust height to make space for the pie chart
                )

            with pie_col:
                st.subheader("Area % Distribution")

                edited_df.loc[edited_df['Category'] == 'Mesh', 'Subcategory'] = 'Mesh'
                edited_df.loc[edited_df['Category'] == 'White Space', 'Subcategory'] = 'White Space'
                edited_df.loc[edited_df['Category'] == 'Analog', 'Subcategory'] = 'Analog'

                # Ensure Area % column is numeric
                edited_df["Area %"] = edited_df["Area %"].str.rstrip('%').astype(float, errors='ignore')

                # Create the pie chart
                fig = px.sunburst(edited_df, path=['Category', 'Subcategory'], values='Area %')
                # fig = px.pie(edited_df, names='Category', values='Area %', hole=0.3)

                # Customize the chart (optional)
                fig.update_traces(textinfo='label+value', textfont_size=12)
                fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))

                # Display the chart
                st.plotly_chart(fig, use_container_width=True, key="chart2")

                edited_df.loc[edited_df['Category'] == 'Mesh', 'Subcategory'] = ''
                edited_df.loc[edited_df['Category'] == 'White Space', 'Subcategory'] = ''
                edited_df.loc[edited_df['Category'] == 'Analog', 'Subcategory'] = ''
                
        else:
            st.error("Uploaded file does not match the template format. Please use the provided template.")
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

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

die_defect_density_df = pd.DataFrame()

if uploaded_file is not None:
    # Add a button to trigger YIELD CALCULATION
    if st.button("Calculate Yield and Display Table"):
        st.write("Button Pressed!")  # Debugging line to confirm button functionality

        # Validate input table data, accommodating for missing values in the selected columns
        # required_columns = ["Area %", "Utilization/Efficiency[%]", "Must Work"]

        # Example: Define specific cells to validate (row, column)
        cells_to_check = [(1, "Area %"), (1, "Utilization/Efficiency[%]"), (1, "Must Work"),
                        (2, "Area %"), (2, "Utilization/Efficiency[%]"), (2, "Must Work"),
                        (3, "Area %"), (4, "Utilization/Efficiency[%]"), 
                        (4, "Area %"), (5, "Utilization/Efficiency[%]"), 
                        (5, "Area %"), (8, "Utilization/Efficiency[%]"), 
                        (6, "Area %"), (9, "Utilization/Efficiency[%]"), 
                        (7, "Area %"), 
                        (8, "Area %"), 
                        (9, "Area %"), 
                        (10, "Area %")]

        # Check if any of these specific cells are missing (NaN or empty string)
        missing_values = any(
            pd.isnull(edited_df.at[row, col]) or edited_df.at[row, col] == ""
            for row, col in cells_to_check
            )

        if not missing_values:
            st.warning("Please fill in the required cells in the DIE CONSTRUCTION/COMPOSITION table.")
        else:
            # YIELD CALCULATION
            die_construction_df = edited_df.copy()
            # Read the data into a pandas DataFrame
            technology_defect_density_df = pd.read_csv(StringIO(tdd_data), sep="\t")
        
            # Remove the '%' sign, convert 'Area %' to floats, and divide by 100
            # Safely remove '%' and convert 'Area %' to floats
            if 'Area %' in die_construction_df.columns:
                die_construction_df['Area %'] = (
                    die_construction_df['Area %']
                    .astype(str)  # Convert to string to allow `.str` operations
                    .str.rstrip('%')  # Remove '%' if present
                    .replace('', '0')  # Replace empty strings with '0'
                    .astype(float) / 100  # Convert to float and normalize to a fraction
                )
            else:
                st.error("'Area %' column is missing in the uploaded data.")
            # die_construction_df['Area %'] = die_construction_df['Area %'].str.rstrip('%').astype(float, errors='ignore') / 100
            # if die_construction_df['Area %'].isnull().any():
            #     st.error("Area % contains invalid or missing values.")
                #raise ValueError("Area % contains invalid or missing values.")
            
            die_defect_density_df["Time"] = technology_defect_density_df["Time"]

            # User input for A (Area of a Die)
            # die_width = float(input("Enter the width of Die in mm: "))
            # die_height = float(input("Enter the height of Die in mm: "))
            die_width = Xdie
            die_height = Ydie
            
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
            for label, area, utilization in zip(die_construction_df['Defectivity Labels'], die_construction_df['Area %'], die_construction_df['Utilization/Efficiency[%]']):
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
            die_defect_density_df['Die Aggregate DD'] = technology_defect_density_df['Die Aggregate DD'].astype(float).astype(str)# + " (def/cm²)"
            die_defect_density_df['MFU'] = round(mfu(die_width, die_height)["MFU%"], 2)
            die_defect_density_df['MFU'] = die_defect_density_df['MFU'].apply(lambda x: f"{round(x, 2)}%")
            # Save the updated DataFrame back to a new CSV file
            # output_file = "die_defect_density_updated.csv"
            # die_defect_density_df.to_csv(output_file, index=False, encoding='utf-8')
            # print(f"\nAggregated Die Defect Density, Yield, GDPW, and MFU columns added and saved to {output_file}")
                
        
        # Display the Die Defect Density Table
        st.subheader("YIELD TABLE")
    
        # Check if the dataframe is not empty
        if not die_defect_density_df.empty:
            # Select only the desired columns
            display_df = die_defect_density_df[["Time", "Die Aggregate DD", "Yield", "GDPW"]]
        
            # Format and style the table for better readability
            styled_display_df = display_df.style.format({
                "Time": "{}",
                "Die Aggregate DD (def/cm^2)": "{}",  # Format DD values to 2 decimals
                "Yield": "{}",
                "GDPW": "{}"  # Format GDPW values to 2 decimals
            }).set_table_styles([
                {'selector': 'td', 'props': [('text-align', 'center')]},
                {'selector': 'th', 'props': [('text-align', 'center'), ('font-weight', 'bold')]}
            ])

            # Render the table with Streamlit's interactive dataframe
            st.dataframe(styled_display_df, use_container_width=True)
            
            # Allow downloading the table as CSV
            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Yield Table as CSV",
                data=csv,
                file_name="die_defect_density_table.csv",
                mime="text/csv",
            )

            # Sample data (replace these with your actual variables from your data)
            time = pd.to_datetime(die_defect_density_df["Time"])  # Example time variable
            # time_dad = die_defect_density_df["Time"][::-1]  # Example time variable
            # die_aggregate_dd = die_defect_density_df["Die Aggregate DD"].values  # Example Die Aggregate DD values
            # yield_data = die_defect_density_df["Yield"].values  # Example Yield percentages
            # Ensure Die Aggregate DD values are numeric
            die_aggregate_dd = pd.to_numeric(die_defect_density_df["Die Aggregate DD"].values[::-1], errors="coerce")

            # Ensure Yield values are numeric, stripping '%' if necessary
            yield_data = [float(y.strip('%')) for y in die_defect_density_df["Yield"].values]

            # Safeguard for NaN values
            if pd.Series(die_aggregate_dd).isnull().any():
                st.error("Die Aggregate DD contains invalid values. Please check your data.")
            elif any(pd.isnull(yield_data)):
                st.error("Yield contains invalid values. Please check your data.")
            else:
                # Create subplots for two adjacent plots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharex=False)

                # Plot Die Aggregate DD vs. Time
                ax1.set_title("Die Aggregate DD vs Time")
                ax1.set_xlabel("Time")
                ax1.set_ylabel("Die Aggregate DD", color="tab:blue")
                ax1.plot(time, die_aggregate_dd, label="Die Aggregate DD", color="tab:blue", marker="o")
                ax1.tick_params(axis="y", labelcolor="tab:blue")
                ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))  # Format x-axis to show Month and Year
                ax1.set_xticks(time[::max(1, len(time) // 6)])  # Set equidistant ticks
                # ax1.set_xticklabels(time, rotation=90)  # Rotate x-axis labels
                ax1.grid(visible=True, linestyle="--", alpha=0.5)

                # Automatically scale y-axis and make it consistent across the plots
                ax1.set_ylim(min(die_aggregate_dd) * 0.9, max(die_aggregate_dd) * 1.1)  # Scale for 10% padding

                # Plot Yield vs. Time
                ax2.set_title("Yield vs Time")
                ax2.set_xlabel("Time")
                ax2.set_ylabel("Yield (%)", color="tab:green")
                ax2.plot(time, yield_data, label="Yield", color="tab:green", marker="x")
                ax2.tick_params(axis="y", labelcolor="tab:green")
                ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))  # Format x-axis to show Month and Year
                ax2.set_xticks(time[::max(1, len(time) // 6)])  # Set equidistant ticks
                # ax2.set_xticks(time)  # Standardize x-axis ticks
                # ax2.set_xticklabels(time, rotation=90)  # Rotate x-axis labels
                ax2.grid(visible=True, linestyle="--", alpha=0.5)

                # Automatically scale y-axis and make it consistent across the plots
                # yield_values = [float(y.strip('%')) for y in yield_data]  # Remove % and convert to float
                yield_values = [float(y.strip('%')) if isinstance(y, str) else float(y) for y in yield_data]

                ax2.set_ylim(min(yield_values) * 0.9, max(yield_values) * 1.1)  # Scale for 10% padding

                # Add legends
                ax1.legend(loc="upper left")
                ax2.legend(loc="upper left")

                # Adjust layout for better spacing
                plt.tight_layout()

                # Display the plots in Streamlit
                st.pyplot(fig)
        else:
            st.warning("No data available to display in the Die Defect Density Table.")