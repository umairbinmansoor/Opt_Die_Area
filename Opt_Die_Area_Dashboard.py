import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define functions for calculations
def Die_count_per_reticle(Xdie, Ydie, Scribe_use_flag=0, Scribe_x_width=0, Scribe_y_width=0):
    Reticle_X = 26  # mm
    Reticle_Y = 33  # mm
    Reticle_A = Reticle_X * Reticle_Y

    if Scribe_use_flag:
        Xdie += 0.001 * Scribe_x_width
        Ydie += 0.001 * Scribe_y_width

    No_of_dies_in_Reticle = Reticle_A // (Xdie * Ydie)
    return Xdie, Ydie, Reticle_A, No_of_dies_in_Reticle

def MFU(Xdie, Ydie, Scribe_use_flag=0, Scribe_x_width=0, Scribe_y_width=0):
    MFU_dict = {}
    Xdie, Ydie, MFU_dict['Reticle_A'], MFU_dict['Die_count'] = Die_count_per_reticle(
        Xdie, Ydie, Scribe_use_flag, Scribe_x_width, Scribe_y_width)
    MFU_dict['Adie'] = Xdie * Ydie
    MFU_dict['MFU%'] = (MFU_dict['Adie'] * MFU_dict['Die_count'] / MFU_dict['Reticle_A']) * 100
    return MFU_dict

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

# Table placeholder
st.subheader("Table Placeholder")

# Create a DataFrame with the specified column names and empty cells
columns = ["Area%", "Utilization/Efficiency[%]", "Must Work", "Redundancy"]
data = [
    ["Logic", "", "", "", ""],
    ["", "short Ht", "", "", ""],
    ["", "Med Ht", "", "", ""],
    ["", "Recovery", "", "", ""],
    ["Memory", "", "", "", ""],
    ["", "HDC", "", "", ""],
    ["", "HCC", "", "", ""],
    ["", "RF", "", "", ""],
    ["", "Recovery", "", "", ""],
    ["Mesh", "", "", "", ""],
    ["White Space", "", "", "", ""],
    ["Analog", "", "", "", ""]
]
placeholder_df = pd.DataFrame(data, columns=[" "] + columns)

# Display table
st.table(placeholder_df)

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
        mfu_data = MFU(xd, yd, Scribe_use_flag, Scribe_x_width, Scribe_y_width)
        data.append({
            "Xdie (mm)": round(xd, 2),
            "Ydie (mm)": round(yd, 2),
            "Adie (mm^2)": round(mfu_data['Adie'], 2),
            "MFU (%)": round(mfu_data['MFU%'], 2),
            "Aspect Ratio": round(yd / xd, 2)
        })

mfu_data = MFU(Xdie, Ydie, Scribe_use_flag, Scribe_x_width, Scribe_y_width)
data.append({
            "Xdie (mm)": round(Xdie, 2),
            "Ydie (mm)": round(Ydie, 2),
            "Adie (mm^2)": round(mfu_data['Adie'], 2),
            "MFU (%)": round(mfu_data['MFU%'], 2),
            "Aspect Ratio": round(Ydie / Xdie, 2)
        })

df = pd.DataFrame(data)

# Display table
st.subheader("Results Table")
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
            Z_grid[i, j] = MFU(X_grid[i, j], Y_grid[i, j], Scribe_use_flag, Scribe_x_width, Scribe_y_width)["MFU%"]

    # Plot contour with scatter points
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X_grid, Y_grid, Z_grid, cmap='viridis', levels=20)
    plt.colorbar(contour, label="MFU (%)")
    ax.scatter(X, Y, color="white", s=10, label="ISO AREA MFU Contour")
    ax.scatter(Xdie, Ydie, color="red", marker='x', s=60, label=f"Xdie={Xdie:.2f}, Ydie={Ydie:.2f}")
    ax.set_xlabel("Xdie (mm)")
    ax.set_ylabel("Ydie (mm)")
    ax.set_title("Contour Plot of MFU with Data Points")
    ax.legend()
    st.pyplot(fig)
