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
st.title("Xdie-Ydie Dashboard")
st.markdown("""
This dashboard allows you to input die dimensions (Xdie and Ydie) and view:
- A table of possible Xdie, Ydie, Adie, MFU, and Aspect Ratio values.
- A contour plot of MFU with Xdie and Ydie axes.
""")

# Input fields
Xdie = st.number_input("Enter Xdie (mm):", min_value=1.0, value=10.0, step=0.1)
Ydie = st.number_input("Enter Ydie (mm):", min_value=1.0, value=8.0, step=0.1)
Scribe_use_flag = st.selectbox("Use Scribe Width?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
Scribe_x_width = st.number_input("Enter Scribe X Width (µm):", min_value=0.0, value=0.0, step=0.1)
Scribe_y_width = st.number_input("Enter Scribe Y Width (µm):", min_value=0.0, value=0.0, step=0.1)

# Generate random values
np.random.seed(42)
Xdie_values = np.random.normal(Xdie, 0.5, 1000)  # Mean = Xdie, Std Dev = 0.5
Ydie_values = np.random.normal(Ydie, 0.5, 1000)  # Mean = Ydie, Std Dev = 0.5
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

df = pd.DataFrame(data)

# Display table
st.subheader("Results Table")
if not df.empty:
    st.dataframe(df)
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

    # Plot contour
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X_grid, Y_grid, Z_grid, cmap='viridis', levels=20)
    plt.colorbar(contour, label="MFU (%)")
    ax.set_xlabel("Xdie (mm)")
    ax.set_ylabel("Ydie (mm)")
    ax.set_title("Contour Plot of MFU")
    st.pyplot(fig)
