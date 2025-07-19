import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Your time array (copy-paste the full array as needed)
time_values = np.array([
    0,
    1.00E-12, 2.30E-12, 3.80E-12, 5.30E-12, 6.80E-12, 8.30E-12, 9.80E-12, 1.13E-11, 1.28E-11, 1.43E-11, 1.58E-11,
    1.73E-11, 1.81E-11, 1.90E-11, 2.03E-11, 2.03E-11, 2.03E-11, 2.04E-11, 2.04E-11, 2.04E-11, 2.05E-11, 2.05E-11,
    2.06E-11, 2.07E-11, 2.08E-11, 2.10E-11, 2.12E-11, 2.15E-11, 2.18E-11, 2.20E-11, 2.23E-11, 2.27E-11, 2.32E-11,
    2.39E-11, 2.45E-11, 2.45E-11, 2.45E-11, 2.45E-11, 2.45E-11, 2.46E-11, 2.46E-11, 2.46E-11, 2.46E-11, 2.46E-11,
    2.46E-11, 2.46E-11, 2.46E-11, 2.46E-11, 2.46E-11, 2.47E-11, 2.47E-11, 2.47E-11, 2.47E-11, 2.47E-11, 2.47E-11,
    2.47E-11, 2.47E-11, 2.47E-11, 2.47E-11, 2.48E-11, 2.48E-11, 2.48E-11, 2.48E-11, 2.48E-11, 2.48E-11, 2.48E-11,
    2.48E-11, 2.48E-11, 2.48E-11, 2.49E-11, 2.49E-11, 2.49E-11, 2.49E-11, 2.49E-11, 2.49E-11, 2.49E-11, 2.49E-11,
    2.49E-11, 2.49E-11, 2.49E-11, 2.50E-11, 2.50E-11, 2.50E-11, 2.50E-11, 2.50E-11, 2.50E-11, 2.50E-11, 2.50E-11,
    2.50E-11, 2.50E-11, 2.51E-11, 2.51E-11, 2.51E-11, 2.51E-11, 2.51E-11, 2.51E-11, 2.51E-11, 2.51E-11, 2.51E-11,
    2.51E-11, 2.52E-11, 2.52E-11, 2.52E-11, 2.52E-11, 2.52E-11, 2.52E-11, 2.52E-11, 2.52E-11, 2.52E-11, 2.52E-11,
    2.53E-11, 2.53E-11, 2.53E-11, 2.53E-11, 2.53E-11, 2.53E-11, 2.53E-11, 2.53E-11, 2.53E-11, 2.53E-11, 2.54E-11,
    2.54E-11, 2.54E-11, 2.54E-11, 2.54E-11, 2.54E-11, 2.54E-11, 2.54E-11, 2.54E-11, 2.54E-11, 2.55E-11, 2.55E-11,
    2.55E-11, 2.55E-11, 2.55E-11, 2.55E-11, 2.55E-11, 2.55E-11, 2.56E-11, 2.56E-11, 2.57E-11, 2.58E-11, 2.59E-11,
    2.60E-11, 2.62E-11, 2.64E-11, 2.67E-11, 2.71E-11, 2.76E-11, 2.83E-11, 2.91E-11, 2.97E-11, 2.97E-11, 2.99E-11,
    3.00E-11, 3.02E-11, 3.05E-11, 3.06E-11, 3.08E-11, 3.11E-11, 3.15E-11, 3.17E-11, 3.20E-11, 3.22E-11, 3.25E-11,
    3.28E-11, 3.33E-11, 3.39E-11, 3.40E-11, 3.42E-11, 3.42E-11, 3.43E-11, 3.45E-11, 3.47E-11, 3.49E-11, 3.50E-11,
    3.51E-11, 3.52E-11, 3.54E-11, 3.55E-11, 3.56E-11, 3.58E-11, 3.60E-11, 3.63E-11, 3.67E-11, 3.72E-11, 3.79E-11,
    3.87E-11, 3.98E-11, 4.13E-11, 4.28E-11, 4.43E-11, 4.58E-11, 4.73E-11, 4.88E-11, 5.00E-11
])

# Load model function
@st.cache_resource
def load_model():
    return joblib.load('finfet_random_model_compressed.pkl')
model = load_model()

# st.latex(r"\text{Time vs.}\ I_{d}\ \text{Curve (User-selected }\phi,\,\theta,\,\text{LET)}")
st.latex(r"\text{FINFET}\ I_{d}\ \text{Drain Current Predictor}")

# Get user parameters
phi = st.number_input("Phi (°):", min_value=0.0, max_value=360.0, value=0.0, format="%.2f")
theta = st.number_input("Theta (°):", min_value=0.0, max_value=360.0, value=0.0, format="%.2f")
let = st.number_input("LET (°):", min_value=0.0, max_value=360.0, value=0.0, format="%.2f")

# Prepare the DataFrame for prediction
input_df = pd.DataFrame({
    "phi": np.full_like(time_values, phi, dtype=float),
    "theta": np.full_like(time_values, theta, dtype=float),
    "let": np.full_like(time_values, let, dtype=float),
    "time": time_values
})

# Predict Id for each time
id_pred = model.predict(input_df)

# Plot
fig, ax = plt.subplots()
ax.plot(time_values, id_pred)
ax.set_xlabel("Time (s)")
ax.set_ylabel(r"$I_{d}$ (A)")
ax.set_title(r"Predicted Time vs. $I_{d}$ ($\phi$, $\theta$, LET selected)")
ax.grid(True)
st.pyplot(fig)

# Optionally display data table
if st.checkbox("Show prediction table"):
    st.dataframe(pd.DataFrame({
        "Time (s)": time_values,
        r"$I_{d}$ (A)": id_pred
    }))
