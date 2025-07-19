import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    return joblib.load('finfet_random_model_compressed.pkl')

model = load_model()

# Title with subscript using LaTeX
st.latex(r"\text{FINFET}\ I_{d}\ \text{Drain Current Predictor}")

st.markdown(
    "Enter the following parameters to predict the Drain Current $I_{d}$ :"
)

phi = st.number_input("Phi° :", min_value=0.0, max_value=360.0, value=0.0, format="%.2f")
theta = st.number_input("Theta° :", min_value=0.0, max_value=360.0, value=0.0, format="%.2f")
let = st.number_input("LET° :", min_value=0.0, max_value=360.0, value=0.0, format="%.2f")
time = st.number_input("Time :", value=0.0, format="%.6f")

if st.button("Predict"):
    input_df = pd.DataFrame([[phi, theta, let, time]], columns=['phi', 'theta', 'let', 'time'])
    prediction = model.predict(input_df)[0]
    formatted_pred = f"{prediction:.5e}"
    st.success(f"**Predicted Id current:** {formatted_pred}")
