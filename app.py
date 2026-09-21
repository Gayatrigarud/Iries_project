import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("iris_model.pkl")

# App title
st.title("🌸 Iris Flower Prediction")

st.write("Enter the flower measurements:")

# Input fields
sepal_length = st.number_input(
    "Sepal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.1,
    key="sepal_length"
)

sepal_width = st.number_input(
    "Sepal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=3.0,
    step=0.1,
    key="sepal_width"
)

petal_length = st.number_input(
    "Petal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=1.5,
    step=0.1,
    key="petal_length"
)

petal_width = st.number_input(
    "Petal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1,
    key="petal_width"
)

# Prediction button
if st.button("Predict", key="predict_button"):

    input_data = np.array([
        [sepal_length, sepal_width, petal_length, petal_width]
    ])

    prediction = model.predict(input_data)

    st.success(f"Predicted Iris Flower: {prediction[0]}")
