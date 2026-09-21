import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("iris_model.pkl")

# Title
st.title("🌸 Iris Flower Prediction")

st.write("Enter the flower measurements:")

# Sepal Length
sepal_length = st.number_input(
    "Sepal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.1,
    key="sepal_length"
)

# Sepal Width
sepal_width = st.number_input(
    "Sepal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=3.0,
    step=0.1,
    key="sepal_width"
)

# Petal Length
petal_length = st.number_input(
    "Petal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=1.5,
    step=0.1,
    key="petal_length"
)

# Petal Width
petal_width = st.number_input(
    "Petal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=0.2,
    step=0.1,
    key="petal_width"
)

# Predict button
if st.button("Predict", key="predict_button"):

    # Create input
    input_data = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    # Prediction
    prediction = model.predict(input_data)[0]

    # Flower names
    flower_names = [
        "Setosa",
        "Versicolor",
        "Virginica"
    ]

    # Show result
    st.success(
        f"🌸 Predicted Iris Flower: {flower_names[prediction]}"
    )
