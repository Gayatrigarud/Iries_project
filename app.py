import streamlit as st
import numpy as np

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Page title
st.title("🌸 Iris Flower Prediction")

st.write("Enter the flower measurements:")

# Inputs
sepal_length = st.number_input(
    "Sepal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=5.1,
    step=0.1
)

sepal_width = st.number_input(
    "Sepal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=3.5,
    step=0.1
)

petal_length = st.number_input(
    "Petal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=1.4,
    step=0.1
)

petal_width = st.number_input(
    "Petal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=0.2,
    step=0.1
)

# Prediction
if st.button("Predict"):

    input_data = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    prediction = model.predict(input_data)[0]

    flower_names = [
        "Setosa",
        "Versicolor",
        "Virginica"
    ]

    st.success(
        f"🌸 Predicted Iris Flower: {flower_names[prediction]}"
    )
