import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("iris_model.pkl")

st.title("🌸 Iris Flower Classification App")



st.write("Enter flower measurements:")

# User inputs
sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)

    # Convert to integer to ensure the comparison works
    val = int(prediction[0].lower().strip())
    

    if val == 0:
        st.success("🌼🌸 Result: Iris Setosa")
    elif val == 1:
        st.warning("🌷⛅ Result: Iris Versicolor")
    elif val == 2:
        st.info("🌹🌿 Result: Iris Virginica")
    else:
        st.error("Unknown Prediction Value")

        