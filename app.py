import streamlit as st
import numpy as np
import joblib

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Iris Flower Prediction",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Iris Flower Species Prediction")
st.write(
    "This application uses a **Logistic Regression model** trained on the Iris dataset. "
    "Enter flower measurements to predict the Iris species."
)

# -------------------------------
# Load Model & Scaler
# -------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("iris_logistic_model.pkl")
    scaler = joblib.load("iris_scaler.pkl")
    return model, scaler

model, scaler = load_model()

# -------------------------------
# Input Features (2 Columns)
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length (cm)", value=5.1)
    petal_length = st.number_input("Petal Length (cm)", value=1.4)

with col2:
    sepal_width = st.number_input("Sepal Width (cm)", value=3.5)
    petal_width = st.number_input("Petal Width (cm)", value=0.2)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Species"):
    try:
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input).max()

        species_map = {
            0: "Iris Setosa",
            1: "Iris Versicolor",
            2: "Iris Virginica"
        }

        st.success(f"🌼 Predicted Species: **{species_map[prediction]}**")
        st.info(f"Confidence: **{probability * 100:.2f}%**")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
