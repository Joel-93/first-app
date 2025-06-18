import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('car.pkl', 'rb') as f:
    model = pickle.load(f)

# App title and description
st.set_page_config(page_title="🚗 Highway MPG Predictor", layout="centered")
st.title("🚗 Highway Fuel Efficiency Predictor")
st.markdown("This app predicts a car's highway MPG based on encoded features.")

# Input form
st.subheader("Enter Encoded Car Features")
new_make = st.number_input("Make (Encoded)", min_value=0, step=1)
new_model = st.number_input("Model (Encoded)", min_value=0, step=1)
city_mpg = st.number_input("City MPG", min_value=1)
combination_mpg = st.number_input("Combined MPG", min_value=1)

# Predict
if st.button("Predict Highway MPG"):
    input_data = pd.DataFrame([[new_make, new_model, city_mpg, combination_mpg]],
                              columns=['new_make', 'new_model', 'city_mpg', 'combination_mpg'])
    prediction = model.predict(input_data)
    st.success(f"Predicted Highway MPG: **{prediction[0]:.2f}**")

# Footer
st.markdown("---")
st.caption("Model trained using Linear Regression on cleaned car dataset.")
