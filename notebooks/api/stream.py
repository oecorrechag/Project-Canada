import streamlit as st
import requests

# URL FastAPI
API_URL = "http://127.0.0.1:8000/predict/"

# Title
st.title("Film prediction")
summary = st.text_area("Please enter your text here:")

# Botón
if st.button("Predict"):
    # dictionary to send predict api
    input_data = {"summary": summary} 
    # enviar a fastapi
    response = requests.post(API_URL, json=input_data)
    prediction = response.json()
    st.write(f"Film prediction: {prediction[0]}")
