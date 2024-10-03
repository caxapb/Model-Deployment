# streamlit_app.py
import streamlit as st
import requests

# FastAPI endpoint
FASTAPI_URL = "http://fastapi:80/predict"

# Streamlit app UI
st.title("Mushroom Classifier")

# Input fields for the Iris flower data
cap_diameter = st.number_input("Cap diameter", min_value=0.0)
cap_shape = st.number_input("Cap shape", min_value=0.0)
gill_attachment = st.number_input("Gill attachment", min_value=0.0)
gill_color = st.number_input("Gill color", min_value=0.0)
stem_height = st.number_input("Stem height", min_value=0.0)
stem_width = st.number_input("Stem width", min_value=0.0)
stem_color = st.number_input("Stem color", min_value=0.0)
season = st.number_input("Season", min_value=0.0)

## cap-diameter,cap-shape,gill-attachment,gill-color,stem-height,stem-width,stem-color,season,class
## 1372,2,2,10,3.8074667544799388,1545,11,1.8042727086281731,1



# Make prediction when the button is clicked
if st.button("Predict"):
    # Prepare the data for the API request
    input_data = {
        "cap_diameter": cap_diameter,
        "cap_shape": cap_shape,
        "gill_attachment": gill_attachment,
        "gill_color": gill_color,
        "stem_height": stem_height,
        "stem_width": stem_width,
        "stem_color": stem_color,
        "season": season
        }
    
    # Send a request to the FastAPI prediction endpoint
    response = requests.post(FASTAPI_URL, json=input_data)
    prediction = response.json()["prediction"]
    
    # Display the result
    st.success(f"The model predicts class: {prediction}")