import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import pickle
import os
import yaml

# Function to load models
def load_models(model_folder):
    models = {}
    for filename in os.listdir(model_folder):
        if filename.endswith(".sav"):
            model_name = filename.split(".")[0]
            models[model_name] = joblib.load(os.path.join(model_folder, filename))
    return models

# Load the models
model_folder = "D:/Fake News Detection/models"
models = load_models(model_folder)

# Set the title and sidebar title
st.title("📰 Fake News Detection")
st.sidebar.title("🛠️ Choose Model")

# Add some description and instructions
st.write("Welcome to the **Fake News Detection** app! This tool helps you identify whether a piece of news is fake or real using machine learning models. Select a model from the sidebar and enter the news text below to get a prediction.")

# Sidebar model selection
model_name = st.sidebar.selectbox("Select a model for prediction", list(models.keys()))

# Display selected model
st.write(f"### Selected Model: **{model_name}**")

# Load the vectorizer
with open('vectorizer.pkl', 'rb') as f:
    loaded_vectorizer = pickle.load(f)

# Get user input for prediction
user_input = st.text_area("🖊️ Enter input for news text:")

if st.button("🔍 Predict"):
    if user_input:
        data = [user_input]
        inputs = loaded_vectorizer.transform(data)
        model = models[model_name]
        predictions = model.predict(inputs)
        if predictions[0] == 0:
            st.write(f"### Prediction: 🚫 **Fake News**")
        else:
            st.write(f"### Prediction: ✅ **Real News**")
    else:
        st.write("⚠️ Please enter input for prediction")

# Add a footer with some links or additional information
st.markdown("""
---
*Developed by [Joseph Mwangi](https://www.linkedin.com/in/josephmwangiboyle/)*
""")
