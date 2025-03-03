import streamlit as st
import joblib
import re
import string

# Load Model and Vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Function to Clean Text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply Custom CSS for Modern UI
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #141E30, #243B55);
        color: white;
        font-family: 'Arial', sans-serif;
    }
    .stApp {
        background: linear-gradient(to right, #1F1C2C, #928DAB);
        padding: 30px;
        border-radius: 10px;
    }
    .stTextArea textarea {
        border-radius: 10px;
        font-size: 16px;
    }
    .stButton button {
        background-color: #ff4b4b;
        color: white;
        font-size: 18px;
        padding: 10px;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title("📝 Sentiment Analysis Web App")
st.write("🔍 Enter a review below and get an AI-powered sentiment prediction.")

# Input Text
user_input = st.text_area("✍️ Enter your review:")

if st.button("Analyze Sentiment"):
    if user_input:
        cleaned_text = preprocess_text(user_input)
        transformed_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(transformed_text)[0]

        # Map Sentiment Labels
        sentiment_labels = {0: "😞 Negative", 1: "😊 Positive", 2: "😐 Neutral"}
        st.subheader(f"📌 Prediction: {sentiment_labels[prediction]}")
    else:
        st.warning("⚠️ Please enter a review before analyzing.")