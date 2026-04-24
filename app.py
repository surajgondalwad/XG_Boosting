import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page configuration
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

# Custom CSS for a clean, aesthetic look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    with open('XG_Boost.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

st.title("🎓 Student Performance Predictor")
st.markdown("Enter the student details below to predict academic outcomes.")

# Creating columns for a smoother layout
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    age = st.number_input("Age", min_value=10, max_value=25, value=18)
    study_hours = st.number_input("Study Hours Per Week", min_value=0, max_value=168, value=15)
    parent_edu = st.selectbox("Parent Education Level", options=[0, 1, 2, 3], help="0: Low to 3: High")
    internet = st.selectbox("Internet Access", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    attendance = st.slider("Attendance Rate (%)", 0.0, 100.0, 85.0)
    extracurricular = st.selectbox("Extracurricular Activities", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    prev_score = st.number_input("Previous Score", min_value=0, max_value=100, value=75)
    final_score = st.number_input("Current/Final Score", min_value=0, max_value=100, value=80)

# Prediction Logic
if st.button("Analyze Performance"):
    # Prepare the feature array based on the model's expected order
    features = np.array([[
        gender, age, study_hours, attendance, 
        parent_edu, internet, extracurricular, 
        prev_score, final_score
    ]])
    
    # Getting prediction and probability
    prediction = model.predict(features)[0]
    # The pkl shows binary:logistic, so we can get probabilities 
    probability = model.predict_proba(features)[0][1]

    st.markdown("---")
    
    # Displaying results in a clean card
    if prediction == 1:
        st.success(f"### High Success Probability: {probability:.2%}")
        st.balloons()
    else:
        st.warning(f"### At-Risk Probability: {1 - probability:.2%}")
        st.info("Recommendation: Consider academic intervention or mentorship.")
