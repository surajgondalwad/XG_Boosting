import streamlit as st
import pandas as pd
import pickle
import numpy as np
import xgboost as xgb

# Page configuration
st.set_page_config(page_title="Student Success Predictor", layout="centered")

# Custom CSS for a clean, aesthetic look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stSelectbox, .stNumberInput, .stSlider {
        margin-bottom: 10px;
    }
    div.stButton > button:first-child {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #0056b3;
        color: white;
    }
    .result-card {
        padding: 25px;
        border-radius: 12px;
        background-color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# Load the model with error handling
@st.cache_resource
def load_model():
    try:
        with open('XG_Boost.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model:
    st.title("🎓 Student Performance Analytics")
    st.write("Predict academic success based on student metrics.")
    
    with st.container():
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            # Map inputs to the feature names found in the pkl metadata [cite: 12, 13]
            gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
            age = st.number_input("Age", min_value=12, max_value=25, value=18)
            study_hours = st.number_input("Study Hours/Week", min_value=0, max_value=100, value=20)
            parent_edu = st.selectbox("Parent Education Level", options=[0, 1, 2, 3, 4], help="Higher values indicate higher education")
            internet = st.selectbox("Internet Access", options=[0, 1], format_func=lambda x: "Available" if x == 1 else "No Access")

        with col2:
            attendance = st.slider("Attendance Rate (%)", 0.0, 100.0, 90.0)
            extracurricular = st.selectbox("Extracurriculars", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            prev_score = st.number_input("Previous Score", min_value=0, max_value=100, value=75)
            final_score = st.number_input("Final Score", min_value=0, max_value=100, value=80)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Generate Prediction"):
        # Features must be in this specific order to match the model [cite: 12]
        features = np.array([[
            gender, age, study_hours, attendance, 
            parent_edu, internet, extracurricular, 
            prev_score, final_score
        ]])
        
        # Binary prediction 
        prediction = model.predict(features)[0]
        # Probability based on binary:logistic objective [cite: 10, 11]
        probability = model.predict_proba(features)[0][1]

        st.markdown("---")
        if prediction == 1:
            st.success(f"### High Performance Likely")
            st.metric("Confidence Level", f"{probability:.1%}")
            st.balloons()
        else:
            st.warning(f"### Potential Academic Risk")
            st.metric("Confidence Level", f"{1-probability:.1%}")
    else:
        st.info("Fill in the student metrics above and click 'Generate Prediction'.")
