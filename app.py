import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Try-except block to handle missing library gracefully in UI
try:
    import xgboost as xgb
except ImportError:
    st.error("The 'xgboost' library is not installed. Please ensure 'xgboost' is listed in your requirements.txt file.")

st.set_page_config(page_title="Academic Insight", page_icon="🎓", layout="wide")

# Smooth Aesthetic Styling
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .stNumberInput, .stSelectbox, .stSlider {
        background-color: white;
        border-radius: 10px;
    }
    .main-card {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        with open('XG_Boost.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file 'XG_Boost.pkl' not found!")
        return None

model = load_model()

st.title("🎓 Student Performance Analytics")
st.write("Leverage machine learning to predict student academic outcomes.")

if model:
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown('<div class="main-card">', unsafe_allow_html=True)
            
            # Feature Inputs
            gndr = st.radio("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female", horizontal=True)
            age = st.slider("Student Age", 10, 25, 18)
            hrs = st.number_input("Weekly Study Hours", 0, 100, 15)
            attn = st.slider("Attendance Rate (%)", 0.0, 100.0, 85.0)
            edu = st.select_slider("Parent Education Level", options=[0, 1, 2, 3])
            net = st.toggle("Internet Access", value=True)
            extra = st.toggle("Extracurricular Activities")
            prev = st.number_input("Previous Academic Score", 0, 100, 70)
            final = st.number_input("Current Final Score", 0, 100, 75)
            
            if st.button("Predict Outcome", use_container_width=True):
                # Prepare data in the exact order found in pkl metadata [cite: 12]
                input_data = np.array([[
                    gndr, age, hrs, attn, edu, 
                    int(net), int(extra), prev, final
                ]])
                
                prediction = model.predict(input_data)[0]
                prob = model.predict_proba(input_data)[0][1]
                
                st.divider()
                if prediction == 1:
                    st.success(f"### Result: Positive Academic Standing")
                    st.progress(prob, text=f"Confidence: {prob:.1%}")
                else:
                    st.warning(f"### Result: Academic Intervention Recommended")
                    st.progress(1-prob, text=f"Risk Probability: {1-prob:.1%}")
            
            st.markdown('</div>', unsafe_allow_html=True)
