import streamlit as st
import pandas as pd
import pickle
import os

# Page configuration for a clean, professional dashboard
st.set_page_config(
    page_title="Academic Performance Analytics",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("Academic Performance Predictor")
st.caption("Enterprise grading inference engine driven by a trained Random Forest configuration.")
st.markdown("---")

# Verify model presence before execution
if not os.path.exists('student_model.pkl'):
    st.error("Execution Error: 'student_model.pkl' target file not found.")
else:
    # Load the trained model artifact
    with open('student_model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)

    st.subheader("Input Parameters")
    
    # Clean two-column layout for parameter adjustment
    col1, col2 = st.columns(2)
    with col1:
        study_hours = st.slider("Weekly Study Hours", min_value=1, max_value=10, value=6)
        attendance = st.slider("Attendance Rate (%)", min_value=50, max_value=100, value=80)
    with col2:
        assignment_score = st.slider("Assignment Score (0-100)", min_value=30, max_value=100, value=70)
        internal_marks = st.number_input("Internal Assessment Marks (0-20)", min_value=0, max_value=20, value=12)

    st.markdown("---")

    # Prediction execution block
    if st.button("Run Evaluation Profile", use_container_width=True):
        # Format the features array array matches training layout
        input_data = [[study_hours, attendance, assignment_score, internal_marks]]
        
        # Inference output extraction
        predicted_grade = loaded_model.predict(input_data)[0]
        
        st.subheader("Analysis Metrics Output")
        
        # Display the result using a clean, professional metric card component
        st.metric(
            label="Predicted Final Target Class", 
            value=f"Grade {predicted_grade}"
        )
        
        # Contextual summary footer message
        st.info(f"System Confirmation: Model successfully computed output matrix mapping to classification category {predicted_grade}.")
