# =====================================================================
# PROJECT FRONTEND: app.py (The Interactive Simple Web UI Portal)
# =====================================================================

import streamlit as st
import pandas as pd
import pickle
import os

# 1. Webpage UI configurations
st.set_page_config(page_title="Grade Predictor Engine", page_icon="🎓", layout="centered")

st.title("🎓 Student Performance Predictor")
st.write("Drag the sliders below to adjust a student's attributes and click predict to calculate their final letter grade.")
st.markdown("---")

# 2. Check if your trained model file exists before running
if not os.path.exists('student_model.pkl'):
    st.error("❌ Error: 'student_model.pkl' not found! Make sure it is in the exact same folder as this app.py file.")
else:
    # Load your existing trained model brain
    with open('student_model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)

    # 3. User Input Layout Form
    st.subheader("📊 Adjust Student Attributes")
    
    col1, col2 = st.columns(2)
    with col1:
        study_hours = st.slider("Weekly Study Hours", min_value=1, max_value=10, value=6)
        attendance = st.slider("Attendance Rate (%)", min_value=50, max_value=100, value=80)
    with col2:
        assignment_score = st.slider("Assignment Score (0-100)", min_value=30, max_value=100, value=70)
        internal_marks = st.number_input("Internal Assessment Marks (0-20)", min_value=0, max_value=20, value=12)

    st.markdown("---")

    # 4. Run Prediction Logic On Click
    if st.button("🔮 Predict Final Grade Summary", use_container_width=True):
        # Format inputs exactly how your model expects it
        user_input_features = [[study_hours, attendance, assignment_score, internal_marks]]
        
        # Query your AI model
        prediction_output = loaded_model.predict(user_input_features)[0]
        
        # 5. Show Beautiful Visual Cards based on the resulting grade
        st.subheader("🎯 Prediction Analysis Summary:")
        
        if prediction_output == 'A':
            st.success(f"### 🎉 Spectacular! The model predicts: **Grade {prediction_output}**")
            st.balloons()
        elif prediction_output == 'B':
            st.info(f"### 👍 Solid Performance! The model predicts: **Grade {prediction_output}**")
        else:
            st.warning(f"### ⚠️ At Risk! The model predicts: **Grade {prediction_output}**")
            st.caption("💡 *Tip: Helping this student improve attendance and internal marks will significantly push them into Grade B or A territory.*")