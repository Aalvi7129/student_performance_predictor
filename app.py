import streamlit as st
import pandas as pd
import joblib
import os

# Page Configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# Load the trained model pipeline
@st.cache_resource
def load_model():
    model_path = os.path.join("models", "best_model.pkl")
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

pipeline = load_model()

# Title and Description
st.title("🎓 Student Final Exam Score Predictor")
st.markdown("""
This interactive web application uses a trained **Random Forest Machine Learning Pipeline** to predict a student's **Final Exam Score** based on study habits, attendance, and academic history.
""")

if pipeline is None:
    st.error("⚠️ Trained model artifact not found at `models/best_model.pkl`. Please run `python src/train.py` first!")
else:
    st.divider()
    st.subheader("📝 Input Student Parameters")

    # Layout inputs in two columns
    col1, col2 = st.columns(2)

    with col1:
        study_hours = st.slider("Weekly Study Hours", 0.0, 20.0, 5.0, 0.5)
        attendance_pct = st.slider("Attendance Percentage (%)", 0.0, 100.0, 75.0, 1.0)
        previous_score = st.slider("Previous Exam Score", 0.0, 100.0, 70.0, 1.0)
        assignments_completed = st.slider("Assignments Completed (%)", 0.0, 100.0, 80.0, 1.0)
        sleep_hours = st.slider("Average Sleep Hours", 2.0, 12.0, 7.0, 0.5)

    with col2:
        extracurricular = st.slider("Extracurricular Hours", 0.0, 20.0, 3.0, 0.5)
        class_participation = st.slider("Class Participation Score", 0.0, 10.0, 5.0, 0.5)
        previous_backlogs = st.number_input("Previous Backlogs", 0, 10, 0)
        # (Post-Exam Confidence slider removed to prevent data leakage)

    # Collect inputs into a DataFrame matching model training features
    input_data = pd.DataFrame({
        "StudyHours": [study_hours],
        "AttendancePercentage": [attendance_pct],
        "PreviousExamScore": [previous_score],
        "AssignmentsCompleted": [assignments_completed],
        "SleepHours": [sleep_hours],
        "ExtracurricularHours": [extracurricular],
        "ClassParticipation": [class_participation],
        "PreviousBacklogs": [previous_backlogs]
    })

    st.divider()

    # Prediction button
    if st.button("🚀 Predict Final Exam Score", type="primary", use_container_width=True):
        prediction = pipeline.predict(input_data)[0]
        
        # Display Result Metric
        st.success(f"### Predicted Final Exam Score: **{prediction:.2f} / 100**")
        
        # Performance tier feedback
        if prediction >= 85:
            st.balloons()
            st.info("🌟 **Top Tier Performance:** This student is on track for an outstanding grade!")
        elif prediction >= 60:
            st.info("👍 **Passing Performance:** Solid standing, with room for minor optimizations in study habits.")
        else:
            st.warning("⚠️ **At Risk:** Early intervention, increased study hours, and tutoring recommended.")