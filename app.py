import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Student Dropout Risk Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Dropout Risk Prediction")

st.sidebar.header("📋 Project Information")

st.sidebar.info(
    """
    **Student Dropout Risk Prediction**

    This web application predicts the likelihood of a student dropping out using Machine Learning.

    **Model**
    - Random Forest Classifier

    **Libraries**
    - Scikit-learn
    - Pandas
    - Streamlit
    - Joblib
    """
)

st.write(
    "This application predicts whether a student is at risk of dropping out using a Random Forest Machine Learning model."
)

# Load trained pipeline
pipeline = joblib.load("models/student_dropout_pipeline.pkl")

st.success("✅ Machine Learning Pipeline Loaded Successfully!")

st.header("Student Information")

# Create two columns
col1, col2 = st.columns(2)

# -----------------------------
# Left Column
# -----------------------------
with col1:

    age = st.number_input(
        "Age",
        min_value=16,
        max_value=40,
        value=20
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male", "Other"]
    )

    year_of_study = st.number_input(
        "Year of Study",
        min_value=1,
        max_value=6,
        value=2
    )

    department = st.selectbox(
        "Department",
        [
            "Arts",
            "Business",
            "Engineering",
            "Law",
            "Medicine",
            "Science"
        ]
    )

    residence = st.selectbox(
        "Residence Type",
        [
            "Day Scholar",
            "Hostel",
            "PG/Rented"
        ]
    )

    attendance = st.slider(
        "Attendance Percentage",
        0.0,
        100.0,
        75.0
    )

    study_hours = st.slider(
        "Study Hours Per Day",
        0.0,
        15.0,
        4.0
    )

    gpa = st.number_input(
        "Previous GPA",
        min_value=0.0,
        max_value=4.0,
        value=3.0
    )

    backlogs = st.number_input(
        "Backlogs",
        min_value=0,
        max_value=20,
        value=0
    )

    sleep = st.slider(
        "Sleep Hours",
        0.0,
        12.0,
        7.0
    )

    screen_time = st.slider(
        "Screen Time (Hours)",
        0.0,
        15.0,
        5.0
    )

    exercise = st.slider(
        "Exercise Frequency Per Week",
        0,
        7,
        3
    )



# -----------------------------
# Right Column
# -----------------------------
with col2:

    social = st.slider(
        "Social Activity Score",
        0,
        10,
        5
    )

    part_time = st.selectbox(
        "Part-Time Job",
        ["No", "Yes"]
    )

    income = st.selectbox(
        "Family Income Bracket",
        [
            "High",
            "Low",
            "Lower-Middle",
            "Middle",
            "Upper-Middle"
        ]
    )

    financial = st.slider(
        "Financial Stress Score",
        0,
        10,
        5
    )

    family_support = st.slider(
        "Family Support Score",
        0,
        10,
        5
    )

    stress = st.slider(
        "Stress Level",
        0,
        10,
        5
    )

    anxiety = st.slider(
        "Anxiety Score",
        0,
        10,
        5
    )

    motivation = st.slider(
        "Motivation Score",
        0,
        10,
        5
    )

    peer = st.slider(
        "Peer Pressure Score",
        0,
        10,
        5
    )

    counseling = st.selectbox(
        "Counseling Access",
        ["No", "Yes"]
    )

    burnout = st.selectbox(
        "Burnout Level",
        [
            "High",
            "Low",
            "Medium"
        ]
    )

    st.divider()

if st.button("Predict Dropout Risk"):

    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Year_of_Study": [year_of_study],
        "Department": [department],
        "Residence_Type": [residence],
        "Attendance_Percent": [attendance],
        "Study_Hours_Per_Day": [study_hours],
        "Previous_GPA": [gpa],
        "Backlogs": [backlogs],
        "Sleep_Hours": [sleep],
        "Screen_Time_Hours": [screen_time],
        "Exercise_Freq_Per_Week": [exercise],
        "Social_Activity_Score": [social],
        "Part_Time_Job": [part_time],
        "Family_Income_Bracket": [income],
        "Financial_Stress_Score": [financial],
        "Family_Support_Score": [family_support],
        "Stress_Level": [stress],
        "Anxiety_Score": [anxiety],
        "Motivation_Score": [motivation],
        "Peer_Pressure_Score": [peer],
        "Counseling_Access": [counseling],
        "Burnout_Level": [burnout]
    })

    prediction = pipeline.predict(input_data)[0]
    probability = pipeline.predict_proba(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("🚨 High Dropout Risk")
    else:
        st.success("✅ Low Dropout Risk")

    st.subheader("Prediction Probabilities")

    st.write(f"🟢 Low Risk: {probability[0]:.2%}")
    st.write(f"🔴 High Risk: {probability[1]:.2%}")

    confidence = max(probability)

    st.progress(float(confidence))

    st.write(f"Model Confidence: {confidence:.2%}")

    st.subheader("Input Summary")
    st.dataframe(input_data)

with st.expander("ℹ️ About the Model"):

    st.write("""
    This application predicts whether a student is at risk of dropping out
    based on academic, lifestyle, and psychological factors.

    **Machine Learning Algorithm**
    - Random Forest Classifier

    **Preprocessing**
    - Missing values handled using SimpleImputer
    - Categorical features encoded using OneHotEncoder
    - Numerical and categorical preprocessing managed with ColumnTransformer
    - Entire workflow combined using a Scikit-learn Pipeline

    **Deployment**
    - Built with Streamlit
    - Model serialized using Joblib
    """)

st.markdown("---")
st.caption("Developed by Sobia Asif | Machine Learning Portfolio Project")