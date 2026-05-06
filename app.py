import streamlit as st
import numpy as np
import joblib

st.title("Multi Disease Prediction")

# Load models
heart_model = joblib.load("models/heart.pkl")
dia_model = joblib.load("models/diabetes.pkl")
liv_model = joblib.load("models/liver.pkl")

st.header("Enter Patient Details")

#  ALL FEATURES COMBINED 
age = st.number_input("Age", value=None)
gender = st.selectbox("Gender", ["Male", "Female"])
bp = st.number_input("Blood Pressure (trestbps)", value=None)
chol = st.number_input("Cholesterol", value=None)
thalach = st.number_input("Max Heart Rate (thalach)", value=None)
oldpeak = st.number_input("Oldpeak", value=None)

glucose = st.number_input("Glucose", value=None)
bmi = st.number_input("BMI", value=None)

bilirubin = st.number_input("Total Bilirubin", value=None)
alk = st.number_input("Alkaline Phosphotase", value=None)
albumin = st.number_input("Albumin", value=None)

# Convert gender
gender_val = 0 if gender == "Male" else 1

def format_output(value):
    if value == 0:
        return "Insufficient Data"
    elif value < 50:
        return "Normal"
    else:
        return f"{value:.2f}% Risk"

# PREDICT
if st.button("Predict"):

    # HEART
    if None not in [age, bp, chol, thalach, oldpeak]:
        heart_input = np.array([[age, gender_val, bp, chol, thalach, oldpeak]])
        heart_pred = heart_model.predict_proba(heart_input)[0][1]*100
    else:
        heart_pred = 0

    # DIABETES
    if None not in [age, glucose, bp, bmi]:
        dia_input = np.array([[age, glucose, bp, bmi]])
        dia_pred = dia_model.predict_proba(dia_input)[0][1]*100
    else:
        dia_pred = 0

    # LIVER
    if None not in [age, bilirubin, alk, albumin]:
        liv_input = np.array([[age, gender_val, bilirubin, alk, albumin]])
        liv_pred = liv_model.predict_proba(liv_input)[0][1]*100
    else:
        liv_pred = 0

    # OUTPUT
    st.subheader("Results")
    
    st.write(f" Heart Disease: {format_output(heart_pred)}")
    st.write(f" Diabetes: {format_output(dia_pred)}")
    st.write(f" Liver Disease: {format_output(liv_pred)}")