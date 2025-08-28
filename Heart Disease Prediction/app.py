import streamlit as st
import pandas as pd
import pickle

# ==================== LOAD SAVED OBJECTS ====================
model = pickle.load(open("random_forest_heart_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
selected_features = pickle.load(open("selected_features.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

st.title("Heart Health Prediction ❤️")
st.markdown("Provide the following details:")

# ==================== USER INPUTS ====================
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ['M', 'F'])
chest_pain = st.selectbox("Chest Pain Type", ['ATA', "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 0, 250, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 0, 700, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ['Normal', "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", ['Y', "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# ==================== PREDICTION ====================
if st.button("Predict"):
    # Build input DataFrame
    input_dict = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex': sex,
        'ChestPainType': chest_pain,
        'RestingECG': resting_ecg,
        'ExerciseAngina': exercise_angina,
        'ST_Slope': st_slope
    }
    input_df = pd.DataFrame([input_dict])

    # Safe label encoding
    for col, le in label_encoders.items():
        if input_df[col].iloc[0] not in le.classes_:
            le_classes = list(le.classes_)
            le_classes.append(input_df[col].iloc[0])
            le.classes_ = le_classes
        input_df[col] = le.transform(input_df[col])

    # Keep only training features
    input_df = input_df[selected_features]

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Predict and get probability
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][prediction]  # confidence of predicted class

    # Show result with confidence
    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease — Confidence: {probability:.2%}")
    else:
        st.success(f"✅ Low Risk of Heart Disease — Confidence: {probability:.2%}")
