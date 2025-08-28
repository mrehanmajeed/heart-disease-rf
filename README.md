# heart-disease-rf

# ❤️ Heart Disease Prediction App

An interactive **Streamlit** web application that predicts the likelihood of heart disease using a **Random Forest Classifier** trained on real patient medical data.

## 📜 Project Overview
This project implements an **end‑to‑end machine learning pipeline**:
1. **Data Preprocessing**  
   - Handles missing/zero values in key features.  
   - Label encoding for categorical features.  
   - Feature scaling for numerical values.

2. **Model Training**  
   - Uses a **Random Forest Classifier** to learn patterns from the data.  
   - Evaluates performance with accuracy, classification reports, and confusion matrices.

3. **Model Saving**  
   - Stores the trained model, scaler, selected features, and label encoders using `pickle`.  
   - Makes them available for real‑time predictions in the web app.

4. **Streamlit App**  
   - User‑friendly form to input patient details.  
   - Encodes and scales inputs exactly as done during training.  
   - Displays **predicted risk** (Low or High) along with **model confidence score**.

---

## 🖥️ How the App Works
- **Step 1:** User enters clinical parameters (age, cholesterol, chest pain type, etc.).  
- **Step 2:** App applies **label encoding** to categorical fields using the saved encoders.  
- **Step 3:** App **scales** the numeric values using the saved scaler.  
- **Step 4:** The preprocessed input is fed to the **Random Forest model**.  
- **Step 5:** Output is shown as either:
  - ✅ *Low Risk of Heart Disease*  
  - ⚠️ *High Risk of Heart Disease*  
  With **confidence percentage**.

---

## 📂 Repository Structure
. ├── app.py # Streamlit application 
  ├── random_forest_heart_model.pkl # Trained Random Forest model 
  ├── scaler.pkl # StandardScaler fitted on training data 
  ├── selected_features.pkl # Features chosen during training 
  ├── label_encoders.pkl # LabelEncoders for categorical variables 
  ├── heart.csv # (Optional) Dataset used for training 
  └── README.md # Project documentation

  
---

## 📦 Dependencies
Python 3.8+ recommended. Install required packages with:
pip install streamlit pandas scikit-learn matplotlib seaborn

##  Running the app
streamlit run app.py

## Model Details
Algorithm: Random Forest Classifier (n_estimators=100, random_state=42)
Preprocessing
        Scaling: StandardScaler
        Encoding: LabelEncoder
Evaluation Metrics: Accuracy, Classification Report, Confusion Matrix
Model Persistence: pickle
