# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle

# # Load the trained model
# model_file = 'pipe.pkl'
# with open(model_file, 'rb') as file:
#     model = pickle.load(file)

# # Function to preprocess input data
# def preprocess_input(age, gender, heart_rate, sys_bp, dia_bp, blood_sugar, ck_mb, troponin):
#     # Preprocessing steps here (e.g., scaling, encoding categorical variables)
#     # For simplicity, we'll just return the input data as is
#     return np.array([[age, gender, heart_rate, sys_bp, dia_bp, blood_sugar, ck_mb, troponin]])

# # Function to make predictions
# def predict_heart_attack(age, gender, heart_rate, sys_bp, dia_bp, blood_sugar, ck_mb, troponin):
#     input_data = preprocess_input(age, gender, heart_rate, sys_bp, dia_bp, blood_sugar, ck_mb, troponin)
#     prediction = model.predict(input_data)
#     return prediction[0]

# # Streamlit app
# def main():
#     st.title('Heart Attack Predictor')

#     # Input features
#     age = st.slider('Age', min_value=18, max_value=100, value=50)
#     gender = st.radio('Gender', ['Male', 'Female'])
#     heart_rate = st.slider('Heart Rate', min_value=40, max_value=200, value=75)
#     sys_bp = st.slider('Systolic Blood Pressure', min_value=80, max_value=220, value=120)
#     dia_bp = st.slider('Diastolic Blood Pressure', min_value=40, max_value=150, value=80)
#     blood_sugar = st.slider('Blood Sugar', min_value=60, max_value=300, value=100)
#     ck_mb = st.slider('CK-MB', min_value=0, max_value=100, value=0)
#     troponin = st.slider('Troponin', min_value=0.0, max_value=10.0, value=0.0)

#     # Convert gender to numerical value
#     gender_num = 1 if gender == 'Male' else 0

#     if st.button('Predict'):
#         prediction = predict_heart_attack(age, gender_num, heart_rate, sys_bp, dia_bp, blood_sugar, ck_mb, troponin)
#         if prediction == 1:
#             st.error('Risk of heart attack: Positive')
#         else:
#             st.success('Risk of heart attack: Negative')

# if __name__ == '__main__':
#     main()
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
model_file = 'pipe.pkl'
with open(model_file, 'rb') as file:
    model = pickle.load(file)

# Function to preprocess input data
def preprocess_input(age, gender, heart_rate, sys_bp, dia_bp, blood_sugar, ck_mb, troponin):
    # Preprocessing steps here (e.g., scaling, encoding categorical variables)
    # For simplicity, we'll just return the input data as is
    return np.array([[age, gender, heart_rate, sys_bp, dia_bp, blood_sugar, ck_mb, troponin]])

# Function to make predictions
def predict_heart_attack(age, gender, heart_rate, sys_bp, dia_bp, blood_sugar, ck_mb, troponin):
    input_data = preprocess_input(age, gender, heart_rate, sys_bp, dia_bp, blood_sugar, ck_mb, troponin)
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit app
def main():
    st.title('Heart Attack Predictor')

    # Input features
    age = st.slider('Age', min_value=18, max_value=100, value=50)
    gender = st.radio('Gender', ['Male', 'Female'])
    heart_rate = st.slider('Heart Rate', min_value=40, max_value=200, value=75)
    sys_bp = st.slider('Systolic Blood Pressure', min_value=80, max_value=220, value=120)
    dia_bp = st.slider('Diastolic Blood Pressure', min_value=40, max_value=150, value=80)
    blood_sugar = st.slider('Blood Sugar', min_value=60, max_value=300, value=100)
    ck_mb = st.slider('CK-MB', min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    troponin = st.slider('Troponin', min_value=0.000, max_value=10.000, value=0.000, step=0.001)

    # Convert gender to numerical value
    gender_num = 1 if gender == 'Male' else 0

    if st.button('Predict'):
        if ck_mb == 0.0 or troponin == 0.0:
            st.warning('Please enter all input features')
        else:
            prediction = predict_heart_attack(age, gender_num, heart_rate, sys_bp, dia_bp, blood_sugar, ck_mb, troponin)
            if prediction == 1:
                st.error('Risk of heart attack: Positive')
            else:
                st.success('Risk of heart attack: Negative')

if __name__ == '__main__':
    main()
