#importing Necessary Libraries 
import numpy as np
import pickle as pkl 
import streamlit as st

# Load trained model
model = pkl.load(open('MIPML.pkl', 'rb'))

st.header('Medical Insurance Premium Predictor')

# Input fields
gender = st.selectbox('Choose Gender', ['Female','Male'])
smoker = st.selectbox('Are you a smoker ?', ['Yes','No'])
region = st.selectbox('Choose Region', ['SouthEast','SouthWest','NorthEast','NorthWest'])
age = st.slider('Enter Age', 5, 80)

# BMI Calculator (only BMI will be used in dataset)
st.subheader("BMI Calculator")
weight = st.number_input("Enter Weight (kg)", min_value=10.0, max_value=300.0, step=0.5)
height = st.number_input("Enter Height (cm)", min_value=50.0, max_value=250.0, step=0.5)

if height > 0:
    bmi = weight / ((height/100) ** 2)
    st.write(f"Your calculated **BMI** is: `{round(bmi, 2)}`")
else:
    bmi = 0
    st.warning("Please enter a valid height to calculate BMI.")

# Prediction button
if st.button('Predict'):
    # Encoding categorical variables
    gender = 0 if gender == 'Female' else 1
    smoker = 1 if smoker == 'Yes' else 0
    
    if region == 'SouthEast':
        region = 0
    elif region == 'SouthWest':
        region = 1
    elif region == 'NorthEast':
        region = 2
    else:
        region = 3   # NorthWest

    # Only dataset features: (age, gender, bmi, smoker, region)
    input_data = (age, gender, bmi, smoker, region)
    input_data_array = np.asarray(input_data).reshape(1, -1)

    # Prediction
    predicted_prem = model.predict(input_data_array)

    display_string = '💰 Estimated Insurance Premium: **' + str(round(predicted_prem[0], 2)) + ' USD**'
    st.success(display_string)
