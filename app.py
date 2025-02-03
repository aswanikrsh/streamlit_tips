import streamlit as st
import pandas as pd
import seaborn as sns
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

with open('tip_prediction_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

le = LabelEncoder()
le.fit(['Lunch', 'Dinner'])

st.title('Tip Prediction App')

total_bill = st.number_input('Enter Total Bill ($)', min_value=0.0, step=0.01)
time = st.selectbox('Select Time', ['Lunch', 'Dinner'])
size = st.number_input('Enter Number of People (Size)', min_value=1, step=1)

if st.button('Predict Tip'):
    try:
        time_encoded = le.transform([time])[0]

        input_data = pd.DataFrame([[total_bill, time_encoded, size]], columns=['total_bill', 'Time', 'size'])

        predicted_tip = model.predict(input_data)
       
        st.write(f"Predicted Tip: ${predicted_tip[0]:.2f}")
    
    except Exception as e:
        st.error(f"Error: {e}")
        