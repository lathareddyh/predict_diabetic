# Load the Libraries
import numpy as np
import joblib
import streamlit as st

#Load the trained model
model = joblib.load('model_scaled.pkl')
scale = joblib.load('scaled.pkl')

#streamlit app title 

st.title('Diabetic Prediction')
st.write('Enter your Medical details to know about your diabetic status')

#Define the input fields
st.sidebar.header('Your Medical Records')

preg = st.sidebar.number_input('Pregnancies', min_value=0.0, max_value=1000.0, value=0.0, step=1.0)
plas = st.sidebar.number_input('Glucose', min_value=0.0, max_value=1000.0, value=50.0, step=0.1)
pres = st.sidebar.number_input('BloodPressure', min_value=0.0, max_value=1000.0, value=50.0, step=0.1)
skin = st.sidebar.number_input('SkinThickness', min_value=0.0, max_value=1000.0, value=50.0, step=0.1)
test = st.sidebar.number_input('Insulin', min_value=0.0, max_value=1000.0, value=50.0, step=0.1)
mass = st.sidebar.number_input('BMI', min_value=0.0, max_value=1000.0, value=50.0, step=0.1)
pedi = st.sidebar.number_input('DiabetesPedigreeFunction', min_value=0.0, max_value=1000.0, value=50.0, step=0.1)
age = st.sidebar.number_input('Age', min_value=0.0, max_value=1000.0, value=50.0, step=0.1)

input_data = np.array([[preg,plas,pres,skin,test,mass,pedi,age]])
scaled_input = scale.transform(input_data)

if st.sidebar.button('Predict'):
    # Model Prediction
    prediction = model.predict(scaled_input)
    if prediction[0] == 0:
        st.success('You are Not Diabetic')
    else:
        # st.success('You are Diabetic')
        # Custom CSS for a red/orange alert box
        st.markdown("""
        <style>
        .diabetic-alert {
            background-color: #FF4B4B; /* A shade of red */
            color: white;
            padding: 15px;
            border-radius: 10px;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown('<div class="diabetic-alert">You are diabetic.</div>', unsafe_allow_html=True)