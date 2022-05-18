# -*- coding: utf-8 -*-
"""
Created on Tue May 17 23:34:03 2022

@author: End User
"""

#%%packages
from tensorflow.keras.models import load_model
import pickle
import os
import numpy as np
import streamlit as st

# static values
MMS_SCALER_SAVE_PATH = os.path.join(os.getcwd(), 'saved_model', "mms_scaler.pkl")
OHE_SCALER_SAVE_PATH = os.path.join(os.getcwd(), 'saved_model', 'ohe_scaler.pkl')
MODEL_SAVE_PATH=os.path.join(os.getcwd(), 'saved_model', 'model.h5')

model=load_model(MODEL_SAVE_PATH)

#model summary
ohe_scaler=pickle.load(open(OHE_SCALER_SAVE_PATH,'rb'))
mms_scaler=pickle.load(open(MMS_SCALER_SAVE_PATH,'rb'))

diabetes_chance={0:'Negative',1:'Positive'}

#%%deployment

patients_info = np.array([5,116,74,0,0,25.6,0.201,30])
patients_info_scaled = mms_scaler.transform(np.expand_dims(patients_info, axis=0))


# model prediction
new_pred = model.predict(patients_info_scaled)
if np.argmax(new_pred) == 1:
    new_pred = [0,1]
    print(diabetes_chance[np.argmax(new_pred)])
else:
    new_pred = [1,0]
    print(diabetes_chance[np.argmax(new_pred)])
    
    
#%%building the app using streamlit 

with st.form('Diabetes Prediction Form'):
    st.write("Patient's Information")
    pregnancies = int(st.number_input("Time of Pregnant")) # only int values
    glucose = st.number_input("Glucose Concentration")     # float value
    bp = st.number_input("Blood Pressure")
    skin_thick = st.number_input("Skin_Thickness")
    insulin_level = st.number_input("Insulin Level")
    bmi = st.number_input("BMI")
    diabetes = st.number_input("Diabetes")
    age = int(st.number_input("Age"))
    
    
    
    
    submitted = st.form_submit_button('Submit')
    
    if submitted == True:
        patience_info = np.array([pregnancies,glucose,bp,skin_thick,insulin_level,
                                  bmi,diabetes,age])
        patients_info = mms_scaler.transform(np.expand_dims(patients_info, axis=0))
        new_pred = model.predict(patients_info_scaled)
        if np.argmax(new_pred) == 1:
            st.warning(f"You are {diabetes_chance[np.argmax(new_pred)]} diabetes")
        else:
            st.snow()
            st.success(f"You are {diabetes_chance[np.argmax(new_pred)]} diabetes")
    






   