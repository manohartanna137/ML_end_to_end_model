import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, f1_score, RocCurveDisplay
import warnings
warnings.filterwarnings("ignore")
import streamlit as st


from pickle import load
scaler = load(open('F:/model/standard_scaler.pkl', 'rb'))
lr_classifier = load(open('F:/model/lr_model.pkl', 'rb'))
rf_classifier = load(open('F:/model/rf_model.pkl', 'rb'))
tree = load(open('F:/model/dt_model.pkl', 'rb'))
#
st.title("Health Prediction App")

# Sidebar with input form
st.sidebar.header("Enter Patient Details")

# Dynamic values for the sidebar inputs

smoking_history_options = [2,3,4,5,6,7]

# Collect user inputs using dynamic values
gender = st.text_input(" gender Male-2 Female-3")
age = st.text_input('Age enter the age')
hypertension = st.text_input("Hypertension enter value")
heart_disease = st.text_input("Heart disease ")
smoking_history = st.selectbox("Smoking History", smoking_history_options)
bmi = st.text_input('enter value for bmi')
HbA1c_level = st.text_input("HbA1c Level")
blood_glucose_level = st.text_input("Blood Glucose Level")


btn_click = st.button("Predict")

if btn_click == True:
    if gender and age and hypertension and heart_disease and smoking_history and bmi and HbA1c_level and blood_glucose_level:
        query_point = np.array([gender,int(age), int(hypertension), int(heart_disease),smoking_history,float(bmi),float(HbA1c_level),int(blood_glucose_level)]).reshape(1, -1)
        query_point_transformed = scaler.transform(query_point)
        pred = lr_classifier.predict(query_point_transformed)
        st.success(pred)






