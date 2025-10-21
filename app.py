import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import pickle
from tensorflow.keras.models import load_model


model=load_model("/Users/kalpkanungo/Desktop/tensor/mode.h5")

with open("/Users/kalpkanungo/Desktop/tensor/label_encoder_geo.pkl",'rb') as file:
    label=pickle.load(file)
with open("/Users/kalpkanungo/Desktop/tensor/OnehotEncoder.pkl",'rb') as file:
    one=pickle.load(file)
with open("/Users/kalpkanungo/Desktop/tensor/Scaler.pkl",'rb') as file:
    scaler=pickle.load(file)

st.title("Customer chrun Predicion")

geo=st.selectbox("Geography",one.categories_[0])
gender=st.selectbox("Gender",label.classes_)
age=st.slider("Age",18,92)
balanace=st.number_input("Balance")
credit_score=st.number_input("Credit Score")
estimated=st.number_input("Estimater Salary")
tenure=st.slider("Tenure",0,10)
num_of_products=st.slider("Number of products",1,4)
has_cred=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])


input=pd.DataFrame({
    "CreditScore":[credit_score],
    'Gender':[label.transform([gender])[0]],
    "Age":[age],
    "Tenure":[tenure],
    "Balance":[balanace],
    "NumOfProducts":[num_of_products],
    "HasCrCard":[has_cred],
    "IsActiveMember":[is_active_member],
    "EstimatedSalary":[estimated]
})

encoded=one.transform([[geo]]).toarray()
enc=pd.DataFrame(encoded,columns=one.get_feature_names_out(["Geography"]))

data=pd.concat([input.reset_index(drop=True),enc],axis=1)

data=scaler.transform(data)

pred=model.predict(data)
predss=pred[0][0]
x=f"{100*float(pred):.2f}"
st.write(f"Churn Probability is {x}%")

if predss>0.5:
    st.write("The Customer is likely to churn")
else:
    st.write("The customer is not likely to churn")
    