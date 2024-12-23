import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle



## Loading all the pickle files:


# Load the trained model
model = tf.keras.models.load_model('model.h5')

## Save the encoders and sscaler
with open('le_feedback.pkl','rb') as file:
    le_feedback=pickle.load(file)

with open('le_smoke.pkl','rb') as file:
    le_smoke=pickle.load(file)

with open('ohe_education.pkl','rb') as file:
    ohe_education=pickle.load(file)

with open('ohe_occupation.pkl','rb') as file:
    ohe_occupation=pickle.load(file)

with open('ohe_location.pkl','rb') as file:
    ohe_location=pickle.load(file)

with open('ohe_property.pkl','rb') as file:
    ohe_property=pickle.load(file)

with open('ohe_policy.pkl','rb') as file:
    ohe_policy=pickle.load(file)
    
with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

with open('pca.pkl','rb') as file:
    pca=pickle.load(file)

with open("preprocessor.pkl", "rb") as file:
    preprocessor = pickle.load(file)


## streamlit app
st.title('Insurance Premium Prediction')



##User input:
annual_Income=st.number_input('Annual Income')
no_of_dependents=st.slider('Number of Dependents',0,10)
education_level=st.selectbox('Education Level',ohe_education.categories_[0])
occupation=st.selectbox('Occupation',ohe_occupation.categories_[0])
health_score=st.number_input('Health Score')
location=st.selectbox('Location',ohe_location.categories_[0])
policy_type=st.selectbox('Policy Type',ohe_policy.categories_[0])
previous_claims=st.selectbox('Previous Claims',[0,1,2,3,4,5,6,7])
vehicle_age=st.slider('Vehicle Age',0,20)
credit_score=st.number_input('Credit Score')
insurance_duration=st.slider('Insurance Duration',0,10)
customer_feedback=st.selectbox('Customer Feedback',le_feedback.classes_)
smoking_status=st.selectbox('Smoking Status',le_smoke.classes_)
property_type=st.selectbox('Property Type',ohe_property.categories_[0])


##Input data:
input_data=pd.DataFrame({
'Annual Income':[annual_Income],
'Number of Dependents':[no_of_dependents],
'Health Score':[health_score],
'Previous Claims':[previous_claims],
'Vehicle Age':[vehicle_age],
'Credit Score':[credit_score],
'Insurance Duration':[insurance_duration]
})

## Label Encoding:
feedback_encoded=le_feedback.transform([customer_feedback])[0]
smoke_encoded=le_smoke.transform([smoking_status])[0]
df_feedback=pd.DataFrame([feedback_encoded],columns=['Customer Feedback'])
df_smoke=pd.DataFrame([smoke_encoded],columns=['Smoking Status'])
df_label_encoded=pd.concat([df_smoke.reset_index(drop=True),df_feedback.reset_index(drop=True),],axis=1)

##One hot encoding:
education_encoded=ohe_education.transform([[education_level]]).toarray()
occupation_encoded=ohe_occupation.transform([[occupation]]).toarray()
location_encoded=ohe_location.transform([[location]]).toarray()
property_encoded=ohe_property.transform([[property_type]]).toarray()
policy_encoded=ohe_policy.transform([[policy_type]]).toarray()

df_education=pd.DataFrame(education_encoded,columns=ohe_education.get_feature_names_out(['Education Level']))
df_occupation=pd.DataFrame(occupation_encoded,columns=ohe_occupation.get_feature_names_out(['Occupation']))
df_location=pd.DataFrame(location_encoded,columns=ohe_location.get_feature_names_out(['Location']))
df_property=pd.DataFrame(property_encoded,columns=ohe_property.get_feature_names_out(['Property Type']))
df_policy=pd.DataFrame(policy_encoded,columns=ohe_policy.get_feature_names_out(['Policy Type']))

df_ohe_encoded=pd.concat([df_education.reset_index(drop=True),df_occupation.reset_index(drop=True),
df_location.reset_index(drop=True),df_policy.reset_index(drop=True),df_property.reset_index(drop=True)],axis=1)



# input_df=pd.concat([input_df.reset_index(drop=True),df_ohe_encoded.reset_index(drop=True)],axis=1)


# numerical_columns=['Annual Income','Number of Dependents','Previous Claims','Vehicle Age','Credit_Score','Insurance Duration']

## Scaling of all numerical columns:]
df_scaled=preprocessor.transform(input_data)
input_df_scaled=pd.DataFrame(df_scaled,columns=['Annual Income','Number of Dependents','Health Score','Previous Claims','Vehicle Age','Credit Score','Insurance Duration'])


##PCA
pca_scaled=pca.transform(input_df_scaled)
data_pca=pd.DataFrame(pca_scaled,columns=['pca_feature1','pca_feature2','pca_feature3'])
df_new_after_pca=pd.concat([data_pca.reset_index(drop=True),
                            df_ohe_encoded.reset_index(drop=True),
                            df_label_encoded.reset_index(drop=True)],axis=1)



# Convert DataFrame to NumPy array
final_data_transformed = df_new_after_pca.values

# Ensure it has the correct shape for prediction (1, number of features)
final_data_transformed = final_data_transformed.reshape(1, -1)

prediction=model.predict(final_data_transformed)



# Validation to check if all fields are filled
if st.button("Show Prediction"):
    if (
        annual_Income == 0 or
        health_score == 0 or
        credit_score == 0 
    ):
        st.error("Please fill in all fields before getting a prediction!")
    else:
       # Make prediction
        prediction = model.predict(final_data_transformed)
        st.success(f"Your Insurance Premium Amount: {prediction[0][0]:.2f}")