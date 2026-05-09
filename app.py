import streamlit as st
import pickle
import pandas as pd

model = pickle.load(open('model.pkl', 'rb'))

st.title('Customer Churn Predictor')

st.subheader('Enter Customer Details')

gender = st.selectbox('Gender', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
senior = st.selectbox('Senior Citizen', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
partner = st.selectbox('Has Partner', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
dependents = st.selectbox('Has Dependents', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
tenure = st.slider('Tenure (months)', 0, 72, 12)
phone = st.selectbox('Phone Service', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
multiple = st.selectbox('Multiple Lines', [0, 1, 2])
internet = st.selectbox('Internet Service', [0, 1, 2])
security = st.selectbox('Online Security', [0, 1, 2])
backup = st.selectbox('Online Backup', [0, 1, 2])
protection = st.selectbox('Device Protection', [0, 1, 2])
support = st.selectbox('Tech Support', [0, 1, 2])
tv = st.selectbox('Streaming TV', [0, 1, 2])
movies = st.selectbox('Streaming Movies', [0, 1, 2])
contract = st.selectbox('Contract Type', [0, 1, 2], format_func=lambda x: ['Month-to-month','One year','Two year'][x])
paperless = st.selectbox('Paperless Billing', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
payment = st.selectbox('Payment Method', [0, 1, 2, 3])
monthly = st.number_input('Monthly Charges', 0.0, 150.0, 50.0)
total = st.number_input('Total Charges', 0.0, 10000.0, 500.0)

if st.button('Predict'):
    input_data = pd.DataFrame([[gender, senior, partner, dependents, tenure,
                                 phone, multiple, internet, security, backup,
                                 protection, support, tv, movies, contract,
                                 paperless, payment, monthly, total]],
                               columns=['gender','SeniorCitizen','Partner','Dependents','tenure',
                                        'PhoneService','MultipleLines','InternetService','OnlineSecurity',
                                        'OnlineBackup','DeviceProtection','TechSupport','StreamingTV',
                                        'StreamingMovies','Contract','PaperlessBilling','PaymentMethod',
                                        'MonthlyCharges','TotalCharges'])
    result = model.predict(input_data)
    if result[0] == 1:
        st.error('⚠️ This customer is likely to CHURN')
    else:
        st.success('✅ This customer is likely to STAY')