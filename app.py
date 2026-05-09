import streamlit as st
import pickle
import pandas as pd

# ---- PAGE CONFIG ----
st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="centered")

# ---- CUSTOM CSS ----
st.markdown("""
    <style>
    /* Background */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 50%, #1a1a2e 100%);
        color: white;
    }

    /* Title */
    h1 {
        color: #4fc3f7;
        text-align: center;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: 1px;
        padding-bottom: 10px;
        border-bottom: 2px solid #4fc3f7;
        margin-bottom: 25px;
    }

    /* Subheader */
    h3 {
        color: #90caf9;
        font-size: 1.1rem;
        margin-top: 20px;
    }

    /* Input labels */
    label {
        color: #cfd8dc !important;
        font-weight: 500;
    }

    /* Selectbox and slider */
    .stSelectbox > div, .stSlider > div {
        background-color: #1e2a3a !important;
        border-radius: 8px;
    }

    /* Number input */
    .stNumberInput > div > div > input {
        background-color: #1e2a3a !important;
        color: white !important;
        border: 1px solid #4fc3f7 !important;
        border-radius: 8px;
    }

    /* Predict Button */
    .stButton > button {
        background: linear-gradient(90deg, #0077b6, #00b4d8);
        color: white;
        font-size: 1.1rem;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 12px 40px;
        width: 100%;
        margin-top: 20px;
        cursor: pointer;
        transition: 0.3s;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #023e8a, #0096c7);
        transform: scale(1.02);
    }

    /* Success box */
    .stSuccess {
        background-color: #1b4332 !important;
        border-left: 5px solid #40916c !important;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: 600;
    }

    /* Error box */
    .stAlert {
        background-color: #3b1a1a !important;
        border-left: 5px solid #e63946 !important;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: 600;
    }

    /* Divider */
    hr {
        border: 1px solid #1e3a5f;
    }
    </style>
""", unsafe_allow_html=True)

# ---- LOAD MODEL ----
model = pickle.load(open('model.pkl', 'rb'))

# ---- HEADER ----
st.title('📊 Customer Churn Predictor')
st.markdown("<p style='text-align:center; color:#90caf9;'>Fill in the customer details below to predict churn</p>", unsafe_allow_html=True)
st.markdown("---")

# ---- INPUT FIELDS ----
st.subheader('👤 Customer Details')

gender = st.selectbox('Gender', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
senior = st.selectbox('Senior Citizen', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
partner = st.selectbox('Has Partner', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
dependents = st.selectbox('Has Dependents', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
tenure = st.slider('Tenure (months)', 0, 72, 12)
phone = st.selectbox('Phone Service', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
multiple = st.selectbox('Multiple Lines', [0, 1, 2], format_func=lambda x: ['No Phone Service', 'No', 'Yes'][x])
internet = st.selectbox('Internet Service', [0, 1, 2], format_func=lambda x: ['DSL', 'Fiber Optic', 'No Internet'][x])
security = st.selectbox('Online Security', [0, 1, 2], format_func=lambda x: ['No', 'Yes', 'No Internet Service'][x])
backup = st.selectbox('Online Backup', [0, 1, 2], format_func=lambda x: ['No', 'Yes', 'No Internet Service'][x])
protection = st.selectbox('Device Protection', [0, 1, 2], format_func=lambda x: ['No', 'Yes', 'No Internet Service'][x])
support = st.selectbox('Tech Support', [0, 1, 2], format_func=lambda x: ['No', 'Yes', 'No Internet Service'][x])
tv = st.selectbox('Streaming TV', [0, 1, 2], format_func=lambda x: ['No', 'Yes', 'No Internet Service'][x])
movies = st.selectbox('Streaming Movies', [0, 1, 2], format_func=lambda x: ['No', 'Yes', 'No Internet Service'][x])
contract = st.selectbox('Contract Type', [0, 1, 2], format_func=lambda x: ['Month-to-month', 'One Year', 'Two Year'][x])
paperless = st.selectbox('Paperless Billing', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
payment = st.selectbox('Payment Method', [0, 1, 2, 3], format_func=lambda x: ['Electronic Check', 'Mailed Check', 'Bank Transfer', 'Credit Card'][x])
monthly = st.number_input('Monthly Charges ($)', 0.0, 150.0, 50.0)
total = st.number_input('Total Charges ($)', 0.0, 10000.0, 500.0)

st.markdown("---")

# ---- PREDICTION ----
if st.button('🔍 Predict Churn'):
    input_data = pd.DataFrame([[gender, senior, partner, dependents, tenure,
                                 phone, multiple, internet, security, backup,
                                 protection, support, tv, movies, contract,
                                 paperless, payment, monthly, total]],
                               columns=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                                        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                                        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                                        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                                        'MonthlyCharges', 'TotalCharges'])
    result = model.predict(input_data)
    st.markdown("---")
    if result[0] == 1:
        st.error('⚠️ This customer is likely to CHURN')
        st.markdown("<p style='color:#ff6b6b; text-align:center;'>Consider offering a discount or better plan to retain this customer.</p>", unsafe_allow_html=True)
    else:
        st.success('✅ This customer is likely to STAY')
        st.markdown("<p style='color:#40916c; text-align:center;'>This customer is satisfied and likely to continue the service.</p>", unsafe_allow_html=True)

# ---- FOOTER ----
st.markdown("---")
st.markdown("<p style='text-align:center; color:#546e7a; font-size:0.8rem;'>Built by Arnav Dasmal | NIT Durgapur | Customer Churn Prediction using XGBoost</p>", unsafe_allow_html=True)
