import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Advanced Fraud Detection Tester", layout="wide")

# Load the model
@st.cache_resource
def load_model():
    return joblib.load('model.joblib')

model = load_model()


st.title('🕵️‍♀️ Advanced Fraud Detection Model Tester')

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Transaction Details")
    
    step = st.slider('Step', min_value=1, max_value=1000, value=1)
    transaction_type = st.selectbox('Transaction Type', ['TRANSFER', 'CASH_OUT', 'PAYMENT', 'DEBIT', 'CASH_IN'])
    amount = st.number_input('Amount ($)', min_value=0.0, value=100.0, format="%.2f")
    
    col1a, col1b = st.columns(2)
    with col1a:
        name_orig = st.text_input('Origin Account', value='C1231006815')
        old_balance_org = st.number_input('Old Balance Origin ($)', min_value=0.0, value=100.0, format="%.2f")
        new_balance_orig = st.number_input('New Balance Origin ($)', min_value=0.0, value=60.0, format="%.2f")
    
    with col1b:
        name_dest = st.text_input('Destination Account', value='M1979787155')
        old_balance_dest = st.number_input('Old Balance Destination ($)', min_value=0.0, value=0.0, format="%.2f")
        new_balance_dest = st.number_input('New Balance Destination ($)', min_value=0.0, value=100.0, format="%.2f")
    
    is_flagged_fraud = st.checkbox('Flagged as Potential Fraud')

with col2:
    st.header("Prediction")
    if st.button('Analyze Transaction', key='predict'):
        input_data = pd.DataFrame({
            'step': [step],
            'type': [transaction_type],
            'amount': [amount],
            'nameOrig': [name_orig],
            'oldbalanceOrg': [old_balance_org],
            'newbalanceOrig': [new_balance_orig],
            'nameDest': [name_dest],
            'oldbalanceDest': [old_balance_dest],
            'newbalanceDest': [new_balance_dest],
            'isFlaggedFraud': [int(is_flagged_fraud)]
        })

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        if prediction[0] == 1:
            st.error("⚠️ Potential Fraud Detected!")
        else:
            st.success("✅ Transaction Appears Legitimate")

        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probability[0][1] * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Fraud Probability"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 75], 'color': "yellow"},
                    {'range': [75, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': probability[0][1] * 100
                }
            }
        ))

        st.plotly_chart(fig)

st.sidebar.header("About")
st.sidebar.info("This advanced fraud detection tester allows you to input transaction details and receive a fraud probability prediction based on a machine learning model.")

# Add this new section
st.sidebar.header("How Fraud is Detected")
st.sidebar.info("""
In this specific dataset, fraudulent behavior is characterized by agents attempting to profit by:

1. Taking control of customer accounts
2. Transferring funds to another account
3. Cashing out of the system

The model analyzes transaction patterns and account behaviors to identify these suspicious activities.
""")
