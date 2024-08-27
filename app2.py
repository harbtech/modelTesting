import streamlit as st
import pandas as pd
from joblib import load
import time

# Load the saved model
model_filename = 'model.joblib'
loaded_model = load(model_filename)
st.write(f"Model loaded from {model_filename}")

# Function to preprocess new data
def preprocess_data(data):
    # Ensure the data has the same columns as used during training
    # Exclude 'isFraud', 'isFlaggedFraud', 'nameOrig', 'nameDest' as they were dropped during training
    columns_to_use = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    return data[columns_to_use]

# Streamlit app
st.title('Fraud Detection Web App')

# Input fields
step = st.number_input('Step', min_value=1, value=1)
transaction_type = st.selectbox('Type', ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT'])
amount = st.number_input('Amount', min_value=0.0, value=1000.0)
userphone = st.text_input('Phone', '0240818849')
oldbalanceOrg = st.number_input('Old Balance Origin', min_value=0.0, value=1001.0)
newbalanceOrig = st.number_input('New Balance Origin', min_value=0.0, value=0.0)
oldbalanceDest = st.number_input('Old Balance Destination', min_value=0.0, value=0.0)
newbalanceDest = st.number_input('New Balance Destination', min_value=0.0, value=1001.0)
nameDest = st.text_input('Name Destination', 'C38997010')



# Button to predict
if st.button('Predict'):
    # Create a DataFrame with the input data
    new_data = pd.DataFrame({
        'step': [step],
        'type': [transaction_type],
        'amount': [amount],
        'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrig': [newbalanceOrig],
        'oldbalanceDest': [oldbalanceDest],
        'newbalanceDest': [newbalanceDest],
        'nameOrig': ['C840083671'],  # These columns will be dropped in preprocessing
        'nameDest': [nameDest],
        'isFraud': [1],
        'isFlaggedFraud': [0]
    })

    # Preprocess the new data
    preprocessed_data = preprocess_data(new_data)

    # Make predictions
    predictions = loaded_model.predict(preprocessed_data)
    prediction_probabilities = loaded_model.predict_proba(preprocessed_data)

    # Display results
    st.write("Predictions:", predictions)
    st.write("Prediction Probabilities:", prediction_probabilities)

    # Interpret results
    for i, pred in enumerate(predictions):
        st.write(f"\nTransaction {i+1}:")
        st.write(f"Predicted class: {'Fraudulent' if pred == 1 else 'Not Fraudulent'}")
        st.write(f"Probability of being fraudulent: {prediction_probabilities[i][1]:.4f}")

        # Add to Triang list if fraudulent
        if pred == 1:
            st.session_state.triang_list.append(userphone)


# Display Triang list
if 'triang_list' not in st.session_state:
    st.session_state.triang_list = []

st.subheader('T-Red List')
st.write(st.session_state.triang_list)


# Hyperloca field and button
hyperloca = st.text_input('Hyperloca', '')
if st.button('Find Hyperloca'):
    if hyperloca:
        # Show progress bar
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.05)  # Simulate a 5-second process
            progress_bar.progress(i + 1)
        st.write("Target located at long:000,lat:000")
