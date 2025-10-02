import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for army green background and styling
st.markdown("""
<style>
    .main {
        background-color: #4B5320;
        color: white;
    }
    .stApp {
        background: linear-gradient(135deg, #4B5320, #556B2F);
    }
    .sidebar .sidebar-content {
        background-color: #3A4520;
    }
    .prediction-box {
        background-color: #2F4F4F;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        border: 2px solid #8FBC8F;
    }
    .title-text {
        color: #F5F5DC;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 30px;
    }
    .section-header {
        color: #F5F5DC;
        border-bottom: 2px solid #8FBC8F;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="title-text">🔮 Customer Churn Prediction</h1>', unsafe_allow_html=True)
st.markdown("Predict whether a customer will churn based on their profile information")

def load_model():
    """Load the trained model and preprocessing objects"""
    try:
        model = joblib.load('customer_churn_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file 'customer_churn_model.pkl' not found. Please make sure it's in the same directory.")
        return None

def preprocess_input(user_input, cat_list, num_list):
    """Preprocess user input to match model training format"""
    # Create DataFrame from user input
    input_df = pd.DataFrame([user_input])
    
    # Ensure numerical columns are numeric
    for col in num_list:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
    
    return input_df

def main():
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Define feature lists (same as your training)
    cat_list = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                'PaperlessBilling', 'PaymentMethod']
    
    num_list = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
    
    # Create two columns for input fields
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="section-header">Personal Information</h3>', unsafe_allow_html=True)
        
        # Personal information
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
    
    with col2:
        st.markdown('<h3 class="section-header">Service Information</h3>', unsafe_allow_html=True)
        
        # Service information
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        
        # Contract and billing
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card"
        ])
    
    # Additional services (in expandable section)
    with st.expander("Additional Services"):
        st.markdown('<h4 class="section-header">Online Services</h4>', unsafe_allow_html=True)
        col3, col4 = st.columns(2)
        
        with col3:
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        
        with col4:
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    
    # Charges information
    st.markdown('<h3 class="section-header">Charges Information</h3>', unsafe_allow_html=True)
    charge_col1, charge_col2 = st.columns(2)
    
    with charge_col1:
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 200.0, 65.0, 0.5)
    
    with charge_col2:
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=1000.0, step=10.0)
    
    # Prepare user input dictionary
    user_input = {
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    # Prediction button
    st.markdown("---")
    predict_button = st.button("🔍 Predict Churn", use_container_width=True)
    
    if predict_button:
        try:
            # Preprocess input
            input_df = preprocess_input(user_input, cat_list, num_list)
            
            # Make prediction
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)
            
            # Display results
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            
            churn_prob = prediction_proba[0][1] * 100
            no_churn_prob = prediction_proba[0][0] * 100
            
            if prediction[0] == 1:
                st.error(f"🚨 **Prediction: CHURN (Yes)**")
                st.write(f"The model predicts this customer is likely to churn.")
            else:
                st.success(f"✅ **Prediction: NO CHURN (No)**")
                st.write(f"The model predicts this customer will stay.")
            
            # Show probability scores
            col_prob1, col_prob2 = st.columns(2)
            with col_prob1:
                st.metric("Probability of No Churn", f"{no_churn_prob:.1f}%")
            with col_prob2:
                st.metric("Probability of Churn", f"{churn_prob:.1f}%")
            
            # Progress bars for visualization
            st.progress(int(no_churn_prob)/100, text="No Churn Confidence")
            st.progress(int(churn_prob)/100, text="Churn Confidence")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please check that all input values are correctly filled.")

if __name__ == "__main__":
    main()
