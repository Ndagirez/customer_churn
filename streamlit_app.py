import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
from tensorflow.keras.models import load_model
import os

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header { font-size: 3rem; color: #2E86AB; text-align: center; margin-bottom: 2rem; font-weight: bold; }
    .sub-header { font-size: 1.5rem; color: #A23B72; margin-bottom: 1rem; font-weight: bold; }
    .prediction-box { padding: 2rem; border-radius: 10px; margin: 2rem 0; text-align: center; font-size: 1.5rem; font-weight: bold; }
    .churn-yes { background-color: #FF6B6B; color: white; }
    .churn-no { background-color: #51CF66; color: white; }
    .stButton>button { background-color: #2E86AB; color: white; border: none; padding: 0.5rem 2rem; border-radius: 5px; font-size: 1.1rem; font-weight: bold; }
    .stButton>button:hover { background-color: #1B5E7B; color: white; }
    .section { background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; border-left: 5px solid #2E86AB; }
</style>
""", unsafe_allow_html=True)

# Load model and preprocessor safely
@st.cache_resource
def load_components():
    model_path = 'customer_churn.keras'
    preprocessor_path = 'customer_churn_preprocessor.pkl'
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None, None
    if not os.path.exists(preprocessor_path):
        st.error(f"Preprocessor file not found: {preprocessor_path}")
        return None, None
    try:
        model = load_model(model_path)
        preprocessor = joblib.load(preprocessor_path)
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading components: {e}")
        return None, None

def main():
    st.markdown('<div class="main-header">📊 Customer Churn Prediction Dashboard</div>', unsafe_allow_html=True)
    model, preprocessor = load_components()
    if model is None or preprocessor is None:
        st.stop()

    # Input sections
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">👤 Customer Demographics</div>', unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">📞 Phone Services</div>', unsafe_allow_html=True)
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"]) if phone_service == "Yes" else "No"
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">🌐 Internet Services</div>', unsafe_allow_html=True)
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        if internet_service != "No":
            online_security = st.selectbox("Online Security", ["No", "Yes"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
        else:
            online_security = online_backup = device_protection = tech_support = streaming_tv = streaming_movies = "No"
        st.markdown('</div>', unsafe_allow_html=True)

    col3, col4 = st.columns([1, 1])
    with col3:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">💳 Billing Information</div>', unsafe_allow_html=True)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">💰 Financial Details</div>', unsafe_allow_html=True)
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 200.0, 65.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=1000.0, step=50.0)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button("🔮 Predict Customer Churn", use_container_width=True, type="primary")

    if predict_button:
        input_data = {
            'gender': gender,
            'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': int(tenure),
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
            'MonthlyCharges': float(monthly_charges),
            'TotalCharges': float(total_charges)
        }
        input_df = pd.DataFrame([input_data])

        # Ensure columns match preprocessor expectation
        expected_cols = getattr(preprocessor, 'feature_names_in_', input_df.columns)
        missing_cols = set(expected_cols) - set(input_df.columns)
        if missing_cols:
            st.error(f"Missing columns for prediction: {missing_cols}")
            st.stop()

        try:
            processed_data = preprocessor.transform(input_df)
            prediction_proba = model.predict(processed_data)
            # Handle output shape robustly
            if prediction_proba.shape[1] == 1:
                churn_probability = prediction_proba[0][0]
            else:
                churn_probability = prediction_proba[0][1]
            prediction = int(churn_probability > 0.5)

            st.markdown("---")
            col_result1, col_result2, col_result3 = st.columns([1, 2, 1])
            with col_result2:
                if prediction == 1:
                    st.markdown(
                        f'<div class="prediction-box churn-yes">⚠️ Prediction: CHURN (Yes)<br>'
                        f'<small>Probability: {churn_probability:.2%}</small></div>',
                        unsafe_allow_html=True
                    )
                    st.warning("This customer is likely to churn. Consider retention strategies!")
                else:
                    st.markdown(
                        f'<div class="prediction-box churn-no">✅ Prediction: NO CHURN<br>'
                        f'<small>Probability: {1-churn_probability:.2%}</small></div>',
                        unsafe_allow_html=True
                    )
                    st.success("This customer is likely to stay with the service!")

            with st.expander("📈 View Prediction Details"):
                col_detail1, col_detail2 = st.columns(2)
                with col_detail1:
                    st.metric("Churn Probability", f"{churn_probability:.2%}")
                    st.metric("Retention Probability", f"{(1-churn_probability):.2%}")
                with col_detail2:
                    st.write("**Key Factors Considered:**")
                    st.write(f"• Tenure: {tenure} months")
                    st.write(f"• Contract: {contract}")
                    st.write(f"• Monthly Charges: ${monthly_charges:.2f}")
                    st.write(f"• Internet Service: {internet_service}")
                st.write("**Customer Summary:**")
                summary_col1, summary_col2 = st.columns(2)
                with summary_col1:
                    st.write(f"**Demographics:**")
                    st.write(f"- Gender: {gender}")
                    st.write(f"- Senior Citizen: {senior_citizen}")
                    st.write(f"- Partner: {partner}")
                    st.write(f"- Dependents: {dependents}")
                with summary_col2:
                    st.write(f"**Services:**")
                    st.write(f"- Phone Service: {phone_service}")
                    st.write(f"- Internet: {internet_service}")
                    st.write(f"- Contract: {contract}")

        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.info("Please check that your input data matches the training data format.")

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Built with ❤️ using Streamlit | Customer Churn Prediction Model"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
