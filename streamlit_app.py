
import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('pcos_model.pkl')
scaler = joblib.load('scaler.pkl')

# Page config
st.set_page_config(page_title="PCOS Risk Predictor", page_icon="🩺", layout="centered")

# Title
st.markdown("<h1 style='text-align: center;'>🩺 PCOS Risk Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter the patient information below to assess PCOS risk.</p>", unsafe_allow_html=True)
st.markdown("---")

# Form
with st.form("pcos_form"):
    st.markdown("### 👤 Patient Information")

    age = st.number_input("Age (yrs)", min_value=10, max_value=50, value=25)
    weight = st.number_input("Weight (Kg)", min_value=30, max_value=150, value=60)
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)
   
    st.markdown("### 🩹 Symptoms and Lifestyle")

    cycle_input = st.radio("Menstrual Cycle Regularity", ["Regular", "Irregular"])
    cycle =( 0 if cycle_input == "Regular" else 1)
    hair_growth = st.radio("Hair Growth?", ['No', 'Yes'], horizontal=True)
    skin_darkening = st.radio("Skin Darkening?", ['No', 'Yes'], horizontal=True)
    pimples = st.radio("Pimples?", ['No', 'Yes'], horizontal=True)
    fast_food = st.radio("Frequent Fast Food?", ['No', 'Yes'], horizontal=True)
    exercise = st.radio("Regular Exercise?", ['No', 'Yes'], horizontal=True)

    # Centered button (use_container_width makes it look clean)
    submitted = st.form_submit_button("🔍 Evaluate", use_container_width=True)

# On submit
if submitted:
    def yn(val): return 1 if val == "Yes" else 0

    input_data = np.array([[
        age, weight, bmi, cycle,
        yn(hair_growth), yn(skin_darkening), yn(pimples),
        yn(fast_food), yn(exercise)
    ]])

    try:
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        st.markdown("<h2 style='text-align: center;'>👩🏼‍⚕️PCOS Risk Prediction</h2>", unsafe_allow_html=True)

        if prediction[0] == 1:
            st.markdown("<h4 style='color: red; text-align: center;'>⚠️ High risk of PCOS detected.</h4>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Please consult a healthcare professional.</p>", unsafe_allow_html=True)
        else:
            st.markdown("<h4 style='color: green; text-align: center;'>✅ Low risk of PCOS.</h4>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Indicators are healthy. Keep maintaining a good lifestyle!</p>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
