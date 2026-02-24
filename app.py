

import streamlit as st
import pandas as pd
import joblib
import os

# Set app layout
st.set_page_config(page_title="🏠 Delhi Rent App", layout="wide")

# Sidebar navigation
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Rent Prediction", "📊 Analysis Report"])

# Shared footer
def footer():
    st.markdown("---")
    st.caption("👨‍💻 Made by Vishal Rai")

# --- PAGE 1: Rent Prediction ---
if page == "🏠 Rent Prediction":
    st.markdown("## 🏠 Rent Prediction for Delhi &nbsp;&nbsp;&nbsp;&nbsp;📍 Fill in the property details below to predict monthly rent", unsafe_allow_html=True)

    # Load model and encoders
    model = joblib.load("house_price_model.pkl")
    full_pipeline = joblib.load("full_pipeline.pkl")
    locality_encoding_map = joblib.load("local_encoder.pkl")

    with st.form("predict_form"):
        size_sq_ft = st.number_input("📏 Area (in sq. ft.)", min_value=100, max_value=10000, value=500)
        bedrooms = st.selectbox("🛏️ Number of Bedrooms", [1, 2, 3, 4, 5], index=1)
        propertyType = st.selectbox("🏢 Property Type", ["apartment", "independent floor"])
        localityName = st.text_input("📌 Locality Name (e.g., geeta colony)").strip().lower()
        suburb = st.selectbox("🗺️ Suburb", [
            "East Delhi", "West Delhi", "North Delhi", 
            "South Delhi", "Central Delhi", "North West Delhi", "South West Delhi"
        ])
        ap_dist = st.slider("📍 Distance from Airport (in km)", 0.0, 50.0, 20.0)

        submitted = st.form_submit_button("🔍 Predict Rent")

    if submitted:
        if localityName not in locality_encoding_map:
            st.error(f"❌ Unknown locality: {localityName}. Please check the spelling.")
        else:
            locality_encoded = locality_encoding_map[localityName]
            user_input = {
                "size_sq_ft": size_sq_ft,
                "bedrooms": bedrooms,
                "propertyType": propertyType,
                "suburbName_clean": suburb,
                "AP_dist_km": ap_dist,
                "locality_encoded": locality_encoded
            }
            input_df = pd.DataFrame([user_input])

            try:
                input_prepared = full_pipeline.transform(input_df)
                predicted_rent = model.predict(input_prepared)[0]
                st.success(f"🏠 Estimated Monthly Rent: ₹{predicted_rent:,.2f}")
            except Exception as e:
                st.error(f"⚠️ Prediction failed: {str(e)}")

    footer()

# --- PAGE 2: Analysis Report ---
elif page == "📊 Analysis Report":
    st.markdown("# 📊 Delhi Rent Analysis Report")

    try:
        with open("Delhi_Rent_insights.html", "r", encoding="utf-8") as f:
            html_data = f.read()
        st.components.v1.html(html_data, height=1000, scrolling=True)
    except FileNotFoundError:
        st.error("❌ Report file not found. Please ensure 'Delhi_Rent_insights.html' is in the same directory.")

    footer()
