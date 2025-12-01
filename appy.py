import streamlit as st
import pandas as pd
import joblib

# load trained pipeline
model = joblib.load("Models")

st.title("Smartphone Price Category Predictor")

st.write("Fill in the specs and click Predict.")

# simple inputs (must match your training columns)
rating = st.slider("Rating", 60.0, 100.0, 80.0)
ram = st.number_input("RAM Size GB", 1.0, 24.0, 8.0)
storage = st.number_input("Storage Size GB", 16.0, 1024.0, 128.0)
battery = st.number_input("Battery Capacity (mAh)", 1000.0, 8000.0, 5000.0)
screen_size = st.number_input("Screen Size (inches)", 4.5, 7.5, 6.5)
refresh = st.number_input("Refresh Rate (Hz)", 60.0, 165.0, 90.0)

dual_sim = st.selectbox("Dual SIM", ["Yes", "No"])
four_g = st.selectbox("4G", ["Yes", "No"])
five_g = st.selectbox("5G", ["Yes", "No"])

processor_brand = st.selectbox("Processor Brand", ["Snapdragon", "Dimensity", "Bionic", "Exynos", "Helio", "Other"])
performance_tier = st.selectbox("Performance Tier", ["Budget", "Mid-range", "High-end", "Flagship"])
ram_tier = st.selectbox("RAM Tier", ["Low", "Medium", "High"])
notch_type = st.selectbox("Notch Type", ["None", "Waterdrop", "Punch-hole", "Other"])
os_name = st.selectbox("OS Name", ["Android", "iOS", "Other"])
os_version = st.text_input("OS Version (e.g. v12)", "v12")
brand = st.text_input("Brand", "Samsung")

if st.button("Predict"):
    # build one-row DataFrame with SAME column names as training X
    data = {
        "rating": [rating],
        "RAM Size GB": [ram],
        "Storage Size GB": [storage],
        "battery_capacity": [battery],
        "Screen_Size": [screen_size],
        "Refresh_Rate": [refresh],
        "Dual_Sim": [dual_sim],
        "4G": [four_g],
        "5G": [five_g],
        "Vo5G": ["No"],
        "NFC": ["No"],
        "IR_Blaster": ["No"],
        "Processor_Brand": [processor_brand],
        "Processor_Series": ["Unknown"],
        "Core_Count": [8],
        "Clock_Speed_GHz": [2.4],
        "Performance_Tier": [performance_tier],
        "RAM Tier": [ram_tier],
        "Resolution_Width": [2400],
        "Resolution_Height": [1080],
        "primary_rear_camera_mp": [50.0],
        "num_rear_cameras": [3],
        "primary_front_camera_mp": [16.0],
        "num_front_cameras": [1],
        "memory_card_support": ["No"],
        "memory_card_size": ["0 GB"],
        "Notch_Type": [notch_type],
        "os_name": [os_name],
        "os_version": [os_version],
        "brand": [brand],
    }

    X_new = pd.DataFrame(data)

    # IMPORTANT: recompute engineered features like in training
    X_new["screen_pixels"] = X_new["Resolution_Width"] * X_new["Resolution_Height"]
    X_new["total_cameras"] = X_new["num_rear_cameras"] + X_new["num_front_cameras"]
    X_new["camera_score"] = X_new["primary_rear_camera_mp"] + X_new["primary_front_camera_mp"]
    X_new["ram_storage_ratio"] = X_new["RAM Size GB"] / (X_new["Storage Size GB"] + 1e-6)

    # model pipeline will handle preprocessing
    pred = model.predict(X_new)[0]
    prob = model.predict_proba(X_new)[0][int(pred)]

    label = "Expensive" if pred == 1 else "Non-expensive"
    st.success(f"Prediction: {label}")
    st.write(f"Estimated confidence: {prob:.2f}")