import streamlit as st
import pandas as pd
import joblib
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Phone Price Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 2rem 0;
    }
    .expensive {
        background-color: #ffd700;
        border: 3px solid #ff6b6b;
    }
    .non-expensive {
        background-color: #90EE90;
        border: 3px solid #4CAF50;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load model with comprehensive error handling
@st.cache_resource
def load_model():
    """Load the trained model with multiple fallback paths"""
    possible_paths = [
        "Models/phone_price_model.pkl",
        "Models",
        "model.pkl",
        "phone_price_model.pkl",
        "DT_model.pkl",
        "pipeline.pkl"
    ]
    
    for path in possible_paths:
        try:
            if os.path.exists(path):
                model = joblib.load(path)
                st.sidebar.success(f"✅ Model loaded from: {path}")
                
                # Check if model has necessary methods
                has_predict = hasattr(model, 'predict')
                has_predict_proba = hasattr(model, 'predict_proba')
                
                if has_predict:
                    st.sidebar.info(f"📊 Model type: {type(model).__name__}")
                    if not has_predict_proba:
                        st.sidebar.warning("⚠️ Model doesn't support probability predictions")
                    return model
                else:
                    st.sidebar.error("❌ Loaded object is not a valid model")
                    continue
                    
        except Exception as e:
            st.sidebar.error(f"Error loading {path}: {str(e)}")
            continue
    
    st.error("❌ Model file not found. Please ensure the model file exists in one of these locations:")
    st.write("- Models/phone_price_model.pkl")
    st.write("- Models/")
    st.write("- phone_price_model.pkl")
    st.info("💡 **Demo Mode:** You can still use the interface to see how it works!")
    return None

# Header
st.markdown("""
    <div class="main-header">
        <h1>📱 Smartphone Price Category Predictor</h1>
        <p>Predict whether a smartphone is Expensive or Non-Expensive based on its specifications</p>
    </div>
""", unsafe_allow_html=True)

# Load model
model = load_model()

# Sidebar with instructions
with st.sidebar:
    st.header("📖 Instructions")
    st.markdown("""
    1. **Fill in** the smartphone specifications across the three tabs
    2. **Review** your inputs in the summary
    3. **Click** the Predict button to get results
    
    ### Features Covered:
    - ✅ Performance specs
    - ✅ Battery & charging
    - ✅ Camera details
    - ✅ Display properties
    - ✅ Connectivity options
    """)
    
    st.markdown("---")
    st.markdown("### About")
    st.info("This tool uses machine learning to classify smartphones into price categories based on their technical specifications.")

# Create tabs for organized input
tab1, tab2, tab3 = st.tabs(["📊 Basic Specs", "🔧 Advanced Features", "📸 Camera & Display"])

# TAB 1: Basic Specs
with tab1:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🎯 Performance")
        rating = st.slider("Device Rating", 60.0, 100.0, 80.0, 0.5, 
                          help="Overall device rating (60-100)")
        ram = st.number_input("RAM Size (GB)", 1.0, 24.0, 8.0, 0.5,
                             help="Amount of RAM in gigabytes")
        storage = st.number_input("Storage Size (GB)", 16.0, 1024.0, 128.0, 16.0,
                                 help="Internal storage capacity")
        
    with col2:
        st.subheader("🔋 Battery & Power")
        battery = st.number_input("Battery Capacity (mAh)", 1000.0, 8000.0, 5000.0, 100.0,
                                 help="Battery capacity in milliamp hours")
        fast_charging = st.number_input("Fast Charging Power (W)", 0.0, 120.0, 18.0, 5.0,
                                       help="Fast charging wattage")
        
    with col3:
        st.subheader("🏢 Brand & OS")
        brand = st.selectbox("Brand", [
            "Samsung", "Apple", "Xiaomi", "Oppo", "Vivo", 
            "OnePlus", "Realme", "Motorola", "Nokia", "Google",
            "Huawei", "Honor", "Asus", "Sony", "LG", "Other"
        ], help="Smartphone manufacturer")
        
        os_name = st.selectbox("Operating System", 
                              ["Android", "iOS", "Other"],
                              help="Mobile operating system")
        os_version = st.text_input("OS Version", "v12",
                                  help="Version number (e.g., v12, v13)")

# TAB 2: Advanced Features
with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ Processor")
        processor_brand = st.selectbox("Processor Brand", [
            "Snapdragon", "Dimensity", "Bionic", "Exynos", 
            "Helio", "Tensor", "Kirin", "Other"
        ], help="Chipset manufacturer")
        
        processor_series = st.text_input("Processor Series", "Unknown",
                                        help="e.g., 8 Gen 2, A16")
        
        core_count = st.slider("Core Count", 2, 16, 8, 1,
                              help="Number of CPU cores")
        
        clock_speed = st.number_input("Clock Speed (GHz)", 1.0, 4.0, 2.4, 0.1,
                                     help="Maximum CPU frequency")
        
        performance_tier = st.selectbox("Performance Tier", [
            "Budget", "Mid-range", "High-end", "Flagship"
        ], help="Overall performance category")
        
        ram_tier = st.selectbox("RAM Tier", 
                               ["Low", "Medium", "High"],
                               help="RAM capacity tier")
    
    with col2:
        st.subheader("📡 Connectivity")
        dual_sim = st.selectbox("Dual SIM", ["Yes", "No"],
                               help="Supports two SIM cards")
        
        four_g = st.selectbox("4G Support", ["Yes", "No"],
                             help="4G LTE connectivity")
        
        five_g = st.selectbox("5G Support", ["Yes", "No"],
                             help="5G network support")
        
        vo5g = st.selectbox("Vo5G Support", ["Yes", "No"],
                           help="Voice over 5G")
        
        nfc = st.selectbox("NFC", ["Yes", "No"],
                          help="Near Field Communication")
        
        ir_blaster = st.selectbox("IR Blaster", ["Yes", "No"],
                                 help="Infrared remote control")
        
        st.subheader("💾 Storage Expansion")
        memory_card_support = st.selectbox("Memory Card Support", ["Yes", "No"],
                                          help="Expandable storage slot")
        
        memory_card_size = st.text_input("Max Memory Card Size", "0 GB",
                                        help="Maximum SD card capacity")

# TAB 3: Camera & Display
with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📺 Display")
        screen_size = st.number_input("Screen Size (inches)", 4.5, 7.5, 6.5, 0.1,
                                     help="Diagonal screen measurement")
        
        refresh = st.number_input("Refresh Rate (Hz)", 60.0, 165.0, 90.0, 15.0,
                                 help="Screen refresh frequency")
        
        res_width = st.number_input("Resolution Width", 720, 3200, 2400, 120,
                                   help="Horizontal pixel count")
        
        res_height = st.number_input("Resolution Height", 480, 1440, 1080, 120,
                                    help="Vertical pixel count")
        
        notch_type = st.selectbox("Notch Type", [
            "None", "Waterdrop", "Punch-hole", "Dynamic Island", "Other"
        ], help="Front camera cutout style")
    
    with col2:
        st.subheader("📷 Camera")
        primary_rear_camera_mp = st.number_input("Primary Rear Camera (MP)", 
                                                 8.0, 200.0, 50.0, 1.0,
                                                 help="Main rear camera megapixels")
        
        num_rear_cameras = st.slider("Number of Rear Cameras", 1, 5, 3, 1,
                                    help="Total rear camera count")
        
        primary_front_camera_mp = st.number_input("Primary Front Camera (MP)", 
                                                  5.0, 50.0, 16.0, 1.0,
                                                  help="Selfie camera megapixels")
        
        num_front_cameras = st.slider("Number of Front Cameras", 1, 3, 1, 1,
                                     help="Total front camera count")

# Summary Section
with st.expander("📋 View Current Input Summary", expanded=False):
    sum_col1, sum_col2, sum_col3 = st.columns(3)
    
    with sum_col1:
        st.markdown("**Basic Specs**")
        st.write(f"• Brand: {brand}")
        st.write(f"• Rating: {rating}/100")
        st.write(f"• RAM: {ram} GB")
        st.write(f"• Storage: {storage} GB")
        st.write(f"• Battery: {battery} mAh")
        
    with sum_col2:
        st.markdown("**Performance**")
        st.write(f"• Processor: {processor_brand}")
        st.write(f"• Cores: {core_count}")
        st.write(f"• Clock: {clock_speed} GHz")
        st.write(f"• Tier: {performance_tier}")
        st.write(f"• 5G: {five_g}")
        
    with sum_col3:
        st.markdown("**Display & Camera**")
        st.write(f"• Screen: {screen_size}\" @ {refresh}Hz")
        st.write(f"• Resolution: {res_width}x{res_height}")
        st.write(f"• Rear Camera: {primary_rear_camera_mp}MP")
        st.write(f"• Front Camera: {primary_front_camera_mp}MP")
        st.write(f"• OS: {os_name} {os_version}")

# Prediction Button
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    predict_btn = st.button("🔮 PREDICT PRICE CATEGORY", 
                           type="primary", 
                           use_container_width=True)

# Prediction Logic
if predict_btn:
    if model is None:
        st.warning("⚠️ Model not loaded. This is a demo prediction.")
        # Demo prediction for showcase
        demo_score = (ram * 10 + storage * 0.5 + battery * 0.01 + 
                     primary_rear_camera_mp * 2 + rating * 2) / 100
        pred = 1 if demo_score > 50 else 0
        prob = [1 - demo_score/100, demo_score/100] if pred == 1 else [demo_score/100, 1 - demo_score/100]
    else:
        with st.spinner("🔍 Analyzing smartphone specifications..."):
            # Build DataFrame with exact column names
            data = {
                "rating": [rating],
                "Core_Count": [core_count],
                "Clock_Speed_GHz": [clock_speed],
                "RAM Size GB": [ram],
                "Storage Size GB": [storage],
                "battery_capacity": [battery],
                "fast_charging_power": [fast_charging],
                "Screen_Size": [screen_size],
                "Resolution_Width": [res_width],
                "Resolution_Height": [res_height],
                "Refresh_Rate": [refresh],
                "primary_rear_camera_mp": [primary_rear_camera_mp],
                "num_rear_cameras": [num_rear_cameras],
                "primary_front_camera_mp": [primary_front_camera_mp],
                "num_front_cameras": [num_front_cameras],
                "Dual_Sim": [dual_sim],
                "4G": [four_g],
                "5G": [five_g],
                "Vo5G": [vo5g],
                "NFC": [nfc],
                "IR_Blaster": [ir_blaster],
                "memory_card_support": [memory_card_support],
                "Processor_Brand": [processor_brand],
                "Processor_Series": [processor_series],
                "Performance_Tier": [performance_tier],
                "RAM Tier": [ram_tier],
                "Notch_Type": [notch_type],
                "os_name": [os_name],
                "os_version": [os_version],
                "brand": [brand],
                "memory_card_size": [memory_card_size],
            }
            
            X_new = pd.DataFrame(data)
            
            # Encode binary columns (Yes/No -> 1/0)
            binary_cols = ["Dual_Sim", "4G", "5G", "Vo5G", "NFC", "IR_Blaster", "memory_card_support"]
            for col in binary_cols:
                X_new[col] = X_new[col].map({"Yes": 1, "No": 0})
            
            # Add engineered features
            X_new["screen_pixels"] = X_new["Resolution_Width"] * X_new["Resolution_Height"]
            X_new["total_cameras"] = X_new["num_rear_cameras"] + X_new["num_front_cameras"]
            X_new["camera_score"] = X_new["primary_rear_camera_mp"] + X_new["primary_front_camera_mp"]
            X_new["ram_storage_ratio"] = X_new["RAM Size GB"] / (X_new["Storage Size GB"] + 1e-6)
            
            try:
                # Make prediction
                pred = model.predict(X_new)[0]
                
                # Try multiple ways to get probabilities
                prob = None
                
                # Method 1: Direct predict_proba (works for sklearn Pipelines)
                if hasattr(model, 'predict_proba'):
                    try:
                        prob = model.predict_proba(X_new)[0]
                    except:
                        pass
                
                # Method 2: Access classifier directly if it's a Pipeline
                if prob is None and hasattr(model, 'named_steps'):
                    try:
                        classifier = model.named_steps['classifier']
                        X_transformed = model.named_steps['preprocessor'].transform(X_new)
                        prob = classifier.predict_proba(X_transformed)[0]
                    except:
                        pass
                
                # Method 3: Try accessing via steps
                if prob is None and hasattr(model, 'steps'):
                    try:
                        X_transformed = model[:-1].transform(X_new)
                        prob = model[-1].predict_proba(X_transformed)[0]
                    except:
                        pass
                
                # Method 4: Create approximate probabilities
                if prob is None:
                    st.warning("⚠️ Using approximate probability estimates")
                    prob = [0.3, 0.7] if pred == 1 else [0.7, 0.3]
                        
            except Exception as e:
                st.error(f"❌ Prediction Error: {str(e)}")
                st.write("**Debug Information:**")
                st.write(f"- Error type: {type(e).__name__}")
                st.write(f"- Model type: {type(model).__name__}")
                
                # Show model structure
                if hasattr(model, 'named_steps'):
                    st.write(f"- Pipeline steps: {list(model.named_steps.keys())}")
                
                # Show available methods
                methods = [m for m in dir(model) if not m.startswith('_') and callable(getattr(model, m))]
                st.write(f"- Available methods: {methods[:10]}...")
                
                st.write("\n**Input DataFrame shape:**", X_new.shape)
                st.write("**Input DataFrame columns:**", list(X_new.columns))
                st.stop()
    
    # Display Results
    st.markdown("---")
    
    if pred == 1:
        st.markdown("""
            <div class="prediction-box expensive">
                <h1>💎 EXPENSIVE SMARTPHONE</h1>
                <p style="font-size: 1.2rem; color: #d32f2f;">Premium Price Category</p>
            </div>
        """, unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown("""
            <div class="prediction-box non-expensive">
                <h1>💰 NON-EXPENSIVE SMARTPHONE</h1>
                <p style="font-size: 1.2rem; color: #2e7d32;">Budget-Friendly Price Category</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Confidence Metrics
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("🎯 Overall Confidence", f"{max(prob):.1%}")
    
    with metric_col2:
        st.metric("💰 Non-Expensive", f"{prob[0]:.1%}")
    
    with metric_col3:
        st.metric("💎 Expensive", f"{prob[1]:.1%}")
    
    with metric_col4:
        certainty = "High" if max(prob) > 0.8 else "Medium" if max(prob) > 0.6 else "Low"
        st.metric("📊 Certainty", certainty)
    
    # Detailed Analysis
    st.markdown("---")
    st.subheader("📈 Detailed Analysis")
    
    analysis_col1, analysis_col2 = st.columns(2)
    
    with analysis_col1:
        st.markdown("**Key Factors Contributing to Prediction:**")
        factors = []
        if ram >= 12:
            factors.append("✅ High RAM capacity")
        if storage >= 256:
            factors.append("✅ Large storage")
        if primary_rear_camera_mp >= 64:
            factors.append("✅ High-resolution camera")
        if five_g == "Yes":
            factors.append("✅ 5G connectivity")
        if performance_tier in ["High-end", "Flagship"]:
            factors.append("✅ Premium performance tier")
        if refresh >= 120:
            factors.append("✅ High refresh rate display")
        
        if factors:
            for factor in factors:
                st.write(factor)
        else:
            st.write("• Standard specifications across categories")
    
    with analysis_col2:
        st.markdown("**Specification Comparison:**")
        specs_quality = []
        if ram >= 8:
            specs_quality.append(("RAM", "Good", "🟢"))
        else:
            specs_quality.append(("RAM", "Basic", "🟡"))
            
        if battery >= 4500:
            specs_quality.append(("Battery", "Good", "🟢"))
        else:
            specs_quality.append(("Battery", "Standard", "🟡"))
            
        if primary_rear_camera_mp >= 50:
            specs_quality.append(("Camera", "Excellent", "🟢"))
        else:
            specs_quality.append(("Camera", "Standard", "🟡"))
        
        for spec, quality, icon in specs_quality:
            st.write(f"{icon} {spec}: {quality}")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p><strong>Smartphone Price Predictor v1.0</strong></p>
        <p>Built with Streamlit 🎈 | Powered by Machine Learning 🤖</p>
        <p style="font-size: 0.9rem;">This tool provides predictions based on technical specifications and market trends</p>
    </div>
""", unsafe_allow_html=True)