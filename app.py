import shap
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

#  Page Config 
st.set_page_config(page_title="Smart Factory 6G Dashboard", layout="wide")

# Load Data and Models 
@st.cache_data
def load_data():
    df = pd.read_csv("Thales_Group_Manufacturing.csv")
    # Basic clean up for visual display
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Timestamp'], format='%d-%m-%Y %H:%M:%S')
    df = df.sort_values('Datetime')
    return df

@st.cache_resource
def load_models():
    model = joblib.load('xgb_efficiency_model.pkl')
    scaler = joblib.load('feature_scaler.pkl')
    encoder = joblib.load('target_encoder.pkl')
    return model, scaler, encoder

df = load_data()
try:
    xgb_model, scaler, le_target = load_models()
    model_loaded = True
except FileNotFoundError:
    st.error("Model files not found. Please run the pipeline script to save the .pkl files.")
    model_loaded = False

# Sidebar Controls 
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/121px-Python-logo-notext.svg.png", width=50)
st.sidebar.title("Factory Controls")

st.sidebar.subheader("Historical Filters")
selected_machines = st.sidebar.multiselect("Select Machine IDs", options=df['Machine_ID'].unique(), default=df['Machine_ID'].unique()[:3])
max_latency = st.sidebar.slider("Max 6G Latency Filter (ms)", min_value=0, max_value=int(df['Network_Latency_ms'].max()), value=100)

st.sidebar.markdown("---")
st.sidebar.subheader("Live What-If Analysis")
st.sidebar.caption("Adjust sliders to simulate live telemetry and predict efficiency.")
sim_temp = st.sidebar.slider("Temperature (°C)", 20.0, 120.0, 75.0)
sim_vib = st.sidebar.slider("Vibration (Hz)", 0.0, 10.0, 3.5)
sim_lat = st.sidebar.slider("6G Latency (ms)", 1.0, 150.0, 15.0)
sim_power = st.sidebar.slider("Power (kW)", 1.0, 20.0, 5.0)

# Main UI 
st.title("🏭 AI-Based Manufacturing Efficiency & 6G Network Monitor")

if model_loaded:
    # 1. Live Prediction Module (What-If Simulation)
    st.subheader("1. Live Prediction Status (Simulated)")
    
    # Create synthetic feature array based on sliders (using median values for non-slider features)
    sim_features = np.array([[
        1, # Operation Mode (1 = Active)
        sim_temp, sim_vib, sim_power, sim_lat, 
        2.0, # Packet Loss
        2.5, # Defect Rate
        350, # Production Speed
        0.8, # Maint Score
        5.0, # Error Rate
        350/sim_power, # Energy Ratio
        5.0/350,       # Error Ratio
        sim_lat * (1 + 2.0/100), # Net Score
        sim_temp * sim_vib       # Stress Index
    ]])
    
    sim_scaled = scaler.transform(sim_features)
    pred_prob = xgb_model.predict_proba(sim_scaled)[0]
    pred_class_idx = np.argmax(pred_prob)
    pred_class_label = le_target.inverse_transform([pred_class_idx])[0]
    confidence = pred_prob[pred_class_idx] * 100

    col1, col2, col3 = st.columns([1, 1, 2])
    
    status_color = "🟢" if pred_class_label == "High" else "🟡" if pred_class_label == "Medium" else "🔴"
    col1.metric("Predicted Efficiency", f"{status_color} {pred_class_label}")
    col2.metric("AI Confidence", f"{confidence:.1f}%")
    
    # Confidence Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number", value = confidence, title = {'text': "Confidence Level"},
        gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "blue"}}
    ))
    fig_gauge.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
    col3.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("---")

    # Historical & Operational Analytics
    st.subheader("2. Operational Analytics")
    
    # Filter Data
    filtered_df = df[(df['Machine_ID'].isin(selected_machines)) & (df['Network_Latency_ms'] <= max_latency)]
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("**Efficiency Trends Over Time**")
        # Sample down for rendering speed if dataset is large
        trend_df = filtered_df.sample(min(1000, len(filtered_df))).sort_values('Datetime')
        fig_trend = px.line(trend_df, x='Datetime', y='Production_Speed_units_per_hr', color='Efficiency_Status', title="Production Output by Efficiency Status")
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with col_b:
        st.markdown("**6G Network vs. Sensor Impact**")
        fig_scatter = px.scatter(filtered_df.sample(min(1000, len(filtered_df))), 
                                 x='Network_Latency_ms', y='Temperature_C', 
                                 color='Efficiency_Status', 
                                 title="Latency vs Temperature Profile")
        st.plotly_chart(fig_scatter, use_container_width=True)

    # 3. Explainability Panel (Root Cause Analysis)
    st.markdown("---")
    st.subheader("3. Explainability Panel (Root Cause Analysis)")
    
    with st.spinner("Calculating live feature contributions..."):
        # Ensure feature names match the exact training order
        feature_names = [
            'Operation_Mode', 'Temperature_C', 'Vibration_Hz', 'Power_Consumption_kW', 
            'Network_Latency_ms', 'Packet_Loss_%', 'Quality_Control_Defect_Rate_%', 
            'Production_Speed_units_per_hr', 'Predictive_Maintenance_Score', 'Error_Rate_%',
            'Energy_Efficiency_Ratio', 'Error_to_Output_Ratio', 'Network_Reliability_Score', 'Sensor_Stress_Index'
        ]
        
        # Initialize the TreeExplainer
        explainer = shap.TreeExplainer(xgb_model)
        
        # Calculate SHAP values for the simulated feature array (the What-If sliders)
        shap_values = explainer.shap_values(sim_scaled)
        
        # Handle modern SHAP 3D array return: (samples, features, classes)
        if isinstance(shap_values, list):
            target_class_shap = shap_values[pred_class_idx][0]
        else:
            target_class_shap = shap_values[0, :, pred_class_idx]
        
        # Create a clean horizontal bar chart using Matplotlib
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Sort features by absolute impact magnitude 
        impact_magnitudes = np.abs(target_class_shap)
        sorted_idx = np.argsort(impact_magnitudes)
        
        # Select the top 7 driving features for a clean UI
        top_features = [feature_names[i] for i in sorted_idx[-7:]]
        top_shap_vals = target_class_shap[sorted_idx[-7:]]
        
        # Color code: Red pushes toward the predicted class, Blue pushes away
        colors = ['#ff4b4b' if val > 0 else '#1f77b4' for val in top_shap_vals]
        
        ax.barh(top_features, top_shap_vals, color=colors)
        ax.set_xlabel(f"Impact on predicting '{pred_class_label}' Status")
        ax.set_title("Top 7 Drivers for Current Prediction (Live What-If)")
        
        # Render the plot in Streamlit
        st.pyplot(fig)
