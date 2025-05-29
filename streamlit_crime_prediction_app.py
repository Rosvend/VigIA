"""
Crime Risk Prediction Streamlit App - Prototype Deployment
A simple GUI prototype for crime risk prediction using the best trained model.
This app allows users to input H3 IDs and get risk scores without complex feature engineering.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Configure Streamlit page
st.set_page_config(
    page_title="Crime Risk Prediction - Medellín",
    page_icon="🚔",
    layout="wide"
)

# App title and description
st.title("🚔 Crime Risk Prediction System - Medellín")
st.markdown("""
This prototype predicts crime risk for specific H3 grid cells in Medellín using machine learning.
Select an H3 ID from the predefined list to see the predicted risk score.
""")

# Sample H3 IDs with pre-computed features for prototype
# These represent different risk levels across Medellín
SAMPLE_H3_DATA = {
    "882a1072e7fffff": {  # High-risk area (Centro)
        "location": "Centro - Downtown",
        "distance_to_police": 0.8,
        "crime_count": 45,
        "crimes_last_1d": 2.5,
        "crimes_last_7d": 8.3,
        "crimes_last_30d": 18.7,
        "barrios_count": 3,
        "ref_hour": 14,
        "ref_day": 2,
        "ref_month": 8,
        "ref_is_weekend": 0,
        "ref_is_day_shift": 1,
        "ref_is_night_shift": 0,
        "expected_risk": "High"
    },
    "882a1072cfffff": {  # Medium-risk area (Poblado)
        "location": "El Poblado - Commercial",
        "distance_to_police": 1.2,
        "crime_count": 23,
        "crimes_last_1d": 1.2,
        "crimes_last_7d": 4.8,
        "crimes_last_30d": 12.3,
        "barrios_count": 2,
        "ref_hour": 14,
        "ref_day": 2,
        "ref_month": 8,
        "ref_is_weekend": 0,
        "ref_is_day_shift": 1,
        "ref_is_night_shift": 0,
        "expected_risk": "Medium"
    },
    "882a10729fffff": {  # Low-risk area (Laureles)
        "location": "Laureles - Residential",
        "distance_to_police": 2.1,
        "crime_count": 8,
        "crimes_last_1d": 0.3,
        "crimes_last_7d": 1.4,
        "crimes_last_30d": 4.2,
        "barrios_count": 1,
        "ref_hour": 14,
        "ref_day": 2,
        "ref_month": 8,
        "ref_is_weekend": 0,
        "ref_is_day_shift": 1,
        "ref_is_night_shift": 0,
        "expected_risk": "Low"
    },
    "882a107367fffff": {  # Medium-High risk area (Bello)
        "location": "Bello - Urban",
        "distance_to_police": 1.8,
        "crime_count": 32,
        "crimes_last_1d": 1.8,
        "crimes_last_7d": 6.2,
        "crimes_last_30d": 15.1,
        "barrios_count": 2,
        "ref_hour": 14,
        "ref_day": 2,
        "ref_month": 8,
        "ref_is_weekend": 0,
        "ref_is_day_shift": 1,
        "ref_is_night_shift": 0,
        "expected_risk": "Medium-High"
    },
    "882a10721fffff": {  # Very Low risk area (Envigado)
        "location": "Envigado - Suburban",
        "distance_to_police": 3.2,
        "crime_count": 3,
        "crimes_last_1d": 0.1,
        "crimes_last_7d": 0.6,
        "crimes_last_30d": 1.8,
        "barrios_count": 1,
        "ref_hour": 14,
        "ref_day": 2,
        "ref_month": 8,
        "ref_is_weekend": 0,
        "ref_is_day_shift": 1,
        "ref_is_night_shift": 0,
        "expected_risk": "Very Low"
    }
}

@st.cache_data
def load_model():
    """Load the best trained model"""
    try:
        # Try to load the best optimized model first
        model_path = "best_crime_model_12h.pkl"
        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
            st.success(f"✅ Loaded optimized model: {model_data.get('model_name', 'Unknown')}")
            return model_data
        
        # Fallback to other model files
        model_files = [f for f in os.listdir('.') if f.endswith('_crime_prediction_12h.pkl')]
        if model_files:
            model_path = model_files[0]  # Use first available model
            model_data = joblib.load(model_path)
            st.warning(f"⚠️ Using fallback model: {model_path}")
            return model_data
        
        st.error("❌ No trained model found! Please run training first.")
        return None
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

def prepare_features(h3_data, time_settings=None):
    """
    Prepare features for prediction from sample H3 data
    
    Args:
        h3_data: Dictionary with H3 cell features
        time_settings: Optional dictionary to override temporal features
    """
    # Base features from H3 data
    features = {
        'distance_to_police': h3_data['distance_to_police'],
        'crime_count': h3_data['crime_count'],
        'crimes_last_1d': h3_data['crimes_last_1d'],
        'crimes_last_7d': h3_data['crimes_last_7d'],
        'crimes_last_30d': h3_data['crimes_last_30d'],
        'barrios_count': h3_data['barrios_count']
    }
    
    # Temporal features (can be overridden)
    if time_settings:
        features.update({
            'ref_hour': time_settings.get('hour', h3_data['ref_hour']),
            'ref_day': time_settings.get('day_of_week', h3_data['ref_day']),
            'ref_month': time_settings.get('month', h3_data['ref_month']),
            'ref_is_weekend': 1 if time_settings.get('day_of_week', h3_data['ref_day']) >= 5 else 0,
            'ref_is_day_shift': 1 if 6 <= time_settings.get('hour', h3_data['ref_hour']) < 18 else 0,
            'ref_is_night_shift': 1 if time_settings.get('hour', h3_data['ref_hour']) < 6 or time_settings.get('hour', h3_data['ref_hour']) >= 18 else 0
        })
    else:
        features.update({
            'ref_hour': h3_data['ref_hour'],
            'ref_day': h3_data['ref_day'],
            'ref_month': h3_data['ref_month'],
            'ref_is_weekend': h3_data['ref_is_weekend'],
            'ref_is_day_shift': h3_data['ref_is_day_shift'],
            'ref_is_night_shift': h3_data['ref_is_night_shift']
        })
    
    return pd.DataFrame([features])

def get_risk_level(risk_score):
    """Convert risk score to risk level"""
    if risk_score < 0.2:
        return "Very Low", "🟢"
    elif risk_score < 0.4:
        return "Low", "🟡"
    elif risk_score < 0.6:
        return "Medium", "🟠"
    elif risk_score < 0.8:
        return "High", "🔴"
    else:
        return "Very High", "🚨"

def create_risk_gauge(risk_score):
    """Create a risk gauge visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Crime Risk Score (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 20], 'color': "lightgreen"},
                {'range': [20, 40], 'color': "yellow"},
                {'range': [40, 60], 'color': "orange"},
                {'range': [60, 80], 'color': "red"},
                {'range': [80, 100], 'color': "darkred"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def main():
    """Main Streamlit app"""
    
    # Load model
    model_data = load_model()
    
    if model_data is None:
        st.stop()
    
    # Sidebar for settings
    st.sidebar.header("⚙️ Prediction Settings")
    
    # H3 ID selection
    st.sidebar.subheader("📍 Location Selection")
    h3_options = {f"{h3_id} ({data['location']})": h3_id for h3_id, data in SAMPLE_H3_DATA.items()}
    selected_display = st.sidebar.selectbox("Select H3 Grid Cell:", list(h3_options.keys()))
    selected_h3 = h3_options[selected_display]
    
    # Time settings
    st.sidebar.subheader("🕐 Time Settings")
    use_custom_time = st.sidebar.checkbox("Override time settings")
    
    time_settings = None
    if use_custom_time:
        current_time = datetime.now()
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            hour = st.slider("Hour", 0, 23, current_time.hour)
            month = st.slider("Month", 1, 12, current_time.month)
        
        with col2:
            day_of_week = st.selectbox("Day of Week", 
                                     options=list(range(7)),
                                     format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x],
                                     index=current_time.weekday())
        
        time_settings = {
            'hour': hour,
            'day_of_week': day_of_week,
            'month': month
        }
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎯 Risk Prediction")
        
        # Get selected H3 data
        h3_data = SAMPLE_H3_DATA[selected_h3]
        
        # Display location info
        st.subheader(f"📍 {h3_data['location']}")
        st.write(f"**H3 ID:** `{selected_h3}`")
        st.write(f"**Expected Risk Level:** {h3_data['expected_risk']}")
        
        # Prepare features and make prediction
        with st.spinner("🔮 Predicting crime risk..."):
            try:
                features_df = prepare_features(h3_data, time_settings)
                
                # Make prediction
                if 'pipeline' in model_data:
                    # Use the complete pipeline
                    risk_score = model_data['pipeline'].predict_proba(features_df)[0, 1]
                else:
                    # Use model + preprocessor separately
                    model = model_data['model']
                    preprocessor = model_data['preprocessor']
                    
                    # Transform features
                    features_processed = preprocessor.transform(features_df)
                    risk_score = model.predict_proba(features_processed)[0, 1]
                
                # Display results
                risk_level, risk_emoji = get_risk_level(risk_score)
                
                st.success("✅ Prediction completed!")
                
                # Risk score display
                col_score1, col_score2, col_score3 = st.columns(3)
                
                with col_score1:
                    st.metric("Risk Score", f"{risk_score:.3f}", f"{risk_score*100:.1f}%")
                
                with col_score2:
                    st.metric("Risk Level", f"{risk_emoji} {risk_level}")
                
                with col_score3:
                    confidence = abs(risk_score - 0.5) * 2  # Distance from uncertain (0.5)
                    st.metric("Confidence", f"{confidence:.3f}", f"{confidence*100:.1f}%")
                
                # Risk gauge
                st.plotly_chart(create_risk_gauge(risk_score), use_container_width=True)
                
                # Feature importance (if available)
                st.subheader("📊 Feature Analysis")
                feature_importance_data = {
                    'Feature': list(features_df.columns),
                    'Value': features_df.iloc[0].values
                }
                st.dataframe(pd.DataFrame(feature_importance_data), use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
    
    with col2:
        st.header("ℹ️ Information")
        
        # Model information
        if model_data:
            st.subheader("🤖 Model Details")
            st.write(f"**Model Type:** {model_data.get('model_name', 'Unknown')}")
            st.write(f"**Prediction Window:** {model_data.get('prediction_window', 12)} hours")
            
            if 'metrics' in model_data:
                metrics = model_data['metrics']
                if 'validation' in metrics:
                    val_metrics = metrics['validation']
                    st.write(f"**Validation PR AUC:** {val_metrics.get('pr_auc', 'N/A'):.3f}")
                    st.write(f"**Validation ROC AUC:** {val_metrics.get('roc_auc', 'N/A'):.3f}")
            
            if 'training_date' in model_data:
                training_date = pd.to_datetime(model_data['training_date']).strftime('%Y-%m-%d %H:%M')
                st.write(f"**Training Date:** {training_date}")
        
        # Risk interpretation
        st.subheader("🎯 Risk Levels")
        st.write("🟢 **Very Low (0-20%):** Minimal crime risk")
        st.write("🟡 **Low (20-40%):** Below average risk")
        st.write("🟠 **Medium (40-60%):** Average risk")
        st.write("🔴 **High (60-80%):** Above average risk")
        st.write("🚨 **Very High (80-100%):** Critical risk")
        
        # Sample locations
        st.subheader("📍 Available Locations")
        for h3_id, data in SAMPLE_H3_DATA.items():
            st.write(f"• {data['location']}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Crime Risk Prediction System for Medellín**  
    *University Project - Machine Learning for Crime Prevention*  
    
    ⚠️ **Disclaimer:** This is a prototype for educational purposes. 
    Risk predictions should not be used as the sole basis for operational decisions.
    """)

if __name__ == "__main__":
    main()