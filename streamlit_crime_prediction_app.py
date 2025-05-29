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
        "comuna": 10,  # Add comuna information
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
        "comuna": 14,  # Add comuna information
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
        "comuna": 11,  # Add comuna information
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
        "comuna": 3,  # Add comuna information
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
        "comuna": 16,  # Add comuna information
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
        # Try to load the fixed model first, then fallback to original
        model_paths = [
            "ml/fixed_crime_model_12h.pkl",  # Try fixed model first
            "ml/best_crime_model_12h.pkl",
            "best_crime_model_12h.pkl",  # fallback for current directory
            "ml/crime_prediction_deployment_pipeline_12h.pkl"
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                model_data = joblib.load(model_path)
                st.success(f"✅ Loaded model: {model_data.get('model_name', 'Unknown')} from {model_path}")
                return model_data
        
        # Fallback to any available model files in ml directory
        ml_dir = "ml"
        if os.path.exists(ml_dir):
            model_files = [f for f in os.listdir(ml_dir) if f.endswith('_crime_prediction_12h.pkl')]
            if model_files:
                model_path = os.path.join(ml_dir, model_files[0])
                model_data = joblib.load(model_path)
                st.warning(f"⚠️ Using fallback model: {model_files[0]}")
                return model_data
        
        st.error("❌ No trained model found! Please run training first.")
        st.info("Looking for models in: ml/fixed_crime_model_12h.pkl")
        return None
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

def prepare_features(h3_data, time_settings=None):
    """
    Prepare features for prediction from sample H3 data to match the expected 26 features
    
    Args:
        h3_data: Dictionary with H3 cell features
        time_settings: Optional dictionary to override temporal features
    """
    # Base numerical features (11 features)
    features = {
        'crime_count': float(h3_data['crime_count']),
        'distance_to_police': float(h3_data['distance_to_police']),
        'barrios_count': int(h3_data['barrios_count']),
        'crimes_last_1d': float(h3_data['crimes_last_1d']),
        'crimes_last_7d': float(h3_data['crimes_last_7d']),
        'crimes_last_30d': float(h3_data['crimes_last_30d'])
    }
    
    # Temporal features (can be overridden)
    if time_settings:
        features.update({
            'ref_hour': int(time_settings.get('hour', h3_data['ref_hour'])),
            'ref_day': int(time_settings.get('day_of_week', h3_data['ref_day'])),
            'ref_is_weekend': int(1 if time_settings.get('day_of_week', h3_data['ref_day']) >= 5 else 0),
            'ref_is_day_shift': int(1 if 6 <= time_settings.get('hour', h3_data['ref_hour']) < 18 else 0),
            'ref_is_night_shift': int(1 if time_settings.get('hour', h3_data['ref_hour']) < 6 or time_settings.get('hour', h3_data['ref_hour']) >= 18 else 0)
        })
    else:
        features.update({
            'ref_hour': int(h3_data['ref_hour']),
            'ref_day': int(h3_data['ref_day']),
            'ref_is_weekend': int(h3_data['ref_is_weekend']),
            'ref_is_day_shift': int(h3_data['ref_is_day_shift']),
            'ref_is_night_shift': int(h3_data['ref_is_night_shift'])
        })
    
    # Add the categorical comuna feature (CRITICAL - this was missing!)
    # Ensure it's treated as a categorical/string, not numeric
    features['comuna'] = str(int(h3_data['comuna']))  # Convert to string to avoid numeric issues
    
    # Create DataFrame
    features_df = pd.DataFrame([features])
    
    # Ensure proper data types
    # Numerical columns
    numerical_cols = [
        'crime_count', 'distance_to_police', 'barrios_count', 'ref_hour', 'ref_day',
        'ref_is_weekend', 'ref_is_day_shift', 'ref_is_night_shift',
        'crimes_last_1d', 'crimes_last_7d', 'crimes_last_30d'
    ]
    for col in numerical_cols:
        if col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
    
    # Categorical column
    if 'comuna' in features_df.columns:
        features_df['comuna'] = features_df['comuna'].astype('category')
    
    # Ensure column order matches training data
    expected_columns = [
        'crime_count', 'distance_to_police', 'barrios_count', 'ref_hour', 'ref_day',
        'ref_is_weekend', 'ref_is_day_shift', 'ref_is_night_shift',
        'crimes_last_1d', 'crimes_last_7d', 'crimes_last_30d', 'comuna'
    ]
    
    # Reorder columns to match expected order
    features_df = features_df[expected_columns]
    
    return features_df

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
                
                # Debug: Show what we're trying to predict with
                st.info(f"Preparing prediction for {len(features_df.columns)} features")
                
                # Make prediction using the correct approach
                risk_score = None
                prediction_method = "unknown"
                
                try:
                    # First, try to use the pipeline if it exists and seems to be fitted
                    if 'pipeline' in model_data and model_data['pipeline'] is not None:
                        pipeline = model_data['pipeline']
                        
                        # Check if pipeline is fitted by trying to get feature names
                        try:
                            # Try a small test to see if pipeline is fitted
                            test_features = features_df.iloc[:1]  # Just first row
                            pipeline.predict_proba(test_features)
                            # If we get here, pipeline works
                            risk_score = pipeline.predict_proba(features_df)[0, 1]
                            prediction_method = "complete_pipeline"
                            st.success("✅ Using complete fitted pipeline")
                            
                        except Exception as pipeline_error:
                            st.warning(f"Pipeline not properly fitted: {str(pipeline_error)}")
                            # Fall back to model + preprocessor approach
                            
                            if 'model' in model_data and 'preprocessor' in model_data:
                                st.info("Falling back to separate model + preprocessor")
                                model = model_data['model']
                                preprocessor = model_data['preprocessor']
                                
                                # Check if preprocessor is fitted
                                try:
                                    # Try to get feature names to verify it's fitted
                                    feature_names = preprocessor.get_feature_names_out()
                                    st.info(f"Preprocessor has {len(feature_names)} feature names")
                                    
                                    # Transform features
                                    features_processed = preprocessor.transform(features_df)
                                    risk_score = model.predict_proba(features_processed)[0, 1]
                                    prediction_method = "separate_fitted_preprocessor"
                                    st.success("✅ Using fitted preprocessor + model")
                                    
                                except Exception as preprocessor_error:
                                    st.warning(f"Preprocessor not fitted: {str(preprocessor_error)}")
                                    
                                    # Last resort: try to use the model directly with raw features
                                    # This only works for simple models that don't require preprocessing
                                    try:
                                        # For XGBoost, we might be able to use numeric features directly
                                        numeric_features = features_df.select_dtypes(include=[np.number])
                                        if len(numeric_features.columns) > 0:
                                            st.info("Attempting direct model prediction with numeric features only")
                                            risk_score = model.predict_proba(numeric_features.values)[0, 1]
                                            prediction_method = "direct_numeric_only"
                                            st.warning("⚠️ Using simplified prediction (numeric features only)")
                                        else:
                                            raise Exception("No numeric features available for direct prediction")
                                            
                                    except Exception as direct_error:
                                        st.error(f"Direct prediction failed: {str(direct_error)}")
                                        risk_score = None
                            else:
                                st.error("No valid model or preprocessor found in model data")
                                risk_score = None
                    
                    else:
                        # Try the separate model + preprocessor approach directly
                        if 'model' in model_data and 'preprocessor' in model_data:
                            st.info("No pipeline found, using separate model + preprocessor")
                            model = model_data['model']
                            preprocessor = model_data['preprocessor']
                            
                            try:
                                feature_names = preprocessor.get_feature_names_out()
                                features_processed = preprocessor.transform(features_df)
                                risk_score = model.predict_proba(features_processed)[0, 1]
                                prediction_method = "separate_model_preprocessor"
                                st.success("✅ Using separate model + preprocessor")
                                
                            except Exception as sep_error:
                                st.error(f"Separate model/preprocessor failed: {str(sep_error)}")
                                risk_score = None
                        else:
                            st.error("No valid prediction method found")
                            risk_score = None
                
                except Exception as prediction_error:
                    st.error(f"Prediction process failed: {str(prediction_error)}")
                    st.info("Attempting emergency fallback...")
                    
                    # Emergency fallback: create a mock prediction based on input features
                    # This is just for demonstration purposes
                    try:
                        # Simple heuristic based on crime count and distance to police
                        crime_count = h3_data.get('crime_count', 0)
                        distance_to_police = h3_data.get('distance_to_police', 5)
                        
                        # Simple risk calculation: higher crime count and farther from police = higher risk
                        base_risk = min(crime_count / 50.0, 0.8)  # Normalize crime count
                        distance_penalty = min(distance_to_police / 10.0, 0.2)  # Distance penalty
                        risk_score = min(base_risk + distance_penalty, 0.95)
                        
                        prediction_method = "emergency_heuristic"
                        st.warning("⚠️ Using emergency heuristic prediction (for demonstration only)")
                        st.info("This is not a real ML prediction - please check model training")
                        
                    except Exception:
                        st.error("All prediction methods failed")
                        risk_score = None
                
                # Only proceed if we have a valid risk score
                if risk_score is not None:
                    # Display prediction method used
                    st.info(f"Prediction method: {prediction_method}")
                    
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
                    
                else:
                    st.error("❌ Unable to generate prediction")
                    st.info("Please check the model training and ensure the pipeline is properly saved")
                    
                    # Show debugging information
                    with st.expander("🔧 Debug Information"):
                        st.write("Model data structure:")
                        for key in model_data.keys():
                            if key != 'model':  # Don't try to display the actual model object
                                try:
                                    st.write(f"- {key}: {type(model_data[key])}")
                                except:
                                    st.write(f"- {key}: <unable to display>")
                        
                        st.write("Input features:")
                        st.dataframe(features_df)
                
            except Exception as e:
                st.error(f"❌ Critical prediction error: {e}")
                st.info("This indicates a serious issue with the model or preprocessing pipeline")
                
                # Emergency information display
                st.subheader("📍 Location Information (No Prediction)")
                st.write(f"**H3 ID:** `{selected_h3}`")
                st.write(f"**Location:** {h3_data['location']}")
                st.write(f"**Expected Risk:** {h3_data['expected_risk']}")
                st.warning("Please retrain the model or check the model saving process")
    
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