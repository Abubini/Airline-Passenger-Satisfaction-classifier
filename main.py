# main.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import base64

# Add src to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import your existing modules
try:
    from preprocess import preprocess_and_scale, CONTINUOUS_FEATURES
    from train import train_all_models
    from evaluate import load_models, load_and_split_data, evaluate_all_models, compare_performance, create_visualizations
    from classify import create_sample_dataframe, preprocess_sample_data, prepare_for_scaled_models, load_all_models, ensure_feature_order, make_predictions
    from summary import PROJECT_DESCRIPTION
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Define constants
TARGET = 'Satisfaction'

def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0a192f, #112240);
        color: #fff;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a192f, #112240);
    }
    
    .dashboard-card {
        background: rgba(17, 34, 64, 0.7);
        padding: 25px;
        border-radius: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        border: 1px solid rgba(64, 224, 208, 0.1);
    }
    
    .metric-card {
        background: rgba(64, 224, 208, 0.1);
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        border: 1px solid rgba(64, 224, 208, 0.2);
    }
    
    .prediction-item {
        display: flex;
        justify-content: space-between;
        margin: 10px 0;
        padding: 8px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stButton button {
        background: linear-gradient(135deg, #40e0d0, #20b2aa);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 50px;
        font-weight: 500;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(64, 224, 208, 0.3);
        width: 100%;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(64, 224, 208, 0.4);
    }
    
    h1, h2, h3 {
        color: #fff;
    }
    
    h1 {
        color: #40e0d0;
        text-align: center;
    }
    
    .css-1d391kg, .css-12oz5g7 {
        background: linear-gradient(135deg, #0a192f, #112240);
    }
    
    .input-section {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(64, 224, 208, 0.2);
    }
    
    .result-container {
        background: rgba(64, 224, 208, 0.08);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid rgba(64, 224, 208, 0.3);
    }
    
    .result-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .result-row:last-child {
        border-bottom: none;
    }
    
    .tab-content {
        padding: 20px 0;
    }
    
    /* Status indicators */
    .status-high {
        color: #40e0d0;
        font-weight: 600;
    }
    
    .status-medium {
        color: #ffa500;
        font-weight: 600;
    }
    
    .status-low {
        color: #ff6b6b;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: rgba(17, 34, 64, 0.9);
        border-right: 1px solid rgba(64, 224, 208, 0.2);
    }
    
    /* Custom metric boxes */
    .custom-metric {
        background: linear-gradient(135deg, rgba(64, 224, 208, 0.1), rgba(32, 178, 170, 0.1));
        border-radius: 10px;
        padding: 15px;
        margin: 10px;
        border: 1px solid rgba(64, 224, 208, 0.3);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #40e0d0, #20b2aa);
    }
    
    /* Dataframe styling */
    .dataframe {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 10px;
    }
    
    .dataframe th {
        background: rgba(64, 224, 208, 0.2);
        color: white;
    }
    
    .dataframe td {
        color: #ddd;
    }
    
    /* Alert styling */
    .stAlert {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    /* Visualization gallery styling */
    .viz-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid rgba(64, 224, 208, 0.2);
        transition: transform 0.3s;
    }
    
    .viz-card:hover {
        transform: translateY(-5px);
        border-color: rgba(64, 224, 208, 0.5);
        box-shadow: 0 5px 20px rgba(64, 224, 208, 0.2);
    }
    
    .viz-title {
        color: #40e0d0;
        font-weight: 600;
        margin-bottom: 10px;
        text-align: center;
    }
    
    /* Image styling */
    .stImage img {
        border-radius: 10px;
        border: 2px solid rgba(64, 224, 208, 0.3);
    }
    
    /* Download button styling */
    .download-btn {
        background: linear-gradient(135deg, #40e0d0, #20b2aa);
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 14px;
        cursor: pointer;
        transition: all 0.3s;
        display: inline-block;
        text-decoration: none;
    }
    
    .download-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(64, 224, 208, 0.4);
    }
        /* Summary tab specific styling */
    .summary-section {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    
    .summary-section h2 {
        color: #40e0d0;
        border-bottom: 2px solid rgba(64, 224, 208, 0.3);
        padding-bottom: 10px;
    }
    
    .summary-section h3 {
        color: #20b2aa;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

def check_dependencies():
    """Check if required files and directories exist"""
    required_dirs = ['data/raw', 'data/processed', 'models', 'results']
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
    
    # Check for raw data
    if not os.path.exists('data/raw/airline_passenger_satisfaction.csv'):
        st.warning("⚠ Raw data file not found. Please upload data to proceed.")
        return False
    return True

def display_model_predictions(predictions):
    """Display model predictions in a formatted way"""
    if not predictions:
        st.error("No predictions available")
        return
    
    satisfied_count = 0
    total_models = 0
    results_data = []
    
    st.markdown("### 📊 Model Predictions")
    
    for model_name, pred in predictions.items():
        if pred is not None:
            total_models += 1
            if pred['prediction'] == 1:
                satisfied_count += 1
            
            confidence_str = f"{pred['confidence']:.1f}%" if pred['confidence'] is not None else "N/A"
            result = pred['result']
            
            results_data.append({
                'Model': model_name,
                'Prediction': result,
                'Confidence': confidence_str,
                'Satisfied': pred['prediction']
            })
    
    # Display as dataframe
    df_results = pd.DataFrame(results_data)
    
    # Color code the predictions
    def color_prediction(val):
        color = '#40e0d0' if val == 'SATISFIED' else '#ff6b6b'
        return f'background-color: {color}; color: white;'
    
    styled_df = df_results.style.applymap(
        lambda x: 'background-color: #40e0d0; color: white;' if x == 'SATISFIED' else 
                 ('background-color: #ff6b6b; color: white;' if x == 'NOT SATISFIED' else ''),
        subset=['Prediction']
    )
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Calculate consensus
    if total_models > 0:
        consensus = (max(satisfied_count, total_models - satisfied_count) / total_models) * 100
        majority = "SATISFIED" if satisfied_count > total_models - satisfied_count else "NOT SATISFIED"
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Models Predicting Satisfied", f"{satisfied_count}/{total_models}")
        with col2:
            st.metric("Majority Decision", majority)
        with col3:
            st.metric("Consensus Level", f"{consensus:.1f}%")
        
        if consensus >= 75:
            st.success("✓ Strong consensus among models")
        elif consensus >= 60:
            st.warning("○ Moderate consensus")
        else:
            st.error("⚠ Low consensus - models disagree")

def get_image_download_link(img_path, filename):
    """Generate a download link for an image"""
    with open(img_path, "rb") as file:
        img_bytes = file.read()
    b64 = base64.b64encode(img_bytes).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}" class="download-btn">📥 Download</a>'
    return href

def display_visualizations_gallery():
    """Display all visualizations from the visuals folder in a gallery format"""
    if not os.path.exists('visuals'):
        st.info("No visualizations found. Run model evaluation to generate visualizations.")
        return
    
    viz_files = [f for f in os.listdir('visuals') if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not viz_files:
        st.info("No image files found in the visuals folder.")
        return
    
    # Sort files for better organization
    viz_files.sort()
    
    # Create tabs for different categories of visualizations
    tab_names = ['📈 Performance', '📊 Confusion Matrices', '⚡ Speed Analysis', '🎯 All Visualizations']
    viz_tabs = st.tabs(tab_names)
    
    # Categorize visualizations
    performance_viz = [f for f in viz_files if 'performance' in f.lower() or 'comparison' in f.lower() 
                      or 'roc' in f.lower() or 'radar' in f.lower()]
    confusion_viz = [f for f in viz_files if 'confusion' in f.lower() or 'matrix' in f.lower()]
    speed_viz = [f for f in viz_files if 'speed' in f.lower() or 'time' in f.lower()]
    
    # Performance tab
    with viz_tabs[0]:
        if performance_viz:
            st.markdown("### Model Performance Visualizations")
            cols = st.columns(2)
            for idx, viz_file in enumerate(performance_viz):
                with cols[idx % 2]:
                    img_path = os.path.join('visuals', viz_file)
                    try:
                        img = Image.open(img_path)
                        
                        # Create an expander for each visualization
                        with st.expander(f"📊 {viz_file.replace('.png', '').replace('_', ' ').title()}", expanded=True):
                            st.image(img, use_container_width=True)
                            
                            # Add download button
                            st.markdown(get_image_download_link(img_path, viz_file), unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Could not load {viz_file}: {str(e)}")
        else:
            st.info("No performance visualizations found.")
    
    # Confusion Matrices tab
    with viz_tabs[1]:
        if confusion_viz:
            st.markdown("### Confusion Matrix Visualizations")
            if len(confusion_viz) > 1:
                # Show multiple confusion matrices
                cols = st.columns(min(3, len(confusion_viz)))
                for idx, viz_file in enumerate(confusion_viz):
                    with cols[idx % 3]:
                        img_path = os.path.join('visuals', viz_file)
                        try:
                            img = Image.open(img_path)
                            st.image(img, caption=viz_file.replace('.png', '').replace('_', ' ').title(), 
                                    use_container_width=True)
                            st.markdown(get_image_download_link(img_path, viz_file), unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Could not load {viz_file}: {str(e)}")
            else:
                # Show single confusion matrix
                viz_file = confusion_viz[0]
                img_path = os.path.join('visuals', viz_file)
                try:
                    img = Image.open(img_path)
                    st.image(img, use_container_width=True)
                    st.markdown(get_image_download_link(img_path, viz_file), unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Could not load {viz_file}: {str(e)}")
        else:
            st.info("No confusion matrix visualizations found.")
    
    # Speed Analysis tab
    with viz_tabs[2]:
        if speed_viz:
            st.markdown("### Prediction Speed Visualizations")
            for viz_file in speed_viz:
                img_path = os.path.join('visuals', viz_file)
                try:
                    img = Image.open(img_path)
                    
                    with st.expander(f"⚡ {viz_file.replace('.png', '').replace('_', ' ').title()}", expanded=True):
                        st.image(img, use_container_width=True)
                        st.markdown(get_image_download_link(img_path, viz_file), unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Could not load {viz_file}: {str(e)}")
        else:
            st.info("No speed analysis visualizations found.")
    
    # All Visualizations tab
    with viz_tabs[3]:
        st.markdown("### All Generated Visualizations")
        
        # Filter to show only unique visualizations (not already shown in other tabs)
        all_viz = [f for f in viz_files if f not in performance_viz + confusion_viz + speed_viz]
        
        if all_viz:
            # Display in a grid
            cols_per_row = 3
            for i in range(0, len(all_viz), cols_per_row):
                cols = st.columns(cols_per_row)
                row_viz = all_viz[i:i + cols_per_row]
                
                for col_idx, viz_file in enumerate(row_viz):
                    if col_idx < len(cols):
                        with cols[col_idx]:
                            img_path = os.path.join('visuals', viz_file)
                            try:
                                img = Image.open(img_path)
                                st.image(img, caption=viz_file.replace('.png', '').replace('_', ' ').title(), 
                                        use_container_width=True)
                                st.markdown(get_image_download_link(img_path, viz_file), unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"Could not load {viz_file}: {str(e)}")
        
        # Also show count of all visualizations
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Visualizations", len(viz_files))
        with col2:
            st.metric("Performance Charts", len(performance_viz))
        with col3:
            st.metric("Other Visualizations", len(all_viz))

def display_model_performance(comparison_df):
    """Display model performance metrics using saved visualizations"""
    st.markdown("### 🎯 Model Performance Analysis")
    
    # Sort by accuracy
    comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
    
    # Display key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Best Accuracy", f"{comparison_df.iloc[0]['Accuracy']:.4f}")
    with col2:
        st.metric("Best F1 Score", f"{comparison_df['F1 Score'].max():.4f}")
    with col3:
        st.metric("Fastest Prediction", f"{comparison_df['Pred Time (ms)'].min():.2f} ms")
    with col4:
        st.metric("Average Accuracy", f"{comparison_df['Accuracy'].mean():.4f}")
    with col5:
        st.metric("Models Evaluated", len(comparison_df))
    
    # Display detailed table
    st.markdown("#### Detailed Metrics Table")
    st.dataframe(comparison_df, use_container_width=True)
    
    # Check for visualizations
    if os.path.exists('visuals'):
        st.markdown("---")
        st.markdown("### 📈 Generated Visualizations")
        
        # Show visualization gallery
        display_visualizations_gallery()
    else:
        st.info("No visualizations found. Visualizations will be generated when you run model evaluation.")

def create_passenger_input_form():
    """Create form for passenger input"""
    with st.form("passenger_form"):
        st.markdown("### 👤 Passenger Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", 18, 100, 35)
            customer_type = st.selectbox("Customer Type", ["Loyal Customer", "Disloyal Customer"])
        
        with col2:
            type_of_travel = st.selectbox("Type of Travel", ["Business", "Personal"])
            travel_class = st.selectbox("Class", ["Business", "Eco Plus", "Eco"])
            flight_distance = st.number_input("Flight Distance (miles)", 0, 10000, 500)
        
        st.markdown("### ✈️ Flight Experience Ratings (0-5)")
        
        rating_cols = st.columns(4)
        ratings = {}
        
        with rating_cols[0]:
            ratings['Departure and Arrival Time Convenience'] = st.slider("Time Convenience", 0, 5, 3)
            ratings['Ease of Online Booking'] = st.slider("Online Booking", 0, 5, 3)
            ratings['Check-in Service'] = st.slider("Check-in", 0, 5, 3)
        
        with rating_cols[1]:
            ratings['Online Boarding'] = st.slider("Online Boarding", 0, 5, 3)
            ratings['Gate Location'] = st.slider("Gate Location", 0, 5, 3)
            ratings['On-board Service'] = st.slider("On-board Service", 0, 5, 3)
        
        with rating_cols[2]:
            ratings['Seat Comfort'] = st.slider("Seat Comfort", 0, 5, 3)
            ratings['Leg Room Service'] = st.slider("Leg Room", 0, 5, 3)
            ratings['Cleanliness'] = st.slider("Cleanliness", 0, 5, 3)
        
        with rating_cols[3]:
            ratings['Food and Drink'] = st.slider("Food & Drink", 0, 5, 3)
            ratings['In-flight Service'] = st.slider("In-flight Service", 0, 5, 3)
            ratings['In-flight Wifi Service'] = st.slider("WiFi Service", 0, 5, 3)
            ratings['In-flight Entertainment'] = st.slider("Entertainment", 0, 5, 3)
            ratings['Baggage Handling'] = st.slider("Baggage", 0, 5, 3)
        
        # Delay information
        st.markdown("### ⏰ Flight Delays")
        delay_col1, delay_col2 = st.columns(2)
        with delay_col1:
            departure_delay = st.number_input("Departure Delay (minutes)", 0, 500, 0)
        with delay_col2:
            arrival_delay = st.number_input("Arrival Delay (minutes)", 0, 500, 0)
        
        submitted = st.form_submit_button("🎯 Predict Satisfaction", use_container_width=True)
        
        if submitted:
            # Create data dictionary
            passenger_data = {
                'Gender': ['male' if gender == 'Male' else 'female'],
                'Age': [age],
                'Customer Type': ['returning' if customer_type == 'Loyal Customer' else 'first time'],
                'Type of Travel': ['Business' if type_of_travel == 'Business' else 'Personal'],
                'Class': ['Business' if travel_class == 'Business' else 
                         ('Eco Plus' if travel_class == 'Eco Plus' else 'Eco')],
                'Flight Distance': [flight_distance],
                'Departure Delay': [departure_delay],
                'Arrival Delay': [arrival_delay],
            }
            
            # Add ratings
            passenger_data.update({k: [v] for k, v in ratings.items()})
            
            return pd.DataFrame(passenger_data)
    
    return None

def run_data_preprocessing():
    """Run data preprocessing pipeline"""
    with st.spinner("🔄 Preprocessing data..."):
        try:
            scaled_data, scaler = preprocess_and_scale()
            if scaled_data is not None:
                st.success(f"✅ Data preprocessing complete! Processed {len(scaled_data)} samples.")
                
                # Show sample of processed data
                with st.expander("View Processed Data Sample"):
                    st.dataframe(scaled_data.head(), use_container_width=True)
                
                # Show data statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Samples", len(scaled_data))
                with col2:
                    st.metric("Features", scaled_data.shape[1] - 1)
                with col3:
                    satisfied_ratio = scaled_data[TARGET].mean()
                    st.metric("Satisfaction Rate", f"{satisfied_ratio:.1%}")
                
                return True
            else:
                st.error("❌ Data preprocessing failed")
                return False
        except Exception as e:
            st.error(f"Error during preprocessing: {str(e)}")
            return False

def run_model_training():
    """Run model training pipeline"""
    with st.spinner("🤖 Training models... This may take a few minutes."):
        try:
            all_models, training_times = train_all_models()
            if all_models:
                st.success(f"✅ All models trained successfully!")
                
                # Display training times
                training_df = pd.DataFrame({
                    'Model': list(training_times.keys()),
                    'Training Time (s)': list(training_times.values())
                }).sort_values('Training Time (s)')
                
                st.markdown("#### Training Times")
                st.dataframe(training_df, use_container_width=True)
                
                # Quick summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Fastest Training", f"{training_df.iloc[0]['Training Time (s)']:.2f}s")
                with col2:
                    st.metric("Slowest Training", f"{training_df.iloc[-1]['Training Time (s)']:.2f}s")
                with col3:
                    st.metric("Total Training Time", f"{training_df['Training Time (s)'].sum():.2f}s")
                
                return True
            else:
                st.error("❌ Model training failed")
                return False
        except Exception as e:
            st.error(f"Error during training: {str(e)}")
            return False

def run_model_evaluation():
    """Run model evaluation pipeline"""
    with st.spinner("📊 Evaluating models and generating visualizations...(this may take a while)"):
        try:
            # Load models
            all_models = load_models()
            if not all_models:
                st.error("No models found. Please train models first.")
                return None
            
            # Load and split data
            test_data = load_and_split_data()
            
            # Evaluate models
            all_metrics = evaluate_all_models(all_models, test_data)
            
            # Compare performance
            comparison_df = compare_performance(all_metrics)
            
            # Create visualizations - this will save images to visuals folder
            create_visualizations(all_metrics, comparison_df)
            
            st.success(f"✅ Model evaluation complete! Generated visualizations in 'visuals/' folder.")
            
            # Show quick preview of generated visualizations
            if os.path.exists('visuals'):
                viz_files = [f for f in os.listdir('visuals') if f.endswith('.png')]
                if viz_files:
                    st.info(f"Generated {len(viz_files)} visualization files:")
                    
                    # Show first 3 visualizations as preview
                    preview_cols = st.columns(min(3, len(viz_files)))
                    for idx, viz_file in enumerate(viz_files[:3]):
                        with preview_cols[idx]:
                            try:
                                img = Image.open(f'visuals/{viz_file}')
                                st.image(img, caption=viz_file.replace('.png', ''), use_container_width=True)
                            except:
                                st.write(f"📄 {viz_file}")
            
            return comparison_df
        except Exception as e:
            st.error(f"Error during evaluation: {str(e)}")
            return None

def predict_passenger_satisfaction(passenger_df):
    """Predict satisfaction for a passenger"""
    with st.spinner("🔮 Making predictions..."):
        try:
            # Preprocess the input data
            processed_df = preprocess_sample_data(passenger_df)
            
            # Prepare both scaled and non-scaled versions
            features_nonscaled = processed_df.drop('Satisfaction', axis=1)
            
            try:
                scaled_df = prepare_for_scaled_models(processed_df)
                features_scaled = scaled_df.drop('Satisfaction', axis=1)
            except:
                st.warning("Scaler not found. Using non-scaled features for all models.")
                features_scaled = features_nonscaled.copy()
            
            # Load all models
            models = load_all_models()
            
            # Make predictions
            predictions = make_predictions(models, features_nonscaled, features_scaled)
            
            return predictions
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            return None

def main():
    st.set_page_config(
        page_title="Airline Satisfaction Dashboard",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    inject_custom_css()
    
    # Initialize session state
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'comparison_df' not in st.session_state:
        st.session_state.comparison_df = None
    
    # Sidebar for system controls
    with st.sidebar:
        st.markdown("<h2 style='color: #40e0d0;'>⚙️ System Controls</h2>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### Data Operations")
        
        if st.button("🔄 Preprocess Data", use_container_width=True):
            if check_dependencies():
                run_data_preprocessing()
        
        if st.button("🤖 Train All Models", use_container_width=True):
            run_model_training()
        
        if st.button("📊 Evaluate Models", use_container_width=True):
            comparison_df = run_model_evaluation()
            if comparison_df is not None:
                st.session_state.comparison_df = comparison_df
        
        st.markdown("---")
        st.markdown("### System Status")
        
        # Check system status
        status_cols = st.columns(2)
        with status_cols[0]:
            data_exists = os.path.exists('data/processed/classification_data.csv')
            st.metric("Data Ready", "✅" if data_exists else "❌")
        with status_cols[1]:
            models_exist = any(os.path.exists(f'models/{model}_model.pkl') 
                             for model in ['logistic_regression', 'random_forest'])
            st.metric("Models Ready", "✅" if models_exist else "❌")
        
        # Check visualizations
        viz_exists = os.path.exists('visuals') and any(f.endswith('.png') for f in os.listdir('visuals'))
        st.metric("Visualizations", "✅" if viz_exists else "❌")
        
        st.markdown("---")
        st.markdown("### Quick Actions")
        
        if st.button("🆕 Use Sample Passenger", use_container_width=True):
            sample_df = create_sample_dataframe()
            predictions = predict_passenger_satisfaction(sample_df)
            if predictions:
                st.session_state.predictions = predictions
                st.success("Sample prediction complete!")
        
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.predictions = None
            st.session_state.comparison_df = None
            st.rerun()
        
        # View visualizations button
        if viz_exists:
            st.markdown("---")
            if st.button("🖼️ View Visualizations", use_container_width=True):
                st.session_state.show_viz = True
    
    # Header Section
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("<h1>✈️ Airline Customer Satisfaction Dashboard</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; opacity: 0.8;'>Predicting and Analyzing Passenger Satisfaction</p>", 
                   unsafe_allow_html=True)
        st.markdown(
            """
            <p style='text-align: center; font-size: 14px; color: #999; margin-top: -10px;'>
            Developed by: <b>biniyam_girma</b> (ATE/7146/14),
            <b>simon_mesfin</b> (ATE/7211/14),
            <b>yosef_ashebir</b> (ATE/4638/14)
            </p>
            """,
            unsafe_allow_html=True
        )
    
    # Main Content Tabs
    tab1, tab2, tab3, tab4= st.tabs(["🎯 Satisfaction Prediction", "📊 Model Analysis", "📁 Data Management", "📋 Project Summary"])
    
    with tab1:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.markdown("<h2>Customer Satisfaction Prediction</h2>", unsafe_allow_html=True)
        
        # Create input form
        passenger_df = create_passenger_input_form()
        
        # If form was submitted, make predictions
        if passenger_df is not None:
            predictions = predict_passenger_satisfaction(passenger_df)
            if predictions:
                st.session_state.predictions = predictions
        
        # Display predictions if available
        if st.session_state.predictions:
            st.markdown("---")
            display_model_predictions(st.session_state.predictions)
            
            # Show passenger data
            with st.expander("📋 View Passenger Data"):
                st.dataframe(passenger_df, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.markdown("<h2>Model Performance Analysis</h2>", unsafe_allow_html=True)
        
        # Check if evaluation results are available
        if st.session_state.comparison_df is not None:
            display_model_performance(st.session_state.comparison_df)
        else:
            st.info("No evaluation results available. Click 'Evaluate Models' in the sidebar to run evaluation.")
            
            # Quick evaluation button
            if st.button("🚀 Run Evaluation", type="primary"):
                with st.spinner("Running evaluation..."):
                    comparison_df = run_model_evaluation()
                    if comparison_df is not None:
                        st.session_state.comparison_df = comparison_df
                        st.rerun()
            
            # Show existing visualizations if available
            # if os.path.exists('visuals'):
            #     viz_files = [f for f in os.listdir('visuals') if f.endswith('.png')]
            #     if viz_files:
            #         st.markdown("---")
            #         st.markdown("### Existing Visualizations")
            #         st.info(f"Found {len(viz_files)} existing visualization files. You can view them by running evaluation or clicking 'View Visualizations' in the sidebar.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab3:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.markdown("<h2>Data Management</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='input-section'>", unsafe_allow_html=True)
            st.markdown("### 📁 Data Information")
            
            # Check data files
            data_files = {
                "Raw Data": "data/raw/airline_passenger_satisfaction.csv",
                "Processed Data": "data/processed/classification_data.csv",
                "Scaled Data": "data/processed/scaled_classification_data.csv"
            }
            
            for name, path in data_files.items():
                exists = os.path.exists(path)
                st.write(f"**{name}:** {'✅ Available' if exists else '❌ Missing'}")
                if exists and st.button(f"View {name}", key=f"view_{name}"):
                    try:
                        df = pd.read_csv(path)
                        st.dataframe(df.head(10), use_container_width=True)
                        st.metric("Rows", len(df))
                        st.metric("Columns", len(df.columns))
                    except:
                        st.error(f"Could not load {name}")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='input-section'>", unsafe_allow_html=True)
            st.markdown("### 🤖 Model Information")
            
            # Check model files
            model_files = {
                'Logistic Regression': 'models/logistic_regression_model.pkl',
                'Decision Tree': 'models/decision_tree_model.pkl',
                "Newton's Method": 'models/newton_method_model.pkl',
                'Random Forest': 'models/random_forest_model.pkl',
                'SVM': 'models/svm_model.pkl',
                'KNN': 'models/knn_model.pkl',
                'Gaussian NB': 'models/gaussian_nb_model.pkl',
                'Neural Network': 'models/neural_network_model.pkl',
                'Scaler': 'models/standard_scaler.pkl'
            }
            
            available_models = []
            for name, path in model_files.items():
                exists = os.path.exists(path)
                if exists:
                    available_models.append(name)
                    st.write(f"✅ **{name}**")
                else:
                    st.write(f"❌ **{name}**")
            
            st.metric("Models Available", f"{len(available_models)}/{len(model_files)}")
            
            # Model statistics
            if available_models:
                st.markdown("---")
                st.markdown("#### 📊 Model Statistics")
                
                # Check training times
                training_times_path = "results/training_times.csv"
                if os.path.exists(training_times_path):
                    training_df = pd.read_csv(training_times_path)
                    fastest = training_df.loc[training_df['Time (s)'].idxmin()]
                    slowest = training_df.loc[training_df['Time (s)'].idxmax()]
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Fastest Model", fastest['Model'])
                    with col_b:
                        st.metric("Slowest Model", slowest['Model'])
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Data management actions
        st.markdown("---")
        st.markdown("### 🛠️ Data Operations")
        
        action_cols = st.columns(4)
        with action_cols[0]:
            if st.button("🔄 Preprocess", use_container_width=True):
                run_data_preprocessing()
        
        with action_cols[1]:
            if st.button("🤖 Train", use_container_width=True):
                run_model_training()
        
        with action_cols[2]:
            if st.button("📊 Evaluate", use_container_width=True):
                comparison_df, all_metrics = run_model_evaluation()
                if comparison_df is not None:
                    st.session_state.comparison_df = comparison_df
                    st.session_state.all_metrics = all_metrics
        
        with action_cols[3]:
            if st.button("🗑️ Clear Cache", use_container_width=True):
                # Clear session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
        with tab4:
            st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
            st.markdown("<h2>Project Summary</h2>", unsafe_allow_html=True)
            
            # Create columns for layout
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Display the project description from summary.py
                st.markdown(PROJECT_DESCRIPTION, unsafe_allow_html=True)
            
            with col2:
                # Add quick stats sidebar
                st.markdown("<div class='input-section'>", unsafe_allow_html=True)
                st.markdown("### 📊 Quick Stats")
                
                # Add some statistics
                if os.path.exists('data/processed/classification_data.csv'):
                    try:
                        df = pd.read_csv('data/processed/classification_data.csv')
                        st.metric("Total Passengers", f"{len(df):,}")
                        st.metric("Satisfaction Rate", f"{df[TARGET].mean():.1%}")
                        st.metric("Features", df.shape[1] - 1)
                    except:
                        st.info("Process data to see stats")
                
                # Model stats if available
                if os.path.exists('results/model_comparison_summary.csv'):
                    try:
                        comparison_df = pd.read_csv('results/model_comparison_summary.csv')
                        best_accuracy = comparison_df['Accuracy'].max()
                        st.metric("Best Accuracy", f"{best_accuracy:.2%}")
                    except:
                        pass
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Add download button for the summary
        st.markdown("---")
        col_a, col_b, col_c = st.columns(3)
        with col_b:
            # Create a downloadable text file
            summary_text = PROJECT_DESCRIPTION
            
            st.download_button(
                label="📥 Download Summary",
                data=summary_text,
                file_name="project_summary.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    # Check dependencies on startup
    if not check_dependencies():
        st.warning("""
        ### ⚠️ Setup Required
        
        Please ensure:
        1. Your raw data file is at `data/raw/airline_passenger_satisfaction.csv`
        2. You have the required Python packages installed
        
        Run these commands in your terminal:
        ```bash
        mkdir -p data/raw data/processed models results visuals
        # Place your CSV file in data/raw/
        pip install streamlit pandas numpy scikit-learn matplotlib seaborn pillow joblib
        ```
        """)
    
    main()