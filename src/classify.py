import pandas as pd
import numpy as np
import joblib
import sys
import os

# Add src to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Define constants
TARGET = 'Satisfaction'
CONTINUOUS_FEATURES = [
    'Departure and Arrival Time Convenience',
    'Ease of Online Booking',
    'Check-in Service',
    'Online Boarding',
    'Gate Location',
    'On-board Service',
    'Seat Comfort',
    'Leg Room Service',
    'Cleanliness',
    'Food and Drink',
    'In-flight Service',
    'In-flight Wifi Service',
    'In-flight Entertainment',
    'Baggage Handling',
    'Flight Distance',
    'Departure Delay',
    'Arrival Delay',
    'Age'
]

def create_sample_dataframe():
    """
    Create a DataFrame with the sample data provided
    """
    sample_data = {
        'Gender': ['male'],
        'Age': [48],
        'Customer Type': ['first time'],
        'Type of Travel': ['Business'],
        'Class': ['Business'],
        'Flight Distance': [821],
        'Departure Delay': [2],
        'Arrival Delay': [5],
        'Departure and Arrival Time Convenience': [3],
        'Ease of Online Booking': [3],
        'Check-in Service': [4],
        'Online Boarding': [3],
        'Gate Location': [3],
        'On-board Service': [3],
        'Seat Comfort': [5],
        'Leg Room Service': [2],
        'Cleanliness': [5],
        'Food and Drink': [5],
        'In-flight Service': [5],
        'In-flight Wifi Service': [3],
        'In-flight Entertainment': [5],
        'Baggage Handling': [5]
    }
    
    return pd.DataFrame(sample_data)

def preprocess_sample_data(df):
    """
    Preprocess the sample data to match the training data format
    """
    df_processed = df.copy()
    
    # Encode categorical variables
    gender_mapping = {'male': 1, 'female': 0}
    df_processed['Gender'] = df_processed['Gender'].map(gender_mapping)
    
    customer_type_mapping = {'first time': 0, 'returning': 1}
    df_processed['Customer Type'] = df_processed['Customer Type'].map(customer_type_mapping)
    
    travel_type_mapping = {'Business': 0, 'Personal': 1}
    df_processed['Type of Travel'] = df_processed['Type of Travel'].map(travel_type_mapping)
    
    # One-hot encode Class column
    df_processed['Class_Business'] = (df_processed['Class'] == 'Business').astype(int)
    df_processed['Class_Eco'] = (df_processed['Class'] == 'Eco').astype(int)
    df_processed['Class_Eco Plus'] = (df_processed['Class'] == 'Eco Plus').astype(int)
    
    # Drop original Class column
    df_processed = df_processed.drop('Class', axis=1)
    
    # Add target column placeholder
    df_processed['Satisfaction'] = 0
    
    return df_processed

def prepare_for_scaled_models(df):
    """
    Prepare data for models that require scaling
    """
    scaler = joblib.load("models/standard_scaler.pkl")
    
    df_scaled = df.copy()
    
    # Scale only the continuous features that exist in the dataframe
    available_features = [f for f in CONTINUOUS_FEATURES if f in df_scaled.columns]
    df_scaled[available_features] = scaler.transform(df_scaled[available_features])
    
    return df_scaled

def load_all_models():
    """
    Load all 8 trained models
    """
    model_files = {
        'Logistic Regression': 'models/logistic_regression_model.pkl',
        'Decision Tree': 'models/decision_tree_model.pkl',
        "Newton's Method": 'models/newton_method_model.pkl',
        'Random Forest': 'models/random_forest_model.pkl',
        'SVM': 'models/svm_model.pkl',
        'KNN': 'models/knn_model.pkl',
        'Gaussian NB': 'models/gaussian_nb_model.pkl',
        'Neural Network': 'models/neural_network_model.pkl'
    }
    
    models = {}
    for model_name, model_path in model_files.items():
        try:
            models[model_name] = joblib.load(model_path)
        except FileNotFoundError:
            print(f"  ✗ {model_name} not found")
            models[model_name] = None
    
    return models

def ensure_feature_order(df, model):
    """
    Ensure the dataframe has the same feature order as the model expects
    """
    if model is None or not hasattr(model, 'feature_names_in_'):
        return df
    
    model_features = list(model.feature_names_in_)
    
    # Get columns that are in model_features but not in df
    missing_cols = set(model_features) - set(df.columns)
    
    # Add missing columns with default value 0
    for col in missing_cols:
        df[col] = 0
    
    # Reorder columns to match model_features
    df = df[model_features]
    
    return df

def make_predictions(models, features_nonscaled, features_scaled):
    """
    Make predictions using all 8 models
    """
    predictions = {}
    
    # Define which models use scaled vs non-scaled data
    scaled_models = ['Logistic Regression', "Newton's Method", 'SVM', 'KNN', 
                    'Gaussian NB', 'Neural Network']
    nonscaled_models = ['Decision Tree', 'Random Forest']
    
    print("\nMaking predictions...")
    
    for model_name, model in models.items():
        if model is None:
            continue
            
        # Choose the right dataset for the model
        if model_name in scaled_models:
            features = features_scaled.copy()
        else:
            features = features_nonscaled.copy()
        
        # Ensure feature order
        features = ensure_feature_order(features, model)
        
        try:
            # Make prediction
            prediction = model.predict(features)[0]
            
            # Get probabilities if available
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(features)[0]
                confidence = max(probability) * 100
            else:
                probability = None
                confidence = None
            
            # Map prediction to readable labels
            prediction_map = {0: "NOT SATISFIED", 1: "SATISFIED"}
            result = prediction_map[prediction]
            
            # Store results
            predictions[model_name] = {
                'prediction': prediction,
                'result': result,
                'confidence': confidence,
                'probability': probability
            }
            
        except Exception as e:
            print(f"  ✗ {model_name}: {str(e)}")
            predictions[model_name] = None
    
    return predictions

def display_predictions(predictions):
    """
    Display all model predictions
    """
    print("\n" + "="*60)
    print("PREDICTION RESULTS - 8 MODELS")
    print("="*60)
    
    satisfied_count = 0
    total_models = 0
    
    print(f"\n{'Model':25} {'Prediction':15} {'Confidence':>10}")
    print("-" * 50)
    
    for model_name, pred in predictions.items():
        if pred is not None:
            total_models += 1
            if pred['prediction'] == 1:
                satisfied_count += 1
            
            confidence_str = f"{pred['confidence']:.1f}%" if pred['confidence'] is not None else "N/A"
            print(f"{model_name:25} {pred['result']:15} {confidence_str:>10}")
    
    # Calculate consensus
    if total_models > 0:
        consensus = (max(satisfied_count, total_models - satisfied_count) / total_models) * 100
        majority = "SATISFIED" if satisfied_count > total_models - satisfied_count else "NOT SATISFIED"
        
        print(f"\nConsensus: {satisfied_count}/{total_models} models predict SATISFIED")
        print(f"Majority: {majority} ({consensus:.1f}% agreement)")
        
        if consensus >= 75:
            print("✓ Strong consensus among models")
        elif consensus >= 60:
            print("○ Moderate consensus")
        else:
            print("⚠ Low consensus")

def display_sample_data(df):
    """
    Display the sample data
    """
    print("\n" + "="*60)
    print("SAMPLE PASSENGER DATA")
    print("="*60)
    
    for column in df.columns:
        print(f"{column:35}: {df[column].iloc[0]}")

def predict_with_all_models():
    """
    Main function to make predictions using all 8 models
    """
    print("="*60)
    print("AIRLINE PASSENGER SATISFACTION PREDICTION")
    print("="*60)
    
    try:
        # Step 1: Create and display sample data
        sample_df = create_sample_dataframe()
        display_sample_data(sample_df)
        
        # Step 2: Preprocess the data
        processed_df = preprocess_sample_data(sample_df)
        
        # Step 3: Prepare both scaled and non-scaled versions
        features_nonscaled = processed_df.drop('Satisfaction', axis=1)
        scaled_df = prepare_for_scaled_models(processed_df)
        features_scaled = scaled_df.drop('Satisfaction', axis=1)
        
        # Step 4: Load all models
        models = load_all_models()
        
        # Step 5: Make predictions
        predictions = make_predictions(models, features_nonscaled, features_scaled)
        
        # Step 6: Display predictions
        display_predictions(predictions)
        
        print("\n" + "="*60)
        print("PREDICTION COMPLETE")
        print("="*60)
        
        return predictions
        
    except FileNotFoundError as e:
        print(f"\nError: {str(e)}")
        print("\nPlease ensure you have run:")
        print("  python src/preprocess.py")
        print("  python src/train.py")
        return None
    except Exception as e:
        print(f"\nError: {str(e)}")
        return None

if __name__ == "__main__":
    predictions = predict_with_all_models()