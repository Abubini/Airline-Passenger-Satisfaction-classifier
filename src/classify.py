import pandas as pd
import numpy as np
import joblib
import sys
import os

# Add src to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def create_sample_dataframe():
    """
    Create a DataFrame with the sample data provided
    """
    # Create a dictionary with the sample data
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
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Print the data
    print("Sample passenger data:")
    print("=" * 60)
    for key, value in sample_data.items():
        print(f"{key:40}: {value[0]}")
    print("=" * 60)
    
    return df

def preprocess_sample_data(df):
    """
    Preprocess the sample data to match the training data format
    """
    try:
        # Create a copy to avoid modifying original
        df_processed = df.copy()
        
        # Encode categorical variables
        gender_mapping = {'male': 1, 'female': 0}
        df_processed['Gender'] = df_processed['Gender'].map(gender_mapping)
        
        # Customer Type: first time=0, returning=1
        customer_type_mapping = {'first time': 0, 'returning': 1}
        df_processed['Customer Type'] = df_processed['Customer Type'].map(customer_type_mapping)
        
        # Type of Travel: Business=1, Personal=0
        travel_type_mapping = {'Business': 1, 'Personal': 0}
        df_processed['Type of Travel'] = df_processed['Type of Travel'].map(travel_type_mapping)
        
        # One-hot encode Class column (matching preprocess.py which dropped 'Class_Economy Plus')
        df_processed['Class_Business'] = (df_processed['Class'] == 'Business').astype(int)
        df_processed['Class_Eco'] = (df_processed['Class'] == 'Eco').astype(int)
        df_processed['Class_Eco Plus'] = (df_processed['Class'] == 'Eco Plus').astype(int)
        
        # Drop original Class column
        df_processed = df_processed.drop('Class', axis=1)
        
        # Add target column (Satisfaction) with placeholder value 0
        df_processed['Satisfaction'] = 0
        
        return df_processed
        
    except Exception as e:
        print(f"Error preprocessing data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def prepare_for_scaled_models(df):
    """
    Prepare data for models that require scaling (Logistic Regression, Newton's Method, SVM)
    """
    try:
        # Load the scaler
        scaler = joblib.load("models/standard_scaler.pkl")
        
        # Define features that need scaling (continuous features)
        continuous_features = [
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
        
        # Create a copy for scaling
        df_scaled = df.copy()
        
        # Scale only the continuous features that exist in the dataframe
        available_features = [f for f in continuous_features if f in df_scaled.columns]
        df_scaled[available_features] = scaler.transform(df_scaled[available_features])
        
        return df_scaled
        
    except Exception as e:
        print(f"Error preparing data for scaled models: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def ensure_feature_order(df, model_features):
    """
    Ensure the dataframe has the same feature order as the model expects
    """
    # Get columns that are in model_features but not in df
    missing_cols = set(model_features) - set(df.columns)
    
    # Add missing columns with default value 0
    for col in missing_cols:
        df[col] = 0
    
    # Reorder columns to match model_features
    df = df[model_features]
    
    return df

def load_all_models():
    """
    Load all 5 trained models
    """
    print("\nLoading all trained models...")
    
    models = {}
    
    # List of all models
    model_files = {
        'Logistic Regression': 'models/logistic_regression_model.pkl',
        'Decision Tree': 'models/decision_tree_model.pkl',
        "Newton's Method": 'models/newton_method_model.pkl',
        'Random Forest': 'models/random_forest_model.pkl',
        'SVM': 'models/svm_model.pkl'
    }
    
    for model_name, model_path in model_files.items():
        try:
            models[model_name] = joblib.load(model_path)
            print(f"✓ Loaded {model_name}")
        except FileNotFoundError:
            print(f"✗ {model_name} not found at {model_path}")
            models[model_name] = None
        except Exception as e:
            print(f"✗ Error loading {model_name}: {str(e)}")
            models[model_name] = None
    
    return models

def get_model_features(models):
    """
    Get feature names expected by each model
    """
    model_features = {}
    
    for model_name, model in models.items():
        if model is not None:
            try:
                if hasattr(model, 'feature_names_in_'):
                    model_features[model_name] = list(model.feature_names_in_)
                else:
                    # Try to infer from training data
                    train_data = pd.read_csv("data/processed/classification_data.csv")
                    model_features[model_name] = [col for col in train_data.columns if col != 'Satisfaction']
            except:
                # Default features if cannot determine
                model_features[model_name] = None
    
    return model_features

def make_predictions_with_all_models(models, model_features, features_nonscaled, features_scaled):
    """
    Make predictions using all 5 models
    """
    print("\n" + "="*60)
    print("MAKING PREDICTIONS WITH ALL MODELS")
    print("="*60)
    
    predictions = {}
    
    for model_name, model in models.items():
        if model is None:
            print(f"\n✗ Skipping {model_name} (model not loaded)")
            continue
            
        print(f"\n{'─'*40}")
        print(f"PREDICTING WITH: {model_name}")
        print(f"{'─'*40}")
        
        try:
            # Choose the right dataset for the model
            if model_name in ['Logistic Regression', "Newton's Method", 'SVM']:
                # Models that use scaled data
                features = features_scaled.copy()
                data_type = "scaled data"
            else:
                # Models that use non-scaled data (Decision Tree, Random Forest)
                features = features_nonscaled.copy()
                data_type = "non-scaled data"
            
            # Ensure feature order
            if model_features.get(model_name) is not None:
                features = ensure_feature_order(features, model_features[model_name])
            
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
                'probabilities': probability,
                'data_type': data_type
            }
            
            # Display result
            print(f"Prediction:  {result}")
            if confidence is not None:
                print(f"Confidence:  {confidence:.1f}%")
                if probability is not None:
                    print(f"Probabilities:")
                    print(f"  - Not Satisfied: {probability[0]:.3f}")
                    print(f"  - Satisfied:     {probability[1]:.3f}")
            
        except Exception as e:
            print(f"✗ Error making prediction with {model_name}: {str(e)}")
            predictions[model_name] = None
    
    return predictions

def display_final_summary(predictions):
    """
    Display a final summary of all model predictions
    """
    print("\n" + "="*70)
    print("FINAL PREDICTION SUMMARY")
    print("="*70)
    
    # Count predictions
    satisfied_count = 0
    not_satisfied_count = 0
    total_models = 0
    
    print(f"\n{'Model':25} {'Prediction':20} {'Confidence':12} {'Data Type':15}")
    print(f"{'-'*25} {'-'*20} {'-'*12} {'-'*15}")
    
    for model_name, pred in predictions.items():
        if pred is not None:
            total_models += 1
            if pred['prediction'] == 1:
                satisfied_count += 1
            else:
                not_satisfied_count += 1
            
            confidence_str = f"{pred['confidence']:.1f}%" if pred['confidence'] is not None else "N/A"
            print(f"{model_name:25} {pred['result']:20} {confidence_str:12} {pred['data_type']:15}")
    
    print(f"\n{'='*70}")
    print("PREDICTION CONSENSUS:")
    print(f"{'='*70}")
    
    if total_models == 0:
        print("No models made successful predictions.")
        return
    
    if satisfied_count > not_satisfied_count:
        majority = "SATISFIED"
        majority_count = satisfied_count
    elif not_satisfied_count > satisfied_count:
        majority = "NOT SATISFIED"
        majority_count = not_satisfied_count
    else:
        print("Models are evenly split (tie)")
        return
    
    consensus_percentage = (majority_count / total_models) * 100
    print(f"\n{majority_count} out of {total_models} models predict: {majority}")
    print(f"Consensus: {consensus_percentage:.1f}% of models agree")
    
    if consensus_percentage >= 80:
        print("✓ STRONG CONSENSUS: High agreement among models")
    elif consensus_percentage >= 60:
        print("○ MODERATE CONSENSUS: Majority agreement")
    else:
        print("⚠ LOW CONSENSUS: Models disagree significantly")
    
    # Show which models agree/disagree
    print(f"\nModels predicting SATISFIED:")
    sat_models = [name for name, pred in predictions.items() 
                  if pred is not None and pred['prediction'] == 1]
    if sat_models:
        for model in sat_models:
            print(f"  - {model}")
    else:
        print("  None")
    
    print(f"\nModels predicting NOT SATISFIED:")
    not_sat_models = [name for name, pred in predictions.items() 
                      if pred is not None and pred['prediction'] == 0]
    if not_sat_models:
        for model in not_sat_models:
            print(f"  - {model}")
    else:
        print("  None")

def get_confidence_ranking(predictions):
    """
    Rank models by prediction confidence
    """
    print(f"\n{'='*60}")
    print("MODEL CONFIDENCE RANKING")
    print(f"{'='*60}")
    
    # Filter models with confidence scores
    models_with_confidence = [(name, pred) for name, pred in predictions.items() 
                              if pred is not None and pred['confidence'] is not None]
    
    if not models_with_confidence:
        print("No models provided confidence scores.")
        return
    
    # Sort by confidence (descending)
    sorted_models = sorted(models_with_confidence, key=lambda x: x[1]['confidence'], reverse=True)
    
    print(f"\n{'Rank':6} {'Model':25} {'Confidence':12} {'Prediction':15}")
    print(f"{'-'*6} {'-'*25} {'-'*12} {'-'*15}")
    
    for rank, (model_name, pred) in enumerate(sorted_models, 1):
        print(f"{rank:6} {model_name:25} {pred['confidence']:.1f}%{'':8} {pred['result']:15}")
    
    # Show most confident model
    most_confident = sorted_models[0]
    print(f"\n✓ Most confident model: {most_confident[0]}")
    print(f"  Prediction: {most_confident[1]['result']} ({most_confident[1]['confidence']:.1f}% confidence)")

def predict_with_all_models():
    """
    Main function to make predictions using all 5 models
    """
    print("\n" + "="*70)
    print("AIRLINE PASSENGER SATISFACTION PREDICTION")
    print("Testing with 5 Machine Learning Models")
    print("="*70)
    
    try:
        # Step 1: Create sample data
        sample_df = create_sample_dataframe()
        
        # Step 2: Preprocess the data
        print("\nPreprocessing data...")
        processed_df = preprocess_sample_data(sample_df)
        
        if processed_df is None:
            print("Failed to preprocess data.")
            return
        
        print(f"✓ Processed data shape: {processed_df.shape}")
        
        # Step 3: Prepare both scaled and non-scaled versions
        print("\nPreparing data versions:")
        print("  - Non-scaled data (for Decision Tree, Random Forest)")
        print("  - Scaled data (for Logistic Regression, Newton's Method, SVM)")
        
        # Create non-scaled features
        features_nonscaled = processed_df.drop('Satisfaction', axis=1) if 'Satisfaction' in processed_df.columns else processed_df.copy()
        
        # Create scaled features
        scaled_df = prepare_for_scaled_models(processed_df)
        if scaled_df is None:
            print("Failed to scale data.")
            return
        
        features_scaled = scaled_df.drop('Satisfaction', axis=1) if 'Satisfaction' in scaled_df.columns else scaled_df.copy()
        
        print(f"✓ Non-scaled features shape: {features_nonscaled.shape}")
        print(f"✓ Scaled features shape: {features_scaled.shape}")
        
        # Step 4: Load all models
        models = load_all_models()
        
        # Step 5: Get feature names for each model
        model_features = get_model_features(models)
        
        # Step 6: Make predictions with all models
        predictions = make_predictions_with_all_models(
            models, model_features, features_nonscaled, features_scaled
        )
        
        # Step 7: Display final summary
        display_final_summary(predictions)
        
        # Step 8: Show confidence ranking
        get_confidence_ranking(predictions)
        
        print("\n" + "="*70)
        print("PREDICTION COMPLETE")
        print("="*70)
        
        return predictions
        
    except FileNotFoundError as e:
        print(f"\nError: {str(e)}")
        print("\nPlease ensure:")
        print("1. You have run 'python src/preprocess.py' to generate the datasets")
        print("2. You have run 'python src/train.py' to train all 5 models")
        print("3. The following model files exist:")
        print("   - models/logistic_regression_model.pkl")
        print("   - models/decision_tree_model.pkl")
        print("   - models/newton_method_model.pkl")
        print("   - models/random_forest_model.pkl")
        print("   - models/svm_model.pkl")
        print("   - models/standard_scaler.pkl")
        print("\nRun these commands first if needed:")
        print("  python src/preprocess.py")
        print("  python src/train.py")
        return None
    except Exception as e:
        print(f"\nError during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def interactive_mode():
    """
    Optional: Allow user to enter custom data
    """
    print("\n" + "="*60)
    print("INTERACTIVE PREDICTION MODE")
    print("="*60)
    
    print("\nWould you like to enter custom passenger data? (y/n)")
    choice = input().lower()
    
    if choice == 'y':
        print("\nEnter passenger details:")
        print("Note: For now, using the default sample data.")
        print("To add custom data input, modify the create_sample_dataframe() function.")
    
    # Use default sample data
    return predict_with_all_models()

if __name__ == "__main__":
    # Run prediction with all models
    predictions = predict_with_all_models()
    
    if predictions:
        print("\n" + "="*60)
        print("USAGE GUIDE")
        print("="*60)
        print("""
To use individual model predictions in your code:

import joblib

# Load specific model
model = joblib.load('models/model_name.pkl')

# Get prediction for a single sample
prediction = model.predict(features)[0]
probability = model.predict_proba(features)[0]  # if available

# 0 = Not Satisfied, 1 = Satisfied
if prediction == 1:
    print("Passenger is SATISFIED")
else:
    print("Passenger is NOT SATISFIED")
    
Available models:
1. Logistic Regression - Uses scaled data
2. Decision Tree - Uses non-scaled data
3. Newton's Method - Uses scaled data
4. Random Forest - Uses non-scaled data
5. SVM - Uses scaled data
        """)