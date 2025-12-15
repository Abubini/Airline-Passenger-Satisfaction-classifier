import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Add src to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Define constants at module level
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

TARGET = 'Satisfaction'
LABEL_COLS = ['Gender', 'Customer Type', 'Type of Travel', 'Satisfaction']

def preprocess_data():
    """
    Preprocess airline passenger satisfaction data for classification task
    Predicts Satisfaction based on passenger and flight features
    """
    try:
        print("Loading and preprocessing data...")
        df = pd.read_csv("data/raw/airline_passenger_satisfaction.csv")
        
        # Handle missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Dropping {missing_count} missing values...")
            df = df.dropna()
        
        # Drop ID column
        df = df.drop('ID', axis=1)
        
        # Encode categorical variables
        le = LabelEncoder()
        for col in LABEL_COLS:
            df[col] = le.fit_transform(df[col])
        
        # One-hot encode 'Class' column
        class_dummies = pd.get_dummies(df['Class'], prefix='Class')
        class_dummies = class_dummies.drop('Class_Economy Plus', axis=1)
        df = pd.concat([df.drop('Class', axis=1), class_dummies], axis=1)
        
        # Save processed data
        os.makedirs("data/processed", exist_ok=True)
        df.to_csv("data/processed/classification_data.csv", index=False)
        
        print(f"Processed dataset shape: {df.shape}")
        print(f"Target distribution:\n{df[TARGET].value_counts(normalize=True).round(3)}")
        
        return df
        
    except FileNotFoundError:
        print("Error: Raw data file not found")
        return None
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        return None

def scale_features():
    """
    Scale continuous features using StandardScaler with proper train-test split protocol.
    """
    try:
        print("\nStarting feature scaling...")
        df = pd.read_csv("data/processed/classification_data.csv")
        
        X = df.drop(columns=[TARGET])
        y = df[TARGET]
        
        # Split to learn scaling parameters
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Filter to available continuous features
        available_features = [f for f in CONTINUOUS_FEATURES if f in X.columns]
        
        # Initialize and fit scaler on training set
        scaler = StandardScaler()
        scaler.fit(X_train[available_features])
        
        # Apply scaling to entire dataset
        X_scaled = X.copy()
        X_scaled[available_features] = scaler.transform(X[available_features])
        
        # Combine scaled features with target
        scaled_df = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
        
        # Save results
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        scaled_df.to_csv("data/processed/scaled_classification_data.csv", index=False)
        joblib.dump(scaler, "models/standard_scaler.pkl")
        
        print(f"Complete scaled dataset saved to: data/processed/scaled_classification_data.csv")
        print(f"Dataset shape: {scaled_df.shape}")
        
        return scaled_df, scaler
        
    except Exception as e:
        print(f"Error during scaling: {str(e)}")
        return None, None

def scale_dataset_from_path(data_path="data/processed/classification_data.csv"):
    """
    Alternative function that directly scales a dataset from a given path
    """
    try:
        df = pd.read_csv(data_path)
        
        X = df.drop(columns=[TARGET])
        y = df[TARGET]
        
        # Split to learn scaling parameters
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Filter to available features
        available_features = [f for f in CONTINUOUS_FEATURES if f in X.columns]
        
        # Fit scaler on training set only
        scaler = StandardScaler()
        scaler.fit(X_train[available_features])
        
        # Apply to entire dataset
        X_scaled = X.copy()
        X_scaled[available_features] = scaler.transform(X[available_features])
        
        # Create and save final dataframe
        scaled_df = pd.concat([X_scaled, y], axis=1)
        
        output_dir = os.path.dirname(data_path)
        output_path = os.path.join(output_dir, "scaled_dataset.csv")
        scaled_df.to_csv(output_path, index=False)
        
        return scaled_df, scaler
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None, None

def preprocess_and_scale():
    """
    Complete preprocessing and scaling pipeline
    """
    processed_data = preprocess_data()
    if processed_data is not None:
        scaled_df, scaler = scale_features()
        return scaled_df, scaler
    return None, None

if __name__ == "__main__":
    # Run complete pipeline
    scaled_data, scaler = preprocess_and_scale()
    
    if scaled_data is not None:
        print("\nFirst 3 rows of scaled dataset:")
        print(scaled_data.head(3))
        
        # Verify binary columns are not scaled
        binary_cols = [col for col in scaled_data.columns 
                      if col not in CONTINUOUS_FEATURES and col != TARGET]
        
        if binary_cols:
            print(f"\nBinary columns ({len(binary_cols)}): {binary_cols}")
            for col in binary_cols[:3]:
                unique_vals = scaled_data[col].unique()
                print(f"{col}: unique values = {sorted(unique_vals)}")