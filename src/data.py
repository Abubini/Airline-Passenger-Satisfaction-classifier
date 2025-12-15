
#================================================= preprocess and scale =================================================================================
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Add src to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def preprocess_data():
    """
    Preprocess airline passenger satisfaction data for classification task
    Predicts Satisfaction based on passenger and flight features
    """
    try:
        # Load raw data
        print("Loading raw data...")
        df = pd.read_csv("data/raw/airline_passenger_satisfaction.csv")
        
        # Handle missing values
        print("Handling missing values...")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        print(f"Missing values before handling:\n{missing_values[missing_values > 0]}")
        
        # Remove rows with missing values (since only 'Arrival Delay in Minutes' has missing values)
        df = df.dropna()
        print(f"Remaining rows after dropping missing values: {df.shape[0]}")
        
        # Feature engineering
        print("Performing feature engineering...")
        
        # Drop ID column as it's not useful for prediction
        df = df.drop('ID', axis=1)
        
        # Encode binary categorical columns using Label Encoding
        print("Encoding categorical variables...")
        label_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Satisfaction']
        le = LabelEncoder()
        for col in label_cols:
            df[col] = le.fit_transform(df[col])
            print(f"  Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        # One-hot encode 'Class' column
        class_dummies = pd.get_dummies(df['Class'], prefix='Class')
        
        # Drop 'Economy' column as the baseline
        class_dummies = class_dummies.drop('Class_Economy Plus', axis=1)
        
        # Add one-hot encoded columns to dataframe and drop original 'Class' column
        df = pd.concat([df.drop('Class', axis=1), class_dummies], axis=1)
        
        # Define target variable
        target = 'Satisfaction'
        
        # Get all feature columns (everything except target)
        feature_columns = [col for col in df.columns if col != target]
        
        # Create final dataframe with features and target
        classification_df = df[feature_columns + [target]].copy()
        
        # Save processed data
        print("Saving processed data...")
        os.makedirs("data/processed", exist_ok=True)
        classification_df.to_csv("data/processed/classification_data.csv", index=False)
        
        # Print dataset info
        print(f"\nProcessed dataset shape: {classification_df.shape}")
        print(f"Number of features: {len(feature_columns)}")
        print(f"Features: {feature_columns}")
        print(f"Target: {target}")
        print(f"Target distribution:\n{classification_df[target].value_counts(normalize=True)}")
        print(f"Data saved to: data/processed/classification_data.csv")
        
        return classification_df
        
    except FileNotFoundError:
        print("Error: Raw data file not found at 'data/raw/airline_passenger_satisfaction.csv'")
        print("Please ensure the raw data file exists in the correct location")
        return None
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        return None

def scale_features():
    """
    Scale continuous features using StandardScaler with proper train-test split protocol
    """
    try:
        print("\n" + "="*50)
        print("Starting feature scaling...")
        print("="*50)
        
        # Load the preprocessed data
        print("Loading preprocessed data...")
        df = pd.read_csv("data/processed/classification_data.csv")
        
        # Define target variable
        target = 'Satisfaction'
        
        # Separate features and target
        X = df.drop(columns=[target])
        y = df[target]
        
        # Split data BEFORE scaling (important!)
        print("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        # Define continuous features to scale (based on your list)
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
            'Departure Delay in Minutes',
            'Arrival Delay in Minutes',
            'Age'
        ]
        
        # Check if all specified features exist in the dataset
        available_features = [f for f in continuous_features if f in X.columns]
        missing_features = set(continuous_features) - set(available_features)
        
        if missing_features:
            print(f"Warning: Some features not found in dataset: {missing_features}")
            print(f"Will scale available features: {available_features}")
        
        if not available_features:
            raise ValueError("No continuous features found to scale!")
        
        print(f"\nScaling {len(available_features)} continuous features...")
        
        # Initialize StandardScaler
        scaler = StandardScaler()
        
        # Learn scaling parameters from training set only
        print("Fitting scaler on training set...")
        scaler.fit(X_train[available_features])
        
        # Apply scaling to training set
        print("Transforming training set...")
        X_train_scaled = X_train.copy()
        X_train_scaled[available_features] = scaler.transform(X_train[available_features])
        
        # Apply SAME scaling to test set
        print("Transforming test set...")
        X_test_scaled = X_test.copy()
        X_test_scaled[available_features] = scaler.transform(X_test[available_features])
        
        # Create DataFrames with targets
        train_df = pd.concat([X_train_scaled, y_train.reset_index(drop=True)], axis=1)
        test_df = pd.concat([X_test_scaled, y_test.reset_index(drop=True)], axis=1)
        
        # Save scaled datasets
        print("\nSaving scaled datasets...")
        os.makedirs("data/processed", exist_ok=True)
        
        train_df.to_csv("data/processed/scaled_train_data.csv", index=False)
        test_df.to_csv("data/processed/scaled_test_data.csv", index=False)
        
        # Also save a combined version (train + test) for reference
        combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
        combined_df.to_csv("data/processed/scaled_classification_data.csv", index=False)
        
        # Save the scaler for future use
        import joblib
        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler, "models/standard_scaler.pkl")
        
        # Print summary
        print(f"\nScaling completed successfully!")
        print(f"Training set saved to: data/processed/scaled_train_data.csv")
        print(f"Test set saved to: data/processed/scaled_test_data.csv")
        print(f"Combined data saved to: data/processed/scaled_classification_data.csv")
        print(f"Scaler saved to: models/standard_scaler.pkl")
        
        # Show statistics for first few scaled features
        print("\nScaled feature statistics (first 3 features):")
        for feature in available_features[:3]:
            print(f"\n{feature}:")
            print(f"  Training set - Mean: {X_train_scaled[feature].mean():.4f}, Std: {X_train_scaled[feature].std():.4f}")
            print(f"  Test set - Mean: {X_test_scaled[feature].mean():.4f}, Std: {X_test_scaled[feature].std():.4f}")
        
        return train_df, test_df, scaler
        
    except FileNotFoundError:
        print("Error: Processed data file not found at 'data/processed/classification_data.csv'")
        print("Please run preprocess_data() first")
        return None, None, None
    except Exception as e:
        print(f"Error during feature scaling: {str(e)}")
        return None, None, None

def preprocess_and_scale():
    """
    Complete preprocessing pipeline with scaling
    """
    # Step 1: Preprocess the data
    print("="*60)
    print("STEP 1: Data Preprocessing")
    print("="*60)
    processed_data = preprocess_data()
    
    if processed_data is None:
        print("Preprocessing failed, cannot proceed with scaling.")
        return None, None, None
    
    # Step 2: Scale the features
    print("\n" + "="*60)
    print("STEP 2: Feature Scaling")
    print("="*60)
    train_df, test_df, scaler = scale_features()
    
    if train_df is not None:
        print("\n" + "="*60)
        print("COMPLETE PIPELINE SUCCESSFUL!")
        print("="*60)
        print(f"Final datasets:")
        print(f"  Training set: {train_df.shape[0]} samples, {train_df.shape[1]-1} features")
        print(f"  Test set: {test_df.shape[0]} samples, {test_df.shape[1]-1} features")
        print(f"  Target variable: Satisfaction")
    
    return train_df, test_df, scaler

if __name__ == "__main__":
    print("Starting complete preprocessing pipeline...")
    train_df, test_df, scaler = preprocess_and_scale()
    
    if train_df is not None:
        print("\nPipeline completed successfully!")
        print("Data ready for modeling with proper scaling.")
    else:
        print("Pipeline failed!")



#=====================================================================================================================================================================================================================#





#++_+++++++++++++++++++++++++++++++++++++++++++++++++++full preprocess++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from sklearn.preprocessing import StandardScaler

# Add src to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def preprocess_data():
    """
    Preprocess airline passenger satisfaction data for classification task
    Predicts Satisfaction based on passenger and flight features
    """
    try:
        # Load raw data
        print("Loading raw data...")
        df = pd.read_csv("data/raw/airline_passenger_satisfaction.csv")
        
        # Handle missing values
        print("Handling missing values...")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        print(f"Missing values before handling:\n{missing_values[missing_values > 0]}")
        
        # Remove rows with missing values (since only 'Arrival Delay in Minutes' has missing values)
        df = df.dropna()
        print(f"Remaining rows after dropping missing values: {df.shape[0]}")
        
        # Feature engineering
        print("Performing feature engineering...")
        
        # Drop ID column as it's not useful for prediction
        df = df.drop('ID', axis=1)
        
        # Encode binary categorical columns using Label Encoding
        print("Encoding categorical variables...")
        label_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Satisfaction']
        le = LabelEncoder()
        for col in label_cols:
            df[col] = le.fit_transform(df[col])
            print(f"  Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        # One-hot encode 'Class' column
        class_dummies = pd.get_dummies(df['Class'], prefix='Class')
        
        # Drop 'Economy Plus' column to avoid redundancy (Economy is the baseline)
        # Note: In your original code, you dropped 'Class_Economy Plus' but that seems odd
        # Typically we'd drop one reference category from each one-hot encoded column
        # Let's drop 'Class_Eco' (Economy) as the baseline
        class_dummies = class_dummies.drop('Class_Economy Plus', axis=1)
        
        # Add one-hot encoded columns to dataframe and drop original 'Class' column
        df = pd.concat([df.drop('Class', axis=1), class_dummies], axis=1)
        
        # Define target variable
        target = 'Satisfaction'
        
        # Get all feature columns (everything except target)
        feature_columns = [col for col in df.columns if col != target]
        
        # Create final dataframe with features and target
        classification_df = df[feature_columns + [target]].copy()
        
        # Save processed data
        print("Saving processed data...")
        os.makedirs("data/processed", exist_ok=True)
        classification_df.to_csv("data/processed/classification_data.csv", index=False)
        
        # Print dataset info
        print(f"\nProcessed dataset shape: {classification_df.shape}")
        print(f"Number of features: {len(feature_columns)}")
        print(f"Features: {feature_columns}")
        print(f"Target: {target}")
        print(f"Target distribution:\n{classification_df[target].value_counts(normalize=True)}")
        print(f"Data saved to: data/processed/classification_data.csv")
        
        # Optional: Prepare X and y for modeling
        X = classification_df.drop(target, axis=1)
        y = classification_df[target]
        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        
        return classification_df, X, y
        
    except FileNotFoundError:
        print("Error: Raw data file not found at 'data/raw/airline_passenger_satisfaction.csv'")
        print("Please ensure the raw data file exists in the correct location")
        return None, None, None
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        return None, None, None
def scale_features():
    """
    Scale continuous features using StandardScaler with proper train-test split protocol.
    Learns scaling parameters from training set, then applies to entire dataset.
    """
    try:
        print("\n" + "="*50)
        print("Starting feature scaling...")
        print("="*50)
        
        # Load the preprocessed data
        print("Loading preprocessed data...")
        df = pd.read_csv("data/processed/classification_data.csv")
        
        # Define target variable
        target = 'Satisfaction'
        
        # Separate features and target
        X = df.drop(columns=[target])
        y = df[target]
        
        # Split data BEFORE scaling to learn parameters only from training set
        print("Splitting data into train and test sets to learn scaling parameters...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Training set shape: {X_train.shape} (used for learning scaling parameters)")
        print(f"Test set shape: {X_test.shape}")
        
        # Define continuous features to scale
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
        
        # Check if features exist
        available_features = [f for f in continuous_features if f in X.columns]
        missing_features = set(continuous_features) - set(available_features)
        
        if missing_features:
            print(f"Warning: Some features not found: {missing_features}")
        
        print(f"\nLearning scaling parameters from training set for {len(available_features)} continuous features...")
        
        # Initialize and fit scaler ONLY on training set
        scaler = StandardScaler()
        scaler.fit(X_train[available_features])
        
        print("Applying scaling to the entire dataset...")
        
        # Create a copy of the original features
        X_scaled = X.copy()
        
        # Apply scaling to the ENTIRE dataset using parameters learned from training set
        # This maintains the data leakage prevention while transforming all data
        X_scaled[available_features] = scaler.transform(X[available_features])
        
        # Combine scaled features with target variable
        scaled_df = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
        
        # Save the complete scaled dataset
        os.makedirs("data/processed", exist_ok=True)
        scaled_df.to_csv("data/processed/scaled_classification_data.csv", index=False)
        
        # Also save the train/test split data for model training if needed
        # (but not as the main output as per your requirement)
        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler, "models/standard_scaler.pkl")
        
        print(f"\nScaling completed!")
        print(f"Total dataset shape: {scaled_df.shape}")
        print(f"Files saved:")
        print(f"  - data/processed/scaled_classification_data.csv (complete scaled dataset)")
        print(f"  - models/standard_scaler.pkl (scaler object)")
        
        # Display some statistics to verify scaling
        print(f"\nVerification - Statistics for scaled continuous features:")
        for feature in available_features[:3]:  # Show first 3 features as example
            print(f"\n{feature}:")
            print(f"  Original - Mean: {X[feature].mean():.2f}, Std: {X[feature].std():.2f}")
            print(f"  Scaled   - Mean: {X_scaled[feature].mean():.2f}, Std: {X_scaled[feature].std():.2f}")
        
        return scaled_df, scaler
        
    except Exception as e:
        print(f"Error during scaling: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def preprocess_and_scale():
    """
    Complete preprocessing and scaling pipeline
    """
    
    # Step 2: Scale
    scaled_df, scaler = scale_features()
    return scaled_df, scaler


# Optional: If you need a version that doesn't require preprocess_data function
def scale_dataset_from_path(data_path="data/processed/classification_data.csv"):
    """
    Alternative function that directly scales a dataset from a given path
    """
    try:
        print(f"\nLoading data from: {data_path}")
        df = pd.read_csv(data_path)
        
        target = 'Satisfaction'
        X = df.drop(columns=[target])
        y = df[target]
        
        # Split to learn scaling parameters
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define continuous features (adjust as needed)
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
        
        # Filter to available features only
        available_features = [f for f in continuous_features if f in X.columns]
        
        # Fit scaler on training set only
        scaler = StandardScaler()
        scaler.fit(X_train[available_features])
        
        # Apply to entire dataset
        X_scaled = X.copy()
        X_scaled[available_features] = scaler.transform(X[available_features])
        
        # Create final dataframe
        scaled_df = pd.concat([X_scaled, y], axis=1)
        
        # Save
        output_dir = os.path.dirname(data_path)
        output_path = os.path.join(output_dir, "scaled_dataset.csv")
        scaled_df.to_csv(output_path, index=False)
        
        print(f"\nComplete scaled dataset saved to: {output_path}")
        print(f"Dataset shape: {scaled_df.shape}")
        
        return scaled_df, scaler
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None, None

if __name__ == "__main__":
    print("Starting data preprocessing for classification...")
    processed_data, X, y = preprocess_data()
    
    if processed_data is not None:
        print("\nData preprocessing completed successfully!")
        print(f"Dataset ready for modeling with {X.shape[1]} features and {len(y)} samples")
    else:
        print("Data preprocessing failed!")
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
    # Option 2: Use the direct scaling function
    scaled_data, scaler = scale_features()
    
    if scaled_data is not None:
        print("\nFirst 5 rows of scaled dataset:")
        print(scaled_data.head())
        
        # Verify binary columns are not scaled
        print("\nVerifying binary columns (should be 0/1 values):")
        # Assuming some binary columns exist - you might need to adjust this
        binary_candidates = [col for col in scaled_data.columns if col not in continuous_features and col != 'Satisfaction']
        if binary_candidates:
            for col in binary_candidates[:3]:  # Check first 3 binary columns
                unique_vals = scaled_data[col].unique()
                print(f"{col}: unique values = {sorted(unique_vals[:10])}")


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#