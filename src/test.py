import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

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
    # Step 1: Preprocess (assuming this function exists)
    processed_data = preprocess_data()
    if processed_data is None:
        return None, None
    
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


# Example usage:
if __name__ == "__main__":
    # Option 1: Use the full pipeline
    # scaled_data, scaler = preprocess_and_scale()
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