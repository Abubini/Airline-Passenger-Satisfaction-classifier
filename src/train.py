import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import time

# Add src to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Define constants
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

def load_datasets():
    """
    Load both scaled and non-scaled datasets
    """
    try:
        print("Loading datasets...")
        
        # Load scaled dataset
        scaled_data_path = "data/processed/scaled_classification_data.csv"
        if os.path.exists(scaled_data_path):
            scaled_df = pd.read_csv(scaled_data_path)
            print(f"✓ Loaded scaled dataset: {scaled_df.shape}")
        else:
            print("✗ Scaled dataset not found. Trying to load classification_data.csv instead...")
            scaled_data_path = "data/processed/classification_data.csv"
            scaled_df = pd.read_csv(scaled_data_path)
            print(f"✓ Loaded classification dataset: {scaled_df.shape}")
        
        # Load non-scaled dataset
        nonscaled_data_path = "data/processed/classification_data.csv"
        nonscaled_df = pd.read_csv(nonscaled_data_path)
        print(f"✓ Loaded non-scaled dataset: {nonscaled_df.shape}")
        
        return scaled_df, nonscaled_df
        
    except FileNotFoundError as e:
        print(f"Error loading datasets: {str(e)}")
        print("Please run preprocess.py first to generate the datasets.")
        return None, None
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None, None

def prepare_data(df, dataset_name=""):
    """
    Prepare features and target for training
    """
    print(f"\nPreparing {dataset_name} data...")
    
    # Separate features and target
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    
    # Print dataset info
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution:")
    print(y.value_counts(normalize=True).round(3))
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test, model_name="", training_time=None):
    """
    Evaluate model performance
    """
    print(f"\n{'='*50}")
    print(f"Evaluating {model_name}")
    print(f"{'='*50}")
    
    if training_time is not None:
        print(f"Training time: {training_time:.2f} seconds")
    
    # Make predictions
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time
    print(f"Prediction time: {prediction_time:.4f} seconds")
    
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    # Calculate ROC-AUC if possible
    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC-AUC:   {roc_auc:.4f}")
    else:
        roc_auc = None
    
    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"[[TN: {cm[0,0]:4d}  FP: {cm[0,1]:4d}]")
    print(f" [FN: {cm[1,0]:4d}  TP: {cm[1,1]:4d}]]")
    
    # Print classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Satisfied', 'Satisfied']))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'training_time': training_time,
        'prediction_time': prediction_time
    }

def train_logistic_regression(scaled_df):
    """
    Train Logistic Regression model on scaled data
    """
    print(f"\n{'='*50}")
    print("Training Logistic Regression Model")
    print(f"{'='*50}")
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(scaled_df, "scaled")
    
    # Create and train model
    print("\nTraining Logistic Regression...")
    start_time = time.time()
    lr_model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        solver='lbfgs',
        class_weight='balanced'
    )
    
    lr_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Evaluate
    metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression", training_time)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/logistic_regression_model.pkl"
    joblib.dump(lr_model, model_path)
    print(f"\n✓ Logistic Regression model saved to: {model_path}")
    
    return lr_model, metrics

def train_decision_tree(nonscaled_df):
    """
    Train Decision Tree model on non-scaled data
    """
    print(f"\n{'='*50}")
    print("Training Decision Tree Model")
    print(f"{'='*50}")
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(nonscaled_df, "non-scaled")
    
    # Create and train model
    print("\nTraining Decision Tree...")
    start_time = time.time()
    dt_model = DecisionTreeClassifier(
        random_state=42,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced'
    )
    
    dt_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Evaluate
    metrics = evaluate_model(dt_model, X_test, y_test, "Decision Tree", training_time)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/decision_tree_model.pkl"
    joblib.dump(dt_model, model_path)
    print(f"\n✓ Decision Tree model saved to: {model_path}")
    
    return dt_model, metrics

def train_newton_method(scaled_df):
    """
    Train Logistic Regression with Newton's method on scaled data
    """
    print(f"\n{'='*50}")
    print("Training Logistic Regression with Newton's Method")
    print(f"{'='*50}")
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(scaled_df, "scaled")
    
    # Create and train model
    print("\nTraining Logistic Regression (Newton's Method)...")
    start_time = time.time()
    newton_model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        solver='newton-cg',  # Newton's Conjugate Gradient
        class_weight='balanced',
        penalty='l2'
    )
    
    newton_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Evaluate
    metrics = evaluate_model(newton_model, X_test, y_test, "Logistic Regression (Newton's Method)", training_time)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/newton_method_model.pkl"
    joblib.dump(newton_model, model_path)
    print(f"\n✓ Newton's Method model saved to: {model_path}")
    
    return newton_model, metrics

def train_random_forest(nonscaled_df):
    """
    Train Random Forest model on non-scaled data
    """
    print(f"\n{'='*50}")
    print("Training Random Forest Model")
    print(f"{'='*50}")
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(nonscaled_df, "non-scaled")
    
    # Create and train model
    print("\nTraining Random Forest...")
    start_time = time.time()
    rf_model = RandomForestClassifier(
        random_state=42,
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        n_jobs=-1  # Use all available cores
    )
    
    rf_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Evaluate
    metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest", training_time)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/random_forest_model.pkl"
    joblib.dump(rf_model, model_path)
    print(f"\n✓ Random Forest model saved to: {model_path}")
    
    return rf_model, metrics

def train_svm(scaled_df):
    """
    Train Support Vector Machine model on scaled data
    """
    print(f"\n{'='*50}")
    print("Training Support Vector Machine (SVM) Model")
    print(f"{'='*50}")
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(scaled_df, "scaled")
    
    # Create and train model
    print("\nTraining SVM (this may take a while)...")
    start_time = time.time()
    svm_model = SVC(
        random_state=42,
        C=1.0,
        kernel='rbf',
        probability=True,  # Enable probability estimates for ROC-AUC
        class_weight='balanced',
        gamma='scale'
    )
    
    svm_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Evaluate
    metrics = evaluate_model(svm_model, X_test, y_test, "Support Vector Machine", training_time)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/svm_model.pkl"
    joblib.dump(svm_model, model_path)
    print(f"\n✓ SVM model saved to: {model_path}")
    
    return svm_model, metrics

def compare_all_models(all_metrics):
    """
    Compare performance of all models
    """
    print(f"\n{'='*70}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    # Create comparison DataFrame
    comparison_data = {}
    model_names = list(all_metrics.keys())
    
    # Extract metrics for each model
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'training_time', 'prediction_time']
    
    for metric in metrics_to_compare:
        values = []
        for model_name in model_names:
            value = all_metrics[model_name].get(metric, np.nan)
            if value is None:
                value = np.nan
            values.append(f"{value:.4f}" if isinstance(value, (int, float)) else "N/A")
        comparison_data[metric.capitalize()] = values
    
    comparison_df = pd.DataFrame(comparison_data, index=model_names)
    
    # Display comparison
    print("\nPerformance Metrics:")
    print("-" * 100)
    print(comparison_df.to_string())
    
    # Find best model for each metric
    print(f"\n{'='*70}")
    print("BEST MODELS FOR EACH METRIC")
    print(f"{'='*70}")
    
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        valid_models = []
        for model_name in model_names:
            value = all_metrics[model_name].get(metric)
            if value is not None:
                valid_models.append((model_name, value))
        
        if valid_models:
            best_model = max(valid_models, key=lambda x: x[1])
            print(f"Best {metric.upper():10}: {best_model[0]:30} ({best_model[1]:.4f})")
    
    # Find fastest model
    valid_times = []
    for model_name in model_names:
        value = all_metrics[model_name].get('training_time')
        if value is not None:
            valid_times.append((model_name, value))
    
    if valid_times:
        fastest_model = min(valid_times, key=lambda x: x[1])
        print(f"Fastest Training: {fastest_model[0]:30} ({fastest_model[1]:.2f}s)")
    
    # Overall best model (based on accuracy)
    valid_accuracies = []
    for model_name in model_names:
        value = all_metrics[model_name].get('accuracy')
        if value is not None:
            valid_accuracies.append((model_name, value))
    
    if valid_accuracies:
        best_overall = max(valid_accuracies, key=lambda x: x[1])
        print(f"\n{'*'*70}")
        print(f"OVERALL BEST MODEL (Accuracy): {best_overall[0]}")
        print(f"Accuracy: {best_overall[1]:.4f}")
        print(f"{'*'*70}")
    
    # Save comparison results
    os.makedirs("results", exist_ok=True)
    comparison_df.to_csv("results/model_comparison_summary.csv")
    
    # Create detailed comparison with more metrics
    detailed_comparison = []
    for model_name, metrics in all_metrics.items():
        detailed_comparison.append({
            'Model': model_name,
            'Accuracy': metrics.get('accuracy', np.nan),
            'Precision': metrics.get('precision', np.nan),
            'Recall': metrics.get('recall', np.nan),
            'F1_Score': metrics.get('f1', np.nan),
            'ROC_AUC': metrics.get('roc_auc', np.nan),
            'Training_Time_s': metrics.get('training_time', np.nan),
            'Prediction_Time_s': metrics.get('prediction_time', np.nan),
            'TP': metrics.get('confusion_matrix', np.zeros((2,2)))[1,1],
            'FP': metrics.get('confusion_matrix', np.zeros((2,2)))[0,1],
            'TN': metrics.get('confusion_matrix', np.zeros((2,2)))[0,0],
            'FN': metrics.get('confusion_matrix', np.zeros((2,2)))[1,0]
        })
    
    detailed_df = pd.DataFrame(detailed_comparison)
    detailed_df.to_csv("results/detailed_model_comparison.csv", index=False)
    
    print(f"\n✓ Comparison results saved to:")
    print(f"  - results/model_comparison_summary.csv")
    print(f"  - results/detailed_model_comparison.csv")
    
    return best_overall[0] if valid_accuracies else None

def save_training_summary(all_metrics, best_model_name):
    """
    Save training summary to a text file
    """
    summary_content = f"""MODEL TRAINING SUMMARY
======================

Dataset Information:
-------------------
- Scaled data used for: Logistic Regression, Newton's Method, SVM
- Non-scaled data used for: Decision Tree, Random Forest
- Test size: 20%
- Random state: 42

Models Trained:
--------------
1. Logistic Regression (LBFGS solver)
2. Decision Tree
3. Logistic Regression with Newton's Method
4. Random Forest
5. Support Vector Machine (SVM)

Model Performances:
------------------
"""
    
    # Add performance for each model
    for model_name, metrics in all_metrics.items():
        summary_content += f"\n{model_name}:\n"
        summary_content += f"- Accuracy:  {metrics.get('accuracy', 'N/A'):.4f}\n"
        summary_content += f"- Precision: {metrics.get('precision', 'N/A'):.4f}\n"
        summary_content += f"- Recall:    {metrics.get('recall', 'N/A'):.4f}\n"
        summary_content += f"- F1 Score:  {metrics.get('f1', 'N/A'):.4f}\n"
        roc_auc = metrics.get('roc_auc', 'N/A')
        if roc_auc is not None:
            summary_content += f"- ROC-AUC:   {roc_auc:.4f}\n"
        else:
            summary_content += f"- ROC-AUC:   N/A\n"
        summary_content += f"- Training Time: {metrics.get('training_time', 'N/A'):.2f}s\n"

    summary_content += f"\n\nBest Model: {best_model_name}\n"
    
    summary_content += """
Files Generated:
---------------
1. models/logistic_regression_model.pkl
2. models/decision_tree_model.pkl
3. models/newton_method_model.pkl
4. models/random_forest_model.pkl
5. models/svm_model.pkl
6. results/model_comparison_summary.csv
7. results/detailed_model_comparison.csv
8. results/training_summary.txt
"""
    
    os.makedirs("results", exist_ok=True)
    with open("results/training_summary.txt", "w") as f:
        f.write(summary_content)
    
    print(f"\n✓ Training summary saved to: results/training_summary.txt")

def train_all_models():
    """
    Main function to train all 5 models
    """
    print("="*80)
    print("STARTING MODEL TRAINING - 5 ALGORITHMS")
    print("="*80)
    
    # Load datasets
    scaled_df, nonscaled_df = load_datasets()
    
    if scaled_df is None or nonscaled_df is None:
        print("Failed to load datasets. Exiting...")
        return None
    
    all_models = {}
    all_metrics = {}
    
    # Train Logistic Regression
    lr_model, lr_metrics = train_logistic_regression(scaled_df)
    all_models['Logistic Regression'] = lr_model
    all_metrics['Logistic Regression'] = lr_metrics
    
    # Train Decision Tree
    dt_model, dt_metrics = train_decision_tree(nonscaled_df)
    all_models['Decision Tree'] = dt_model
    all_metrics['Decision Tree'] = dt_metrics
    
    # Train Newton's Method
    newton_model, newton_metrics = train_newton_method(scaled_df)
    all_models["Newton's Method"] = newton_model
    all_metrics["Newton's Method"] = newton_metrics
    
    # Train Random Forest
    rf_model, rf_metrics = train_random_forest(nonscaled_df)
    all_models['Random Forest'] = rf_model
    all_metrics['Random Forest'] = rf_metrics
    
    # Train SVM
    svm_model, svm_metrics = train_svm(scaled_df)
    all_models['SVM'] = svm_model
    all_metrics['SVM'] = svm_metrics
    
    # Compare all models
    best_model_name = compare_all_models(all_metrics)
    
    # Save training summary
    save_training_summary(all_metrics, best_model_name)
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print("\n✓ All 5 models trained and saved")
    print("✓ Performance comparison generated")
    print("✓ Results saved to 'results/' directory")
    print("\nAvailable models:")
    print("1. Logistic Regression (models/logistic_regression_model.pkl)")
    print("2. Decision Tree (models/decision_tree_model.pkl)")
    print("3. Newton's Method (models/newton_method_model.pkl)")
    print("4. Random Forest (models/random_forest_model.pkl)")
    print("5. SVM (models/svm_model.pkl)")
    
    return all_models, all_metrics

if __name__ == "__main__":
    # Run training pipeline for all models
    all_models, all_metrics = train_all_models()
    
    if all_models:
        print("\nTo use the models for prediction:")
        print("""
        import joblib
        
        # Load any model
        model = joblib.load('models/model_name.pkl')
        
        # For scaled data models (Logistic Regression, Newton's Method, SVM):
        # Make sure to scale your data first using the saved scaler
        scaler = joblib.load('models/standard_scaler.pkl')
        X_scaled = scaler.transform(X[continuous_features])
        
        # For non-scaled data models (Decision Tree, Random Forest):
        # Use data directly without scaling
        
        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)  # if available
        """)