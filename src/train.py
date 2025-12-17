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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import time
import warnings
warnings.filterwarnings('ignore')

# Add src to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Define constants
TARGET = 'Satisfaction'

def load_and_split_data():
    """
    Load and split data once for all models
    """
    print("Loading and splitting data...")
    
    try:
        # Load scaled dataset
        scaled_data_path = "data/processed/scaled_classification_data.csv"
        if os.path.exists(scaled_data_path):
            scaled_df = pd.read_csv(scaled_data_path)
        else:
            scaled_data_path = "data/processed/classification_data.csv"
            scaled_df = pd.read_csv(scaled_data_path)
        
        # Load non-scaled dataset
        nonscaled_data_path = "data/processed/classification_data.csv"
        nonscaled_df = pd.read_csv(nonscaled_data_path)
        
        # Split scaled data
        X_scaled = scaled_df.drop(columns=[TARGET])
        y_scaled = scaled_df[TARGET]
        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42, stratify=y_scaled
        )
        
        # Split non-scaled data
        X_nonscaled = nonscaled_df.drop(columns=[TARGET])
        y_nonscaled = nonscaled_df[TARGET]
        X_train_ns, X_test_ns, y_train_ns, y_test_ns = train_test_split(
            X_nonscaled, y_nonscaled, test_size=0.2, random_state=42, stratify=y_nonscaled
        )
        
        print(f"  Scaled data: {X_train_s.shape} train, {X_test_s.shape} test")
        print(f"  Non-scaled data: {X_train_ns.shape} train, {X_test_ns.shape} test")
        
        data = {
            'scaled': {
                'X_train': X_train_s,
                'X_test': X_test_s,
                'y_train': y_train_s,
                'y_test': y_test_s
            },
            'nonscaled': {
                'X_train': X_train_ns,
                'X_test': X_test_ns,
                'y_train': y_train_ns,
                'y_test': y_test_ns
            }
        }
        
        return data
        
    except FileNotFoundError:
        print("Error: Please run preprocess.py first")
        return None

def train_logistic_regression(data):
    """
    Train Logistic Regression model
    """
    print("\n[1/8] Training Logistic Regression...")
    
    X_train = data['scaled']['X_train']
    y_train = data['scaled']['y_train']
    
    start_time = time.time()
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        solver='lbfgs',
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    save_model(model, "logistic_regression")
    return model, training_time

def train_decision_tree(data):
    """
    Train Decision Tree model
    """
    print("\n[2/8] Training Decision Tree...")
    
    X_train = data['nonscaled']['X_train']
    y_train = data['nonscaled']['y_train']
    
    start_time = time.time()
    model = DecisionTreeClassifier(
        random_state=42,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    save_model(model, "decision_tree")
    return model, training_time

def train_newton_method(data):
    """
    Train Logistic Regression with Newton's method
    """
    print("\n[3/8] Training Newton's Method...")
    
    X_train = data['scaled']['X_train']
    y_train = data['scaled']['y_train']
    
    start_time = time.time()
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        solver='newton-cg',
        class_weight='balanced',
        penalty='l2'
    )
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    save_model(model, "newton_method")
    return model, training_time

def train_random_forest(data):
    """
    Train Random Forest model
    """
    print("\n[4/8] Training Random Forest...")
    
    X_train = data['nonscaled']['X_train']
    y_train = data['nonscaled']['y_train']
    
    start_time = time.time()
    model = RandomForestClassifier(
        random_state=42,
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    save_model(model, "random_forest")
    return model, training_time

def train_svm(data):
    """
    Train Support Vector Machine model
    """
    print("\n[5/8] Training SVM...")
    
    X_train = data['scaled']['X_train']
    y_train = data['scaled']['y_train']
    
    start_time = time.time()
    model = SVC(
        random_state=42,
        C=1.0,
        kernel='rbf',
        probability=True,
        class_weight='balanced',
        gamma='scale'
    )
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    save_model(model, "svm")
    return model, training_time

def train_knn(data):
    """
    Train K-Nearest Neighbors model
    """
    print("\n[6/8] Training KNN...")
    
    X_train = data['scaled']['X_train']
    y_train = data['scaled']['y_train']
    
    start_time = time.time()
    model = KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',
        algorithm='auto',
        leaf_size=30,
        p=2,
        metric='minkowski',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    save_model(model, "knn")
    return model, training_time

def train_gaussian_naive_bayes(data):
    """
    Train Gaussian Naive Bayes model
    """
    print("\n[7/8] Training Gaussian Naive Bayes...")
    
    X_train = data['scaled']['X_train']
    y_train = data['scaled']['y_train']
    
    start_time = time.time()
    model = GaussianNB(var_smoothing=1e-9)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    save_model(model, "gaussian_nb")
    return model, training_time

def train_neural_network(data):
    """
    Train Neural Network model
    """
    print("\n[8/8] Training Neural Network...")
    
    X_train = data['scaled']['X_train']
    y_train = data['scaled']['y_train']
    
    start_time = time.time()
    n_features = X_train.shape[1]
    hidden_layer1 = max(50, n_features * 2)
    hidden_layer2 = max(25, n_features)
    
    model = MLPClassifier(
        hidden_layer_sizes=(hidden_layer1, hidden_layer2),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=False
    )
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    save_model(model, "neural_network")
    return model, training_time

def save_model(model, model_name):
    """
    Save trained model to disk
    """
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{model_name}_model.pkl"
    joblib.dump(model, model_path)
    
    
    return model_path

def compare_training_times(training_times):
    """
    Compare training times of all models
    """
    print("\n" + "="*60)
    print("TRAINING TIME COMPARISON")
    print("="*60)
    
    df = pd.DataFrame({
        'Model': list(training_times.keys()),
        'Time (s)': list(training_times.values())
    }).sort_values('Time (s)')
    
    for idx, row in df.iterrows():
        print(f"{row['Model']:25} {row['Time (s)']:>10.4f}s")
    
    print("\nSummary:")
    print(f"Fastest: {df.iloc[0]['Model']} ({df.iloc[0]['Time (s)']:.4f}s)")
    print(f"Slowest: {df.iloc[-1]['Model']} ({df.iloc[-1]['Time (s)']:.4f}s)")
    print(f"Average: {df['Time (s)'].mean():.4f}s")
    print(f"Total:   {df['Time (s)'].sum():.4f}s")
    
    # Save comparison
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/training_times.csv", index=False)

def train_all_models():
    """
    Main function to train all 8 models
    """
    print("="*60)
    print("TRAINING 8 MODELS")
    print("="*60)
    
    # Load and split data once
    data = load_and_split_data()
    if data is None:
        return
    
    all_models = {}
    training_times = {}
    total_start = time.time()
    
    # Train each model using the pre-split data
    lr_model, lr_time = train_logistic_regression(data)
    all_models['Logistic Regression'] = lr_model
    training_times['Logistic Regression'] = lr_time
    
    dt_model, dt_time = train_decision_tree(data)
    all_models['Decision Tree'] = dt_model
    training_times['Decision Tree'] = dt_time
    
    nm_model, nm_time = train_newton_method(data)
    all_models["Newton's Method"] = nm_model
    training_times["Newton's Method"] = nm_time
    
    rf_model, rf_time = train_random_forest(data)
    all_models['Random Forest'] = rf_model
    training_times['Random Forest'] = rf_time
    
    svm_model, svm_time = train_svm(data)
    all_models['SVM'] = svm_model
    training_times['SVM'] = svm_time
    
    knn_model, knn_time = train_knn(data)
    all_models['KNN'] = knn_model
    training_times['KNN'] = knn_time
    
    gnb_model, gnb_time = train_gaussian_naive_bayes(data)
    all_models['Gaussian NB'] = gnb_model
    training_times['Gaussian NB'] = gnb_time
    
    nn_model, nn_time = train_neural_network(data)
    all_models['Neural Network'] = nn_model
    training_times['Neural Network'] = nn_time
    
    total_time = time.time() - total_start
    
    # Compare training times
    compare_training_times(training_times)
    
    print(f"\n" + "="*60)
    print(f"ALL 8 MODELS TRAINED IN {total_time:.2f}s")
    print("="*60)
    print("\nModels saved to 'models/' directory")
    print("Run evaluate.py for performance comparison")
    
    return all_models, training_times

if __name__ == "__main__":
    train_all_models()