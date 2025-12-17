import pandas as pd
import numpy as np
import os
import sys
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Add src to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Define constants
TARGET = 'Satisfaction'

def load_models():
    """
    Load all trained models
    """
    print("Loading trained models...")
    
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
    
    all_models = {}
    loaded_count = 0
    
    for model_name, model_path in model_files.items():
        if os.path.exists(model_path):
            all_models[model_name] = joblib.load(model_path)
            loaded_count += 1
            print(f"  ✓ {model_name}")
        else:
            print(f"  ✗ {model_name} (not found)")
    
    print(f"\nLoaded {loaded_count}/8 models")
    return all_models

def load_and_split_data():
    """
    Load and split data for evaluation
    """
    print("\nLoading and splitting data...")
    
    # Load scaled dataset for scaled models
    scaled_data_path = "data/processed/scaled_classification_data.csv"
    if os.path.exists(scaled_data_path):
        scaled_df = pd.read_csv(scaled_data_path)
    else:
        scaled_data_path = "data/processed/classification_data.csv"
        scaled_df = pd.read_csv(scaled_data_path)
    
    # Load non-scaled dataset for tree-based models
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
    
    test_data = {
        'scaled': (X_test_s, y_test_s),
        'nonscaled': (X_test_ns, y_test_ns)
    }
    
    print(f"  Scaled test set: {X_test_s.shape}")
    print(f"  Non-scaled test set: {X_test_ns.shape}")
    
    return test_data

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a single model
    """
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # ROC-AUC
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    except:
        roc_auc = None
    
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'prediction_time': prediction_time
    }

def evaluate_all_models(all_models, test_data):
    """
    Evaluate all loaded models
    """
    print("\n" + "="*60)
    print("EVALUATING MODELS")
    print("="*60)
    
    all_metrics = {}
    
    for model_name, model in all_models.items():
        print(f"\n{model_name}:")
        
        # Select appropriate test data
        if model_name in ['Decision Tree', 'Random Forest']:
            X_test, y_test = test_data['nonscaled']
        else:
            X_test, y_test = test_data['scaled']
        
        metrics = evaluate_model(model, X_test, y_test, model_name)
        all_metrics[model_name] = metrics
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        if metrics['roc_auc']:
            print(f"  ROC-AUC:  {metrics['roc_auc']:.4f}")
    
    return all_metrics

def compare_performance(all_metrics):
    """
    Compare performance of all models
    """
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)
    
    # Create comparison table
    comparison = []
    
    for model_name, metrics in all_metrics.items():
        comparison.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1 Score': metrics['f1'],
            'ROC-AUC': metrics['roc_auc'] or np.nan,
            'Pred Time (ms)': metrics['prediction_time'] * 1000
        })
    
    df = pd.DataFrame(comparison)
    
    # Sort by accuracy
    df_sorted = df.sort_values('Accuracy', ascending=False)
    
    print("\nRanked by Accuracy:")
    print("-" * 80)
    print(df_sorted.to_string(index=False))
    
    # Best models for each metric
    print("\n" + "="*80)
    print("BEST MODELS")
    print("="*80)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    for metric in metrics:
        best = df.loc[df[metric].idxmax()]
        print(f"{metric:12}: {best['Model']:25} ({best[metric]:.4f})")
    
    # Fastest prediction
    fastest = df.loc[df['Pred Time (ms)'].idxmin()]
    print(f"Fastest    : {fastest['Model']:25} ({fastest['Pred Time (ms)']:.2f} ms)")
    
    return df

def save_results(comparison_df):
    """
    Save evaluation results
    """
    os.makedirs("results", exist_ok=True)
    
    # Save comparison
    comparison_df.to_csv("results/model_comparison_summary.csv", index=False)
    
    print(f"\nResults saved to 'results/' directory")

def main():
    """
    Main evaluation function
    """
    print("="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Load models
    all_models = load_models()
    if not all_models:
        print("No models found. Please run train.py first.")
        return
    
    # Load and split data
    test_data = load_and_split_data()
    
    # Evaluate models
    all_metrics = evaluate_all_models(all_models, test_data)
    
    # Compare performance
    comparison_df = compare_performance(all_metrics)
    
    # Save results
    save_results(comparison_df)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)

def create_visualizations(all_metrics, comparison_df):
    """
    Create and save important visualizations
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap
    
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # Create visuals directory
    os.makedirs("visuals", exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 1. Model Performance Comparison Bar Chart
    print("Creating performance comparison chart...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        sorted_df = comparison_df.sort_values(metric, ascending=True)
        bars = ax.barh(sorted_df['Model'], sorted_df[metric])
        ax.set_xlabel(metric)
        ax.set_title(f'{metric} by Model')
        ax.set_xlim(0, 1)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig('visuals/model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. ROC-AUC Comparison (for models that support it)
    print("Creating ROC-AUC comparison chart...")
    models_with_auc = [(name, metrics) for name, metrics in all_metrics.items() 
                      if metrics['roc_auc'] is not None]
    
    if models_with_auc:
        fig, ax = plt.subplots(figsize=(12, 8))
        models_sorted = sorted(models_with_auc, key=lambda x: x[1]['roc_auc'], reverse=True)
        
        model_names = [m[0] for m in models_sorted]
        auc_scores = [m[1]['roc_auc'] for m in models_sorted]
        
        bars = ax.barh(model_names, auc_scores)
        ax.set_xlabel('ROC-AUC Score')
        ax.set_title('ROC-AUC Comparison by Model')
        ax.set_xlim(0, 1)
        
        # Color bars by score
        cmap = LinearSegmentedColormap.from_list('score_cmap', ['red', 'yellow', 'green'])
        norm = plt.Normalize(0.5, 1.0)
        
        for bar, score in zip(bars, auc_scores):
            bar.set_color(cmap(norm(score)))
            ax.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{score:.4f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig('visuals/roc_auc_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Confusion Matrix Heatmaps
    print("Creating confusion matrix heatmaps...")
    n_models = len(all_metrics)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')
    
    for idx, (model_name, metrics) in enumerate(all_metrics.items()):
        row = idx // n_cols
        col = idx % n_cols
        
        if n_rows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]
        
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   cbar_kws={'label': 'Count'}, ax=ax)
        ax.set_title(model_name)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_xticklabels(['Negative', 'Positive'])
        ax.set_yticklabels(['Negative', 'Positive'])
    
    # Hide empty subplots
    for idx in range(len(all_metrics), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        if n_rows == 1:
            axes[col].axis('off')
        else:
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('visuals/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Radar Chart for Top 5 Models
    print("Creating radar chart for top models...")
    top_n = min(5, len(comparison_df))
    top_models = comparison_df.nlargest(top_n, 'Accuracy')
    
    if len(top_models) >= 3:
        metrics_radar = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        num_vars = len(metrics_radar)
        
        # Normalize data for radar chart
        normalized_data = top_models[metrics_radar].copy()
        for metric in metrics_radar:
            normalized_data[metric] = (normalized_data[metric] - normalized_data[metric].min()) / \
                                     (normalized_data[metric].max() - normalized_data[metric].min())
        
        # Compute angle for each axis
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]  # Close the loop
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Draw one axe per variable + add labels
        plt.xticks(angles[:-1], metrics_radar, size=12)
        
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.5", "0.75", "1.0"], color="grey", size=10)
        plt.ylim(0, 1)
        
        # Plot each model
        for idx, row in normalized_data.iterrows():
            values = row[metrics_radar].tolist()
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, linestyle='solid', 
                   label=top_models.loc[idx, 'Model'])
            ax.fill(angles, values, alpha=0.1)
        
        plt.title('Top Models Performance Comparison (Normalized)', size=16, y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.tight_layout()
        plt.savefig('visuals/top_models_radar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Prediction Speed Comparison
    print("Creating prediction speed comparison chart...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    speed_df = comparison_df.copy().sort_values('Pred Time (ms)', ascending=True)
    bars = ax.barh(speed_df['Model'], speed_df['Pred Time (ms)'])
    ax.set_xlabel('Prediction Time (milliseconds)')
    ax.set_title('Model Prediction Speed Comparison')
    ax.set_xlim(0, speed_df['Pred Time (ms)'].max() * 1.1)
    
    # Color bars - faster is better (greener)
    times = speed_df['Pred Time (ms)'].values
    norm = plt.Normalize(times.min(), times.max())
    cmap = LinearSegmentedColormap.from_list('speed_cmap', ['green', 'yellow', 'red'])
    
    for bar, time_val in zip(bars, times):
        bar.set_color(cmap(norm(time_val)))
        ax.text(time_val + (times.max() * 0.01), bar.get_y() + bar.get_height()/2, 
               f'{time_val:.2f} ms', ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig('visuals/prediction_speed_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Summary Dashboard
    print("Creating summary dashboard...")
    fig = plt.figure(figsize=(20, 15))
    
    # Layout for dashboard
    gs = fig.add_gridspec(3, 3)
    
    # 1. Top-left: Best model highlight
    ax1 = fig.add_subplot(gs[0, 0])
    best_model = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
    ax1.text(0.5, 0.5, f"🏆 Best Model\n{best_model['Model']}\nAccuracy: {best_model['Accuracy']:.4f}", 
            ha='center', va='center', fontsize=14, transform=ax1.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    ax1.set_title('Best Performing Model', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 2. Top-middle: Fastest model
    ax2 = fig.add_subplot(gs[0, 1])
    fastest_model = comparison_df.loc[comparison_df['Pred Time (ms)'].idxmin()]
    ax2.text(
        0.5, 0.5,
        f"⚡ Fastest Model\n{fastest_model['Model']}\n{fastest_model['Pred Time (ms)']:.2f} ms",
        ha='center', va='center', fontsize=14, transform=ax2.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7)
    )
    ax2.set_title('Fastest Prediction', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # 3. Top-right: Model count
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.text(0.5, 0.5, f"📊 Models Evaluated\n{len(comparison_df)} Total", 
            ha='center', va='center', fontsize=14, transform=ax3.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
    ax3.set_title('Evaluation Summary', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # 4. Middle row: Accuracy comparison (horizontal bar)
    ax4 = fig.add_subplot(gs[1, :])
    accuracy_sorted = comparison_df.sort_values('Accuracy', ascending=True)
    bars = ax4.barh(accuracy_sorted['Model'], accuracy_sorted['Accuracy'])
    ax4.set_xlabel('Accuracy')
    ax4.set_title('Model Accuracy Ranking', fontsize=14, fontweight='bold')
    ax4.set_xlim(0, 1)
    
    # Color bars by accuracy
    for bar, acc in zip(bars, accuracy_sorted['Accuracy']):
        color = plt.cm.RdYlGn(acc)  # Red-Yellow-Green colormap
        bar.set_color(color)
    
    # 5. Bottom row: Key metrics for top 3 models
    ax5 = fig.add_subplot(gs[2, :])
    top_3 = comparison_df.nlargest(3, 'Accuracy')
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    
    for i, (idx, row) in enumerate(top_3.iterrows()):
        values = [row[m] for m in metrics_to_plot]
        ax5.bar(x + i*width, values, width, label=row['Model'])
    
    ax5.set_xlabel('Metrics')
    ax5.set_ylabel('Score')
    ax5.set_title('Top 3 Models - Key Metrics', fontsize=14, fontweight='bold')
    ax5.set_xticks(x + width)
    ax5.set_xticklabels(metrics_to_plot)
    ax5.legend()
    ax5.set_ylim(0, 1)
    
    plt.suptitle('Model Evaluation Dashboard', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('visuals/evaluation_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ All visualizations saved to 'visuals/' directory")
    print(f"  Created {len([f for f in os.listdir('visuals') if f.endswith('.png')])} visualization files")

if __name__ == "__main__":
    # main()

    all_models = load_models()
    if all_models:
        test_data = load_and_split_data()
        all_metrics = evaluate_all_models(all_models, test_data)
        comparison_df = compare_performance(all_metrics)
        create_visualizations(all_metrics, comparison_df)