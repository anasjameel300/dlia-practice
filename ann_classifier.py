import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_airlines_data():
    """Load and preprocess the airlines flights dataset"""
    print("Loading airlines flights dataset...")
    df = pd.read_csv('airlines_flights_data.csv')
    
    # Remove unnecessary columns
    df = df.drop(['index', 'flight'], axis=1)
    
    # Handle categorical variables
    categorical_cols = ['airline', 'source_city', 'departure_time', 'stops', 
                       'arrival_time', 'destination_city', 'class']
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # Convert price to binary classification (expensive vs cheap)
    price_median = df['price'].median()
    df['price_category'] = (df['price'] > price_median).astype(int)
    
    # Select features for classification
    feature_cols = ['airline', 'source_city', 'departure_time', 'stops', 
                   'arrival_time', 'destination_city', 'class', 'duration', 'days_left']
    
    X = df[feature_cols].values
    y = df['price_category'].values
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_cols}")
    print(f"Price categories: Cheap (0): {(y == 0).sum()}, Expensive (1): {(y == 1).sum()}")
    
    return X, y, feature_cols

def train_and_evaluate_model(X_train, X_test, y_train, y_test, hidden_layers, learning_rate, max_iter=200):
    """Train and evaluate MLPClassifier with given configuration"""
    
    # Create model with specified configuration
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        learning_rate_init=learning_rate,
        max_iter=max_iter,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=False
    )
    
    # Measure training time
    start_time = time.time()
    
    # Train the model
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Get confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return accuracy, training_time, cm, model

def plot_results(results):
    """Create visualization plots for the results"""
    
    # Prepare data for plotting
    configs = []
    accuracies = []
    training_times = []
    
    for result in results:
        config_name = f"{result['config']}\nLR={result['learning_rate']}"
        configs.append(config_name)
        accuracies.append(result['accuracy'])
        training_times.append(result['training_time'])
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy bar plot
    bars1 = ax1.bar(range(len(configs)), accuracies, color='skyblue', alpha=0.7)
    ax1.set_title('Accuracy by Configuration', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Accuracy')
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels(configs, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Training time bar plot
    bars2 = ax2.bar(range(len(configs)), training_times, color='lightcoral', alpha=0.7)
    ax2.set_title('Training Time by Configuration', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Training Time (seconds)')
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels(configs, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, time_val in zip(bars2, training_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrices(results):
    """Plot confusion matrices for all configurations"""
    import math
    n_configs = len(results)
    n_cols = 3
    n_rows = math.ceil(n_configs / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    
    # Handle single row case
    if n_rows == 1:
        axes = [axes] if n_configs == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, result in enumerate(results):
        cm = result['confusion_matrix']
        config_name = f"{result['config']}\nLR={result['learning_rate']}"
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Cheap', 'Expensive'],
                   yticklabels=['Cheap', 'Expensive'],
                   ax=axes[i])
        axes[i].set_title(f'Confusion Matrix\n{config_name}', fontsize=10)
        axes[i].set_ylabel('True Label')
        axes[i].set_xlabel('Predicted Label')
    
    # Hide unused subplots
    for i in range(n_configs, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def main():
    print("="*80)
    print("3-LAYER ARTIFICIAL NEURAL NETWORK CLASSIFICATION")
    print("Using scikit-learn MLPClassifier")
    print("="*80)
    
    # Load and preprocess data
    X, y, feature_cols = load_and_preprocess_airlines_data()
    
    # Split data (70% train, 15% validation, 15% test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # Combine train and validation for final training
    X_train_final = np.vstack([X_train, X_val])
    y_train_final = np.hstack([y_train, y_val])
    
    # Scale features
    scaler = StandardScaler()
    X_train_final = scaler.fit_transform(X_train_final)
    X_test = scaler.transform(X_test)
    
    print(f"\nTraining set size: {X_train_final.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Number of features: {X_train_final.shape[1]}")
    
    # Part 1: Different hidden layer configurations
    hidden_configs = [
        ([25, 15], "Small (<50 neurons)"),
        ([128, 64, 32], "Large (>100 neurons)")
    ]
    
    # Part 2: Different learning rates
    learning_rates = [0.0001, 0.001, 0.01]
    
    results = []
    
    print("\n" + "="*80)
    print("EXPERIMENTING WITH SMALL AND LARGE CONFIGURATIONS")
    print("="*80)
    
    for hidden_layers, config_name in hidden_configs:
        for lr in learning_rates:
            print(f"\n{'='*60}")
            print(f"Configuration: {config_name}")
            print(f"Hidden layers: {hidden_layers}")
            print(f"Learning rate: {lr}")
            print(f"{'='*60}")
            
            # Train and evaluate model
            accuracy, training_time, cm, model = train_and_evaluate_model(
                X_train_final, X_test, y_train_final, y_test, 
                hidden_layers, lr
            )
            
            # Count total parameters (approximate)
            total_params = 0
            layer_sizes = [X_train_final.shape[1]] + list(hidden_layers) + [1]
            for i in range(len(layer_sizes) - 1):
                total_params += layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]  # weights + biases
            
            print(f"Total parameters: {total_params:,}")
            print(f"Training time: {training_time:.2f} seconds")
            print(f"Test accuracy: {accuracy:.4f}")
            
            # Store results
            result = {
                'config': config_name,
                'hidden_layers': hidden_layers,
                'learning_rate': lr,
                'total_params': total_params,
                'training_time': training_time,
                'accuracy': accuracy,
                'confusion_matrix': cm,
                'model': model
            }
            results.append(result)
    
    # Summary table
    print("\n" + "="*100)
    print("SUMMARY OF ALL EXPERIMENTS")
    print("="*100)
    
    summary_data = []
    for result in results:
        summary_data.append({
            'Configuration': result['config'],
            'Hidden Layers': str(result['hidden_layers']),
            'Learning Rate': result['learning_rate'],
            'Parameters': f"{result['total_params']:,}",
            'Training Time (s)': f"{result['training_time']:.2f}",
            'Test Accuracy': f"{result['accuracy']:.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Find best configuration
    best_result = max(results, key=lambda x: x['accuracy'])
    print(f"\nBest Configuration:")
    print(f"Config: {best_result['config']}")
    print(f"Hidden layers: {best_result['hidden_layers']}")
    print(f"Learning rate: {best_result['learning_rate']}")
    print(f"Test accuracy: {best_result['accuracy']:.4f}")
    print(f"Training time: {best_result['training_time']:.2f} seconds")
    
    # Time complexity analysis
    print(f"\nTime Complexity Analysis:")
    for result in results:
        params_per_second = result['total_params'] / result['training_time']
        print(f"{result['config']} - LR={result['learning_rate']}: "
              f"{params_per_second:.0f} parameters/second")
    
    # Create visualizations
    print(f"\nGenerating visualizations...")
    plot_results(results)
    plot_confusion_matrices(results)
    
    print(f"\n✅ All experiments completed successfully!")

if __name__ == "__main__":
    main()
