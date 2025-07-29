import os
import cv2
import numpy as np
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier

# Parameters
img_size = 128
dataset_path = r"C:\Users\anasj\Desktop\dlia-practice\Concrete Crack Images 250"
classes = ['Negative', 'Positive']

# LBP parameters
radius = 1
n_points = 8 * radius
method = 'uniform'

def extract_hog_features(image):
    """Extract only HOG features from an image"""
    image = cv2.resize(image, (img_size, img_size))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # HOG features only
    hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
    
    return hog_features

def extract_lbp_features(image):
    """Extract only LBP features from an image"""
    image = cv2.resize(image, (img_size, img_size))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # LBP features only
    lbp = local_binary_pattern(gray, n_points, radius, method)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3),
                          range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    
    return hist

def extract_fusion_features(image):
    """Extract both HOG and LBP features from an image"""
    image = cv2.resize(image, (img_size, img_size))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # HOG features
    hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
    
    # LBP features
    lbp = local_binary_pattern(gray, n_points, radius, method)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3),
                          range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    
    # Combine features
    return np.hstack([hog_features, hist])

def extract_features(image):
    """Legacy function - kept for compatibility"""
    return extract_fusion_features(image)

def load_dataset(feature_method='fusion'):
    """Load and preprocess the dataset with specified feature extraction method"""
    X, y = [], []
    
    # Choose feature extraction method
    if feature_method == 'hog':
        extract_func = extract_hog_features
    elif feature_method == 'lbp':
        extract_func = extract_lbp_features
    elif feature_method == 'fusion':
        extract_func = extract_fusion_features
    else:
        raise ValueError("feature_method must be 'hog', 'lbp', or 'fusion'")
    
    for label_idx, label in enumerate(classes):
        folder = os.path.join(dataset_path, label)
        print(f"Loading {label} images with {feature_method.upper()} features...")
        
        for file in tqdm(os.listdir(folder)):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path)
            
            if img is not None:
                try:
                    features = extract_func(img)
                    X.append(features)
                    y.append(label_idx)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    return np.array(X), np.array(y)

def load_hog_dataset():
    """Load dataset with HOG features only"""
    return load_dataset('hog')

def load_lbp_dataset():
    """Load dataset with LBP features only"""
    return load_dataset('lbp')

def load_fusion_dataset():
    """Load dataset with HOG+LBP fusion features"""
    return load_dataset('fusion')

def train_svm(X_train, y_train):
    """Train SVM classifier"""
    print("\nTraining SVM classifier...")
    svm = SVC(kernel='linear', probability=True)
    svm.fit(X_train, y_train)
    return svm

def train_ann(X_train, y_train):
    """Train ANN classifier"""
    print("\nTraining ANN classifier...")
    ann = MLPClassifier(hidden_layer_sizes=(256, 128, 64),
                       activation='relu',
                       solver='adam',
                       learning_rate_init=0.001,
                       max_iter=500,
                       random_state=42,
                       verbose=True)
    ann.fit(X_train, y_train)
    return ann

def train_naive_bayes(X_train, y_train):
    """Train Naive Bayes classifier"""
    print("\nTraining Naive Bayes classifier...")
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    return nb

def train_knn(X_train, y_train):
    """Train KNN classifier"""
    print("\nTraining KNN classifier...")
    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    knn.fit(X_train, y_train)
    return knn

def train_decision_tree(X_train, y_train):
    """Train Decision Tree classifier"""
    print("\nTraining Decision Tree classifier...")
    dt = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt.fit(X_train, y_train)
    return dt

def plot_confusion_matrix(y_true, y_pred, title):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

def compare_classifiers(y_test, predictions_dict):
    """Compare all classifiers and find the best performing one"""
    print("\n" + "="*60)
    print("CLASSIFIER COMPARISON AND RESULTS")
    print("="*60)
    
    results = {}
    
    for name, y_pred in predictions_dict.items():
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        print(f"\n{name.upper()} RESULTS:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
    
    # Find the best classifier based on F1-score (most balanced metric)
    best_classifier = max(results.items(), key=lambda x: x[1]['f1_score'])
    
    print("\n" + "="*60)
    print("🏆 BEST PERFORMING CLASSIFIER")
    print("="*60)
    print(f"Winner: {best_classifier[0].upper()}")
    print(f"F1-Score: {best_classifier[1]['f1_score']:.4f}")
    print(f"Accuracy: {best_classifier[1]['accuracy']:.4f}")
    print(f"Precision: {best_classifier[1]['precision']:.4f}")
    print(f"Recall: {best_classifier[1]['recall']:.4f}")
    
    # Create a comparison table
    print("\n" + "="*80)
    print("DETAILED COMPARISON TABLE")
    print("="*80)
    print(f"{'Classifier':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 80)
    
    for name, metrics in results.items():
        print(f"{name:<15} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f}")
    
    return best_classifier[0], results

def evaluate_feature_method(feature_method, X_train, X_test, y_train, y_test):
    """Evaluate all classifiers with a specific feature extraction method"""
    print(f"\n{'='*60}")
    print(f"EVALUATING WITH {feature_method.upper()} FEATURES")
    print(f"{'='*60}")
    
    # Train all classifiers
    print(f"\nTraining classifiers with {feature_method.upper()} features...")
    
    svm_model = train_svm(X_train, y_train)
    ann_model = train_ann(X_train, y_train)
    nb_model = train_naive_bayes(X_train, y_train)
    knn_model = train_knn(X_train, y_train)
    dt_model = train_decision_tree(X_train, y_train)
    
    # Make predictions
    svm_predictions = svm_model.predict(X_test)
    ann_predictions = ann_model.predict(X_test)
    nb_predictions = nb_model.predict(X_test)
    knn_predictions = knn_model.predict(X_test)
    dt_predictions = dt_model.predict(X_test)
    
    # Print individual results
    print(f"\n--- SVM with {feature_method.upper()} ---")
    print(classification_report(y_test, svm_predictions, target_names=classes))
    plot_confusion_matrix(y_test, svm_predictions, f"SVM with {feature_method.upper()}")
    
    print(f"\n--- ANN with {feature_method.upper()} ---")
    print(classification_report(y_test, ann_predictions, target_names=classes))
    plot_confusion_matrix(y_test, ann_predictions, f"ANN with {feature_method.upper()}")
    
    print(f"\n--- Naive Bayes with {feature_method.upper()} ---")
    print(classification_report(y_test, nb_predictions, target_names=classes))
    plot_confusion_matrix(y_test, nb_predictions, f"Naive Bayes with {feature_method.upper()}")
    
    print(f"\n--- KNN with {feature_method.upper()} ---")
    print(classification_report(y_test, knn_predictions, target_names=classes))
    plot_confusion_matrix(y_test, knn_predictions, f"KNN with {feature_method.upper()}")
    
    print(f"\n--- Decision Tree with {feature_method.upper()} ---")
    print(classification_report(y_test, dt_predictions, target_names=classes))
    plot_confusion_matrix(y_test, dt_predictions, f"Decision Tree with {feature_method.upper()}")
    
    # Compare classifiers for this feature method
    predictions_dict = {
        "SVM": svm_predictions,
        "ANN": ann_predictions,
        "Naive Bayes": nb_predictions,
        "KNN": knn_predictions,
        "Decision Tree": dt_predictions
    }
    
    best_classifier_name, results = compare_classifiers(y_test, predictions_dict)
    
    return best_classifier_name, results


# Main execution
if __name__ == "__main__":
    print("CONCRETE CRACK CLASSIFICATION WITH MULTIPLE FEATURE EXTRACTION METHODS")
    print("="*80)
    
    # Load datasets for each feature method
    print("\n1. Loading HOG dataset...")
    X_hog, y_hog = load_hog_dataset()
    print(f"HOG dataset shape: {X_hog.shape}")
    
    print("\n2. Loading LBP dataset...")
    X_lbp, y_lbp = load_lbp_dataset()
    print(f"LBP dataset shape: {X_lbp.shape}")
    
    print("\n3. Loading Fusion dataset...")
    X_fusion, y_fusion = load_fusion_dataset()
    print(f"Fusion dataset shape: {X_fusion.shape}")
    
    # Split datasets
    print("\n4. Splitting datasets...")
    X_hog_train, X_hog_test, y_hog_train, y_hog_test = train_test_split(
        X_hog, y_hog, test_size=0.2, random_state=42, stratify=y_hog)
    
    X_lbp_train, X_lbp_test, y_lbp_train, y_lbp_test = train_test_split(
        X_lbp, y_lbp, test_size=0.2, random_state=42, stratify=y_lbp)
    
    X_fusion_train, X_fusion_test, y_fusion_train, y_fusion_test = train_test_split(
        X_fusion, y_fusion, test_size=0.2, random_state=42, stratify=y_fusion)
    
    # Evaluate each feature method
    print("\n5. Evaluating HOG features...")
    best_hog_classifier, hog_results = evaluate_feature_method('hog', X_hog_train, X_hog_test, y_hog_train, y_hog_test)
    
    print("\n6. Evaluating LBP features...")
    best_lbp_classifier, lbp_results = evaluate_feature_method('lbp', X_lbp_train, X_lbp_test, y_lbp_train, y_lbp_test)
    
    print("\n7. Evaluating Fusion features...")
    best_fusion_classifier, fusion_results = evaluate_feature_method('fusion', X_fusion_train, X_fusion_test, y_fusion_train, y_fusion_test)
    
    # Final comparison across all feature methods
    print("\n" + "="*80)
    print("FINAL COMPARISON: BEST CLASSIFIER FOR EACH FEATURE METHOD")
    print("="*80)
    print(f"HOG Best Classifier: {best_hog_classifier}")
    print(f"LBP Best Classifier: {best_lbp_classifier}")
    print(f"Fusion Best Classifier: {best_fusion_classifier}")
    
    # Find the overall best combination
    all_results = {
        'HOG': hog_results[best_hog_classifier]['f1_score'],
        'LBP': lbp_results[best_lbp_classifier]['f1_score'],
        'Fusion': fusion_results[best_fusion_classifier]['f1_score']
    }
    
    best_feature_method = max(all_results.items(), key=lambda x: x[1])
    
    print(f"\n🏆 OVERALL BEST COMBINATION:")
    print(f"Feature Method: {best_feature_method[0]}")
    print(f"Best Classifier: {best_hog_classifier if best_feature_method[0] == 'HOG' else best_lbp_classifier if best_feature_method[0] == 'LBP' else best_fusion_classifier}")
    print(f"F1-Score: {best_feature_method[1]:.4f}")
    
    plt.show()
