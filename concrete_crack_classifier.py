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

# Parameters
img_size = 128
dataset_path = r"C:\Users\anasj\Desktop\dlia-practice\Concrete Crack Images 250"
classes = ['Negative', 'Positive']

# LBP parameters
radius = 1
n_points = 8 * radius
method = 'uniform'

def extract_features(image):
    """Extract HOG and LBP features from an image"""
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

def load_dataset():
    """Load and preprocess the dataset"""
    X, y = [], []
    
    for label_idx, label in enumerate(classes):
        folder = os.path.join(dataset_path, label)
        print(f"Loading {label} images...")
        
        for file in tqdm(os.listdir(folder)):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path)
            
            if img is not None:
                try:
                    features = extract_features(img)
                    X.append(features)
                    y.append(label_idx)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    return np.array(X), np.array(y)

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


# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    print("Loading and preprocessing dataset...")
    X, y = load_dataset()
    print("Dataset shape:", X.shape)
    print("Class distribution:", Counter(y))
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train and evaluate SVM
    print("\n=== SVM Classification ===")
    svm_model = train_svm(X_train, y_train)
    svm_predictions = svm_model.predict(X_test)
    
    print("\nSVM Classification Report:")
    print(classification_report(y_test, svm_predictions, target_names=classes))
    plot_confusion_matrix(y_test, svm_predictions, "SVM Confusion Matrix")
    
    # Train and evaluate ANN
    print("\n=== ANN Classification ===")
    ann_model = train_ann(X_train, y_train)
    ann_predictions = ann_model.predict(X_test)
    
    print("\nANN Classification Report:")
    print(classification_report(y_test, ann_predictions, target_names=classes))
    plot_confusion_matrix(y_test, ann_predictions, "ANN Confusion Matrix")

    # Train and evaluate Naive Bayes
    print("\n=== Naive Bayes Classification ===")
    nb_model = train_naive_bayes(X_train, y_train)
    nb_predictions = nb_model.predict(X_test)
    
    print("\nNaive Bayes Classification Report:")
    print(classification_report(y_test, nb_predictions, target_names=classes))
    plot_confusion_matrix(y_test, nb_predictions, "Naive Bayes Confusion Matrix")

    # Train and evaluate KNN
    print("\n=== KNN Classification ===")
    knn_model = train_knn(X_train, y_train)
    knn_predictions = knn_model.predict(X_test)
    
    print("\nKNN Classification Report:")
    print(classification_report(y_test, knn_predictions, target_names=classes))
    plot_confusion_matrix(y_test, knn_predictions, "KNN Confusion Matrix")

    # Compare all classifiers
    predictions_dict = {
        "SVM": svm_predictions,
        "ANN": ann_predictions,
        "Naive Bayes": nb_predictions,
        "KNN": knn_predictions
    }
    best_classifier_name, results = compare_classifiers(y_test, predictions_dict)

    plt.show()
