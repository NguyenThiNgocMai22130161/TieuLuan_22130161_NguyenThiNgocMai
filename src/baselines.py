import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

# Paths relative to this script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
FN_DIR = os.path.join(PROJECT_ROOT, "fn")
PKL_DIR = os.path.join(PROJECT_ROOT, "pkl")

def train_baselines():
    """
    Trains TF-IDF + Naive Bayes & SVM on cleaned datasets, saves models, and evaluates.
    """
    os.makedirs(PKL_DIR, exist_ok=True)
    
    train_path = os.path.join(FN_DIR, "cleaned_train.csv")
    test_path = os.path.join(FN_DIR, "cleaned_test.csv")
    
    print("=" * 60)
    print("Starting Baseline Training Pipeline (TF-IDF + NB / SVM)...")
    print("=" * 60)
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Error: Cleaned train or test files not found. Please run preprocessing first!")
        return

    # 1. Load preprocessed datasets
    print("⌛ Loading cleaned datasets...")
    train_df = pd.read_csv(train_path).fillna('')
    test_df = pd.read_csv(test_path).fillna('')
    
    X_train, y_train = train_df['comment_cleaned'], train_df['label']
    X_test, y_test = test_df['comment_cleaned'], test_df['label']
    
    # 2. Extract TF-IDF features
    print("⌛ Extracting TF-IDF features (max_features=5000, n-gram=(1,2))...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Save Vectorizer
    vectorizer_path = os.path.join(PKL_DIR, "tfidf_vectorizer.pkl")
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Saved TfidfVectorizer to {vectorizer_path}")
    
    # 3. Multinomial Naive Bayes
    print("⌛ Training Multinomial Naive Bayes...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    
    nb_pred = nb_model.predict(X_test_tfidf)
    nb_acc = accuracy_score(y_test, nb_pred)
    
    nb_model_path = os.path.join(PKL_DIR, "naive_bayes_model.pkl")
    joblib.dump(nb_model, nb_model_path)
    print(f"Saved Naive Bayes to {nb_model_path}")
    
    # 4. Support Vector Machine (LinearSVC)
    print("⌛ Training Linear Support Vector Machine...")
    svm_model = LinearSVC(C=0.1, max_iter=2000, random_state=42, dual=False)
    svm_model.fit(X_train_tfidf, y_train)
    
    svm_pred = svm_model.predict(X_test_tfidf)
    svm_acc = accuracy_score(y_test, svm_pred)
    
    svm_model_path = os.path.join(PKL_DIR, "svm_best_model.pkl")
    joblib.dump(svm_model, svm_model_path)
    print(f"Saved SVM to {svm_model_path}")
    
    # 5. Display Evaluation Results
    target_labels = ['Tiêu cực (0)', 'Trung tính (1)', 'Tích cực (2)']
    
    print("\n" + "=" * 40)
    print("RESULTS FOR MULTINOMIAL NAIVE BAYES")
    print("=" * 40)
    print(f"Test Accuracy: {nb_acc * 100:.2f}%")
    print(classification_report(y_test, nb_pred, target_names=target_labels, digits=4))
    
    print("\n" + "=" * 40)
    print("RESULTS FOR LINEAR SVM")
    print("=" * 40)
    print(f"Test Accuracy: {svm_acc * 100:.2f}%")
    print(classification_report(y_test, svm_pred, target_names=target_labels, digits=4))
    
    print("=" * 60)
    print("Baseline Training completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    train_baselines()
