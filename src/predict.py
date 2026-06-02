import os
import argparse
import joblib
import json
import torch
import torch.nn.functional as F

# Local imports
from preprocessing import clean_vietnamese_text
from lstm import BiLSTMClassifier, text_to_sequence

# Paths relative to this script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
PKL_DIR = os.path.join(PROJECT_ROOT, "pkl")

# Label mappings
labels_map = {0: "Tiêu cực (Negative)", 1: "Trung tính (Neutral)", 2: "Tích cực (Positive)"}
emoji_map = {0: "", 1: "", 2: ""}

def predict_sentiment(text, model_type="svm"):
    """
    Cleans the input text, runs it through the selected model, and returns the prediction.
    """
    # 1. Preprocess and segment input text
    cleaned_text = clean_vietnamese_text(text)
    
    if not cleaned_text:
        return "Văn bản rỗng sau tiền xử lý!", 0.0
        
    # 2. Prediction based on model type
    if model_type in ["nb", "svm"]:
        # Load TF-IDF & Classic Machine Learning Model
        tfidf_path = os.path.join(PKL_DIR, "tfidf_vectorizer.pkl")
        model_file = "naive_bayes_model.pkl" if model_type == "nb" else "svm_best_model.pkl"
        model_path = os.path.join(PKL_DIR, model_file)
        
        if not os.path.exists(tfidf_path) or not os.path.exists(model_path):
            return f"Error: Model or vectorizer files not found in {PKL_DIR}. Please train baselines first!", 0.0
            
        vectorizer = joblib.load(tfidf_path)
        model = joblib.load(model_path)
        
        # Transform and predict
        features = vectorizer.transform([cleaned_text])
        prediction = model.predict(features)[0]
        
        # Calculate confidence
        if model_type == "nb":
            probs = model.predict_proba(features)[0]
            confidence = probs[prediction]
        else:
            # SVM does not natively support probabilities without probability=True (which slows down training)
            # We can use decision_function values as a proxy or set default
            decision = model.decision_function(features)[0]
            # Simple softmax approximation for decision scores
            exp_dec = torch.exp(torch.tensor(decision))
            probs = (exp_dec / torch.sum(exp_dec)).numpy()
            confidence = probs[prediction]
            
        return prediction, confidence
        
    elif model_type == "lstm":
        # Load vocabulary & PyTorch LSTM weights
        vocab_path = os.path.join(PKL_DIR, "lstm_vocab.json")
        lstm_weights_path = os.path.join(PKL_DIR, "best_lstm_model_correct.pth")
        
        if not os.path.exists(vocab_path) or not os.path.exists(lstm_weights_path):
            return f"Error: LSTM weights or vocabulary files not found in {PKL_DIR}. Please train Bi-LSTM first!", 0.0
            
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
            
        # Convert text to sequence
        seq = text_to_sequence(cleaned_text, vocab)
        seq_tensor = torch.tensor([seq], dtype=torch.long)
        
        # Initialize and load model
        model = BiLSTMClassifier(vocab_size=len(vocab))
        model.load_state_dict(torch.load(lstm_weights_path, map_location='cpu'))
        model.eval()
        
        # Predict
        with torch.no_grad():
            outputs = model(seq_tensor)
            probs = F.softmax(outputs, dim=1)[0]
            prediction = torch.argmax(probs).item()
            confidence = probs[prediction].item()
            
        return prediction, confidence
        
    else:
        return f"Model '{model_type}' không được hỗ trợ!", 0.0

def interactive_loop(model_type):
    print("=" * 60)
    print("CHƯƠNG TRÌNH PHÂN TÍCH CẢM XÚC PHẢN HỒI SINH VIÊN")
    print(f"Sử dụng mô hình: {model_type.upper()}")
    print("Nhập 'exit' hoặc 'quit' để thoát chương trình.")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\nNhập câu phản hồi của bạn: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Tạm biệt!")
                break
                
            pred_class, conf = predict_sentiment(user_input, model_type)
            
            if isinstance(pred_class, str): # Error message
                print(pred_class)
                continue
                
            label_text = labels_map[pred_class]
            emoji = emoji_map[pred_class]
            
            print(f"Kết quả phân tích: {label_text} (Độ tin cậy: {conf*100:.2f}%)")
            
        except KeyboardInterrupt:
            print("\nTạm biệt!")
            break
        except Exception as e:
            print(f"Đã xảy ra lỗi: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict sentiment of Vietnamese student feedbacks.")
    parser.add_argument("--text", type=str, default=None, help="The input text to classify")
    parser.add_argument("--model", type=str, default="svm", choices=["nb", "svm", "lstm"],
                        help="Model to use for prediction: nb (Naive Bayes), svm (SVM), lstm (Bi-LSTM)")
    args = parser.parse_args()
    
    if args.text is not None:
        pred_class, conf = predict_sentiment(args.text, args.model)
        if isinstance(pred_class, str):
            print(pred_class)
        else:
            label_text = labels_map[pred_class]
            emoji = emoji_map[pred_class]
            print(f"Câu: \"{args.text}\"")
            print(f"Kết quả: {label_text} (Độ tin cậy: {conf*100:.2f}%)")
    else:
        interactive_loop(args.model)
