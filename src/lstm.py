import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

# Seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Paths relative to this script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
FN_DIR = os.path.join(PROJECT_ROOT, "fn")
PKL_DIR = os.path.join(PROJECT_ROOT, "pkl")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM Classifier for Sentiment Analysis.
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_classes=3):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.spatial_dropout = nn.Dropout2d(0.2)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.lstm_dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        
        # Spatial Dropout 1D (requires [batch_size, channels, length, 1])
        embedded = embedded.permute(0, 2, 1).unsqueeze(3)
        embedded = self.spatial_dropout(embedded)
        embedded = embedded.squeeze(3).permute(0, 2, 1)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)
        
        # Global Average Pooling
        out = torch.mean(lstm_out, dim=1)
        out = self.lstm_dropout(out)
        
        # FC Layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

def text_to_sequence(text, vocab, max_len=100):
    """
    Converts a space-separated text string to a sequence of vocabulary indices.
    Pads or truncates to max_len.
    """
    tokens = str(text).split()
    seq = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    if len(seq) < max_len:
        seq = seq + [vocab['<PAD>']] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq

def train_lstm():
    """
    Trains Bi-LSTM with Class Weights and Early Stopping, saves models, and evaluates.
    """
    os.makedirs(PKL_DIR, exist_ok=True)
    
    train_path = os.path.join(FN_DIR, "cleaned_train.csv")
    val_path = os.path.join(FN_DIR, "cleaned_validation.csv")
    test_path = os.path.join(FN_DIR, "cleaned_test.csv")
    
    print("=" * 60)
    print("Starting Bi-LSTM Training Pipeline (PyTorch)...")
    print("=" * 60)
    print(f"Device: {device}")
    
    if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        print("❌ Error: Cleaned data files not found. Please run preprocessing first!")
        return

    # 1. Load preprocessed datasets
    train_df = pd.read_csv(train_path).fillna('')
    val_df = pd.read_csv(val_path).fillna('')
    test_df = pd.read_csv(test_path).fillna('')
    
    X_train_raw = train_df['comment_cleaned'].values
    y_train_raw = train_df['label'].values
    X_val_raw = val_df['comment_cleaned'].values
    y_val_raw = val_df['label'].values
    X_test_raw = test_df['comment_cleaned'].values
    y_test_raw = test_df['label'].values
    
    # 2. Build and save vocabulary
    print("⌛ Building vocabulary from training set...")
    words = []
    for text in X_train_raw:
        words.extend(text.split())
    word_counts = Counter(words)
    
    # Filter noise: keep words appearing at least twice
    filtered_words = [word for word, count in word_counts.items() if count >= 2]
    vocab = {word: idx + 2 for idx, word in enumerate(filtered_words)}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    
    vocab_size = len(vocab)
    print(f"✅ Vocabulary size: {vocab_size}")
    
    vocab_path = os.path.join(PKL_DIR, "lstm_vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)
    print(f"✅ Saved vocabulary to {vocab_path}")
    
    # 3. Create sequence matrices
    X_train_seq = np.array([text_to_sequence(t, vocab) for t in X_train_raw])
    X_val_seq = np.array([text_to_sequence(t, vocab) for t in X_val_raw])
    X_test_seq = np.array([text_to_sequence(t, vocab) for t in X_test_raw])
    
    # Create DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train_seq, dtype=torch.long), torch.tensor(y_train_raw, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val_seq, dtype=torch.long), torch.tensor(y_val_raw, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test_seq, dtype=torch.long), torch.tensor(y_test_raw, dtype=torch.long))
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 4. Compute Class Weights to tackle imbalance
    print("⌛ Computing class weights...")
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_raw), y=y_train_raw)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"✅ Class weights: {class_weights} (Neutral penalized more)")
    
    # 5. Initialize Model, Loss and Optimizer
    model = BiLSTMClassifier(vocab_size=vocab_size).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 6. Training loop with Early Stopping
    epochs = 10
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    model_path = os.path.join(PKL_DIR, "best_lstm_model_correct.pth")
    
    print("⌛ Training model...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()
            
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc*100:.2f}% | Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc*100:.2f}%")
        
        # Early stopping logic
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_path)
            print(f"  --> Saved new best model to {model_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  --> Early stopping triggered at epoch {epoch+1}")
                break
                
    # 7. Evaluation on Test set
    print("\n⌛ Evaluating on Test Set...")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())
            
    target_labels = ['Tiêu cực (0)', 'Trung tính (1)', 'Tích cực (2)']
    print("\n" + "=" * 40)
    print("RESULTS FOR BI-LSTM MODEL")
    print("=" * 40)
    print(f"Test Accuracy: {accuracy_score(all_labels, all_preds) * 100:.2f}%")
    print(classification_report(all_labels, all_preds, target_names=target_labels, digits=4))
    print("=" * 60)
    print("LSTM Training and Evaluation completed!")
    print("=" * 60)

if __name__ == "__main__":
    train_lstm()
