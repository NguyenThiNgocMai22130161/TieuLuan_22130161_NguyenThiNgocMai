import os
import re
import unicodedata
import pandas as pd
from underthesea import word_tokenize

# Robust path handling relative to this script's directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FN_DIR = os.path.join(PROJECT_ROOT, "fn")

TEENCODE_PATH = os.path.join(DATA_DIR, "teencode.txt")

def load_teencode_dict(file_path=TEENCODE_PATH):
    """
    Loads teencode and abbreviation dictionary mapping from a tab-separated text file.
    """
    teencode_dict = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    teencode_dict[parts[0].strip()] = parts[1].strip()
        print(f"[Preprocessing] Loaded teencode dictionary with {len(teencode_dict)} mappings.")
    except FileNotFoundError:
        print(f"[Preprocessing] Warning: Teencode dictionary file not found at {file_path}")
    return teencode_dict

# Load dictionary once upon module import
TEENCODE_DICT = load_teencode_dict()

def normalize_teencode(text, teencode_dict=TEENCODE_DICT):
    """
    Splits text by space and maps each token using the teencode dictionary.
    """
    words = str(text).split()
    normalized_words = [teencode_dict.get(word, word) for word in words]
    return " ".join(normalized_words)

def clean_vietnamese_text(text):
    """
    Performs standard preprocessing for Vietnamese student feedbacks:
    1. Fill empty comments (NaN)
    2. Lowercase and Unicode NFC normalization
    3. Remove character repetition (> 2 times, e.g. 'đẹpppp' -> 'đẹpp')
    4. Translate teencode / abbreviations
    5. Strip special characters, keeping letters, digits, and spaces
    6. Compact consecutive white spaces
    7. Word segmentation using Underthesea library
    """
    if pd.isna(text):
        return ""
    
    # 2. Lowercase & Normalize Unicode NFC
    text = unicodedata.normalize("NFC", str(text)).lower()
    
    # 3. Remove consecutive repeated characters (e.g. dạaaaaa -> dạa)
    text = re.sub(r'([a-zà-ỹđ])\1{2,}', r'\1\1', text)
    
    # 4. Standardize teencode and slang words
    text = normalize_teencode(text)
    
    # 5. Remove punctuation and special symbols
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # 6. Normalize whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 7. Word Segmentation (word_tokenize with format='text' connects words with underscores)
    if text:
        text_segmented = word_tokenize(text, format="text")
    else:
        text_segmented = ""
        
    return text_segmented

def preprocess_files():
    """
    Reads datasets from code/data/, cleans comment column, and saves to code/fn/.
    """
    os.makedirs(FN_DIR, exist_ok=True)
    datasets = ['train.csv', 'validation.csv', 'test.csv']
    
    print("=" * 60)
    print("Starting Preprocessing Pipeline for Vietnamese Student Feedbacks...")
    print("=" * 60)
    
    for file_name in datasets:
        input_path = os.path.join(DATA_DIR, file_name)
        output_path = os.path.join(FN_DIR, f"cleaned_{file_name}")
        
        print(f"⌛ Processing file: {file_name}...")
        if not os.path.exists(input_path):
            print(f"Error: File not found at {input_path}")
            continue
            
        try:
            df = pd.read_csv(input_path)
            
            # Count words before
            raw_len = len(df)
            
            # Clean and segment
            df['comment_cleaned'] = df['comment'].apply(clean_vietnamese_text)
            
            # Fill empty comments with empty string
            df['comment_cleaned'] = df['comment_cleaned'].fillna('')
            
            # Save preprocessed dataset
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"Success! Cleaned dataset saved to {output_path} ({raw_len} samples)")
            
        except Exception as e:
            print(f"Error occurred while processing {file_name}: {e}")
            
    print("=" * 60)
    print("Preprocessing completed!")
    print("=" * 60)

if __name__ == "__main__":
    preprocess_files()
