import os
import argparse
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score

# Paths relative to this script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
FN_DIR = os.path.join(PROJECT_ROOT, "fn")
PKL_DIR = os.path.join(PROJECT_ROOT, "pkl")

def compute_metrics(eval_pred):
    """
    Offline metric computation using scikit-learn.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")
    return {"accuracy": acc, "f1": f1}

def train_phobert(method="lora", epochs=5, batch_size=16, lr=2e-5):
    """
    Fine-tunes vinai/phobert-base on UIT-VSFC.
    """
    train_path = os.path.join(FN_DIR, "cleaned_train.csv")
    val_path = os.path.join(FN_DIR, "cleaned_validation.csv")
    test_path = os.path.join(FN_DIR, "cleaned_test.csv")
    
    print("=" * 60)
    print(f"Starting PhoBERT Training Pipeline ({method.upper()})...")
    print("=" * 60)
    
    if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        print("❌ Error: Cleaned data files not found. Please run preprocessing first!")
        return

    # 1. Load data
    print("⌛ Loading cleaned datasets...")
    df_train = pd.read_csv(train_path).fillna('')
    df_val = pd.read_csv(val_path).fillna('')
    df_test = pd.read_csv(test_path).fillna('')
    
    dataset = DatasetDict({
        "train": Dataset.from_pandas(df_train[['comment_cleaned', 'label']]),
        "validation": Dataset.from_pandas(df_val[['comment_cleaned', 'label']]),
        "test": Dataset.from_pandas(df_test[['comment_cleaned', 'label']])
    })
    
    # 2. Tokenization
    print("⌛ Loading PhoBERT-base Tokenizer and encoding texts...")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    
    def tokenize_function(examples):
        return tokenizer(examples["comment_cleaned"], padding="max_length", truncation=True, max_length=256)
        
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # 3. Load Model
    print("⌛ Loading pre-trained PhoBERT-base model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "vinai/phobert-base",
        num_labels=3,
        id2label={0: "Tiêu cực", 1: "Trung tính", 2: "Tích cực"},
        label2id={"Tiêu cực": 0, "Trung tính": 1, "Tích cực": 2}
    )
    
    output_dir = os.path.join(PROJECT_ROOT, f"phobert_{method}_checkpoints")
    final_model_dir = os.path.join(PROJECT_ROOT, f"phobert_{method}_final")
    
    # 4. Apply LoRA if requested
    if method == "lora":
        print("⌛ Applying LoRA (PEFT) configurations...")
        try:
            from peft import get_peft_model, LoraConfig, TaskType
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["query", "value"]
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            
            # Default LR for LoRA is larger
            if lr == 2e-5:
                lr = 2e-4
        except ImportError:
            print("❌ Error: 'peft' library is not installed. Defaulting to Full Fine-Tuning.")
            method = "full"
            
    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch_cuda_available(), # Enable mixed precision if GPU is present
        report_to="none"
    )
    
    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    # 7. Train and Save
    print(f"⌛ Starting training for {epochs} epochs with LR={lr}...")
    trainer.train()
    
    print(f"✅ Training completed! Saving best model to {final_model_dir}...")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    # 8. Evaluate on test set
    print("⌛ Evaluating best model on Test Set...")
    eval_results = trainer.evaluate(tokenized_datasets["test"])
    print("\n" + "=" * 40)
    print("EVALUATION RESULTS ON TEST SET")
    print("=" * 40)
    print(f"Test Accuracy: {eval_results.get('eval_accuracy', 0) * 100:.2f}%")
    print(f"Test F1-Score (Macro): {eval_results.get('eval_f1', 0) * 100:.2f}%")
    print("=" * 60)

def torch_cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune PhoBERT model for sentiment analysis.")
    parser.add_argument("--method", type=str, default="lora", choices=["lora", "full"],
                        help="Fine-tuning method: lora (PEFT) or full (Full fine-tuning)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    args = parser.parse_args()
    
    # Adjust learning rate defaults based on method if not explicitly modified
    # Default LoRA learning rate is 2e-4
    lr = args.lr
    if args.method == "lora" and lr == 2e-5:
        lr = 2e-4
        
    train_phobert(method=args.method, epochs=args.epochs, batch_size=args.batch_size, lr=lr)
