# train_model_a.py
# Run this in Colab (GPU runtime: Runtime > Change runtime type > GPU)
#
# WHAT THIS DOES:
# 1. Loads distilbert-base-uncased (no sentiment knowledge yet)
# 2. Fine-tunes it on standard_train.csv ONLY (clean/formal sentences)
# 3. Saves the result as model_a/
# 4. Tests it on test_set.csv, reporting accuracy separately for
#    "standard" style rows vs "drift" style rows.
#    -> This gap IS the systemic drift problem we're demonstrating.

# ---- 0. Install deps (Colab usually has these, but just in case) ----
# !pip install -q transformers datasets scikit-learn accelerate

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "model_a"

# ---- 1. Load data ----
train_df = pd.read_csv("data/standard_train.csv")   # ONLY standard style
test_df = pd.read_csv("data/test_set.csv")           # mixed standard + drift

train_ds = Dataset.from_pandas(train_df[["text", "label"]])
test_ds = Dataset.from_pandas(test_df[["text", "label"]])

# ---- 2. Tokenize ----
# Tokenizing = converting text into numeric IDs the model can read.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)

train_ds = train_ds.map(tokenize_fn, batched=True)
test_ds = test_ds.map(tokenize_fn, batched=True)

train_ds = train_ds.rename_column("label", "labels")
test_ds = test_ds.rename_column("label", "labels")
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# ---- 3. Load base model + classification head ----
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# ---- 4. Train (standard cross-entropy fine-tuning, no contrastive loss) ----
training_args = TrainingArguments(
    output_dir="model_a_checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    logging_steps=25,
    save_strategy="no",
    report_to="none",
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)

trainer.train()

# ---- 5. Save Model A ----
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nModel A saved to ./{OUTPUT_DIR}/")

# ---- 6. Evaluate split by style: standard vs drift ----
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def predict_batch(texts):
    enc = tokenizer(texts, truncation=True, padding=True, max_length=64, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    return torch.argmax(logits, dim=1).cpu().numpy()

results = {}
for style in ["standard", "drift"]:
    subset = test_df[test_df["style"] == style]
    preds = predict_batch(list(subset["text"]))
    acc = accuracy_score(subset["label"], preds)
    f1 = f1_score(subset["label"], preds)
    results[style] = {"accuracy": acc, "f1": f1, "n": len(subset)}

print("\n=== MODEL A RESULTS (baseline, standard-only training) ===")
for style, r in results.items():
    print(f"  {style:8s}: n={r['n']:3d}  accuracy={r['accuracy']:.3f}  f1={r['f1']:.3f}")

gap = results["standard"]["accuracy"] - results["drift"]["accuracy"]
print(f"\nDrift gap (standard_acc - drift_acc): {gap:.3f}")
print("This gap is the 'systemic drift vulnerability' we'll fix in Model B.")
