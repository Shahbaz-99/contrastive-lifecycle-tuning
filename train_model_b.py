# train_model_b.py
# Run this in Colab AFTER train_model_a.py (needs ./model_a/ to exist)
#
# WHAT THIS DOES:
# 1. Loads Model A (already trained) as the starting point -- this is the
#    "lifecycle tuning" part: we're continuing an existing model's life,
#    not starting over.
# 2. Fine-tunes further on full_train.csv (standard + drift combined) using
#    a COMBINED loss:
#       total_loss = cross_entropy_loss + lambda * supervised_contrastive_loss
#    - cross_entropy_loss: normal "get the right label" loss
#    - supervised_contrastive_loss: pulls same-label sentence embeddings
#      together, pushes different-label embeddings apart, regardless of
#      whether the sentence is "standard" or "drift" style.
# 3. Saves as model_b/
# 4. Evaluates on the same test_set.csv, split by style, so you can compare
#    directly against Model A's numbers.

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

MODEL_A_DIR = "model_a"      # starting point (lifecycle continuation)
OUTPUT_DIR = "model_b"
CONTRASTIVE_WEIGHT = 0.5      # lambda: how much the contrastive loss matters
                               # vs the classification loss. 0.5 is a
                               # reasonable middle ground for a demo.

# ---- 1. Load data (standard + drift combined this time) ----
train_df = pd.read_csv("data/full_train.csv")
test_df = pd.read_csv("data/test_set.csv")

train_ds = Dataset.from_pandas(train_df[["text", "label"]])
test_ds = Dataset.from_pandas(test_df[["text", "label"]])

tokenizer = AutoTokenizer.from_pretrained(MODEL_A_DIR)

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)

train_ds = train_ds.map(tokenize_fn, batched=True)
test_ds = test_ds.map(tokenize_fn, batched=True)
train_ds = train_ds.rename_column("label", "labels")
test_ds = test_ds.rename_column("label", "labels")
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# ---- 2. Load Model A as the starting point ----
model = AutoModelForSequenceClassification.from_pretrained(MODEL_A_DIR, num_labels=2)


# ---- 3. Supervised Contrastive Loss ----
# For each sentence in a batch: treat all OTHER sentences in the batch with
# the SAME label as "positives" (pull close) and all with a DIFFERENT label
# as "negatives" (push apart). This uses the labels we already have, so no
# need for data augmentation tricks.
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        device = embeddings.device
        embeddings = F.normalize(embeddings, dim=1)
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        labels = labels.view(-1, 1)
        mask_same_label = torch.eq(labels, labels.T).float().to(device)
        mask_self = torch.eye(labels.shape[0], device=device)
        mask_positive = mask_same_label - mask_self  # exclude comparing to itself

        # numerical stability
        sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0].detach()
        exp_sim = torch.exp(sim_matrix) * (1 - mask_self)  # exclude self from denom
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        num_positives = mask_positive.sum(dim=1)
        num_positives = torch.clamp(num_positives, min=1)  # avoid divide-by-zero
        loss_per_sample = -(mask_positive * log_prob).sum(dim=1) / num_positives
        return loss_per_sample.mean()


# ---- 4. Custom Trainer that combines CrossEntropy + SupCon ----
class ContrastiveTrainer(Trainer):
    def __init__(self, *args, contrastive_weight=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.contrastive_loss_fn = SupConLoss(temperature=0.1)
        self.contrastive_weight = contrastive_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs, output_hidden_states=True)

        ce_loss = outputs.loss  # cross-entropy, computed automatically since labels were passed

        # Use the [CLS] token's last hidden state as the "embedding" for
        # contrastive loss -- it's DistilBERT's summary of the whole sentence.
        last_hidden = outputs.hidden_states[-1]
        cls_embeddings = last_hidden[:, 0, :]  # (batch_size, hidden_dim)

        contrastive_loss = self.contrastive_loss_fn(cls_embeddings, labels)

        total_loss = ce_loss + self.contrastive_weight * contrastive_loss

        return (total_loss, outputs) if return_outputs else total_loss


# ---- 5. Train ----
training_args = TrainingArguments(
    output_dir="model_b_checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=32,   # larger batch = more pairs for contrastive loss
    per_device_eval_batch_size=32,
    learning_rate=1e-5,                # slightly lower since we're continuing training
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

trainer = ContrastiveTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
    contrastive_weight=CONTRASTIVE_WEIGHT,
)

trainer.train()

# ---- 6. Save Model B ----
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nModel B saved to ./{OUTPUT_DIR}/")

# ---- 7. Evaluate split by style, same as Model A, for direct comparison ----
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

print("\n=== MODEL B RESULTS (lifecycle-tuned with contrastive loss) ===")
for style, r in results.items():
    print(f"  {style:8s}: n={r['n']:3d}  accuracy={r['accuracy']:.3f}  f1={r['f1']:.3f}")

gap = results["standard"]["accuracy"] - results["drift"]["accuracy"]
print(f"\nDrift gap (standard_acc - drift_acc): {gap:.3f}")
print("Compare this gap to Model A's 0.410 gap -- a smaller gap here means")
print("the contrastive lifecycle tuning reduced the drift vulnerability.")
