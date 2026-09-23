# train_chat_model_b.py
# Run AFTER train_chat_model_a.py (needs ./chat_model_a/ to exist):
#   python train_chat_model_b.py
#
# Same idea as train_qa_model_b.py: continue from Model A, train on
# standard + drift combined, using generation_loss + contrastive_loss
# on the encoder's pooled representation.

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

MODEL_A_DIR = "chat_model_a"
OUTPUT_DIR = "chat_model_b"
MAX_INPUT_LEN = 64
MAX_TARGET_LEN = 24
CONTRASTIVE_WEIGHT = 0.3

train_df = pd.read_csv("data_chat/chat_full_train.csv")
test_df = pd.read_csv("data_chat/chat_test_set.csv")

train_df["input_text"] = "respond: " + train_df["review"]
test_df["input_text"] = "respond: " + test_df["review"]

train_ds = Dataset.from_pandas(train_df[["input_text", "reply", "label"]])
test_ds = Dataset.from_pandas(test_df[["input_text", "reply"]])

tokenizer = T5Tokenizer.from_pretrained(MODEL_A_DIR)

def tokenize_fn(batch):
    model_inputs = tokenizer(batch["input_text"], truncation=True, max_length=MAX_INPUT_LEN, padding="max_length")
    labels = tokenizer(batch["reply"], truncation=True, max_length=MAX_TARGET_LEN, padding="max_length")
    label_ids = [
        [(tok if tok != tokenizer.pad_token_id else -100) for tok in seq]
        for seq in labels["input_ids"]
    ]
    model_inputs["labels"] = label_ids
    return model_inputs

train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["input_text", "reply"])
test_ds = test_ds.map(tokenize_fn, batched=True, remove_columns=["input_text", "reply"])

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "label"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

model = T5ForConditionalGeneration.from_pretrained(MODEL_A_DIR)


def collate_fn(features):
    batch = {
        "input_ids": torch.stack([f["input_ids"] for f in features]),
        "attention_mask": torch.stack([f["attention_mask"] for f in features]),
        "labels": torch.stack([f["labels"] for f in features]),
    }
    if "label" in features[0]:
        batch["sentiment_label"] = torch.stack([f["label"] for f in features])
    return batch


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
        mask_positive = mask_same_label - mask_self
        sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0].detach()
        exp_sim = torch.exp(sim_matrix) * (1 - mask_self)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
        num_positives = torch.clamp(mask_positive.sum(dim=1), min=1)
        loss_per_sample = -(mask_positive * log_prob).sum(dim=1) / num_positives
        return loss_per_sample.mean()


def mean_pool(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class ContrastiveSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, contrastive_weight=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.contrastive_loss_fn = SupConLoss(temperature=0.1)
        self.contrastive_weight = contrastive_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        sentiment_label = inputs.pop("sentiment_label", None)
        outputs = model(**inputs)
        gen_loss = outputs.loss
        if sentiment_label is not None:
            encoder_hidden = outputs.encoder_last_hidden_state
            pooled = mean_pool(encoder_hidden, inputs["attention_mask"])
            contrastive_loss = self.contrastive_loss_fn(pooled, sentiment_label)
            total_loss = gen_loss + self.contrastive_weight * contrastive_loss
        else:
            total_loss = gen_loss
        return (total_loss, outputs) if return_outputs else total_loss


training_args = Seq2SeqTrainingArguments(
    output_dir="chat_model_b_checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=1.5e-4,
    logging_steps=25,
    save_strategy="no",
    predict_with_generate=False,
    report_to="none",
    remove_unused_columns=False,
)

trainer = ContrastiveSeq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    data_collator=collate_fn,
    contrastive_weight=CONTRASTIVE_WEIGHT,
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nChat Model B saved to ./{OUTPUT_DIR}/")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

POSITIVE_WORDS = ["glad", "great", "awesome", "love", "wonderful", "impressed", "happy", "thanks for sharing"]
NEGATIVE_WORDS = ["sorry", "frustrat", "apolog", "disappoint", "trouble", "didn't work", "let you down"]

def generate_reply(review_text):
    input_text = "respond: " + review_text
    enc = tokenizer(input_text, truncation=True, max_length=MAX_INPUT_LEN, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(**enc, max_length=MAX_TARGET_LEN, num_beams=2)
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)

def sentiment_of_reply(text):
    t = text.lower()
    has_pos = any(w in t for w in POSITIVE_WORDS)
    has_neg = any(w in t for w in NEGATIVE_WORDS)
    if has_neg and not has_pos:
        return 0
    if has_pos and not has_neg:
        return 1
    return None

results = {}
for style in ["standard", "drift"]:
    subset = test_df[test_df["style"] == style]
    correct = 0
    for _, row in subset.iterrows():
        gen = generate_reply(row["review"])
        pred = sentiment_of_reply(gen)
        if pred == row["label"]:
            correct += 1
    acc = correct / len(subset)
    results[style] = {"accuracy": acc, "hallucination_rate": 1 - acc, "n": len(subset)}

print("\n=== CHAT MODEL B RESULTS ===")
for style, r in results.items():
    print(f"  {style:8s}: n={r['n']:3d}  correct_tone_rate={r['accuracy']:.3f}  hallucination_rate={r['hallucination_rate']:.3f}")

gap = results["drift"]["hallucination_rate"] - results["standard"]["hallucination_rate"]
print(f"\nHallucination gap (drift - standard): {gap:.3f}")
print("Compare this to Chat Model A's gap.")

print("\n--- Sample drift-style replies (same as Model A's demo) ---")
sample = test_df[test_df["style"] == "drift"].head(6)
for _, row in sample.iterrows():
    gen = generate_reply(row["review"])
    pred = sentiment_of_reply(gen)
    match = "✓ correct tone" if pred == row["label"] else "✗ HALLUCINATED / WRONG TONE"
    print(f"User: {row['review']}")
    print(f"Bot:  {gen}   [{match}]\n")
