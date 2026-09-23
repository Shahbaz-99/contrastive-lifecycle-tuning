# train_chat_model_a.py
# Run:
#   python train_chat_model_a.py
#
# WHAT THIS DOES:
# Trains a small T5 model to act like a single-turn chatbot: read ONE
# message (a review/comment), generate a natural reply -- no separate
# question field, no rigid template. Trained ONLY on standard-style
# messages, so it should struggle on slang/sarcasm/emoji messages.

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

MODEL_NAME = "t5-small"
OUTPUT_DIR = "chat_model_a"
MAX_INPUT_LEN = 64
MAX_TARGET_LEN = 24

train_df = pd.read_csv("data_chat/chat_standard_train.csv")
test_df = pd.read_csv("data_chat/chat_test_set.csv")

train_df["input_text"] = "respond: " + train_df["review"]
test_df["input_text"] = "respond: " + test_df["review"]

train_ds = Dataset.from_pandas(train_df[["input_text", "reply"]])
test_ds = Dataset.from_pandas(test_df[["input_text", "reply"]])

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

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

model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

training_args = Seq2SeqTrainingArguments(
    output_dir="chat_model_a_checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=3e-4,
    logging_steps=25,
    save_strategy="no",
    predict_with_generate=False,
    report_to="none",
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nChat Model A saved to ./{OUTPUT_DIR}/")

# ---- Evaluate: does the reply's TONE match the true sentiment? ----
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
    return None  # ambiguous / neither -- counts as a miss

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

print("\n=== CHAT MODEL A RESULTS ===")
for style, r in results.items():
    print(f"  {style:8s}: n={r['n']:3d}  correct_tone_rate={r['accuracy']:.3f}  hallucination_rate={r['hallucination_rate']:.3f}")

gap = results["drift"]["hallucination_rate"] - results["standard"]["hallucination_rate"]
print(f"\nHallucination gap (drift - standard): {gap:.3f}")

print("\n--- Sample drift-style replies ---")
sample = test_df[test_df["style"] == "drift"].head(6)
for _, row in sample.iterrows():
    gen = generate_reply(row["review"])
    pred = sentiment_of_reply(gen)
    match = "✓ correct tone" if pred == row["label"] else "✗ HALLUCINATED / WRONG TONE"
    print(f"User: {row['review']}")
    print(f"Bot:  {gen}   [{match}]\n")
