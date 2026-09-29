# train_chat_lifecycle.py
#   python train_chat_lifecycle.py ablation      -> chat_model_b_prime  (drift data, generation loss ONLY)
#   python train_chat_lifecycle.py contrastive   -> chat_model_b       (generation + lambda * SupCon)
# Identical data/seed/hparams; only the contrastive term differs. Saves per-epoch encoder
# embeddings (epoch 0 = Model A) of the reference set + candidate pool, and per-step loss logs.
# train_chat_lifecycle.py
#   python train_chat_lifecycle.py   -> chat_model_b (generation + lambda * SupCon)
# Saves per-epoch encoder embeddings (epoch 0 = Model A) of the reference set + candidate pool,
# and per-step loss logs.
import os, json
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from datasets import Dataset
from transformers import (T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments,
                          Seq2SeqTrainer, TrainerCallback, set_seed)

MODEL_A_DIR = "models/chat_model_a"
OUTPUT_DIR, TAG, CW = "models/chat_model_b", "b", 0.3
MAX_IN, MAX_OUT = 64, 24
SNAP = "snapshots"; os.makedirs(SNAP, exist_ok=True)
set_seed(42)

train_df = pd.read_csv("data_chat/chat_full_train.csv")
test_df = pd.read_csv("data_chat/chat_test_set.csv")
pool_df = pd.read_csv("data_chat/candidate_pool.csv")
tokenizer = T5Tokenizer.from_pretrained(MODEL_A_DIR)


def enc(texts):
    e = tokenizer(["respond: " + t for t in texts], truncation=True, max_length=MAX_IN,
                  padding="max_length", return_tensors="pt")
    return e["input_ids"], e["attention_mask"]


REF, POOL = enc(test_df.review.tolist()), enc(pool_df.text.tolist())

train_df["input_text"] = "respond: " + train_df["review"]
ds = Dataset.from_pandas(train_df[["input_text", "reply", "label"]])


def tok_fn(b):
    mi = tokenizer(b["input_text"], truncation=True, max_length=MAX_IN, padding="max_length")
    lab = tokenizer(b["reply"], truncation=True, max_length=MAX_OUT, padding="max_length")
    mi["labels"] = [[t if t != tokenizer.pad_token_id else -100 for t in s] for s in lab["input_ids"]]
    return mi


ds = ds.map(tok_fn, batched=True, remove_columns=["input_text", "reply"])
ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "label"])
model = T5ForConditionalGeneration.from_pretrained(MODEL_A_DIR)


def collate_fn(fs):
    return {"input_ids": torch.stack([f["input_ids"] for f in fs]),
            "attention_mask": torch.stack([f["attention_mask"] for f in fs]),
            "labels": torch.stack([f["labels"] for f in fs]),
            "sentiment_label": torch.stack([f["label"] for f in fs])}


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__(); self.t = temperature

    def forward(self, emb, labels):
        dev = emb.device
        emb = F.normalize(emb, dim=1)
        sim = emb @ emb.T / self.t
        labels = labels.view(-1, 1)
        eye = torch.eye(labels.shape[0], device=dev)
        pos = torch.eq(labels, labels.T).float().to(dev) - eye
        sim = sim - sim.max(dim=1, keepdim=True)[0].detach()
        exp = torch.exp(sim) * (1 - eye)
        logp = sim - torch.log(exp.sum(1, keepdim=True) + 1e-12)
        return (-(pos * logp).sum(1) / pos.sum(1).clamp(min=1)).mean()


def mean_pool(h, m):
    m = m.unsqueeze(-1).float()
    return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)


@torch.no_grad()
def embed(model, ids, mask, bs=128):
    was = model.training; model.eval(); out = []
    for i in range(0, len(ids), bs):
        i_, m_ = ids[i:i + bs].to(model.device), mask[i:i + bs].to(model.device)
        out.append(mean_pool(model.encoder(input_ids=i_, attention_mask=m_).last_hidden_state, m_).cpu())
    model.train(was)
    return torch.cat(out).numpy()


def snapshot(m, ep):
    np.savez(f"{SNAP}/{TAG}_epoch{ep}.npz", ref=embed(m, *REF), pool=embed(m, *POOL))


class Snap(TrainerCallback):
    def on_epoch_end(self, args, state, control, model=None, **kw):
        snapshot(model, int(round(state.epoch)))


class CTrainer(Seq2SeqTrainer):
    def __init__(self, *a, cw=0.3, **k):
        super().__init__(*a, **k)
        self.sup, self.cw = SupConLoss(0.1), cw
        self.hist = {"step": [], "gen": [], "con": []}

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        y = inputs.pop("sentiment_label")
        out = model(**inputs)
        gen = out.loss
        pooled = mean_pool(out.encoder_last_hidden_state, inputs["attention_mask"])
        con = self.sup(pooled, y)
        total = gen + self.cw * con
        if model.training:
            self.hist["step"].append(len(self.hist["step"]) + 1)
            self.hist["gen"].append(float(gen.item())); self.hist["con"].append(float(con.item()))
        return (total, out) if return_outputs else total


args = Seq2SeqTrainingArguments(
    output_dir=f"{OUTPUT_DIR}_checkpoints", num_train_epochs=3, per_device_train_batch_size=16,
    learning_rate=1.5e-4, logging_steps=25, save_strategy="no", report_to="none",
    remove_unused_columns=False, seed=42)
trainer = CTrainer(model=model, args=args, train_dataset=ds, data_collator=collate_fn,
                   callbacks=[Snap()], cw=CW)

snapshot(trainer.model, 0)  # epoch 0 = Model A
trainer.train()
json.dump(trainer.hist, open(f"{SNAP}/{TAG}_losses.json", "w"))
model.save_pretrained(OUTPUT_DIR); tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nSaved ./{OUTPUT_DIR}/ (mode={MODE}) + snapshots in ./{SNAP}/")
