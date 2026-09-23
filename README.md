# Contrastive Lifecycle Tuning — Fixing Systemic Drift in Sentiment Models

A demo project showing how **supervised contrastive loss + lifecycle fine-tuning**
can reduce a sentiment classifier's vulnerability to *systemic drift* — the
gap between how a model performs on the text style it was trained on vs.
new, unseen styles (slang, sarcasm, emojis).

Two models are trained and compared side-by-side in a live UI:

| | Model A (Baseline) | Model B (Lifecycle-Tuned) |
|---|---|---|
| Base model | `distilbert-base-uncased` | Continued from Model A |
| Training data | Standard/formal text only | Standard + drift-style text |
| Loss function | Cross-entropy only | Cross-entropy + Supervised Contrastive Loss |
| Standard-style test accuracy | 100% | 100% |
| Drift-style test accuracy | 59% | 100% |
| **Drift gap** | **0.41** | **0.00** |

## The Problem: Systemic Drift

Most classifiers are trained once on a static data distribution. When
real-world input shifts — new slang, sarcasm, typos, emoji usage — the
model's decision boundaries, fit tightly to the training distribution,
no longer generalize. This is easy to demonstrate but often invisible
until it causes production failures.

## The Fix: Lifecycle Tuning via Contrastive Loss

Instead of retraining from scratch, **Model B continues training from
Model A's weights** (a "lifecycle" step) on a wider dataset, using a
**Supervised Contrastive Loss** alongside the normal classification loss.

In simple terms: contrastive loss pulls sentence embeddings of the
*same label* closer together and pushes *different-label* embeddings
apart — regardless of surface style. This pushes the model to encode
meaning rather than surface word patterns, which is what closes the
drift gap.

```
total_loss = cross_entropy_loss + λ * supervised_contrastive_loss
```

## Project Structure

```
.
├── generate_dataset.py      # Builds the synthetic labeled dataset
├── train_model_a.py         # Trains baseline model on standard-style data only
├── train_model_b.py         # Continues training with contrastive loss on full data
├── app.py                   # Gradio UI — two tabs, Model A vs Model B, live comparison
├── requirements.txt
├── data/                    # Generated CSVs (train/test splits)
│   ├── standard_train.csv
│   ├── drift_train.csv
│   ├── full_train.csv
│   └── test_set.csv
├── model_a/                 # Trained baseline model (not committed — see below)
└── model_b/                 # Trained lifecycle-tuned model (not committed — see below)
```

> **Note on model weights:** `model_a/` and `model_b/` contain multi-hundred-MB
> model files and are excluded via `.gitignore`. Either regenerate them by
> running the training scripts, or use [Git LFS](https://git-lfs.com/) if you
> want to version them.

## Dataset

~2000 synthetically generated, labeled (positive/negative) sentences across
two styles:
- **Standard style** — formal/clean text (e.g. *"I found this laptop to be excellent."*)
- **Drift style** — slang, sarcasm, emojis (e.g. *"ngl this app is actually fire 🔥"*)

A held-out test set (mixed 50/50 across styles) is used to evaluate both
models identically.

## Setup & Usage

```bash
# 1. Clone and enter the repo
git clone https://github.com/<your-username>/contrastive-lifecycle-tuning.git
cd contrastive-lifecycle-tuning

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate the dataset
python generate_dataset.py

# 5. Train Model A (baseline)
python train_model_a.py

# 6. Train Model B (lifecycle-tuned, continues from Model A)
python train_model_b.py

# 7. Launch the comparison UI
python app.py
```

The app launches a local Gradio server (and optionally a public share link)
where you can enter any sentence and compare both models' predictions and
confidence scores side-by-side.

## Tech Stack
- **Model:** DistilBERT (`distilbert-base-uncased`) via Hugging Face `transformers`
- **Training:** Hugging Face `Trainer` API, custom loss combining cross-entropy + Supervised Contrastive Loss (SupCon)
- **UI:** Gradio
- **Data:** Synthetically generated with Python, `pandas`

## Possible Extensions
- Evaluate on a *third*, held-out drift style never seen in training, to test true generalization rather than in-distribution recall.
- Replace synthetic data with a real-world sentiment dataset for a stronger case study.
- Add embedding-space visualization (e.g. t-SNE) showing how contrastive tuning reshapes the representation space.

## License
MIT