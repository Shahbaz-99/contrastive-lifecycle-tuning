# Contrastive Lifecycle Tuning: Fixing Systemic Drift in Sentiment Models

A research demo showing how **supervised contrastive loss (SupCon) + lifecycle fine-tuning**
reduces a model's vulnerability to *systemic drift*: the gap between how it performs on the
text style it was trained on and on new styles (slang, emoji, sarcasm).

Two model families are compared side by side, and every comparison is **Model A (baseline)
vs Model B (lifecycle-tuned)**:

| | Model A (Baseline) | Model B (Lifecycle-Tuned) |
|---|---|---|
| Training data | Standard/formal text only | Standard + drift-style text |
| Starting point | Pretrained base model | Continued from Model A |
| Loss | Task loss only | Task loss + λ · Supervised Contrastive loss |

**Two systems:**
- **Classifier:** DistilBERT (`distilbert-base-uncased`), cross-entropy + SupCon.
- **Chatbot:** T5 (`T5ForConditionalGeneration`) that replies to a customer message; SupCon is applied on the mean-pooled encoder output (`L = L_gen + 0.3 · L_SupCon`, τ = 0.1).

## The Problem: Systemic Drift

Models trained once on a fixed distribution learn decision boundaries tied to that style.
When the same sentiment arrives in a new style, its embedding lands away from its standard
counterpart, near the middle of the space, and the model becomes unreliable. In this project
that shows up as a large gap between standard-text and drift-text performance for Model A.

## The Fix: Lifecycle Tuning via Contrastive Loss

Model B continues from Model A's weights on standard + drift data. SupCon treats texts with the
**same sentiment label as positives** (pulled together) and **opposite labels as negatives**
(pushed apart), regardless of surface style:

```
L_i = -(1/|P(i)|) Σ_{p∈P(i)} log [ exp(z_i·z_p / τ) / Σ_{a≠i} exp(z_i·z_a / τ) ]
total_loss = task_loss + λ · L_SupCon
```

## Results (reference set, 200 texts, single seed)

| Metric | Model A | Model B |
|---|---|---|
| Classifier: standard accuracy | 100% | 100% |
| Classifier: drift accuracy | 59% | 100% |
| Chatbot: tone accuracy, standard | 100% | 100% |
| Chatbot: tone accuracy, drift | 66% | 100% |
| Chatbot: silhouette (cosine) | 0.197 | 0.903 |
| Chatbot: kNN (fit standard → test drift) | 0.74 | 1.00 |
| Chatbot: cross-style centroid alignment | 0.91 | 0.986 |

These are in-distribution numbers: the reference set shares templates with the drift training
data. See **Limitations**.

## The Demo App (3 tabs)

1. **Classifier:** one sentence, both classifiers, label + confidence side by side.
2. **Chatbot:** one message, both chatbots reply in parallel.
3. **Embedding Space** (the research view):
   1. *How systematic drift appears:* Model A's embedding space projected onto a sentiment axis (standard = circles, drift = diamonds), arrows from standard to drift centroid, and gauges showing the % of texts on the wrong or unclear side.
   2. *Same input, two models:* 20 curated novel queries (or your own): replies, sentiment score, and position in both models' space, on one shared projection frame.
   3. *Contrastive lifecycle:* an epoch slider (0 = Model A → 3 = Model B) showing the anchor, its positives (pull) and negatives (push) moving across training.
   4. *The math:* the SupCon formula plus an interactive toy on a unit circle with a temperature (τ) slider and step buttons that show the actual gradient pulling positives in and pushing negatives out.

**Projection:** linear (x = sentiment axis μ₊ − μ₋, y = top orthogonal PC), fit once on Model A using standard reference points only, and reused for Model B and every epoch, so all panels share one frame.

## Project Structure

```
.
├── generate_dataset.py        # Classifier dataset
├── generate_chat_dataset.py   # Chatbot dataset (standard, drift, test)
├── gen_candidates.py          # Novel candidate pool (unseen slang/emoji/sarcasm), deduped vs training
├── train_model_a.py           # Classifier Model A
├── train_model_b.py           # Classifier Model B (CE + SupCon)
├── train_chat_model_a.py      # Chatbot Model A
├── train_chat_lifecycle.py    # Chatbot Model B (gen + λ·SupCon); saves per-epoch embedding snapshots
├── train_chat_model_b.py      # Earlier chatbot Model B script, superseded by train_chat_lifecycle.py
├── geometry.py                # Shared basis, projection, score, tone check
├── analysis.py                # Runs A and B, computes metrics, curates queries -> demo_data/bundle.json
├── embedding_tab.py           # "Embedding Space" tab
├── app.py                     # Gradio UI
├── requirements.txt
├── data/, data_chat/          # Generated CSVs
├── models/                    # model_a, model_b, chat_model_a, chat_model_b (not committed)
├── snapshots/                 # Per-epoch embeddings + loss logs (epoch 0 = Model A)
└── demo_data/                 # bundle.json + basis files produced by analysis.py
```

> Model weights are large and excluded via `.gitignore`. Regenerate them with the scripts below.

## Setup & Run Order

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt   # includes plotly, scikit-learn

python generate_dataset.py
python generate_chat_dataset.py
python gen_candidates.py          # BEFORE training: snapshots embed this pool

python train_model_a.py
python train_model_b.py
python train_chat_model_a.py
python train_chat_lifecycle.py    # -> models/chat_model_b + snapshots/

python analysis.py                # -> demo_data/bundle.json
python app.py
```

Sanity checks printed by `analysis.py`: the epoch-0 snapshot should match Model A
(max diff ≈ 0), and the pos–neg cosine gap should widen across epochs.

## Tech Stack
Hugging Face `transformers` (DistilBERT, T5), PyTorch, Gradio, Plotly, scikit-learn, pandas.

## Limitations
- **Synthetic, template-based data.** The reference set shares templates with the drift training data, so the 100% drift results are in-distribution and overstate real-world generalization.
- **Novel slang is not fixed.** On the held-out candidate pool, Model B improves on held-out sarcasm, but error on novel slang is not lower than Model A's. The pool-level error rates are shown in the app.
- **No ablation.** Model B changes two things at once (sees drift data + contrastive loss). A drift-data-only control was dropped from this demo, so the results do not separate the two effects.
- **Single seed (42).** Small gaps could be noise.
- **Keyword-based tone check.** Chatbot tone accuracy is judged by keyword matching on replies, which is noisy.
- **2D projections lose information.** Read them together with the numbers.
- **Curated queries.** The 20 showcase queries are selected (mostly "A wrong → B right"); the app discloses the selection and the full-pool error rates.

## References
Khosla et al. 2020 (SupCon) · Wang & Isola 2020 (alignment/uniformity) · Gao et al. 2021 (SimCSE) · Cha et al. 2021 (Co2L).

## License
MIT