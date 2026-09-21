# app.py
# Run in Colab AFTER model_a/ and model_b/ exist:
#   !pip install -q gradio
#   !python app.py
# Launches a public share link (share=True).
#
# DESIGN NOTES (what changed vs the first version, and why):
# - Custom Gradio theme (indigo accent, slate neutrals) instead of default
#   orange -- looks more "product" than "tutorial demo".
# - gr.Label component instead of plain textboxes for results: Gradio
#   renders it as clean horizontal confidence bars automatically, no
#   custom chart code needed.
# - Cards (gr.Group) with consistent spacing instead of loose stacked
#   components -- gives visual structure without extra libraries.
# - A small style "badge" next to each example so it's visually obvious
#   which examples are standard vs drift, instead of relying on a text
#   label above the example list.
# - Muted, consistent typography via CSS instead of Gradio's defaults.

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer_a = AutoTokenizer.from_pretrained("model_a")
model_a = AutoModelForSequenceClassification.from_pretrained("model_a").to(DEVICE).eval()

tokenizer_b = AutoTokenizer.from_pretrained("model_b")
model_b = AutoModelForSequenceClassification.from_pretrained("model_b").to(DEVICE).eval()

LABELS = {0: "Negative", 1: "Positive"}


def predict(text, tokenizer, model):
    if not text or not text.strip():
        return None
    enc = tokenizer(text, truncation=True, padding=True, max_length=64, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1)[0]
    return {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}


def predict_a(text):
    return predict(text, tokenizer_a, model_a)


def predict_b(text):
    return predict(text, tokenizer_b, model_b)


standard_examples = [
    "I found this laptop to be excellent.",
    "The customer support was terrible.",
    "This course was genuinely impressive.",
    "The hotel room was disappointing overall.",
    "My experience with this app was fantastic.",
]

# 10 drift examples, deliberately covering different DRIFT SUBTYPES so the
# demo shows the problem isn't just "slang" -- it's several distinct ways
# real-world text differs from clean training data:
#   1-2  slang (positive / negative)
#   3-4  emoji-heavy (positive / negative)
#   5-6  sarcasm (says positive words, means negative / vice versa)
#   7-8  abbreviations & typos (positive / negative)
#   9-10 very short / minimal social-media style (positive / negative)
drift_examples = [
    "ngl this app is actually fire 🔥",                              # 1. slang - positive
    "the delivery was mid tbh",                                      # 2. slang - negative
    "obsessed w this laptop rn 😭❤️",                                 # 3. emoji - positive
    "this restaurant was a disaster 💀🚮",                            # 4. emoji - negative
    "oh great, this software update broke again, love that for me",  # 5. sarcasm - actually negative
    "wasn't expecting much but this course actually delivered",      # 6. sarcasm - actually positive
    "gr8 service ngl, wud recommend fr",                              # 7. abbreviation/typo - positive
    "tbh dis phone kinda trash not gonna lie",                        # 8. abbreviation/typo - negative
    "coffee?? 10/10.",                                                # 9. short/minimal - positive
    "the flight. never again.",                                       # 10. short/minimal - negative
]

# ---------------------------------------------------------------------
# THEME: a clean, minimal indigo/slate palette instead of Gradio default
# ---------------------------------------------------------------------
theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="*neutral_50",
    block_background_fill="white",
    block_border_width="1px",
    block_border_color="*neutral_200",
    block_shadow="0 1px 3px 0 rgba(0,0,0,0.06)",
    block_radius="14px",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    button_primary_text_color="white",
)

CSS = """
#header {text-align: center; padding: 8px 0 4px 0;}
#header h1 {font-weight: 700; margin-bottom: 2px;}
#header p {color: var(--body-text-color-subdued); font-size: 15px; margin-top: 0;}
.tab-desc {color: var(--body-text-color-subdued); font-size: 14px; margin-bottom: 8px;}
.badge-standard {
  display:inline-block; padding:2px 10px; border-radius:999px;
  background:#eef2ff; color:#4338ca; font-size:12px; font-weight:600; margin-bottom:6px;
}
.badge-drift {
  display:inline-block; padding:2px 10px; border-radius:999px;
  background:#fef2f2; color:#b91c1c; font-size:12px; font-weight:600; margin-bottom:6px;
}
footer {display: none !important;}
"""

with gr.Blocks(css=CSS, theme=theme, title="Systemic Drift Demo") as demo:
    with gr.Column(elem_id="header"):
        gr.Markdown(
            "# Sentiment Drift: Before &amp; After\n"
            "<p>The same sentence, run through two models. "
            "Model A learns clean, formal text and breaks on real-world slang. "
            "Model B is lifecycle-tuned with a contrastive loss to close that gap.</p>"
        )

    with gr.Tabs():
        # ---------------- TAB 1: MODEL A ----------------
        with gr.Tab("Model A · Baseline"):
            gr.Markdown(
                "<span class='tab-desc'>Trained only on standard, formal-style text. "
                "Expect strong results on clean sentences and weaker results on slang, "
                "sarcasm, or emoji-heavy text.</span>"
            )
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Group():
                        text_a = gr.Textbox(
                            label="Input text",
                            placeholder="Type a sentence to analyze...",
                            lines=3,
                        )
                        btn_a = gr.Button("Analyze", variant="primary")
                    gr.Markdown("<span class='badge-standard'>Standard style</span>")
                    gr.Examples(examples=standard_examples, inputs=text_a)
                    gr.Markdown("<span class='badge-drift'>Drift style</span>")
                    gr.Examples(examples=drift_examples, inputs=text_a)
                with gr.Column(scale=2):
                    with gr.Group():
                        gr.Markdown("**Result**")
                        result_a = gr.Label(num_top_classes=2, label=None, show_label=False)
            btn_a.click(fn=predict_a, inputs=text_a, outputs=result_a)

        # ---------------- TAB 2: MODEL B ----------------
        with gr.Tab("Model B · Lifecycle-Tuned"):
            gr.Markdown(
                "<span class='tab-desc'>Continued from Model A, fine-tuned further on a "
                "standard + drift mix using supervised contrastive loss.</span>"
            )
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Group():
                        text_b = gr.Textbox(
                            label="Input text",
                            placeholder="Type a sentence to analyze...",
                            lines=3,
                        )
                        btn_b = gr.Button("Analyze", variant="primary")
                    gr.Markdown("<span class='badge-standard'>Standard style</span>")
                    gr.Examples(examples=standard_examples, inputs=text_b)
                    gr.Markdown("<span class='badge-drift'>Drift style</span>")
                    gr.Examples(examples=drift_examples, inputs=text_b)
                with gr.Column(scale=2):
                    with gr.Group():
                        gr.Markdown("**Result**")
                        result_b = gr.Label(num_top_classes=2, label=None, show_label=False)
            btn_b.click(fn=predict_b, inputs=text_b, outputs=result_b)

if __name__ == "__main__":
    demo.launch(share=True)