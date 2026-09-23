# # app.py
# # Run AFTER models/model_a, models/model_b, models/chat_model_a,
# # models/chat_model_b all exist:
# #   pip install -r requirements.txt
# #   python app.py
# #
# # 2 TABS, each comparing Model A (baseline) vs Model B (lifecycle-tuned)
# # SIDE BY SIDE on the exact same input, at the same time:
# # 1. Classifier : one sentence in -> both models' sentiment + confidence
# # 2. Chatbot    : one message in -> both models' generated reply

# import torch
# import gradio as gr
# from transformers import (
#     AutoTokenizer,
#     AutoModelForSequenceClassification,
#     T5Tokenizer,
#     T5ForConditionalGeneration,
# )

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # ---- Load all 4 models once at startup ----
# clf_tokenizer_a = AutoTokenizer.from_pretrained("models/model_a")
# clf_model_a = AutoModelForSequenceClassification.from_pretrained("models/model_a").to(DEVICE).eval()

# clf_tokenizer_b = AutoTokenizer.from_pretrained("models/model_b")
# clf_model_b = AutoModelForSequenceClassification.from_pretrained("models/model_b").to(DEVICE).eval()

# chat_tokenizer_a = T5Tokenizer.from_pretrained("models/chat_model_a")
# chat_model_a = T5ForConditionalGeneration.from_pretrained("models/chat_model_a").to(DEVICE).eval()

# chat_tokenizer_b = T5Tokenizer.from_pretrained("models/chat_model_b")
# chat_model_b = T5ForConditionalGeneration.from_pretrained("models/chat_model_b").to(DEVICE).eval()

# CLF_LABELS = {0: "Negative", 1: "Positive"}


# # ---- Classifier ----
# def classify(text, tokenizer, model):
#     if not text or not text.strip():
#         return None
#     enc = tokenizer(text, truncation=True, padding=True, max_length=64, return_tensors="pt").to(DEVICE)
#     with torch.no_grad():
#         logits = model(**enc).logits
#         probs = torch.softmax(logits, dim=1)[0]
#     return {CLF_LABELS[i]: float(probs[i]) for i in range(len(CLF_LABELS))}


# def classify_both(text):
#     return classify(text, clf_tokenizer_a, clf_model_a), classify(text, clf_tokenizer_b, clf_model_b)


# # ---- Chatbot ----
# def generate_reply(message, tokenizer, model):
#     input_text = "respond: " + message
#     enc = tokenizer(input_text, truncation=True, max_length=64, return_tensors="pt").to(DEVICE)
#     with torch.no_grad():
#         out_ids = model.generate(**enc, max_length=24, num_beams=2)
#     return tokenizer.decode(out_ids[0], skip_special_tokens=True)


# def add_user_both(message, hist_a, hist_b):
#     """Step 1: echo the message into BOTH chat panels at once, clear and
#     disable the shared input while both bots 'think'."""
#     if not message or not message.strip():
#         return hist_a, hist_b, gr.update()
#     hist_a = hist_a + [{"role": "user", "content": message}]
#     hist_b = hist_b + [{"role": "user", "content": message}]
#     return hist_a, hist_b, gr.update(value="", interactive=False, placeholder="Both bots are typing...")


# def bot_reply_both(hist_a, hist_b):
#     """Step 2: generate each model's reply independently and drop it into
#     its own panel -- this is where the two models can diverge."""
#     if hist_a and hist_a[-1]["role"] == "user":
#         reply_a = generate_reply(hist_a[-1]["content"], chat_tokenizer_a, chat_model_a)
#         hist_a = hist_a + [{"role": "assistant", "content": reply_a}]
#     if hist_b and hist_b[-1]["role"] == "user":
#         reply_b = generate_reply(hist_b[-1]["content"], chat_tokenizer_b, chat_model_b)
#         hist_b = hist_b + [{"role": "assistant", "content": reply_b}]
#     return hist_a, hist_b, gr.update(interactive=True, placeholder="Type a message and press Enter...")


# def clear_both_chats():
#     return [], [], ""


# # ---- Shared example sets ----
# clf_standard_examples = [
#     "I found this laptop to be excellent.",
#     "The customer support was terrible.",
#     "This course was genuinely impressive.",
#     "The hotel room was disappointing overall.",
#     "My experience with this app was fantastic.",
# ]
# clf_drift_examples = [
#     "ngl this app is actually fire 🔥",
#     "the delivery was mid tbh",
#     "obsessed w this laptop rn 😭❤️",
#     "this restaurant was a disaster 💀🚮",
#     "oh great, this software update broke again, love that for me",
#     "wasn't expecting much but this course actually delivered",
#     "gr8 service ngl, wud recommend fr",
#     "tbh dis phone kinda trash not gonna lie",
#     "coffee?? 10/10.",
#     "the flight. never again.",
# ]
# chat_standard_examples = [
#     "I found this laptop to be excellent.",
#     "The customer support was terrible.",
#     "This course was genuinely impressive.",
# ]
# chat_drift_examples = [
#     "the hotel room is goated no cap",
#     "obsessed w the service rn 😭❤️",
#     "the museum 2/10 do not recommend",
#     "wow the gym membership really said let me disappoint you",
#     "okay the delivery is actually kinda amazing ngl",
#     "the flight... yeah no thx 💀",
# ]

# # ---------------------------------------------------------------------
# # THEME
# # ---------------------------------------------------------------------
# theme = gr.themes.Base(
#     primary_hue=gr.themes.colors.emerald,
#     secondary_hue=gr.themes.colors.teal,
#     neutral_hue=gr.themes.colors.slate,
#     font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
# ).set(
#     body_background_fill="#0A0F0D",
#     body_text_color="#E5E9E8",
#     block_background_fill="rgba(255,255,255,0.035)",
#     block_border_width="1px",
#     block_border_color="rgba(255,255,255,0.08)",
#     block_shadow="0 8px 32px rgba(0,0,0,0.35)",
#     block_radius="18px",
#     input_background_fill="rgba(255,255,255,0.04)",
#     input_border_color="rgba(255,255,255,0.10)",
#     button_primary_background_fill="linear-gradient(135deg, #10B981 0%, #0D9488 100%)",
#     button_primary_background_fill_hover="linear-gradient(135deg, #34D399 0%, #14B8A6 100%)",
#     button_primary_text_color="#04120E",
#     button_primary_border_color="rgba(16,185,129,0.4)",
# )

# CSS = """
# @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

# * { font-family: 'Plus Jakarta Sans', ui-sans-serif, system-ui, sans-serif !important; }

# .gradio-container {
#   background:
#     radial-gradient(ellipse 900px 500px at 20% -10%, rgba(16,185,129,0.16), transparent 60%),
#     radial-gradient(ellipse 700px 500px at 100% 0%, rgba(20,184,166,0.10), transparent 55%),
#     #0A0F0D !important;
# }

# /* ---- Top brand bar ---- */
# #topbar {
#   display: flex; align-items: center; justify-content: space-between;
#   padding: 6px 4px 18px 4px; margin-bottom: 4px;
#   border-bottom: 1px solid rgba(255,255,255,0.06);
# }
# #topbar .brand { display:flex; align-items:center; gap:10px; }
# #topbar .logo-mark {
#   width: 30px; height: 30px; border-radius: 9px;
#   background: linear-gradient(135deg, #10B981, #0D9488);
#   display:flex; align-items:center; justify-content:center;
#   font-weight: 800; font-size: 14px; color: #04120E;
#   box-shadow: 0 0 18px rgba(16,185,129,0.45);
# }
# #topbar .brand-name { font-weight: 700; font-size: 15px; color: #F1F5F4; letter-spacing: -0.01em; }
# #topbar .badge-pill {
#   padding: 5px 14px; border-radius: 999px; font-size: 12px; font-weight: 600;
#   background: rgba(16,185,129,0.12); color: #6EE7B7; border: 1px solid rgba(16,185,129,0.28);
# }

# /* ---- Hero ---- */
# #hero { text-align: center; padding: 18px 8px 10px 8px; }
# #hero h1 {
#   font-weight: 800; font-size: 34px; letter-spacing: -0.02em;
#   color: #F8FAFA; margin-bottom: 10px; line-height: 1.15;
# }
# #hero h1 .accent {
#   background: linear-gradient(135deg, #34D399, #2DD4BF);
#   -webkit-background-clip: text; background-clip: text; color: transparent;
# }
# #hero p.sub {
#   color: #94A3A0; font-size: 15.5px; max-width: 660px; margin: 0 auto 18px auto; line-height: 1.55;
# }
# .feature-row { display:flex; flex-wrap:wrap; gap:8px; justify-content:center; margin-bottom: 6px; }
# .feature-pill {
#   padding: 7px 14px; border-radius: 999px; font-size: 12.5px; font-weight: 600;
#   background: rgba(255,255,255,0.035); border: 1px solid rgba(255,255,255,0.08); color: #C7D0CD;
#   display:inline-flex; align-items:center; gap:6px;
# }

# .tab-desc { color: #8B9C97; font-size: 13.5px; margin-bottom: 14px; }
# .badge-standard {
#   display:inline-block; padding:3px 12px; border-radius:999px;
#   background: rgba(16,185,129,0.12); color:#6EE7B7; border: 1px solid rgba(16,185,129,0.28);
#   font-size:12px; font-weight:700; margin-bottom:8px;
# }
# .badge-drift {
#   display:inline-block; padding:3px 12px; border-radius:999px;
#   background: rgba(244,63,94,0.10); color:#FB7185; border: 1px solid rgba(244,63,94,0.25);
#   font-size:12px; font-weight:700; margin-bottom:8px;
# }

# /* ---- Model column tags (this is what makes the UI self-explanatory) ---- */
# .model-col {
#   border-radius: 18px; padding: 14px 16px 18px 16px;
#   background: rgba(255,255,255,0.025);
#   border: 1px solid rgba(255,255,255,0.07);
# }
# .model-col.baseline { border-top: 3px solid #FB7185; }
# .model-col.tuned { border-top: 3px solid #34D399; }
# .model-tag {
#   display:inline-flex; align-items:center; gap:6px;
#   font-size: 11.5px; font-weight: 800; letter-spacing: 0.04em; text-transform: uppercase;
#   padding: 4px 10px; border-radius: 8px; margin-bottom: 8px;
# }
# .model-tag.baseline { background: rgba(244,63,94,0.12); color: #FDA4AF; }
# .model-tag.tuned { background: rgba(16,185,129,0.14); color: #6EE7B7; }
# .model-sub { color: #7C8D89; font-size: 12.5px; margin-bottom: 10px; line-height: 1.4; }

# /* ---- Tabs ---- */
# .tabs > .tab-nav { border-bottom: 1px solid rgba(255,255,255,0.08) !important; gap: 4px; }
# .tabs > .tab-nav button {
#   font-weight: 700 !important; font-size: 14.5px !important; color: #8B9C97 !important; border: none !important;
#   background: transparent !important; padding: 10px 20px !important;
# }
# .tabs > .tab-nav button.selected {
#   color: #34D399 !important; border-bottom: 2px solid #34D399 !important;
# }

# /* ---- Buttons ---- */
# button.primary { font-weight: 700 !important; letter-spacing: -0.01em; }

# /* ---- Footer ---- */
# #footerbar {
#   text-align:center; padding: 22px 0 6px 0; color: #5B6B67; font-size: 12.5px;
#   border-top: 1px solid rgba(255,255,255,0.06); margin-top: 18px;
# }

# footer { display: none !important; }
# """

# with gr.Blocks(title="Systemic Drift Lab · Ethical Intelligence") as demo:
#     gr.HTML(
#         """
#         <div id="topbar">
#           <div class="brand">
#             <div class="logo-mark">EI</div>
#             <div class="brand-name">Ethical Intelligence</div>
#           </div>
#           <div class="badge-pill">Internal Demo</div>
#         </div>
#         """
#     )
#     with gr.Column(elem_id="hero"):
#         gr.Markdown(
#             "<h1>One input. Two models. <span class='accent'>See the drift live.</span></h1>"
#             "<p class='sub'>Enter a sentence once — the baseline model and the contrastive "
#             "lifecycle-tuned model respond side by side, at the same time, so the difference "
#             "is obvious without switching tabs.</p>"
#             "<div class='feature-row'>"
#             "<span class='feature-pill'>⚖️ Side-by-Side</span>"
#             "<span class='feature-pill'>💬 Live Chatbot</span>"
#             "<span class='feature-pill'>🧬 Contrastive Loss</span>"
#             "<span class='feature-pill'>🔁 Lifecycle Fine-tune</span>"
#             "</div>"
#         )

#     with gr.Tabs():
#         # ==================== TAB 1: CLASSIFIER ====================
#         with gr.Tab("Classifier"):
#             gr.Markdown(
#                 "<span class='tab-desc'>Type a sentence once. Both models classify it "
#                 "instantly, side by side — compare confidence directly.</span>"
#             )
#             with gr.Group():
#                 text_clf = gr.Textbox(label="Input text", placeholder="Type a sentence to analyze...", lines=2)
#                 btn_clf = gr.Button("Analyze with both models", variant="primary")

#             with gr.Row():
#                 with gr.Column():
#                     gr.HTML("<div class='model-tag baseline'>● Model A · Baseline</div>")
#                     gr.Markdown("<div class='model-sub'>Trained only on standard/formal-style text.</div>")
#                     result_a = gr.Label(num_top_classes=2, show_label=False)
#                 with gr.Column():
#                     gr.HTML("<div class='model-tag tuned'>● Model B · Lifecycle-Tuned</div>")
#                     gr.Markdown("<div class='model-sub'>Continued from Model A with supervised contrastive loss on standard + drift data.</div>")
#                     result_b = gr.Label(num_top_classes=2, show_label=False)

#             gr.Markdown("<span class='badge-standard'>Standard style</span>")
#             gr.Examples(examples=clf_standard_examples, inputs=text_clf)
#             gr.Markdown("<span class='badge-drift'>Drift style</span>")
#             gr.Examples(examples=clf_drift_examples, inputs=text_clf)

#             btn_clf.click(fn=classify_both, inputs=text_clf, outputs=[result_a, result_b])
#             text_clf.submit(fn=classify_both, inputs=text_clf, outputs=[result_a, result_b])

#         # ==================== TAB 2: CHATBOT ====================
#         with gr.Tab("Chatbot"):
#             gr.Markdown(
#                 "<span class='tab-desc'>Send one message. Both bots reply at the same time in "
#                 "their own panel — watch the baseline get the tone wrong while the tuned model "
#                 "stays grounded.</span>"
#             )
#             with gr.Row():
#                 with gr.Column():
#                     gr.HTML("<div class='model-tag baseline'>● Model A · Baseline</div>")
#                     gr.Markdown("<div class='model-sub'>Trained only on standard-style messages.</div>")
#                     chatbot_a = gr.Chatbot(height=340, label=None, show_label=False)
#                 with gr.Column():
#                     gr.HTML("<div class='model-tag tuned'>● Model B · Lifecycle-Tuned</div>")
#                     gr.Markdown("<div class='model-sub'>Continued from Model A, tuned on standard + drift with contrastive loss.</div>")
#                     chatbot_b = gr.Chatbot(height=340, label=None, show_label=False)

#             with gr.Row():
#                 msg = gr.Textbox(placeholder="Type a message and press Enter...", show_label=False, scale=5, container=False)
#                 send_btn = gr.Button("Send", variant="primary", scale=1)
#                 clear_btn = gr.Button("Clear", scale=1)

#             gr.Markdown("<span class='badge-standard'>Standard style</span>")
#             gr.Examples(examples=chat_standard_examples, inputs=msg)
#             gr.Markdown("<span class='badge-drift'>Drift style</span>")
#             gr.Examples(examples=chat_drift_examples, inputs=msg)

#             msg.submit(fn=add_user_both, inputs=[msg, chatbot_a, chatbot_b], outputs=[chatbot_a, chatbot_b, msg]).then(
#                 fn=bot_reply_both, inputs=[chatbot_a, chatbot_b], outputs=[chatbot_a, chatbot_b, msg]
#             )
#             send_btn.click(fn=add_user_both, inputs=[msg, chatbot_a, chatbot_b], outputs=[chatbot_a, chatbot_b, msg]).then(
#                 fn=bot_reply_both, inputs=[chatbot_a, chatbot_b], outputs=[chatbot_a, chatbot_b, msg]
#             )
#             clear_btn.click(fn=clear_both_chats, outputs=[chatbot_a, chatbot_b, msg])

#     gr.HTML("<div id='footerbar'>Built by Ethical Intelligence · Systemic Drift Lab</div>")

# if __name__ == "__main__":
#     demo.launch(share=True, theme=theme, css=CSS)

















































# app.py
# Run AFTER models/model_a, models/model_b, models/chat_model_a,
# models/chat_model_b all exist:
#   pip install -r requirements.txt
#   python app.py
#
# 2 TABS, each comparing Model A (baseline) vs Model B (lifecycle-tuned)
# SIDE BY SIDE on the exact same input, at the same time:
# 1. Classifier : one sentence in -> both models' sentiment + confidence
# 2. Chatbot    : one message in -> both models' generated reply

import torch
import gradio as gr
from concurrent.futures import ThreadPoolExecutor
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    T5Tokenizer,
    T5ForConditionalGeneration,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Load all 4 models once at startup ----
clf_tokenizer_a = AutoTokenizer.from_pretrained("models/model_a")
clf_model_a = AutoModelForSequenceClassification.from_pretrained("models/model_a").to(DEVICE).eval()

clf_tokenizer_b = AutoTokenizer.from_pretrained("models/model_b")
clf_model_b = AutoModelForSequenceClassification.from_pretrained("models/model_b").to(DEVICE).eval()

chat_tokenizer_a = T5Tokenizer.from_pretrained("models/chat_model_a")
chat_model_a = T5ForConditionalGeneration.from_pretrained("models/chat_model_a").to(DEVICE).eval()

chat_tokenizer_b = T5Tokenizer.from_pretrained("models/chat_model_b")
chat_model_b = T5ForConditionalGeneration.from_pretrained("models/chat_model_b").to(DEVICE).eval()

CLF_LABELS = {0: "Negative", 1: "Positive"}


# ---- Classifier ----
def classify(text, tokenizer, model):
    if not text or not text.strip():
        return None
    enc = tokenizer(text, truncation=True, padding=True, max_length=64, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1)[0]
    return {CLF_LABELS[i]: float(probs[i]) for i in range(len(CLF_LABELS))}


def classify_both(text):
    return classify(text, clf_tokenizer_a, clf_model_a), classify(text, clf_tokenizer_b, clf_model_b)


# ---- Chatbot ----
def _extract_message_text(message):
    """Normalize Gradio message content to plain text."""
    if message is None:
        return ""

    if isinstance(message, str):
        return message.strip()

    if isinstance(message, dict):
        return _extract_message_text(message.get("content", ""))

    if isinstance(message, (list, tuple)):
        parts = []
        for item in message:
            if isinstance(item, dict):
                parts.append(_extract_message_text(item.get("content", "")))
            elif isinstance(item, str):
                parts.append(item)
            elif isinstance(item, (list, tuple)) and item:
                parts.append(_extract_message_text(item[0]))
        return " ".join(p for p in parts if p).strip()

    return str(message).strip()


def generate_reply(message, tokenizer, model):
    """Generate a response from one chatbot model."""
    message = _extract_message_text(message)

    if not message:
        return "Please enter a message."

    # Use the raw message instead of adding `respond:`. This avoids a
    # prompt-format mismatch when the T5 checkpoint was trained on
    # direct input -> response pairs.
    enc = tokenizer(
        message,
        truncation=True,
        max_length=128,
        padding=False,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        out_ids = model.generate(
            **enc,
            max_new_tokens=48,
            num_beams=4,
            early_stopping=True,
            repetition_penalty=1.12,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
        )

    reply = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
    return reply or "I couldn't generate a response."


def _generate_model_a(message):
    return generate_reply(message, chat_tokenizer_a, chat_model_a)


def _generate_model_b(message):
    return generate_reply(message, chat_tokenizer_b, chat_model_b)


def chat_compare(message, hist_a, hist_b):
    """Run both chatbot models against the exact same input in parallel."""
    message = _extract_message_text(message)

    if not message:
        return hist_a or [], hist_b or [], gr.update(value="", interactive=True)

    hist_a = (hist_a or []) + [{"role": "user", "content": message}]
    hist_b = (hist_b or []) + [{"role": "user", "content": message}]

    # Both model inference calls are started together.
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(_generate_model_a, message)
        future_b = executor.submit(_generate_model_b, message)
        reply_a = future_a.result()
        reply_b = future_b.result()

    hist_a.append({"role": "assistant", "content": reply_a})
    hist_b.append({"role": "assistant", "content": reply_b})

    return (
        hist_a,
        hist_b,
        gr.update(
            value="",
            interactive=True,
            placeholder="Type a message and press Enter..."
        ),
    )


def clear_both_chats():
    return [], [], ""


# ---- Shared example sets ----
clf_standard_examples = [
    "I found this laptop to be excellent.",
    "The customer support was terrible.",
    "This course was genuinely impressive.",
    "The hotel room was disappointing overall.",
    "My experience with this app was fantastic.",
]
clf_drift_examples = [
    "ngl this app is actually fire 🔥",
    "the delivery was mid tbh",
    "obsessed w this laptop rn 😭❤️",
    "this restaurant was a disaster 💀🚮",
    "oh great, this software update broke again, love that for me",
    "wasn't expecting much but this course actually delivered",
    "gr8 service ngl, wud recommend fr",
    "tbh dis phone kinda trash not gonna lie",
    "coffee?? 10/10.",
    "the flight. never again.",
]
chat_standard_examples = [
    "I found this laptop to be excellent.",
    "The customer support was terrible.",
    "This course was genuinely impressive.",
]
chat_drift_examples = [
    "the hotel room is goated no cap",
    "obsessed w the service rn 😭❤️",
    "the museum 2/10 do not recommend",
    "wow the gym membership really said let me disappoint you",
    "okay the delivery is actually kinda amazing ngl",
    "the flight... yeah no thx 💀",
]

# ---------------------------------------------------------------------
# THEME
# ---------------------------------------------------------------------
theme = gr.themes.Base(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#0A0F0D",
    body_text_color="#E5E9E8",
    block_background_fill="rgba(255,255,255,0.035)",
    block_border_width="1px",
    block_border_color="rgba(255,255,255,0.08)",
    block_shadow="0 8px 32px rgba(0,0,0,0.35)",
    block_radius="18px",
    input_background_fill="rgba(255,255,255,0.04)",
    input_border_color="rgba(255,255,255,0.10)",
    button_primary_background_fill="linear-gradient(135deg, #10B981 0%, #0D9488 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #34D399 0%, #14B8A6 100%)",
    button_primary_text_color="#04120E",
    button_primary_border_color="rgba(16,185,129,0.4)",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

* { font-family: 'Plus Jakarta Sans', ui-sans-serif, system-ui, sans-serif !important; }

.gradio-container {
  background:
    radial-gradient(ellipse 900px 500px at 20% -10%, rgba(16,185,129,0.16), transparent 60%),
    radial-gradient(ellipse 700px 500px at 100% 0%, rgba(20,184,166,0.10), transparent 55%),
    #0A0F0D !important;
}

/* ---- Top brand bar ---- */
#topbar {
  display: flex; align-items: center; justify-content: space-between;
  padding: 6px 4px 18px 4px; margin-bottom: 4px;
  border-bottom: 1px solid rgba(255,255,255,0.06);
}
#topbar .brand { display:flex; align-items:center; gap:10px; }
#topbar .logo-mark {
  width: 30px; height: 30px; border-radius: 9px;
  background: linear-gradient(135deg, #10B981, #0D9488);
  display:flex; align-items:center; justify-content:center;
  font-weight: 800; font-size: 14px; color: #04120E;
  box-shadow: 0 0 18px rgba(16,185,129,0.45);
}
#topbar .brand-name { font-weight: 700; font-size: 15px; color: #F1F5F4; letter-spacing: -0.01em; }
#topbar .badge-pill {
  padding: 5px 14px; border-radius: 999px; font-size: 12px; font-weight: 600;
  background: rgba(16,185,129,0.12); color: #6EE7B7; border: 1px solid rgba(16,185,129,0.28);
}

/* ---- Hero ---- */
#hero { text-align: center; padding: 18px 8px 10px 8px; }
#hero h1 {
  font-weight: 800; font-size: 34px; letter-spacing: -0.02em;
  color: #F8FAFA; margin-bottom: 10px; line-height: 1.15;
}
#hero h1 .accent {
  background: linear-gradient(135deg, #34D399, #2DD4BF);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
#hero p.sub {
  color: #94A3A0; font-size: 15.5px; max-width: 660px; margin: 0 auto 18px auto; line-height: 1.55;
}
.feature-row { display:flex; flex-wrap:wrap; gap:8px; justify-content:center; margin-bottom: 6px; }
.feature-pill {
  padding: 7px 14px; border-radius: 999px; font-size: 12.5px; font-weight: 600;
  background: rgba(255,255,255,0.035); border: 1px solid rgba(255,255,255,0.08); color: #C7D0CD;
  display:inline-flex; align-items:center; gap:6px;
}

.tab-desc { color: #8B9C97; font-size: 13.5px; margin-bottom: 14px; }
.badge-standard {
  display:inline-block; padding:3px 12px; border-radius:999px;
  background: rgba(16,185,129,0.12); color:#6EE7B7; border: 1px solid rgba(16,185,129,0.28);
  font-size:12px; font-weight:700; margin-bottom:8px;
}
.badge-drift {
  display:inline-block; padding:3px 12px; border-radius:999px;
  background: rgba(244,63,94,0.10); color:#FB7185; border: 1px solid rgba(244,63,94,0.25);
  font-size:12px; font-weight:700; margin-bottom:8px;
}

/* ---- Model column tags (this is what makes the UI self-explanatory) ---- */
.model-col {
  border-radius: 18px; padding: 14px 16px 18px 16px;
  background: rgba(255,255,255,0.025);
  border: 1px solid rgba(255,255,255,0.07);
}
.model-col.baseline { border-top: 3px solid #FB7185; }
.model-col.tuned { border-top: 3px solid #34D399; }
.model-tag {
  display:inline-flex; align-items:center; gap:6px;
  font-size: 11.5px; font-weight: 800; letter-spacing: 0.04em; text-transform: uppercase;
  padding: 4px 10px; border-radius: 8px; margin-bottom: 8px;
}
.model-tag.baseline { background: rgba(244,63,94,0.12); color: #FDA4AF; }
.model-tag.tuned { background: rgba(16,185,129,0.14); color: #6EE7B7; }
.model-sub { color: #7C8D89; font-size: 12.5px; margin-bottom: 10px; line-height: 1.4; }

/* ---- Tabs ---- */
.tabs > .tab-nav { border-bottom: 1px solid rgba(255,255,255,0.08) !important; gap: 4px; }
.tabs > .tab-nav button {
  font-weight: 700 !important; font-size: 14.5px !important; color: #8B9C97 !important; border: none !important;
  background: transparent !important; padding: 10px 20px !important;
}
.tabs > .tab-nav button.selected {
  color: #34D399 !important; border-bottom: 2px solid #34D399 !important;
}

/* ---- Buttons ---- */
button.primary { font-weight: 700 !important; letter-spacing: -0.01em; }

/* ---- Footer ---- */
#footerbar {
  text-align:center; padding: 22px 0 6px 0; color: #5B6B67; font-size: 12.5px;
  border-top: 1px solid rgba(255,255,255,0.06); margin-top: 18px;
}

footer { display: none !important; }
"""

with gr.Blocks(title="Systemic Drift Lab · Ethical Intelligence") as demo:
    gr.HTML(
        """
        <div id="topbar">
          <div class="brand">
            <div class="logo-mark">EI</div>
            <div class="brand-name">Ethical Intelligence</div>
          </div>
          <div class="badge-pill">Internal Demo</div>
        </div>
        """
    )
    with gr.Column(elem_id="hero"):
        gr.Markdown(
            "<h1>One input. Two models. <span class='accent'>See the drift live.</span></h1>"
            "<p class='sub'>A controlled model-comparison environment that places the baseline "
            "and lifecycle-tuned systems side by side, making behavioral differences visible on "
            "identical inputs across classification and conversation.</p>"
            "<div class='feature-row'>"
            "<span class='feature-pill'>⚖️ Side-by-Side</span>"
            "<span class='feature-pill'>💬 Live Chatbot</span>"
            "<span class='feature-pill'>🧬 Contrastive Loss</span>"
            "<span class='feature-pill'>🔁 Lifecycle Fine-tune</span>"
            "</div>"
        )

    with gr.Tabs():
        # ==================== TAB 1: CLASSIFIER ====================
        with gr.Tab("Classifier"):
            gr.Markdown(
                "<span class='tab-desc'>A side-by-side evaluation view for the baseline and "
                "lifecycle-tuned classifiers across standard and drift-style language.</span>"
            )
            with gr.Group():
                text_clf = gr.Textbox(label="Input text", placeholder="Type a sentence to analyze...", lines=2)
                btn_clf = gr.Button("Analyze with both models", variant="primary")

            with gr.Row():
                with gr.Column():
                    gr.HTML("<div class='model-tag baseline'>● Model A · Baseline</div>")
                    gr.Markdown("<div class='model-sub'>Trained only on standard/formal-style text.</div>")
                    result_a = gr.Label(num_top_classes=2, show_label=False)
                with gr.Column():
                    gr.HTML("<div class='model-tag tuned'>● Model B · Lifecycle-Tuned</div>")
                    gr.Markdown("<div class='model-sub'>Continued from Model A with supervised contrastive loss on standard + drift data.</div>")
                    result_b = gr.Label(num_top_classes=2, show_label=False)

            gr.Markdown("<span class='badge-standard'>Standard style</span>")
            gr.Examples(examples=clf_standard_examples, inputs=text_clf)
            gr.Markdown("<span class='badge-drift'>Drift style</span>")
            gr.Examples(examples=clf_drift_examples, inputs=text_clf)

            btn_clf.click(fn=classify_both, inputs=text_clf, outputs=[result_a, result_b])
            text_clf.submit(fn=classify_both, inputs=text_clf, outputs=[result_a, result_b])

        # ==================== TAB 2: CHATBOT ====================
        with gr.Tab("Chatbot"):
            gr.Markdown(
                "<span class='tab-desc'>A parallel conversational evaluation view comparing "
                "the baseline chatbot with the lifecycle-tuned chatbot on identical inputs.</span>"
            )
            with gr.Row():
                with gr.Column():
                    gr.HTML("<div class='model-tag baseline'>● Model A · Baseline</div>")
                    gr.Markdown("<div class='model-sub'>Trained only on standard-style messages.</div>")
                    chatbot_a = gr.Chatbot(height=340, label=None, show_label=False)
                with gr.Column():
                    gr.HTML("<div class='model-tag tuned'>● Model B · Lifecycle-Tuned</div>")
                    gr.Markdown("<div class='model-sub'>Continued from Model A, tuned on standard + drift with contrastive loss.</div>")
                    chatbot_b = gr.Chatbot(height=340, label=None, show_label=False)

            with gr.Row():
                msg = gr.Textbox(placeholder="Type a message and press Enter...", show_label=False, scale=5, container=False)
                send_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("Clear", scale=1)

            gr.Markdown("<span class='badge-standard'>Standard style</span>")
            gr.Examples(examples=chat_standard_examples, inputs=msg)
            gr.Markdown("<span class='badge-drift'>Drift style</span>")
            gr.Examples(examples=chat_drift_examples, inputs=msg)

            msg.submit(
                fn=chat_compare,
                inputs=[msg, chatbot_a, chatbot_b],
                outputs=[chatbot_a, chatbot_b, msg]
            )
            send_btn.click(
                fn=chat_compare,
                inputs=[msg, chatbot_a, chatbot_b],
                outputs=[chatbot_a, chatbot_b, msg]
            )
            clear_btn.click(fn=clear_both_chats, outputs=[chatbot_a, chatbot_b, msg])

    gr.HTML("<div id='footerbar'>Built by Ethical Intelligence · Systemic Drift Lab</div>")

if __name__ == "__main__":
    demo.launch(share=True, theme=theme, css=CSS)