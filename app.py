# app.py
# Run AFTER models/chat_model_a, models/chat_model_b and demo_data/ exist
# (demo_data is produced by analysis.py):
#   pip install -r requirements.txt
#   python app.py
#
# Single-page demo: Model A (baseline) vs Model B (lifecycle-tuned)
# shown in embedding space -- how systematic drift appears and how
# contrastive lifecycle tuning fixes it.
# Themes: light (soft beige) + dark, switchable via the toggle in the top bar.

import gradio as gr
from embedding_tab import build_embedding_tab

# ---------------------------------------------------------------------
# THEME  (light = soft beige, *_dark = dark mode)
# ---------------------------------------------------------------------
theme = gr.themes.Base(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    # light (beige)
    body_background_fill="#F5EFE4",
    body_text_color="#2E2C26",
    block_background_fill="rgba(255,255,255,0.55)",
    block_border_width="1px",
    block_border_color="rgba(70,55,30,0.14)",
    block_shadow="0 8px 28px rgba(90,70,30,0.10)",
    block_radius="18px",
    input_background_fill="rgba(255,255,255,0.75)",
    input_border_color="rgba(70,55,30,0.18)",
    # dark
    body_background_fill_dark="#0A0F0D",
    body_text_color_dark="#E5E9E8",
    block_background_fill_dark="rgba(255,255,255,0.035)",
    block_border_color_dark="rgba(255,255,255,0.08)",
    block_shadow_dark="0 8px 32px rgba(0,0,0,0.35)",
    input_background_fill_dark="rgba(255,255,255,0.04)",
    input_border_color_dark="rgba(255,255,255,0.10)",
    # buttons (both modes)
    button_primary_background_fill="linear-gradient(135deg, #10B981 0%, #0D9488 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #34D399 0%, #14B8A6 100%)",
    button_primary_background_fill_dark="linear-gradient(135deg, #10B981 0%, #0D9488 100%)",
    button_primary_background_fill_hover_dark="linear-gradient(135deg, #34D399 0%, #14B8A6 100%)",
    button_primary_text_color="#04120E",
    button_primary_text_color_dark="#04120E",
    button_primary_border_color="rgba(16,185,129,0.4)",
    button_primary_border_color_dark="rgba(16,185,129,0.4)",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

* { font-family: 'Plus Jakarta Sans', ui-sans-serif, system-ui, sans-serif !important; }

/* ---------- Color tokens ---------- */
:root {
  --bg: #F5EFE4;
  --glow1: rgba(16,185,129,0.14);
  --glow2: rgba(217,180,120,0.28);
  --heading: #1F1D18;
  --text: #2E2C26;
  --muted: #6B665A;
  --faint: #8A8476;
  --card: rgba(255,255,255,0.55);
  --card-hover: rgba(255,255,255,0.9);
  --border: rgba(70,55,30,0.14);
  --border-strong: rgba(13,148,136,0.45);
  --pill-bg: rgba(13,148,136,0.10);
  --pill-text: #0F766E;
  --pill-border: rgba(13,148,136,0.30);
  --accent-a: #059669;
  --accent-b: #0D9488;
  --shadow: 0 8px 28px rgba(90,70,30,0.10);
  --shadow-hover: 0 14px 36px rgba(90,70,30,0.18);
}
.dark {
  --bg: #0A0F0D;
  --glow1: rgba(16,185,129,0.16);
  --glow2: rgba(20,184,166,0.10);
  --heading: #F8FAFA;
  --text: #E5E9E8;
  --muted: #94A3A0;
  --faint: #5B6B67;
  --card: rgba(255,255,255,0.035);
  --card-hover: rgba(255,255,255,0.07);
  --border: rgba(255,255,255,0.08);
  --border-strong: rgba(52,211,153,0.45);
  --pill-bg: rgba(16,185,129,0.12);
  --pill-text: #6EE7B7;
  --pill-border: rgba(16,185,129,0.28);
  --accent-a: #34D399;
  --accent-b: #2DD4BF;
  --shadow: 0 8px 32px rgba(0,0,0,0.35);
  --shadow-hover: 0 14px 40px rgba(0,0,0,0.55);
}

/* ---------- Animations ---------- */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(14px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes shimmer {
  0%   { background-position: 0% 50%; }
  100% { background-position: 200% 50%; }
}
@keyframes logoGlow {
  0%, 100% { box-shadow: 0 0 10px rgba(16,185,129,0.35); }
  50%      { box-shadow: 0 0 22px rgba(16,185,129,0.7); }
}

.gradio-container {
  background:
    radial-gradient(ellipse 900px 500px at 20% -10%, var(--glow1), transparent 60%),
    radial-gradient(ellipse 700px 500px at 100% 0%, var(--glow2), transparent 55%),
    var(--bg) !important;
  transition: background-color .35s ease, color .35s ease;
}

/* ---- Top brand bar ---- */
#topbar {
  display: flex; align-items: center; justify-content: space-between;
  padding: 6px 4px 18px 4px; margin-bottom: 4px;
  border-bottom: 1px solid var(--border);
  animation: fadeUp .6s ease both;
}
#topbar .brand { display:flex; align-items:center; gap:10px; }
#topbar .logo-mark {
  width: 30px; height: 30px; border-radius: 9px;
  background: linear-gradient(135deg, #10B981, #0D9488);
  display:flex; align-items:center; justify-content:center;
  font-weight: 800; font-size: 14px; color: #04120E;
  animation: logoGlow 3.2s ease-in-out infinite;
  transition: transform .25s ease;
}
#topbar .brand:hover .logo-mark { transform: rotate(-6deg) scale(1.08); }
#topbar .brand-name { font-weight: 700; font-size: 15px; color: var(--heading); letter-spacing: -0.01em; }
#topbar .right { display:flex; align-items:center; gap:10px; }
#topbar .badge-pill {
  padding: 5px 14px; border-radius: 999px; font-size: 12px; font-weight: 600;
  background: var(--pill-bg); color: var(--pill-text); border: 1px solid var(--pill-border);
}

/* ---- Theme toggle ---- */
#theme-toggle {
  width: 36px; height: 36px; border-radius: 50%; cursor: pointer;
  display:inline-flex; align-items:center; justify-content:center;
  background: var(--card); border: 1px solid var(--border); color: var(--heading);
  font-size: 16px; line-height: 1; padding: 0;
  transition: transform .3s ease, background .25s ease, border-color .25s ease, box-shadow .25s ease;
}
#theme-toggle:hover {
  transform: rotate(20deg) scale(1.1);
  background: var(--card-hover); border-color: var(--border-strong); box-shadow: var(--shadow);
}
#theme-toggle:active { transform: scale(0.94); }
#theme-toggle .ico-sun { display: none; }
.dark #theme-toggle .ico-sun { display: inline; }
.dark #theme-toggle .ico-moon { display: none; }

/* ---- Hero ---- */
#hero { text-align: center; padding: 18px 8px 10px 8px; align-items: center; }
#hero > *, #hero .prose, #hero .md, #hero .markdown { width: 100%; max-width: 100%; text-align: center !important; margin-left: auto; margin-right: auto; }
#hero h1 { text-align: center !important; width: 100%; margin-left: auto; margin-right: auto; }
#hero p.sub { text-align: center !important; }
#hero h1 {
  font-weight: 800; font-size: 34px; letter-spacing: -0.02em;
  color: var(--heading); margin-bottom: 10px; line-height: 1.15;
  animation: fadeUp .7s ease .1s both;
}
#hero h1 .accent {
  display: inline-block; padding-bottom: .08em;
  color: var(--accent-a);
}
@supports ((-webkit-background-clip: text) or (background-clip: text)) {
  #hero h1 .accent {
    background-image: linear-gradient(90deg, var(--accent-a), var(--accent-b), var(--accent-a));
    background-size: 200% auto;
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; color: transparent;
    animation: shimmer 5s linear infinite;
  }
}
#hero p.sub {
  color: var(--muted); font-size: 15.5px; max-width: 660px; margin: 0 auto 18px auto; line-height: 1.55;
  animation: fadeUp .7s ease .2s both;
}
.feature-row { display:flex; flex-wrap:wrap; gap:8px; justify-content:center; align-items:center; width:100%; margin: 0 auto 6px auto; }
.feature-pill {
  padding: 7px 14px; border-radius: 999px; font-size: 12.5px; font-weight: 600;
  background: var(--card); border: 1px solid var(--border); color: var(--text);
  display:inline-flex; align-items:center; gap:6px; cursor: default;
  animation: fadeUp .7s ease both;
  transition: transform .25s ease, border-color .25s ease, background .25s ease, box-shadow .25s ease;
}
.feature-pill:nth-child(1) { animation-delay: .30s; }
.feature-pill:nth-child(2) { animation-delay: .38s; }
.feature-pill:nth-child(3) { animation-delay: .46s; }
.feature-pill:hover {
  transform: translateY(-3px);
  border-color: var(--border-strong); background: var(--card-hover); box-shadow: var(--shadow);
}

.tab-desc { color: var(--muted); font-size: 13.5px; margin-bottom: 14px; }
.badge-standard {
  display:inline-block; padding:3px 12px; border-radius:999px;
  background: var(--pill-bg); color: var(--pill-text); border: 1px solid var(--pill-border);
  font-size:12px; font-weight:700; margin-bottom:8px;
}
.badge-drift {
  display:inline-block; padding:3px 12px; border-radius:999px;
  background: rgba(244,63,94,0.10); color:#E11D48; border: 1px solid rgba(244,63,94,0.25);
  font-size:12px; font-weight:700; margin-bottom:8px;
}
.dark .badge-drift { color:#FB7185; }

/* ---- Model column tags ---- */
.model-col {
  border-radius: 18px; padding: 14px 16px 18px 16px;
  background: var(--card); border: 1px solid var(--border);
  transition: transform .25s ease, box-shadow .25s ease, background .25s ease;
}
.model-col:hover { transform: translateY(-3px); box-shadow: var(--shadow-hover); background: var(--card-hover); }
.model-col.baseline { border-top: 3px solid #FB7185; }
.model-col.tuned { border-top: 3px solid #34D399; }
.model-tag {
  display:inline-flex; align-items:center; gap:6px;
  font-size: 11.5px; font-weight: 800; letter-spacing: 0.04em; text-transform: uppercase;
  padding: 4px 10px; border-radius: 8px; margin-bottom: 8px;
}
.model-tag.baseline { background: rgba(244,63,94,0.12); color: #E11D48; }
.model-tag.tuned { background: rgba(16,185,129,0.14); color: #047857; }
.dark .model-tag.baseline { color: #FDA4AF; }
.dark .model-tag.tuned { color: #6EE7B7; }
.model-sub { color: var(--faint); font-size: 12.5px; margin-bottom: 10px; line-height: 1.4; }

/* ---- Cards (plots, inputs, tables) ---- */
.gradio-container .block {
  transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease, background .25s ease;
}
.gradio-container .block:hover {
  border-color: var(--border-strong);
  box-shadow: var(--shadow-hover);
}
.gradio-container .block:has(.js-plotly-plot):hover { transform: translateY(-2px); }

/* Section headings */
.gradio-container h2 { color: var(--heading); animation: fadeUp .6s ease both; }
.gradio-container h2, .gradio-container h3 { letter-spacing: -0.01em; }

/* ---- Tabs ---- */
.tabs > .tab-nav { border-bottom: 1px solid var(--border) !important; gap: 4px; }
.tabs > .tab-nav button {
  font-weight: 700 !important; font-size: 14.5px !important; color: var(--muted) !important; border: none !important;
  background: transparent !important; padding: 10px 20px !important;
  transition: color .2s ease, transform .2s ease;
}
.tabs > .tab-nav button:hover { color: var(--accent-a) !important; transform: translateY(-1px); }
.tabs > .tab-nav button.selected {
  color: var(--accent-a) !important; border-bottom: 2px solid var(--accent-a) !important;
}

/* ---- Buttons ---- */
button { transition: transform .2s ease, box-shadow .2s ease, filter .2s ease, background .2s ease !important; }
button.primary { font-weight: 700 !important; letter-spacing: -0.01em; }
button.primary:hover { transform: translateY(-2px); box-shadow: 0 8px 22px rgba(16,185,129,0.35); }
button.secondary:hover { transform: translateY(-2px); box-shadow: var(--shadow); }
button:active { transform: translateY(0) scale(0.97) !important; }

/* ---- Footer ---- */
#footerbar {
  text-align:center; padding: 22px 0 6px 0; color: var(--faint); font-size: 12.5px;
  border-top: 1px solid var(--border); margin-top: 18px;
  transition: color .25s ease;
}
#footerbar:hover { color: var(--accent-a); }

footer { display: none !important; }

/* ---- Typography (clear, consistent sizes) ---- */
.gradio-container { font-size: 16px; }
.gradio-container .prose, .gradio-container .md, .gradio-container .markdown {
  font-size: 16px !important; line-height: 1.7 !important; color: var(--text);
}
.gradio-container .prose strong, .gradio-container .md strong { color: var(--heading); font-weight: 700; }
.gradio-container h2 {
  font-size: 28px !important; font-weight: 800 !important; margin: 36px 0 12px 0 !important; color: var(--heading);
}
.gradio-container h3 { font-size: 20px !important; font-weight: 700 !important; }
span[data-testid="block-info"] {
  font-size: 14px !important; font-weight: 600 !important; color: var(--muted) !important;
}
.gradio-container input, .gradio-container textarea, .gradio-container select {
  font-size: 15.5px !important;
}
.gradio-container button { font-size: 15px; }
.gradio-container button.primary { font-size: 16px; padding-top: 12px; padding-bottom: 12px; }

/* ---- Tables (Query result etc.) ---- */
.gradio-container .prose table, .gradio-container .md table {
  width: 100%; border-collapse: separate !important; border-spacing: 0;
  border: 1px solid var(--border) !important; border-radius: 14px; overflow: hidden;
  font-size: 15px; margin: 14px 0;
}
.gradio-container .prose th, .gradio-container .md th {
  background: var(--pill-bg); color: var(--heading); font-weight: 700; text-align: left;
  padding: 12px 16px !important; border: none !important; border-bottom: 1px solid var(--border) !important;
}
.gradio-container .prose td, .gradio-container .md td {
  padding: 12px 16px !important; border: none !important; border-bottom: 1px solid var(--border) !important;
}
.gradio-container .prose tr:last-child td, .gradio-container .md tr:last-child td { border-bottom: none !important; }
.gradio-container .prose tbody tr, .gradio-container .md tbody tr { transition: background .2s ease; }
.gradio-container .prose tbody tr:hover td, .gradio-container .md tbody tr:hover td { background: var(--card-hover); }

/* ---- Plots: theme-aware text + grid (works in light and dark) ---- */
.js-plotly-plot .main-svg text:not(.number) { fill: var(--text) !important; }
.js-plotly-plot .gridlayer path, .js-plotly-plot .zerolinelayer path { stroke: var(--border) !important; }
.js-plotly-plot .modebar { opacity: 0; transition: opacity .2s ease; }
.js-plotly-plot:hover .modebar { opacity: 1; }

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
"""

TOGGLE_JS = "const d=document.body.classList.toggle('dark');try{localStorage.setItem('sd-theme',d?'dark':'light')}catch(e){}"

# Default = light beige; restores the last choice saved by the toggle.
LOAD_JS = """() => {
  let t = null;
  try { t = localStorage.getItem('sd-theme'); } catch (e) {}
  document.body.classList.toggle('dark', t === 'dark');
}"""

with gr.Blocks(title="Systemic Drift Lab · Ethical Intelligence") as demo:
    gr.HTML(
        f"""
        <div id="topbar">
          <div class="brand">
            <div class="logo-mark">EI</div>
            <div class="brand-name">Ethical Intelligence</div>
          </div>
          <div class="right">
            <div class="badge-pill">Internal Demo</div>
            <button id="theme-toggle" type="button" title="Toggle light / dark"
                    onclick="{TOGGLE_JS}">
              <span class="ico-moon">🌙</span><span class="ico-sun">☀️</span>
            </button>
          </div>
        </div>
        """
    )
    with gr.Column(elem_id="hero"):
        gr.Markdown(
            "<h1>One input. Two models. <span class='accent'>See the drift live.</span></h1>"
            "<p class='sub'>A controlled model-comparison environment that places the baseline "
            "and lifecycle-tuned models side by side, making systematic drift and its "
            "contrastive correction visible in embedding space.</p>"
            "<div class='feature-row'>"
            "<span class='feature-pill'>⚖️ Side-by-Side</span>"
            "<span class='feature-pill'>🧬 Contrastive Loss</span>"
            "<span class='feature-pill'>🔁 Lifecycle Fine-tune</span>"
            "</div>"
        )

    build_embedding_tab()

    gr.HTML("<div id='footerbar'>Built by Ethical Intelligence · Systemic Drift Lab</div>")

    demo.load(fn=None, inputs=None, outputs=None, js=LOAD_JS)

if __name__ == "__main__":
    demo.launch(share=True, theme=theme, css=CSS)