# # embedding_tab.py -- "Embedding Space" tab (A vs B). In app.py:
# #     with gr.Tab("Embedding Space"):
# #         build_embedding_tab()
# import json
# import numpy as np, pandas as pd, torch, gradio as gr
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# from geometry import project, score
# import math
# LATEX = [{"left": "$$", "right": "$$", "display": True}, {"left": "$", "right": "$", "display": False}]





# D = json.load(open("demo_data/bundle.json", encoding="utf-8"))
# MODELS = ["A", "B"]
# NAMES = {"A": "Model A (standard only)", "B": "Model B (drift + SupCon)"}
# DIRS = {"A": "models/chat_model_a", "B": "models/chat_model_b"}
# META, LC = D["ref"]["meta"], D["lifecycle"]
# NEP = max(len(LC) - 1, 0)
# POS_C, NEG_C = "#1f77b4", "#ff7f0e"  # SupCon positive pair (pull) / negative pair (push)

# _all = np.vstack([np.array(D["ref"][m]) for m in MODELS] + [np.array(e["ref"]) for e in LC] +
#                  [np.array(e["q"]) for e in LC] + [np.array([r["models"][m]["xy"] for r in D["queries"]]) for m in MODELS])
# XR = [_all[:, 0].min() - .05, _all[:, 0].max() + .05]
# YR = [_all[:, 1].min() - .05, _all[:, 1].max() + .05]
# _cache = {}


# def _load(m):
#     if m not in _cache:
#         _cache[m] = (T5Tokenizer.from_pretrained(DIRS[m]),
#                      T5ForConditionalGeneration.from_pretrained(DIRS[m]).eval())
#     return _cache[m]


# def _live(text):
#     out = {}
#     for m in MODELS:
#         tok, mod = _load(m)
#         e = tok("respond: " + text, return_tensors="pt", truncation=True, max_length=64)
#         with torch.no_grad():
#             emb = mod.encoder(**e).last_hidden_state.mean(1).numpy()
#             g = tok.decode(mod.generate(**e, max_length=24, num_beams=2)[0], skip_special_tokens=True)
#         b = np.load(f"demo_data/basis_{m}.npz")
#         out[m] = {"reply": g, "correct": None, "xy": project(emb, b)[0].tolist(), "score": float(score(emb, b)[0])}
#     return out


# def _ref_traces(fig, xy, col, legend):
#     xy = np.array(xy)
#     for lab, c in ((1, "#2ca02c"), (0, "#d62728")):
#         for st, sym in (("standard", "circle"), ("drift", "diamond")):
#             idx = [j for j, r in enumerate(META) if r["label"] == lab and r["style"] == st]
#             fig.add_trace(go.Scatter(x=xy[idx, 0], y=xy[idx, 1], mode="markers",
#                                      name=f"{'positive' if lab else 'negative'} · {st}", legendgroup=f"{lab}{st}",
#                                      showlegend=legend, marker=dict(color=c, symbol=sym, size=6, opacity=.5)),
#                           row=1, col=col)
#     fig.add_vline(x=0, line_dash="dot", line_color="gray", row=1, col=col)


# def _star(fig, xy, col, legend):
#     fig.add_trace(go.Scatter(x=[xy[0]], y=[xy[1]], mode="markers", name="query / anchor", legendgroup="q",
#                              showlegend=legend, marker=dict(symbol="star", size=20, color="gold",
#                                                             line=dict(color="black", width=1.5))), row=1, col=col)


# def _links(fig, a, xy, idx, color, dash, name, col, legend):
#     xs, ys = [], []
#     for j in idx:
#         xs += [a[0], xy[j][0], None]; ys += [a[1], xy[j][1], None]
#     fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, dash=dash, width=1.6), name=name,
#                              legendgroup=name, showlegend=legend, hoverinfo="skip"), row=1, col=col)
#     fig.add_trace(go.Scatter(x=[xy[j][0] for j in idx], y=[xy[j][1] for j in idx], mode="markers",
#                              marker=dict(size=13, color="rgba(0,0,0,0)", line=dict(color=color, width=2.5)),
#                              showlegend=False, hoverinfo="skip"), row=1, col=col)


# # ---------------- Section 1: how drift appears ----------------
# def drift_fig():
#     fig = make_subplots(rows=1, cols=1)
#     xy = np.array(D["ref"]["A"])
#     _ref_traces(fig, xy, 1, True)
#     for lab, c in ((1, "#2ca02c"), (0, "#d62728")):
#         cs = [xy[[j for j, r in enumerate(META) if r["label"] == lab and r["style"] == st]].mean(0)
#               for st in ("standard", "drift")]
#         fig.add_annotation(x=cs[1][0], y=cs[1][1], ax=cs[0][0], ay=cs[0][1], xref="x", yref="y", axref="x",
#                            ayref="y", showarrow=True, arrowhead=3, arrowwidth=3, arrowcolor=c)
#     fig.update_xaxes(range=XR, title_text="sentiment axis (fit on Model A, standard only)")
#     fig.update_yaxes(range=YR, title_text="PC2 (⟂ axis)")
#     fig.update_layout(height=420, margin=dict(t=50, b=40), legend=dict(orientation="h", y=-0.25),
#                       title="Model A · arrow = standard centroid → drift centroid (same sentiment)")
#     return fig



# AMB = 0.05  # sentiment-axis distance from 0 below which a point counts as ambiguous


# def _bad_pct(style):
#     xy = np.array(D["ref"]["A"])
#     idx = [j for j, r in enumerate(META) if r["style"] == style]
#     x = xy[idx, 0]
#     lab = np.array([META[j]["label"] for j in idx])
#     sgn = np.where(lab == 1, x, -x)  # > 0 means correct side
#     return 100 * float((sgn < AMB).mean())


# def gauge_fig():
#     fig = make_subplots(rows=2, cols=1, specs=[[{"type": "indicator"}], [{"type": "indicator"}]],
#                         vertical_spacing=0.3)
#     for i, (st, nm) in enumerate((("standard", "Standard text"), ("drift", "Drift text")), 1):
#         v = _bad_pct(st)
#         col = "#2ca02c" if v < 20 else "#e6a700" if v < 50 else "#d62728"
#         fig.add_trace(go.Indicator(
#             mode="gauge+number", value=v,
#             number={"suffix": "%", "font": {"size": 46, "color": col}},
#             title={"text": f"<b>{nm}</b><br><span style='font-size:12px;color:#666'>"
#                            f"texts on wrong / unclear side</span>"},
#             gauge={"axis": {"range": [0, 100], "tickvals": [0, 20, 50, 100], "tickcolor": "#999"},
#                    "bar": {"color": col, "thickness": 0.3}, "borderwidth": 0, "bgcolor": "white",
#                    "steps": [{"range": [0, 20], "color": "#d9f0d9"},
#                              {"range": [20, 50], "color": "#fff1bf"},
#                              {"range": [50, 100], "color": "#f7cfcf"}],
#                    "threshold": {"line": {"color": "#222", "width": 3}, "thickness": 0.85, "value": v}}), i, 1)
#     fig.update_layout(height=520, margin=dict(t=80, b=30, l=30, r=30))
#     return fig
#     return fig

# def acc_fig():
#     M, ps = D["metrics"], D["pool_stats"]["error_by_subtype"]
#     fig = make_subplots(rows=1, cols=2, subplot_titles=["Tone accuracy (reference set)",
#                                                         "Error rate by subtype (novel pool)"])
#     for k, nm, c in (("tone_acc_standard", "standard", "#7f7f7f"), ("tone_acc_drift", "drift", "#9467bd")):
#         fig.add_trace(go.Bar(x=[NAMES[m] for m in MODELS], y=[M[m][k] for m in MODELS], name=nm, marker_color=c), 1, 1)
#     subs = list(ps["A"])
#     for m, c in (("A", "#d62728"), ("B", "#2ca02c")):
#         fig.add_trace(go.Bar(x=subs, y=[ps[m][s] for s in subs], name=NAMES[m], marker_color=c), 1, 2)
#     fig.update_layout(barmode="group", height=340, margin=dict(t=50, b=40), legend=dict(orientation="h", y=-0.3))
#     return fig


# # ---------------- Section 2: query walkthrough ----------------
# def scatter_fig(q):
#     fig = make_subplots(rows=1, cols=2, subplot_titles=[NAMES[m] for m in MODELS])
#     pts = np.array([q[m]["xy"] for m in MODELS])
#     xr = [min(XR[0], pts[:, 0].min() - .05), max(XR[1], pts[:, 0].max() + .05)]
#     yr = [min(YR[0], pts[:, 1].min() - .05), max(YR[1], pts[:, 1].max() + .05)]
#     for i, m in enumerate(MODELS, 1):
#         _ref_traces(fig, D["ref"][m], i, i == 1)
#         _star(fig, q[m]["xy"], i, i == 1)
#         fig.update_xaxes(range=xr, title_text="sentiment axis (shared frame)", row=1, col=i)
#         fig.update_yaxes(range=yr, title_text="PC2 (⟂ axis)", row=1, col=i)
#     fig.update_layout(height=430, margin=dict(t=50, b=40), legend=dict(orientation="h", y=-0.25))
#     return fig


# def bars_fig(q):
#     s = [q[m]["score"] for m in MODELS]
#     fig = go.Figure(go.Bar(x=[NAMES[m] for m in MODELS], y=s,
#                            marker_color=["#2ca02c" if v > 0 else "#d62728" for v in s],
#                            text=[f"{v:+.3f}" for v in s], textposition="outside"))
#     fig.update_layout(height=300, margin=dict(t=50, b=40), yaxis_title="score",
#                       title="Sentiment score  cos(q, μ₊) − cos(q, μ₋)   (>0 positive side, ≈0 ambiguous)")
#     return fig


# # ---------------- Section 3: lifecycle (anchor / positives / negatives) ----------------
# def life_fig(qi, ep):
#     if not LC:
#         return go.Figure()
#     ep = min(int(ep), len(LC) - 1)
#     ttl = ["Epoch 0 · Model A", f"Epoch {ep} · " + ("Model B" if ep == NEP else "mid-tuning")]
#     fig = make_subplots(rows=1, cols=2, subplot_titles=ttl)
#     for i, e in enumerate((0, ep), 1):
#         E = LC[e]
#         _ref_traces(fig, E["ref"], i, i == 1)
#         if qi is not None:
#             a, r = E["q"][qi], D["queries"][qi]
#             _links(fig, a, E["ref"], r["pos"], POS_C, "solid", "positive pair (pull)", i, i == 1)
#             _links(fig, a, E["ref"], r["neg"], NEG_C, "dash", "negative pair (push)", i, i == 1)
#             _star(fig, a, i, i == 1)
#         fig.update_xaxes(range=XR, title_text="sentiment axis (shared frame)", row=1, col=i)
#         fig.update_yaxes(range=YR, row=1, col=i)
#     fig.update_layout(height=450, margin=dict(t=50, b=40), legend=dict(orientation="h", y=-0.25))
#     return fig


# def cos_fig(qi):
#     fig = go.Figure()
#     if LC:
#         xs = list(range(len(LC)))
#         fig.add_trace(go.Scatter(x=xs, y=[np.mean(e["cos_pos"]) for e in LC], name="all curated · positives",
#                                  line=dict(color=POS_C, dash="dot")))
#         fig.add_trace(go.Scatter(x=xs, y=[np.mean(e["cos_neg"]) for e in LC], name="all curated · negatives",
#                                  line=dict(color=NEG_C, dash="dot")))
#         if qi is not None:
#             fig.add_trace(go.Scatter(x=xs, y=[e["cos_pos"][qi] for e in LC], name="this anchor · positives",
#                                      line=dict(color=POS_C, width=3)))
#             fig.add_trace(go.Scatter(x=xs, y=[e["cos_neg"][qi] for e in LC], name="this anchor · negatives",
#                                      line=dict(color=NEG_C, width=3)))
#     fig.update_layout(height=320, margin=dict(t=50, b=40), xaxis_title="epoch", yaxis_title="mean cosine (full space)",
#                       title="Anchor ↔ positives / negatives similarity across epochs", xaxis=dict(dtick=1))
#     return fig


# def life_views(qi, ep):
#     return life_fig(qi, ep)


# def _qi(sel, custom):
#     if (custom or "").strip():
#         return None
#     return next((i for i, r in enumerate(D["queries"]) if r["text"] == sel), None)


# def views(sel, custom, ep):
#     custom = (custom or "").strip()
#     qi = _qi(sel, custom)
#     if custom:
#         q = _live(custom); head = f"**Custom query:** {custom}"
#     else:
#         rec = D["queries"][qi]; q = rec["models"]
#         head = (f"**Query:** {sel}  \n**True sentiment:** {'positive' if rec['label'] else 'negative'} · "
#                 f"**Type:** {rec['subtype']} · **Bucket:** {rec['category']}")
#     ok = lambda c: "✅" if c else ("❌" if c is False else "—")
#     rows = "\n".join(f"| {NAMES[m]} | {q[m]['reply']} | {q[m]['score']:+.3f} | {ok(q[m]['correct'])} |" for m in MODELS)
#     md = head + "\n\n| Model | Reply | Score | Tone |\n|---|---|---|---|\n" + rows
#     return md, scatter_fig(q), bars_fig(q), life_views(qi, ep)


# def loss_fig():
#     L = D["losses"]
#     fig = make_subplots(rows=1, cols=2, subplot_titles=["Generation loss", "SupCon loss"])
#     if L:
#         w = 10
#         for col, k in ((1, "gen"), (2, "con")):
#             v = np.array(L[k]); st = np.array(L["step"])
#             fig.add_trace(go.Scatter(x=st, y=v, line=dict(color="#1f77b4", width=1), opacity=.3, showlegend=False), 1, col)
#             if len(v) >= w:
#                 fig.add_trace(go.Scatter(x=st[w - 1:], y=np.convolve(v, np.ones(w) / w, mode="valid"),
#                                          line=dict(color="#1f77b4", width=3), showlegend=False), 1, col)
#     fig.update_xaxes(title_text="step"); fig.update_layout(height=320, margin=dict(t=50, b=40))
#     return fig


# def _metrics_df():
#     rows = [("tone_acc_standard", "Tone accuracy, standard ↑"), ("tone_acc_drift", "Tone accuracy, drift ↑"),
#             ("silhouette", "Silhouette (cosine) ↑"), ("silhouette_drift", "Silhouette, drift only ↑"),
#             ("intra_cos", "Intra-class cosine ↑"), ("inter_cos", "Inter-class cosine ↓"),
#             ("cross_style_align", "Cross-style centroid alignment ↑"), ("alignment", "Alignment (Wang&Isola) ↓"),
#             ("uniformity", "Uniformity (Wang&Isola) ↓"), ("knn_std_to_drift", "kNN acc, fit std → test drift ↑")]
#     return pd.DataFrame([[lab] + [round(D["metrics"][m][k], 3) for m in MODELS] for k, lab in rows],
#                         columns=["Metric"] + [NAMES[m] for m in MODELS])


# # ---------------- Section 4: SupCon math + live toy demo ----------------
# FORMULA = r"""
# $$\mathcal{L}_i=-\frac{1}{|P(i)|}\sum_{p\in P(i)}\log\frac{\exp(z_i\!\cdot\! z_p/\tau)}{\sum_{a\neq i}\exp(z_i\!\cdot\! z_a/\tau)}$$

# - $z$ = unit vector of a text, so $z_i\cdot z_p$ is their cosine similarity. $p$ = same-sentiment texts, $a$ = all other texts.
# - **Pull:** the numerator has the positives. The loss drops when $z_i\cdot z_p$ grows.
# - **Push:** the denominator also has the negatives. The loss drops when their similarity shrinks.
# - **τ:** small τ sharpens the softmax, so the closest (hardest) negatives get the biggest push.

# **How the update happens** (gradient descent on the anchor, $w_a$ = softmax share of point $a$ in the denominator):

# $$z_i \leftarrow z_i+\frac{\eta}{\tau}\Big(\underbrace{\overline{z_p}}_{\text{toward positives}}-\underbrace{\textstyle\sum_a w_a z_a}_{\text{away from crowded / close points}}\Big)$$
# """


# def _loss(theta, tau):
#     z = torch.stack([torch.cos(theta), torch.sin(theta)], 1)
#     cs = z[1:] @ z[0]
#     s = cs / tau
#     return -(s[:2] - torch.logsumexp(s, 0)).mean(), cs, torch.softmax(s, 0)


# def toy_reset(tau):
#     th = [math.radians(d) for d in (90, 200, 320, 60, 130, 20)]  # anchor, P1, P2, N1, N2, N3
#     L = float(_loss(torch.tensor(th, dtype=torch.float64), tau)[0])
#     return {"th": th, "hist": [L]}


# def toy_step(state, tau, n):
#     th, hist = torch.tensor(state["th"], dtype=torch.float64), list(state["hist"])
#     for _ in range(int(n)):
#         th = th.clone().requires_grad_(True)
#         (g,) = torch.autograd.grad(_loss(th, tau)[0], th)
#         th = (th - (0.05 * g).clamp(-0.3, 0.3)).detach()
#         hist.append(float(_loss(th, tau)[0]))
#     return {"th": th.tolist(), "hist": hist}


# def toy_view(state, tau):
#     th = torch.tensor(state["th"], dtype=torch.float64)
#     L, cs, w = _loss(th, tau)
#     cs, w = cs.tolist(), w.tolist()
#     names = ["P1", "P2", "N1", "N2", "N3"]
#     cols = ["#1f77b4"] * 2 + ["#ff7f0e"] * 3
#     xy = np.stack([np.cos(state["th"]), np.sin(state["th"])], 1)
#     fig = make_subplots(rows=1, cols=2, column_widths=[.5, .5],
#                         subplot_titles=["Toy unit circle (2D hypersphere)", "SupCon loss per step"])
#     t = np.linspace(0, 2 * np.pi, 200)
#     fig.add_trace(go.Scatter(x=np.cos(t), y=np.sin(t), mode="lines", line=dict(color="#bbb"), showlegend=False), 1, 1)
#     for k in range(5):
#         fig.add_trace(go.Scatter(x=[xy[0, 0], xy[k + 1, 0]], y=[xy[0, 1], xy[k + 1, 1]], mode="lines",
#                                  line=dict(color=cols[k], dash="solid" if k < 2 else "dash", width=2),
#                                  showlegend=False, hoverinfo="skip"), 1, 1)
#         fig.add_trace(go.Scatter(x=[xy[k + 1, 0]], y=[xy[k + 1, 1]], mode="markers+text", text=[names[k]],
#                                  textposition="top center", showlegend=False,
#                                  marker=dict(size=14, color=cols[k])), 1, 1)
#     fig.add_trace(go.Scatter(x=[xy[0, 0]], y=[xy[0, 1]], mode="markers+text", text=["anchor"], textposition="top center",
#                              showlegend=False, marker=dict(symbol="star", size=22, color="gold",
#                                                            line=dict(color="black", width=1.5))), 1, 1)
#     fig.update_xaxes(range=[-1.3, 1.3], visible=False, row=1, col=1)
#     fig.update_yaxes(range=[-1.3, 1.3], visible=False, scaleanchor="x", scaleratio=1, row=1, col=1)
#     fig.add_trace(go.Scatter(x=list(range(len(state["hist"]))), y=state["hist"], mode="lines+markers",
#                              line=dict(color="#1f77b4"), showlegend=False), 1, 2)
#     fig.update_xaxes(title_text="step", row=1, col=2); fig.update_yaxes(title_text="loss", row=1, col=2)
#     fig.update_layout(height=420, margin=dict(t=50, b=30))
#     rows = "\n".join(f"| {names[k]} | {'positive' if k < 2 else 'negative'} | {cs[k]:+.2f} | {w[k]:.0%} |" for k in range(5))
#     md = (f"**Loss = {float(L):.3f}** · avg cos to positives **{np.mean(cs[:2]):+.2f}** · "
#           f"avg cos to negatives **{np.mean(cs[2:]):+.2f}**\n\n"
#           f"| Point | Type | cos(anchor, point) | share of denominator |\n|---|---|---|---|\n{rows}\n\n"
#           "*Bigger share = bigger push. The closest negative gets the most.*")
#     return md, fig


# def toy_do(state, tau, n):
#     state = toy_step(state, tau, n)
#     return (state,) + toy_view(state, tau)


# def toy_retau(state, tau):
#     th = torch.tensor(state["th"], dtype=torch.float64)
#     state = {"th": state["th"], "hist": state["hist"][:-1] + [float(_loss(th, tau)[0])]}
#     return (state,) + toy_view(state, tau)


# def toy_restart(tau):
#     state = toy_reset(tau)
#     return (state,) + toy_view(state, tau)



# GLOSSARY = """
# **Setup.** Encoder output is mean-pooled into **z** and L2-normalized → points live on a **unit hypersphere**.
# Total loss: **L = L_gen + λ · L_SupCon**, λ = 0.3, temperature τ = 0.1.

# **SupCon** (Khosla et al., 2020) for each **anchor** i:
# `L_i = −(1/|P(i)|) Σ_{p∈P(i)} log [ exp(z_i·z_p / τ) / Σ_{a≠i} exp(z_i·z_a / τ) ]`
# - **Positives P(i):** other samples in the batch with the same sentiment label (standard *and* drift → pulls slang/emoji toward standard phrasing of the same sentiment).
# - **Negatives:** samples with the opposite label (pushed apart). Low τ sharpens the penalty on hard negatives.

# **Anchor view (Section 3).** Training uses in-batch positives/negatives. For illustration we pick, per anchor, the K hard positives (least similar same-label texts) and K hard negatives (most similar opposite-label texts) in Model A's space and track their cosine over epochs.

# **Projection.** Linear: x = sentiment axis (μ₊ − μ₋), y = top PC orthogonal to it; fit once on Model A using **standard reference points only** and reused for Model B and every epoch, so all panels share one frame. Score uses each model's own standard-only centroids.

# **Metrics.** *Silhouette / intra-inter cosine*: cluster quality. *Cross-style alignment*: cosine between standard and drift centroids of the same class. *Alignment* / *Uniformity*: Wang & Isola, 2020. *kNN std→drift*: does the standard-style geometry classify drift points?

# **Refs.** Khosla et al. 2020 (SupCon) · Wang & Isola 2020 · Gao et al. 2021 (SimCSE) · Cha et al. 2021 (Co2L).
# """


# def build_embedding_tab():
#     ps = D["pool_stats"]; er = ps["error_rate"]
#     qs = [r["text"] for r in D["queries"]]
#     init = views(qs[0], "", NEP)

#     gr.Markdown("## 1 · How systematic drift appears")
#     gr.Markdown(f"Model A is trained on standard phrasing only. Drift text (slang, emoji, sarcasm) of the *same* sentiment "
#                 f"lands away from its standard counterpart, and tone accuracy drops.")
#     # gr.Plot(drift_fig())
#     with gr.Row():
#         with gr.Column(scale=3):
#             gr.Plot(drift_fig())
#         with gr.Column(scale=2):
#             gr.Plot(gauge_fig())
#             gr.Markdown("Green = model places the text clearly on its sentiment side. "
#             "Red = text lands on the wrong side or too near the middle to tell.")
#     # gr.Plot(acc_fig())

#     gr.Markdown("## 2 · Same input, two models")
#     gr.Markdown(f"**Disclosure:** {ps['n_selected']} queries curated from {ps['n_candidates']} novel candidates "
#                 f"(new subjects/templates, deduped vs. training; incl. held-out sarcasm structures). Over the full pool, "
#                 f"tone error — A: {er['A']:.0%}, B: {er['B']:.0%}. Selected buckets: {ps['selected_buckets']}.")
#     with gr.Row():
#         sel = gr.Dropdown(qs, value=qs[0], label="Curated query")
#         custom = gr.Textbox(label="…or type your own (overrides; loads models on first use)")
#     btn = gr.Button("Project query", variant="primary")
#     info, sp, bars = gr.Markdown(init[0]), gr.Plot(init[1]), gr.Plot(init[2])

#     gr.Markdown("## 3 · Contrastive lifecycle: anchor, positives, negatives")
#     ep = gr.Slider(0, NEP, step=1, value=NEP, label="Epoch (0 = Model A)")
#     lp = gr.Plot(init[3])

#     ins, outs = [sel, custom, ep], [info, sp, bars, lp]
#     btn.click(views, ins, outs); sel.change(views, ins, outs)
#     ep.change(lambda s, c, e: life_views(_qi(s, c), e), ins, lp)
    
#     gr.Markdown("## 4 · The math: how SupCon pulls and pushes", latex_delimiters=LATEX)
#     gr.Markdown(FORMULA, latex_delimiters=LATEX)
#     st0 = toy_reset(0.1)
#     st = gr.State(st0)
#     tau = gr.Slider(0.05, 1.0, step=0.05, value=0.1, label="Temperature τ")
#     with gr.Row():
#         b1 = gr.Button("Step ×1", variant="primary"); b10 = gr.Button("Step ×10"); rs = gr.Button("Reset")
#     d0 = toy_view(st0, 0.1)
#     tinfo, tplot = gr.Markdown(d0[0], latex_delimiters=LATEX), gr.Plot(d0[1])
#     t_out = [st, tinfo, tplot]
#     b1.click(lambda s, t: toy_do(s, t, 1), [st, tau], t_out)
#     b10.click(lambda s, t: toy_do(s, t, 10), [st, tau], t_out)
#     rs.click(toy_restart, tau, t_out)
#     tau.change(toy_retau, [st, tau], t_out)
    
#     # with gr.Accordion("Geometry metrics (reference set)", open=False):
#     #     gr.Dataframe(_metrics_df(), interactive=False)
#     # with gr.Accordion("Training loss curves", open=False):
#     #     gr.Plot(loss_fig())
    
#     with gr.Accordion("Glossary: contrastive terminology", open=False):
#         gr.Markdown(GLOSSARY)



# embedding_tab.py -- "Embedding Space" tab (A vs B). In app.py:
#     build_embedding_tab()
import json
import numpy as np, pandas as pd, torch, gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import T5Tokenizer, T5ForConditionalGeneration
from geometry import project, score
import math
LATEX = [{"left": "$$", "right": "$$", "display": True}, {"left": "$", "right": "$", "display": False}]


D = json.load(open("demo_data/bundle.json", encoding="utf-8"))
MODELS = ["A", "B"]
NAMES = {"A": "Model A (standard only)", "B": "Model B (drift + SupCon)"}
DIRS = {"A": "models/chat_model_a", "B": "models/chat_model_b"}
META, LC = D["ref"]["meta"], D["lifecycle"]
NEP = max(len(LC) - 1, 0)
POS_C, NEG_C = "#1f77b4", "#ff7f0e"  # SupCon positive pair (pull) / negative pair (push)

# ---- shared plot styling (same font as the page, larger and readable) ----
FONT = "Plus Jakarta Sans, ui-sans-serif, system-ui, sans-serif"
GRID = "rgba(127,127,127,0.22)"


def _style(fig):
    """Apply one consistent look to every figure: page font, readable sizes,
    transparent paper so it blends with the light/dark card behind it."""
    fig.update_layout(
        font=dict(family=FONT, size=14),
        title_font=dict(family=FONT, size=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(127,127,127,0.07)",
        legend=dict(font=dict(size=13), itemsizing="constant", tracegroupgap=6),
        hoverlabel=dict(font=dict(family=FONT, size=13)),
    )
    fig.update_annotations(font=dict(family=FONT, size=16))  # subplot titles
    fig.update_xaxes(title_font=dict(size=14), tickfont=dict(size=13), gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(title_font=dict(size=14), tickfont=dict(size=13), gridcolor=GRID, zerolinecolor=GRID)
    return fig


_all = np.vstack([np.array(D["ref"][m]) for m in MODELS] + [np.array(e["ref"]) for e in LC] +
                 [np.array(e["q"]) for e in LC] + [np.array([r["models"][m]["xy"] for r in D["queries"]]) for m in MODELS])
XR = [_all[:, 0].min() - .05, _all[:, 0].max() + .05]
YR = [_all[:, 1].min() - .05, _all[:, 1].max() + .05]
_cache = {}


def _load(m):
    if m not in _cache:
        _cache[m] = (T5Tokenizer.from_pretrained(DIRS[m]),
                     T5ForConditionalGeneration.from_pretrained(DIRS[m]).eval())
    return _cache[m]


def _live(text):
    out = {}
    for m in MODELS:
        tok, mod = _load(m)
        e = tok("respond: " + text, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            emb = mod.encoder(**e).last_hidden_state.mean(1).numpy()
            g = tok.decode(mod.generate(**e, max_length=24, num_beams=2)[0], skip_special_tokens=True)
        b = np.load(f"demo_data/basis_{m}.npz")
        out[m] = {"reply": g, "correct": None, "xy": project(emb, b)[0].tolist(), "score": float(score(emb, b)[0])}
    return out


def _ref_traces(fig, xy, col, legend):
    xy = np.array(xy)
    for lab, c in ((1, "#2ca02c"), (0, "#d62728")):
        for st, sym in (("standard", "circle"), ("drift", "diamond")):
            idx = [j for j, r in enumerate(META) if r["label"] == lab and r["style"] == st]
            fig.add_trace(go.Scatter(x=xy[idx, 0], y=xy[idx, 1], mode="markers",
                                     name=f"{'positive' if lab else 'negative'} · {st}", legendgroup=f"{lab}{st}",
                                     showlegend=legend, marker=dict(color=c, symbol=sym, size=9, opacity=.6)),
                          row=1, col=col)
    fig.add_vline(x=0, line_dash="dot", line_color="gray", row=1, col=col)


def _star(fig, xy, col, legend):
    fig.add_trace(go.Scatter(x=[xy[0]], y=[xy[1]], mode="markers", name="query / anchor", legendgroup="q",
                             showlegend=legend, marker=dict(symbol="star", size=24, color="gold",
                                                            line=dict(color="black", width=1.5))), row=1, col=col)


def _links(fig, a, xy, idx, color, dash, name, col, legend):
    xs, ys = [], []
    for j in idx:
        xs += [a[0], xy[j][0], None]; ys += [a[1], xy[j][1], None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, dash=dash, width=2), name=name,
                             legendgroup=name, showlegend=legend, hoverinfo="skip"), row=1, col=col)
    fig.add_trace(go.Scatter(x=[xy[j][0] for j in idx], y=[xy[j][1] for j in idx], mode="markers",
                             marker=dict(size=15, color="rgba(0,0,0,0)", line=dict(color=color, width=2.5)),
                             showlegend=False, hoverinfo="skip"), row=1, col=col)


# ---------------- Section 1: how drift appears ----------------
def drift_fig():
    fig = make_subplots(rows=1, cols=1)
    xy = np.array(D["ref"]["A"])
    _ref_traces(fig, xy, 1, True)
    for lab, c in ((1, "#2ca02c"), (0, "#d62728")):
        cs = [xy[[j for j, r in enumerate(META) if r["label"] == lab and r["style"] == st]].mean(0)
              for st in ("standard", "drift")]
        fig.add_annotation(x=cs[1][0], y=cs[1][1], ax=cs[0][0], ay=cs[0][1], xref="x", yref="y", axref="x",
                           ayref="y", showarrow=True, arrowhead=3, arrowwidth=3.5, arrowcolor=c)
    fig.update_xaxes(range=XR, title_text="sentiment axis (fit on Model A, standard only)")
    fig.update_yaxes(range=YR, title_text="PC2 (⟂ axis)")
    fig.update_layout(height=560, margin=dict(t=70, b=60, l=60, r=20),
                      legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
                      title="Model A · arrow = standard centroid → drift centroid (same sentiment)")
    return _style(fig)


AMB = 0.05  # sentiment-axis distance from 0 below which a point counts as ambiguous


def _bad_pct(style):
    xy = np.array(D["ref"]["A"])
    idx = [j for j, r in enumerate(META) if r["style"] == style]
    x = xy[idx, 0]
    lab = np.array([META[j]["label"] for j in idx])
    sgn = np.where(lab == 1, x, -x)  # > 0 means correct side
    return 100 * float((sgn < AMB).mean())


def gauge_fig():
    fig = make_subplots(rows=2, cols=1, specs=[[{"type": "indicator"}], [{"type": "indicator"}]],
                        vertical_spacing=0.28)
    for i, (st, nm) in enumerate((("standard", "Standard text"), ("drift", "Drift text")), 1):
        v = _bad_pct(st)
        col = "#2ca02c" if v < 20 else "#e6a700" if v < 50 else "#d62728"
        fig.add_trace(go.Indicator(
            mode="gauge+number", value=v,
            number={"suffix": "%", "font": {"size": 56, "color": col, "family": FONT}},
            title={"text": f"<b>{nm}</b><br><span style='font-size:13px;color:#8a8a8a'>"
                           f"texts on wrong / unclear side</span>", "font": {"size": 18, "family": FONT}},
            gauge={"axis": {"range": [0, 100], "tickvals": [0, 20, 50, 100], "tickcolor": "#999",
                            "tickfont": {"size": 13}},
                   "bar": {"color": col, "thickness": 0.3}, "borderwidth": 0, "bgcolor": "rgba(0,0,0,0)",
                   "steps": [{"range": [0, 20], "color": "#d9f0d9"},
                             {"range": [20, 50], "color": "#fff1bf"},
                             {"range": [50, 100], "color": "#f7cfcf"}],
                   "threshold": {"line": {"color": "#6b7280", "width": 3}, "thickness": 0.85, "value": v}}), i, 1)
    fig.update_layout(height=560, margin=dict(t=90, b=30, l=40, r=40))
    return _style(fig)


def acc_fig():
    M, ps = D["metrics"], D["pool_stats"]["error_by_subtype"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Tone accuracy (reference set)",
                                                        "Error rate by subtype (novel pool)"])
    for k, nm, c in (("tone_acc_standard", "standard", "#7f7f7f"), ("tone_acc_drift", "drift", "#9467bd")):
        fig.add_trace(go.Bar(x=[NAMES[m] for m in MODELS], y=[M[m][k] for m in MODELS], name=nm, marker_color=c), 1, 1)
    subs = list(ps["A"])
    for m, c in (("A", "#d62728"), ("B", "#2ca02c")):
        fig.add_trace(go.Bar(x=subs, y=[ps[m][s] for s in subs], name=NAMES[m], marker_color=c), 1, 2)
    fig.update_layout(barmode="group", height=420, margin=dict(t=60, b=60), legend=dict(orientation="h", y=-0.25))
    return _style(fig)


# ---------------- Section 2: query walkthrough ----------------
def scatter_fig(q):
    fig = make_subplots(rows=1, cols=2, subplot_titles=[NAMES[m] for m in MODELS], horizontal_spacing=0.08)
    pts = np.array([q[m]["xy"] for m in MODELS])
    xr = [min(XR[0], pts[:, 0].min() - .05), max(XR[1], pts[:, 0].max() + .05)]
    yr = [min(YR[0], pts[:, 1].min() - .05), max(YR[1], pts[:, 1].max() + .05)]
    for i, m in enumerate(MODELS, 1):
        _ref_traces(fig, D["ref"][m], i, i == 1)
        _star(fig, q[m]["xy"], i, i == 1)
        fig.update_xaxes(range=xr, title_text="sentiment axis (shared frame)", row=1, col=i)
        fig.update_yaxes(range=yr, title_text="PC2 (⟂ axis)", row=1, col=i)
    fig.update_layout(height=540, margin=dict(t=70, b=60, l=60, r=20),
                      legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"))
    return _style(fig)


def bars_fig(q):
    s = [q[m]["score"] for m in MODELS]
    fig = go.Figure(go.Bar(x=[NAMES[m] for m in MODELS], y=s,
                           marker_color=["#2ca02c" if v > 0 else "#d62728" for v in s],
                           text=[f"{v:+.3f}" for v in s], textposition="outside",
                           textfont=dict(size=15), width=0.35))
    fig.update_layout(height=400, margin=dict(t=80, b=50, l=60, r=20), yaxis_title="score",
                      title="Sentiment score  cos(q, μ₊) − cos(q, μ₋)   (>0 positive side, ≈0 ambiguous)")
    return _style(fig)


# ---------------- Section 3: lifecycle (anchor / positives / negatives) ----------------
def life_fig(qi, ep):
    if not LC:
        return go.Figure()
    ep = min(int(ep), len(LC) - 1)
    ttl = ["Epoch 0 · Model A", f"Epoch {ep} · " + ("Model B" if ep == NEP else "mid-tuning")]
    fig = make_subplots(rows=1, cols=2, subplot_titles=ttl, horizontal_spacing=0.08)
    for i, e in enumerate((0, ep), 1):
        E = LC[e]
        _ref_traces(fig, E["ref"], i, i == 1)
        if qi is not None:
            a, r = E["q"][qi], D["queries"][qi]
            _links(fig, a, E["ref"], r["pos"], POS_C, "solid", "positive pair (pull)", i, i == 1)
            _links(fig, a, E["ref"], r["neg"], NEG_C, "dash", "negative pair (push)", i, i == 1)
            _star(fig, a, i, i == 1)
        fig.update_xaxes(range=XR, title_text="sentiment axis (shared frame)", row=1, col=i)
        fig.update_yaxes(range=YR, row=1, col=i)
    fig.update_layout(height=560, margin=dict(t=70, b=60, l=60, r=20),
                      legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"))
    return _style(fig)


def cos_fig(qi):
    fig = go.Figure()
    if LC:
        xs = list(range(len(LC)))
        fig.add_trace(go.Scatter(x=xs, y=[np.mean(e["cos_pos"]) for e in LC], name="all curated · positives",
                                 line=dict(color=POS_C, dash="dot")))
        fig.add_trace(go.Scatter(x=xs, y=[np.mean(e["cos_neg"]) for e in LC], name="all curated · negatives",
                                 line=dict(color=NEG_C, dash="dot")))
        if qi is not None:
            fig.add_trace(go.Scatter(x=xs, y=[e["cos_pos"][qi] for e in LC], name="this anchor · positives",
                                     line=dict(color=POS_C, width=3)))
            fig.add_trace(go.Scatter(x=xs, y=[e["cos_neg"][qi] for e in LC], name="this anchor · negatives",
                                     line=dict(color=NEG_C, width=3)))
    fig.update_layout(height=400, margin=dict(t=70, b=50), xaxis_title="epoch", yaxis_title="mean cosine (full space)",
                      title="Anchor ↔ positives / negatives similarity across epochs", xaxis=dict(dtick=1))
    return _style(fig)


def life_views(qi, ep):
    return life_fig(qi, ep)


def _qi(sel, custom):
    if (custom or "").strip():
        return None
    return next((i for i, r in enumerate(D["queries"]) if r["text"] == sel), None)


def views(sel, custom, ep):
    custom = (custom or "").strip()
    qi = _qi(sel, custom)
    if custom:
        q = _live(custom); head = f"**Custom query:** {custom}"
    else:
        rec = D["queries"][qi]; q = rec["models"]
        head = (f"**Query:** {sel}  \n**True sentiment:** {'positive' if rec['label'] else 'negative'} · "
                f"**Type:** {rec['subtype']} · **Bucket:** {rec['category']}")
    ok = lambda c: "✅" if c else ("❌" if c is False else "—")
    rows = "\n".join(f"| {NAMES[m]} | {q[m]['reply']} | {q[m]['score']:+.3f} | {ok(q[m]['correct'])} |" for m in MODELS)
    md = head + "\n\n| Model | Reply | Score | Tone |\n|---|---|---|---|\n" + rows
    return md, scatter_fig(q), bars_fig(q), life_views(qi, ep)


def loss_fig():
    L = D["losses"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Generation loss", "SupCon loss"])
    if L:
        w = 10
        for col, k in ((1, "gen"), (2, "con")):
            v = np.array(L[k]); st = np.array(L["step"])
            fig.add_trace(go.Scatter(x=st, y=v, line=dict(color="#1f77b4", width=1), opacity=.3, showlegend=False), 1, col)
            if len(v) >= w:
                fig.add_trace(go.Scatter(x=st[w - 1:], y=np.convolve(v, np.ones(w) / w, mode="valid"),
                                         line=dict(color="#1f77b4", width=3), showlegend=False), 1, col)
    fig.update_xaxes(title_text="step"); fig.update_layout(height=400, margin=dict(t=60, b=50))
    return _style(fig)


def _metrics_df():
    rows = [("tone_acc_standard", "Tone accuracy, standard ↑"), ("tone_acc_drift", "Tone accuracy, drift ↑"),
            ("silhouette", "Silhouette (cosine) ↑"), ("silhouette_drift", "Silhouette, drift only ↑"),
            ("intra_cos", "Intra-class cosine ↑"), ("inter_cos", "Inter-class cosine ↓"),
            ("cross_style_align", "Cross-style centroid alignment ↑"), ("alignment", "Alignment (Wang&Isola) ↓"),
            ("uniformity", "Uniformity (Wang&Isola) ↓"), ("knn_std_to_drift", "kNN acc, fit std → test drift ↑")]
    return pd.DataFrame([[lab] + [round(D["metrics"][m][k], 3) for m in MODELS] for k, lab in rows],
                        columns=["Metric"] + [NAMES[m] for m in MODELS])


# ---------------- Section 4: SupCon math + live toy demo ----------------
FORMULA = r"""
$$\mathcal{L}_i=-\frac{1}{|P(i)|}\sum_{p\in P(i)}\log\frac{\exp(z_i\!\cdot\! z_p/\tau)}{\sum_{a\neq i}\exp(z_i\!\cdot\! z_a/\tau)}$$

- $z$ = unit vector of a text, so $z_i\cdot z_p$ is their cosine similarity. $p$ = same-sentiment texts, $a$ = all other texts.
- **Pull:** the numerator has the positives. The loss drops when $z_i\cdot z_p$ grows.
- **Push:** the denominator also has the negatives. The loss drops when their similarity shrinks.
- **τ:** small τ sharpens the softmax, so the closest (hardest) negatives get the biggest push.

**How the update happens** (gradient descent on the anchor, $w_a$ = softmax share of point $a$ in the denominator):

$$z_i \leftarrow z_i+\frac{\eta}{\tau}\Big(\underbrace{\overline{z_p}}_{\text{toward positives}}-\underbrace{\textstyle\sum_a w_a z_a}_{\text{away from crowded / close points}}\Big)$$
"""


def _loss(theta, tau):
    z = torch.stack([torch.cos(theta), torch.sin(theta)], 1)
    cs = z[1:] @ z[0]
    s = cs / tau
    return -(s[:2] - torch.logsumexp(s, 0)).mean(), cs, torch.softmax(s, 0)


def toy_reset(tau):
    th = [math.radians(d) for d in (90, 200, 320, 60, 130, 20)]  # anchor, P1, P2, N1, N2, N3
    L = float(_loss(torch.tensor(th, dtype=torch.float64), tau)[0])
    return {"th": th, "hist": [L]}


def toy_step(state, tau, n):
    th, hist = torch.tensor(state["th"], dtype=torch.float64), list(state["hist"])
    for _ in range(int(n)):
        th = th.clone().requires_grad_(True)
        (g,) = torch.autograd.grad(_loss(th, tau)[0], th)
        th = (th - (0.05 * g).clamp(-0.3, 0.3)).detach()
        hist.append(float(_loss(th, tau)[0]))
    return {"th": th.tolist(), "hist": hist}


def toy_view(state, tau):
    th = torch.tensor(state["th"], dtype=torch.float64)
    L, cs, w = _loss(th, tau)
    cs, w = cs.tolist(), w.tolist()
    names = ["P1", "P2", "N1", "N2", "N3"]
    cols = ["#1f77b4"] * 2 + ["#ff7f0e"] * 3
    xy = np.stack([np.cos(state["th"]), np.sin(state["th"])], 1)
    fig = make_subplots(rows=1, cols=2, column_widths=[.5, .5], horizontal_spacing=0.1,
                        subplot_titles=["Toy unit circle (2D hypersphere)", "SupCon loss per step"])
    t = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(go.Scatter(x=np.cos(t), y=np.sin(t), mode="lines", line=dict(color="#bbb"), showlegend=False), 1, 1)
    for k in range(5):
        fig.add_trace(go.Scatter(x=[xy[0, 0], xy[k + 1, 0]], y=[xy[0, 1], xy[k + 1, 1]], mode="lines",
                                 line=dict(color=cols[k], dash="solid" if k < 2 else "dash", width=2.5),
                                 showlegend=False, hoverinfo="skip"), 1, 1)
        fig.add_trace(go.Scatter(x=[xy[k + 1, 0]], y=[xy[k + 1, 1]], mode="markers+text", text=[names[k]],
                                 textposition="top center", textfont=dict(size=14), showlegend=False,
                                 marker=dict(size=17, color=cols[k])), 1, 1)
    fig.add_trace(go.Scatter(x=[xy[0, 0]], y=[xy[0, 1]], mode="markers+text", text=["anchor"], textposition="top center",
                             textfont=dict(size=14), showlegend=False,
                             marker=dict(symbol="star", size=26, color="gold",
                                         line=dict(color="black", width=1.5))), 1, 1)
    fig.update_xaxes(range=[-1.35, 1.35], visible=False, row=1, col=1)
    fig.update_yaxes(range=[-1.35, 1.35], visible=False, scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(len(state["hist"]))), y=state["hist"], mode="lines+markers",
                             line=dict(color="#1f77b4", width=3), marker=dict(size=8), showlegend=False), 1, 2)
    fig.update_xaxes(title_text="step", row=1, col=2); fig.update_yaxes(title_text="loss", row=1, col=2)
    fig.update_layout(height=500, margin=dict(t=70, b=50, l=60, r=20))
    rows = "\n".join(f"| {names[k]} | {'positive' if k < 2 else 'negative'} | {cs[k]:+.2f} | {w[k]:.0%} |" for k in range(5))
    md = (f"**Loss = {float(L):.3f}** · avg cos to positives **{np.mean(cs[:2]):+.2f}** · "
          f"avg cos to negatives **{np.mean(cs[2:]):+.2f}**\n\n"
          f"| Point | Type | cos(anchor, point) | share of denominator |\n|---|---|---|---|\n{rows}\n\n"
          "*Bigger share = bigger push. The closest negative gets the most.*")
    return md, _style(fig)


def toy_do(state, tau, n):
    state = toy_step(state, tau, n)
    return (state,) + toy_view(state, tau)


def toy_retau(state, tau):
    th = torch.tensor(state["th"], dtype=torch.float64)
    state = {"th": state["th"], "hist": state["hist"][:-1] + [float(_loss(th, tau)[0])]}
    return (state,) + toy_view(state, tau)


def toy_restart(tau):
    state = toy_reset(tau)
    return (state,) + toy_view(state, tau)


GLOSSARY = """
**Setup.** Encoder output is mean-pooled into **z** and L2-normalized → points live on a **unit hypersphere**.
Total loss: **L = L_gen + λ · L_SupCon**, λ = 0.3, temperature τ = 0.1.

**SupCon** (Khosla et al., 2020) for each **anchor** i:
`L_i = −(1/|P(i)|) Σ_{p∈P(i)} log [ exp(z_i·z_p / τ) / Σ_{a≠i} exp(z_i·z_a / τ) ]`
- **Positives P(i):** other samples in the batch with the same sentiment label (standard *and* drift → pulls slang/emoji toward standard phrasing of the same sentiment).
- **Negatives:** samples with the opposite label (pushed apart). Low τ sharpens the penalty on hard negatives.

**Anchor view (Section 3).** Training uses in-batch positives/negatives. For illustration we pick, per anchor, the K hard positives (least similar same-label texts) and K hard negatives (most similar opposite-label texts) in Model A's space and track their cosine over epochs.

**Projection.** Linear: x = sentiment axis (μ₊ − μ₋), y = top PC orthogonal to it; fit once on Model A using **standard reference points only** and reused for Model B and every epoch, so all panels share one frame. Score uses each model's own standard-only centroids.

**Metrics.** *Silhouette / intra-inter cosine*: cluster quality. *Cross-style alignment*: cosine between standard and drift centroids of the same class. *Alignment* / *Uniformity*: Wang & Isola, 2020. *kNN std→drift*: does the standard-style geometry classify drift points?

**Refs.** Khosla et al. 2020 (SupCon) · Wang & Isola 2020 · Gao et al. 2021 (SimCSE) · Cha et al. 2021 (Co2L).
"""


def build_embedding_tab():
    ps = D["pool_stats"]; er = ps["error_rate"]
    qs = [r["text"] for r in D["queries"]]
    init = views(qs[0], "", NEP)

    gr.Markdown("## 1 · How systematic drift appears")
    gr.Markdown(f"Model A is trained on standard phrasing only. Drift text (slang, emoji, sarcasm) of the *same* sentiment "
                f"lands away from its standard counterpart, and tone accuracy drops.")
    with gr.Row():
        with gr.Column(scale=3):
            gr.Plot(drift_fig())
        with gr.Column(scale=2):
            gr.Plot(gauge_fig())
            gr.Markdown("Green = model places the text clearly on its sentiment side. "
                        "Red = text lands on the wrong side or too near the middle to tell.")

    gr.Markdown("## 2 · Same input, two models")
    gr.Markdown(f"**Disclosure:** {ps['n_selected']} queries curated from {ps['n_candidates']} novel candidates "
                f"(new subjects/templates, deduped vs. training; incl. held-out sarcasm structures). Over the full pool, "
                f"tone error — A: {er['A']:.0%}, B: {er['B']:.0%}. Selected buckets: {ps['selected_buckets']}.")
    with gr.Row():
        sel = gr.Dropdown(qs, value=qs[0], label="Curated query")
        custom = gr.Textbox(label="…or type your own (overrides; loads models on first use)")
    btn = gr.Button("Project query", variant="primary")
    info, sp, bars = gr.Markdown(init[0]), gr.Plot(init[1]), gr.Plot(init[2])

    gr.Markdown("## 3 · Contrastive lifecycle: anchor, positives, negatives")
    ep = gr.Slider(0, NEP, step=1, value=NEP, label="Epoch (0 = Model A)")
    lp = gr.Plot(init[3])

    ins, outs = [sel, custom, ep], [info, sp, bars, lp]
    btn.click(views, ins, outs); sel.change(views, ins, outs)
    ep.change(lambda s, c, e: life_views(_qi(s, c), e), ins, lp)

    gr.Markdown("## 4 · The math: how SupCon pulls and pushes", latex_delimiters=LATEX)
    gr.Markdown(FORMULA, latex_delimiters=LATEX)
    st0 = toy_reset(0.1)
    st = gr.State(st0)
    tau = gr.Slider(0.05, 1.0, step=0.05, value=0.1, label="Temperature τ")
    with gr.Row():
        b1 = gr.Button("Step ×1", variant="primary"); b10 = gr.Button("Step ×10"); rs = gr.Button("Reset")
    d0 = toy_view(st0, 0.1)
    tinfo, tplot = gr.Markdown(d0[0], latex_delimiters=LATEX), gr.Plot(d0[1])
    t_out = [st, tinfo, tplot]
    b1.click(lambda s, t: toy_do(s, t, 1), [st, tau], t_out)
    b10.click(lambda s, t: toy_do(s, t, 10), [st, tau], t_out)
    rs.click(toy_restart, tau, t_out)
    tau.change(toy_retau, [st, tau], t_out)

    with gr.Accordion("Glossary: contrastive terminology", open=False):
        gr.Markdown(GLOSSARY)