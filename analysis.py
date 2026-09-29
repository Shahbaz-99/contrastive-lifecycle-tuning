# analysis.py -> demo_data/bundle.json + demo_data/basis_{A,B}.npz
# Shared basis: fit on Model A, STANDARD points only; all models/epochs are projected onto it.
import json, os, glob, random
import numpy as np, pandas as pd, torch
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from transformers import T5Tokenizer, T5ForConditionalGeneration
from geometry import nrm, fit_basis, class_means, project, score, sentiment_of_reply

random.seed(7)
os.makedirs("demo_data", exist_ok=True)
DIRS = {"A": "models/chat_model_a", "B": "models/chat_model_b"}
MODELS = list(DIRS)
K = 4  # positives / negatives shown per anchor
dev = "cuda" if torch.cuda.is_available() else "cpu"
ref = pd.read_csv("data_chat/chat_test_set.csv")
pool = pd.read_csv("data_chat/candidate_pool.csv")
y, sty = ref.label.values, ref["style"].values


@torch.no_grad()
def run_model(m, texts, bs=64):
    tok = T5Tokenizer.from_pretrained(DIRS[m])
    mod = T5ForConditionalGeneration.from_pretrained(DIRS[m]).to(dev).eval()
    E, G = [], []
    for i in range(0, len(texts), bs):
        e = tok(["respond: " + t for t in texts[i:i + bs]], return_tensors="pt", padding=True,
                truncation=True, max_length=64).to(dev)
        h = mod.encoder(**e).last_hidden_state
        mk = e["attention_mask"].unsqueeze(-1).float()
        E.append(((h * mk).sum(1) / mk.sum(1)).cpu().numpy())
        G += tok.batch_decode(mod.generate(**e, max_length=24, num_beams=2), skip_special_tokens=True)
    return np.concatenate(E), G


def correct(gens, labels):
    return np.array([sentiment_of_reply(g) == l for g, l in zip(gens, labels)])


R, P, basis = {}, {}, {}
for m in MODELS:
    print("running", m)
    R[m] = run_model(m, ref.review.tolist())
    P[m] = run_model(m, pool.text.tolist())

std = sty == "standard"
shared = fit_basis(R["A"][0], y, std)  # one frame for everything
for m in MODELS:
    mp, mn = class_means(R[m][0], y, std)  # per-model standard-only centroids (for the score)
    basis[m] = {**shared, "mp": mp, "mn": mn}
    np.savez(f"demo_data/basis_{m}.npz", **basis[m])


# ---------- geometry metrics on the reference set ----------
def metrics(m):
    E = nrm(R[m][0]); S = E @ E.T; d2 = 2 - 2 * S
    same = y[:, None] == y[None, :]; off = ~np.eye(len(y), dtype=bool)
    cen = lambda l, s: nrm(E[(y == l) & (sty == s)].mean(0))
    dr, st = sty == "drift", sty == "standard"
    c = correct(R[m][1], y)
    return {
        "tone_acc_standard": c[st].mean(), "tone_acc_drift": c[dr].mean(),
        "silhouette": silhouette_score(E, y, metric="cosine"),
        "silhouette_drift": silhouette_score(E[dr], y[dr], metric="cosine"),
        "intra_cos": S[same & off].mean(), "inter_cos": S[~same].mean(),
        "cross_style_align": np.mean([cen(1, "standard") @ cen(1, "drift"), cen(0, "standard") @ cen(0, "drift")]),
        "alignment": d2[same & off].mean(), "uniformity": np.log(np.exp(-2 * d2[off]).mean()),
        "knn_std_to_drift": KNeighborsClassifier(5, metric="cosine").fit(E[st], y[st]).score(E[dr], y[dr]),
    }


MET = {m: metrics(m) for m in MODELS}

# ---------- pool stats + curation ----------
pl = pool.label.values
C = {m: correct(P[m][1], pl) for m in MODELS}
pool_stats = {"n_candidates": len(pool), "error_rate": {m: float(1 - C[m].mean()) for m in MODELS},
              "error_by_subtype": {m: {s: float(1 - C[m][(pool.subtype == s).values].mean())
                                       for s in pool.subtype.unique()} for m in MODELS}}
recs = []
for i, r in pool.iterrows():
    a, b = bool(C["A"][i]), bool(C["B"][i])
    cat = ("A wrong -> B right" if (not a and b) else "both right" if (a and b)
           else "both wrong" if (not a and not b) else "A right -> B wrong")
    recs.append({"idx": i, "text": r.text, "label": int(r.label), "subtype": r.subtype,
                 "category": cat, "group": f"{r.subtype}|{r.label}"})


def pick(items, n):  # round-robin over (subtype,label) groups for diversity
    g = {}
    for c in items: g.setdefault(c["group"], []).append(c)
    for v in g.values(): random.shuffle(v)
    out = []
    while len(out) < n and any(g.values()):
        for k in list(g):
            if g[k] and len(out) < n: out.append(g[k].pop())
    return out


by = lambda cat: [c for c in recs if c["category"] == cat]
sel = pick(by("A wrong -> B right"), 15) + pick(by("both right"), 3) + pick(by("both wrong"), 2)
if len(sel) < 20:
    print(f"WARNING: only {len(sel)} from target buckets; topping up")
    taken = {c["idx"] for c in sel}
    sel += pick([c for c in recs if c["idx"] not in taken], 20 - len(sel))
random.shuffle(sel)
pool_stats["n_selected"] = len(sel)
pool_stats["selected_buckets"] = {k: sum(c["category"] == k for c in sel) for k in set(c["category"] for c in sel)}
print(pool_stats["error_rate"], pool_stats["selected_buckets"])

# ---------- lifecycle + anchor / positives / negatives ----------
files = sorted(glob.glob("snapshots/b_epoch*.npz"), key=lambda p: int(p.split("epoch")[1].split(".")[0]))
snaps = [dict(np.load(f)) for f in files]
qi = [c["idx"] for c in sel]
LC = []
if snaps:
    print("epoch0 snapshot vs Model A max diff:", float(np.abs(nrm(snaps[0]["ref"]) - nrm(R["A"][0])).max()))
    S0, Q0 = nrm(snaps[0]["ref"]), nrm(snaps[0]["pool"][qi])
    for k, c in enumerate(sel):
        sim = S0 @ Q0[k]
        same, opp = np.where(y == c["label"])[0], np.where(y != c["label"])[0]
        c["pos"] = same[np.argsort(sim[same])[:K]].tolist()   # hard positives: least similar, same label
        c["neg"] = opp[np.argsort(-sim[opp])[:K]].tolist()    # hard negatives: most similar, opposite label
    for s in snaps:
        Rn, Qn = nrm(s["ref"]), nrm(s["pool"][qi])
        LC.append({"ref": project(s["ref"], shared).tolist(), "q": project(s["pool"][qi], shared).tolist(),
                   "cos_pos": [float((Rn[c["pos"]] @ Qn[k]).mean()) for k, c in enumerate(sel)],
                   "cos_neg": [float((Rn[c["neg"]] @ Qn[k]).mean()) for k, c in enumerate(sel)]})
    print("mean cos gap (pos-neg) per epoch:",
          [round(float(np.mean(e["cos_pos"]) - np.mean(e["cos_neg"])), 3) for e in LC])
LOSS = json.load(open("snapshots/b_losses.json")) if os.path.exists("snapshots/b_losses.json") else {}

queries = []
for c in sel:
    q = dict(c); q["models"] = {}
    for m in MODELS:
        e = P[m][0][c["idx"]][None]
        q["models"][m] = {"reply": P[m][1][c["idx"]], "correct": bool(C[m][c["idx"]]),
                          "xy": project(e, basis[m])[0].tolist(), "score": float(score(e, basis[m])[0])}
    queries.append(q)

bundle = {"ref": {"meta": [{"text": r.review, "label": int(r.label), "style": r["style"]} for _, r in ref.iterrows()],
                  **{m: project(R[m][0], basis[m]).tolist() for m in MODELS}},
          "metrics": MET, "pool_stats": pool_stats, "queries": queries, "lifecycle": LC, "losses": LOSS}


def _d(o):
    if isinstance(o, (np.floating, np.integer, np.bool_)): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(type(o))


json.dump(bundle, open("demo_data/bundle.json", "w", encoding="utf-8"), default=_d)
print("Saved demo_data/bundle.json")
for m in MODELS: print(m, {k: round(float(v), 3) for k, v in MET[m].items()})