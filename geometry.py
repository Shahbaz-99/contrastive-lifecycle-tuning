# geometry.py -- shared helpers: sentiment axis basis, projection, score, tone check
import numpy as np

POSITIVE_WORDS = ["glad", "great", "awesome", "love", "wonderful", "impressed", "happy", "thanks for sharing"]
NEGATIVE_WORDS = ["sorry", "frustrat", "apolog", "disappoint", "trouble", "didn't work", "let you down"]


def sentiment_of_reply(text):
    t = text.lower()
    p = any(w in t for w in POSITIVE_WORDS)
    n = any(w in t for w in NEGATIVE_WORDS)
    if n and not p:
        return 0
    if p and not n:
        return 1
    return None


def nrm(E):
    E = np.asarray(E, dtype=np.float64)
    return E / np.clip(np.linalg.norm(E, axis=-1, keepdims=True), 1e-12, None)


def class_means(E, y, mask=None):
    """(mu+, mu-) of L2-normalized embeddings, optionally on a subset (mask)."""
    E, y = nrm(E), np.asarray(y)
    if mask is not None:
        E, y = E[mask], y[mask]
    return E[y == 1].mean(0), E[y == 0].mean(0)


def fit_basis(E, y, mask=None):
    """2D linear basis: x = sentiment axis (mu+ - mu-), y = top PC orthogonal to it. Fit on mask subset if given."""
    E, y = nrm(E), np.asarray(y)
    if mask is not None:
        E, y = E[mask], y[mask]
    mp, mn = E[y == 1].mean(0), E[y == 0].mean(0)
    center = (mp + mn) / 2
    ax = mp - mn
    ax /= np.linalg.norm(ax)
    R = E - center
    R = R - np.outer(R @ ax, ax)
    pc = np.linalg.svd(R, full_matrices=False)[2][0]
    return {"center": center, "ax": ax, "pc": pc, "mp": mp, "mn": mn}


def project(E, b):
    R = nrm(E) - b["center"]
    return np.stack([R @ b["ax"], R @ b["pc"]], axis=1)


def score(E, b):
    """cos(q, mu+) - cos(q, mu-): >0 positive side, <0 negative side, ~0 ambiguous."""
    E = nrm(E)
    return E @ nrm(b["mp"]) - E @ nrm(b["mn"])