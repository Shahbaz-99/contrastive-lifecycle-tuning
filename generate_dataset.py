"""
Generates a synthetic sentiment dataset for the Systemic Drift demo project.

SIMPLE EXPLANATION:
- We build word banks: subjects (things being reviewed), positive/negative
  words (standard style), and slang/emoji/sarcasm snippets (drift style).
- We build EVERY possible combination of subject x word x template up front
  (a full list), shuffle it once, then just slice off how many rows we need.
  This avoids any "keep retrying until we find a new one" loop that can hang.

OUTPUT FILES (in ./data/):
- standard_train.csv  -> 1200 rows, style=standard   (train Model A on this)
- drift_train.csv     -> 800 rows,  style=drift       (add this for Model B)
- full_train.csv      -> standard_train + drift_train (train Model B on this)
- test_set.csv         -> 200 rows, 50/50 mix of both styles, held out
"""

import csv
import os
import random

random.seed(42)
OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

subjects = [
    "the movie", "this phone", "the restaurant", "this laptop", "the service",
    "this book", "the hotel room", "this app", "the concert", "this course",
    "the customer support", "this game", "the coffee", "this jacket",
    "the flight", "this software update", "the new album", "this recipe",
    "the delivery", "this show", "the podcast", "this camera", "the museum",
    "this bakery", "the gym membership",
]

standard_positive_adj = [
    "excellent", "wonderful", "impressive", "outstanding", "great",
    "well-made", "enjoyable", "fantastic", "reliable", "satisfying",
    "superb", "top-notch",
]
standard_negative_adj = [
    "disappointing", "terrible", "poorly made", "frustrating", "mediocre",
    "unreliable", "boring", "awful", "unsatisfactory", "subpar",
    "underwhelming", "second-rate",
]
standard_templates = [
    "I found {subj} to be {adj}.",
    "{subj_cap} was {adj}, overall.",
    "In my opinion, {subj} is {adj}.",
    "Honestly, {subj} turned out {adj}.",
    "{subj_cap} was genuinely {adj}.",
    "My experience with {subj} was {adj}.",
    "{subj_cap} left me feeling it was {adj}.",
]

drift_positive_snippets = [
    "ngl {subj} is actually fire 🔥", "{subj} slaps fr fr", "lowkey obsessed with {subj} 😍",
    "{subj} is goated no cap", "can't stop thinking about {subj}, so good!!",
    "{subj} hit different 🙌", "{subj} was a whole vibe, loved it",
    "not me loving {subj} this much lol", "{subj} 10/10 would recommend",
    "okay {subj} is actually kinda amazing ngl", "{subj} deadass changed my life lol",
    "{subj} is elite fr", "obsessed w {subj} rn 😭❤️",
    "{subj} is actually so underrated", "not gonna lie {subj} is kinda perfect",
    "{subj} really said main character energy", "{subj} is giving excellence tbh",
    "shoutout to {subj}, genuinely great", "{subj} got me smiling like an idiot ngl",
]
drift_negative_snippets = [
    "{subj} was mid tbh", "lol {subj} is straight up trash", "{subj}... yeah no thx 💀",
    "{subj} lowkey ruined my day ngl", "not {subj} being this bad 😭",
    "{subj} 2/10 do not recommend", "wow {subj} really said let me disappoint you",
    "{subj} was so overhyped, big letdown", "{subj} mid af ngl",
    "cant believe {subj} was this disappointing lol", "{subj} straight up flopped ngl",
    "{subj} is a hard pass fr", "big yikes, {subj} was rough 😬",
]
drift_sarcastic_negative = [
    "oh great, {subj} broke again, love that for me",
    "wow {subj} really outdid itself in disappointing me",
    "yeah {subj} was amazing, said no one ever",
    "totally thrilled that {subj} wasted my time",
    "great job {subj}, truly inspiring levels of bad",
]
drift_sarcastic_positive = [
    "somehow {subj} didn't disappoint for once, shocking",
    "wasn't expecting much but {subj} actually delivered",
    "color me surprised, {subj} was actually decent",
]


def cap(s):
    return s[0].upper() + s[1:]


def build_standard_pool(adjectives, label):
    pool = []
    for subj in subjects:
        for adj in adjectives:
            for tmpl in standard_templates:
                text = tmpl.format(subj=subj, subj_cap=cap(subj), adj=adj)
                pool.append((text, label, "standard"))
    return pool


def build_drift_pool(snippets, label):
    pool = []
    for subj in subjects:
        for snippet in snippets:
            pool.append((snippet.format(subj=subj), label, "drift"))
    return pool


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label", "style"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {path}")


if __name__ == "__main__":
    std_pos_pool = build_standard_pool(standard_positive_adj, 1)
    std_neg_pool = build_standard_pool(standard_negative_adj, 0)
    random.shuffle(std_pos_pool)
    random.shuffle(std_neg_pool)

    drift_pos_pool = build_drift_pool(drift_positive_snippets, 1) + build_drift_pool(drift_sarcastic_positive, 1)
    drift_neg_pool = build_drift_pool(drift_negative_snippets, 0) + build_drift_pool(drift_sarcastic_negative, 0)
    random.shuffle(drift_pos_pool)
    random.shuffle(drift_neg_pool)

    std_pos_train, std_pos_test = std_pos_pool[:600], std_pos_pool[600:650]
    std_neg_train, std_neg_test = std_neg_pool[:600], std_neg_pool[600:650]
    drift_pos_train, drift_pos_test = drift_pos_pool[:400], drift_pos_pool[400:450]
    drift_neg_train, drift_neg_test = drift_neg_pool[:400], drift_neg_pool[400:450]

    standard_train = std_pos_train + std_neg_train
    drift_train = drift_pos_train + drift_neg_train
    full_train = standard_train + drift_train
    test_set = std_pos_test + std_neg_test + drift_pos_test + drift_neg_test

    random.shuffle(standard_train)
    random.shuffle(drift_train)
    random.shuffle(full_train)
    random.shuffle(test_set)

    write_csv(f"{OUT_DIR}/standard_train.csv", standard_train)
    write_csv(f"{OUT_DIR}/drift_train.csv", drift_train)
    write_csv(f"{OUT_DIR}/full_train.csv", full_train)
    write_csv(f"{OUT_DIR}/test_set.csv", test_set)

    print("\nClass balance check:")
    for name, rows in [("standard_train", standard_train), ("drift_train", drift_train),
                        ("full_train", full_train), ("test_set", test_set)]:
        pos = sum(1 for r in rows if r[1] == 1)
        neg = sum(1 for r in rows if r[1] == 0)
        print(f"  {name}: total={len(rows)} pos={pos} neg={neg}")
