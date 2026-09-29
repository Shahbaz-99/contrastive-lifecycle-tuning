# gen_candidates.py -> data_chat/candidate_pool.csv
# Novel subjects x novel slang/emoji templates + held-out sarcasm structures,
# deduped against every training/test review. Run BEFORE training (snapshots embed this pool).
import pandas as pd

subjects = [
    "the pizza place", "this headset", "the theme park", "this vacuum", "the sequel",
    "this keyboard", "the streaming service", "this sneaker brand", "the yoga class",
    "the ride share", "this tablet", "the food truck", "the online store", "this playlist",
    "the season finale", "this smartwatch", "the campsite", "the tutoring session",
    "this board game", "the check-in process",
]
pos_slang = [
    "{s} went crazy good 🤩", "ok {s} eats no crumbs", "{s} is bussin fr 🔥",
    "{s} understood the assignment 👏", "ate and left no crumbs, {s}", "big W for {s} 🙌",
    "{s} lowkey saved my week 😭", "{s} is peak, no notes", "y'all {s} is unreal rn",
    "{s} carried hard today ngl", "shook at how good {s} was 😳", "{s} is chef's kiss 💯",
    "{s} = instant fave ✨", "{s} did NOT miss 🔥", "{s} lives rent free in my head 😍",
]
neg_slang = [
    "{s} is cooked fr 💀", "{s} fumbled hard ngl", "{s} is a big L 😩", "{s} ain't it chief",
    "{s} was giving nothing 😑", "{s} is straight cringe", "ick. just ick. {s}",
    "{s} fell off badly 📉", "{s} was a total dud, pass", "{s} cost me my sanity 🙃",
    "yikes on bikes, {s}", "{s} gets zero stars from me 👎", "{s} is such a snooze fest tbh",
    "{s} is not it, dawg", "{s} tanked my mood 😤",
]
sarc_neg = [
    "thanks {s}, exactly the disaster I needed", "{s} broke in five minutes, what a masterpiece",
    "nothing says quality like {s} dying on day one", "five stars if you enjoy pain, {s}",
    "{s} was so good I almost fell asleep", "bravo {s}, you ruined it again",
    "love how {s} just gave up on me", "{s} totally lived up to the hype, if the hype was zero",
]
sarc_pos = [
    "expected a mess but {s} was weirdly great", "thought {s} would flop, I was so wrong",
    "grudgingly admitting {s} is good", "against all odds, {s} rocked",
    "fine, {s} won me over, happy now?", "did not expect to love {s} but here we are",
]

seen = set()
for f in ["chat_standard_train", "chat_drift_train", "chat_full_train", "chat_test_set"]:
    seen |= set(pd.read_csv(f"data_chat/{f}.csv")["review"].str.lower())

rows = []
def add(tmpls, label, sarcasm=False):
    for s in subjects:
        for t in tmpls:
            text = t.format(s=s)
            if text.lower() in seen:
                continue
            seen.add(text.lower())
            sub = "held_out_sarcasm" if sarcasm else ("novel_emoji" if any(ord(c) > 0x2500 for c in text) else "novel_slang")
            rows.append((text, label, sub))

add(pos_slang, 1); add(neg_slang, 0); add(sarc_neg, 0, True); add(sarc_pos, 1, True)
pd.DataFrame(rows, columns=["text", "label", "subtype"]).to_csv("data_chat/candidate_pool.csv", index=False)
print(f"Wrote {len(rows)} candidates -> data_chat/candidate_pool.csv")
