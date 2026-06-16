# %%
import pandas as pd
import re
import time
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("\n🚀 STEP 1: Loading data...")

def read_file(path):
    return pd.read_csv(
        path,
        sep="|",
        header=None,
        names=["code", "text"],
        encoding="latin1",
        engine="python"
    )

cot = read_file("OENACE2025_DE_COT.csv")
cal = read_file("OENACE2025_DE_CAL.csv")
unt = read_file("OENACE2025_DE_UNT.csv")

cot["source"] = "cot"
cal["source"] = "cal"
unt["source"] = "unt"

print(f"COT rows: {len(cot)}")
print(f"CAL rows: {len(cal)}")
print(f"UNT rows: {len(unt)}")

# %%
print("\n🔗 STEP 2: Prepare data...")

def normalize_text(x):
    x = str(x).lower()
    x = x.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    return x.strip()

for df_ in [cot, cal, unt]:
    df_["text"] = df_["text"].apply(normalize_text)
    df_["code"] = df_["code"].astype(str).str.strip()

df = pd.concat([cot, cal], ignore_index=True)

df["weight"] = df["source"].map({
    "cal": 1.2,
    "cot": 1.0
})

# %%
print("\n🧠 STEP 3: Load model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# %%
print("\n📦 STEP 4: Encode embeddings...")

start = time.time()

embeddings = model.encode(
    df["text"].tolist(),
    batch_size=32,
    convert_to_numpy=True
)

# ✅ NEW: encode UNT (critical)
unt_embeddings = model.encode(
    unt["text"].tolist(),
    batch_size=32,
    convert_to_numpy=True
)

print(f"✅ Encoding done in {time.time() - start:.2f}s")

# %%
print("\n📖 STEP 5: Code mapping...")

code_to_desc = (
    cot.groupby("code")["text"]
    .first()
    .to_dict()
)

# %%
print("\n📚 STEP 6: Keyword index (with UNT)...")

code_keywords = defaultdict(set)

def add_keywords(df_):
    for _, row in df_.iterrows():
        words = re.findall(r"\b[a-z]{3,}\b", row["text"])
        for w in words:
            code_keywords[row["code"]].add(w)

add_keywords(cot)
add_keywords(cal)
add_keywords(unt)

# remove very common words
word_freq = defaultdict(int)
for words in code_keywords.values():
    for w in words:
        word_freq[w] += 1

COMMON_THRESHOLD = 80

for code in code_keywords:
    code_keywords[code] = {
        w for w in code_keywords[code]
        if word_freq[w] < COMMON_THRESHOLD
    }

# %%
print("\n✂️ STEP 7: Input extraction...")

def extract_parts(text):
    text = normalize_text(text)

    noise_patterns = [
        r"sehr geehrte.*",
        r"mit freundlichen.*",
        r"ich bitte.*",
        r"vielen dank.*",
        r"\d+ ?%",
        r"umsatz.*"
    ]

    for p in noise_patterns:
        text = re.sub(p, "", text)

    parts = re.split(r"[.,;!?]| und | sowie | auch | bzw", text)

    parts = [p.strip() for p in parts if len(p.strip()) > 3]

    return parts if parts else [text]

# %%
print("\n🔑 STEP 8: Keyword scoring...")

def keyword_scores(query):
    words = set(re.findall(r"\b[a-z]{3,}\b", normalize_text(query)))
    scores = defaultdict(float)

    for code, kws in code_keywords.items():
        overlap = words & kws
        if overlap:
            scores[code] += min(len(overlap), 5) * 0.08

    return scores

# %%
print("\n🔍 STEP 9: Prediction...")

def predict_top_k(query, k=3):
    query = normalize_text(query)
    parts = extract_parts(query)

    code_scores = defaultdict(list)

    # ✅ batch encode parts
    part_embs = model.encode(parts, convert_to_numpy=True)

    for emb in part_embs:
        sims = cosine_similarity([emb], embeddings)[0]

        for i, s in enumerate(sims):
            code = df.iloc[i]["code"]
            weight = df.iloc[i]["weight"]
            code_scores[code].append(s * weight)

    final_scores = {}

    # base embedding scores (weakened)
    for code, scores in code_scores.items():
        final_scores[code] = (
            max(scores) * 0.5 +
            (sum(scores)/len(scores)) * 0.2 +
            0.04 * len(scores)
        )

    # ✅ keyword boost (stronger)
    kw = keyword_scores(query)
    for code, s in kw.items():
        if code in final_scores:
            final_scores[code] += s
        else:
            final_scores[code] = s * 0.7

    # ✅ NEW: UNT similarity (MOST IMPORTANT)
    query_emb = model.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(query_emb, unt_embeddings)[0]

    top_idx = sims.argsort()[-20:]

    for idx in top_idx:
        code = unt.iloc[idx]["code"]
        final_scores[code] = final_scores.get(code, 0) + sims[idx] * 0.6

    # ranking
    top = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:k]

    return {
        "label": query,
        "predictions": [
            {
                "code": c,
                "description": code_to_desc.get(c, ""),
                "score": float(s)
            }
            for c, s in top
        ]
    }

# %%
print("\n📊 STEP 10: Evaluation...")

def evaluate(n=200):
    sample = unt.sample(n, random_state=42)

    top1 = 0
    top3 = 0

    for _, row in sample.iterrows():
        result = predict_top_k(row["text"], 3)
        preds = [p["code"] for p in result["predictions"]]

        if preds and preds[0] == row["code"]:
            top1 += 1
        if row["code"] in preds:
            top3 += 1

    print(f"\n✅ Accuracy@1: {top1/n:.2%}")
    print(f"✅ Accuracy@3: {top3/n:.2%}")

# %%
print("\n🧪 TEST...")

examples = [
    "Hotel mit Restaurant und Spa",
    "Taxi service und Transport von Personen",
    "Food truck mit takeaway",
    "Wir betreiben einen Taxibetrieb und Krankenfahrten"
]

for e in examples:
    print("\n" + "="*60)
    print("INPUT:", e)
    r = predict_top_k(e)
    for p in r["predictions"]:
        print(p)

# %%
evaluate(200)

print("\n✅ DONE")
