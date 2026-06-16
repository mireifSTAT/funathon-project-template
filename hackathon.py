# %%
import pandas as pd
import re
import time
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("\n🚀 STEP 1: Loading data...")

cot = pd.read_csv(
    "OENACE2025_DE_COT.csv",
    sep="|",
    header=None,
    names=["code", "text"],
    encoding="latin1"   # ✅ FIX
)

cal = pd.read_csv(
    "OENACE2025_DE_CAL.csv",
    sep="|",
    header=None,
    names=["code", "text"],
    encoding="latin1"
)

unt = pd.read_csv(
    "OENACE2025_DE_UNT.csv",
    sep="|",
    header=None,
    names=["code", "text"],
    encoding="latin1"
)

cot["source"] = "cot"
cal["source"] = "cal"
unt["source"] = "unt"  # NEW

print(f"COT rows: {len(cot)}")
print(f"CAL rows: {len(cal)}")
print(f"UNT rows: {len(unt)}")

# %%
print("\n🔗 STEP 2: Combine data...")

df = pd.concat([cot, cal], ignore_index=True)

df["text"] = df["text"].astype(str).str.lower().str.strip()
df["code"] = df["code"].astype(str).str.strip()

df["weight"] = df["source"].map({
    "cal": 1.2,
    "cot": 1.0
})

# (IMPORTANT) We do NOT include UNT in embeddings
# → avoid leaking noisy long texts directly

print(f"Total embedding rows: {len(df)}")

# %%
print("\n🧠 STEP 3: Load embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model loaded")

# %%
print("\n📦 STEP 4: Encode knowledge base...")
start = time.time()

embeddings = model.encode(
    df["text"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

print(f"✅ Encoding done in {time.time() - start:.2f}s")

# %%
print("\n📖 STEP 5: Code → description mapping...")

code_to_desc = (
    cot.groupby("code")["text"]
    .first()
    .to_dict()
)

# %%
print("\n📚 STEP 6: Build keyword index (COT + CAL + UNT)...")

code_keywords = defaultdict(set)


def add_keywords_from_df(dataframe, boost_factor=1.0):
    for _, row in dataframe.iterrows():
        code = str(row["code"])
        text = str(row["text"]).lower()

        words = re.findall(r"\b[a-zäöüß]{3,}\b", text)

        for w in words:
            code_keywords[code].add(w)


# base knowledge
add_keywords_from_df(cot)
add_keywords_from_df(cal)

# ✅ NEW: real-world enrichment
add_keywords_from_df(unt)

# remove overly common words
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

print("✅ Keyword index built (with UNT)")

# %%
print("\n✂️ STEP 7: Robust input extraction...")


def extract_relevant_parts(text):
    text = str(text).lower()

    noise_patterns = [
        r"sehr geehrte.*",
        r"mit freundlichen grüßen.*",
        r"ich bitte.*",
        r"bitte.*ändern.*",
        r"vielen dank.*",
        r"\d+ ?%",
        r"umsatz.*",
        r"haupttätigkeit.*",
        r"nebentätigkeit.*",
    ]

    for p in noise_patterns:
        text = re.sub(p, "", text)

    parts = re.split(
        r"[.,;!?]| und | sowie | außerdem | auch | bzw",
        text
    )

    parts = [
        p.strip()
        for p in parts
        if len(p.strip()) > 3
    ]

    return parts if parts else [text]


# %%
print("\n🔑 STEP 8: Keyword scoring...")


def keyword_score(query):
    query_words = set(re.findall(r"\b[a-zäöüß]{3,}\b", query.lower()))

    scores = defaultdict(float)

    for code, keywords in code_keywords.items():
        overlap = query_words & keywords

        if overlap:
            scores[code] += len(overlap) * 0.04  # slightly stronger now

    return scores


# %%
print("\n🔍 STEP 9: Prediction function...")


def predict_top_k(query, k=3):
    parts = extract_relevant_parts(query)

    code_scores = {}

    for part in parts:
        query_emb = model.encode([part], convert_to_numpy=True)

        sims = cosine_similarity(query_emb, embeddings)[0]

        for i, score in enumerate(sims):
            code = df.iloc[i]["code"]
            weight = df.iloc[i]["weight"]

            weighted_score = score * weight

            if code not in code_scores:
                code_scores[code] = []

            code_scores[code].append(weighted_score)

    final_scores = {}

    for code, scores in code_scores.items():
        max_score = max(scores)
        mean_score = sum(scores) / len(scores)
        vote_bonus = 0.04 * len(scores)

        final_scores[code] = (
            max_score * 0.7 +
            mean_score * 0.3 +
            vote_bonus
        )

    # ✅ keyword boost (now enriched with UNT)
    kw_scores = keyword_score(query)

    for code, kw_score in kw_scores.items():
        if code in final_scores:
            final_scores[code] += kw_score

    top_codes = sorted(
        final_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:k]

    results = []

    for code, score in top_codes:
        results.append({
            "code": code,
            "description": code_to_desc.get(code, ""),
            "score": float(score)
        })

    return {
        "label": query,
        "predictions": results
    }


# %%
print("\n📊 STEP 10: Evaluation using UNT (CRITICAL)...")

unt["text"] = unt["text"].astype(str).str.lower().str.strip()
unt["code"] = unt["code"].astype(str).str.strip()


def evaluate(sample_size=200):
    sample = unt.sample(sample_size, random_state=42)

    correct_top1 = 0
    correct_top3 = 0

    for _, row in sample.iterrows():
        result = predict_top_k(row["text"], k=3)

        preds = [p["code"] for p in result["predictions"]]

        if row["code"] == preds[0]:
            correct_top1 += 1

        if row["code"] in preds:
            correct_top3 += 1

    print(f"\n✅ Accuracy@1: {correct_top1/sample_size:.2%}")
    print(f"✅ Accuracy@3: {correct_top3/sample_size:.2%}")


# %%
print("\n🧪 STEP 11: Test examples...")

examples = [
    "Hotel mit Restaurant und Spa",
    "Taxi service und Transport von Personen",
    "Food truck mit takeaway",
    "Hallo Statistik, wir haben pools und bieten essen an und bringen leute über nacht unter lg firma"
]

for text in examples:
    print("\n" + "="*70)
    print(f"INPUT: {text}")

    result = predict_top_k(text, k=3)

    for p in result["predictions"]:
        print(f"  {p['code']} | {p['description']} ({p['score']:.3f})")

# %%
# run evaluation
evaluate(sample_size=200)

# %%
print("\n✅ ✅ SCRIPT FINISHED SUCCESSFULLY ✅")