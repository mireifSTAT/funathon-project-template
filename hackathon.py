# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ÖNACE 2025 top-3 classifier (interactive / VS Code notebook-style version)

Improvements vs. baseline:
- Strict train/test split for UNT examples (prevents evaluation leakage)
- One enriched document per code (COT + CAL + train UNT)
- Dense retrieval with multilingual embeddings (E5 by default)
- Sparse retrieval with TF-IDF (1-2 grams)
- Controlled example-neighbour boost from train UNT only
- Lightweight domain synonym expansion and rule boosts
- Batch prediction for JSON / CSV / Excel input
- Top-1 / Top-3 evaluation on held-out UNT test split
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

print("✅ Imports loaded")

# %%
# -----------------------------
# Configuration
# -----------------------------
DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-small"
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.20
DEFAULT_TOP_K = 3

WEIGHTS = {
    "dense": 0.60,
    "tfidf": 0.25,
    "examples": 0.15,
}

RULE_BOOSTS = {
    "taxi": [("49320", 0.18)],
    "taxibetrieb": [("49320", 0.18)],
    "krankenfahrten": [("49320", 0.18)],
    "hotel": [("55101", 0.18)],
    "hotels": [("55101", 0.18)],
    "beherbergung": [("55101", 0.14)],
    "restaurant": [("56111", 0.18)],
    "restaurants": [("56111", 0.18)],
    "gaststaette": [("56111", 0.18)],
    "gaststaetten": [("56111", 0.18)],
    "imbiss": [("56113", 0.14), ("56121", 0.10)],
    "takeaway": [("56121", 0.18)],
    "mitnahme": [("56121", 0.14)],
    "foodtruck": [("56113", 0.18), ("56121", 0.12)],
    "food_truck": [("56113", 0.18), ("56121", 0.12)],
    "spa": [("96040", 0.08), ("55101", 0.06)],
    "wellness": [("96040", 0.10), ("55101", 0.06)],
}

SYNONYMS = {
    "taxibetrieb": "taxi personenbefoerderung",
    "krankenfahrten": "taxi personenbefoerderung krankentransport",
    "krankenfahrt": "taxi personenbefoerderung krankentransport",
    "takeaway": "mitnahme imbiss",
    "to-go": "mitnahme imbiss",
    "to go": "mitnahme imbiss",
    "food truck": "foodtruck imbiss mobil",
    "gaststaette": "restaurant",
    "gaststaetten": "restaurants",
    "spa": "wellness",
    "beherbergung": "hotel unterkunft",
}

NOISE_PATTERNS = [
    r"sehr geehrte[^\n]*",
    r"mit freundlichen[^\n]*",
    r"vielen dank[^\n]*",
    r"\b\d+\s?%\b",
]

print("✅ Config loaded")

# %%
# -----------------------------
# Utilities
# -----------------------------
def read_pipe_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="|",
        header=None,
        names=["code", "text"],
        encoding="latin1",
        engine="python",
        dtype={"code": str, "text": str},
    )


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = (
        text.replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
    )
    text = text.replace("&", " und ")
    text = re.sub(r"[^a-z0-9\s\-/,.;:+]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_noise(text: str) -> str:
    text = normalize_text(text)
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def apply_synonyms(text: str) -> str:
    out = text
    for src in sorted(SYNONYMS.keys(), key=len, reverse=True):
        dst = SYNONYMS[src]
        pattern = r"\b" + re.escape(src) + r"\b"
        out = re.sub(pattern, f"{src} {dst}", out)
    return re.sub(r"\s+", " ", out).strip()


def preprocess_text(text: str) -> str:
    return apply_synonyms(strip_noise(text))


def smart_split(text: str) -> List[str]:
    """
    Split only on stronger punctuation first, not on every 'und',
    to preserve business meaning like 'forschung und entwicklung'.
    """
    text = preprocess_text(text)
    parts = [p.strip() for p in re.split(r"[.;!?]+", text) if len(p.strip()) >= 4]
    return parts if parts else [text]


def minmax_scale(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    lo = float(np.min(scores))
    hi = float(np.max(scores))
    if math.isclose(lo, hi):
        return np.zeros_like(scores, dtype=float)
    return (scores - lo) / (hi - lo)


def dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen and str(value).strip():
            seen.add(value)
            out.append(value)
    return out


def infer_section(code: str) -> str:
    code = str(code)
    if not code:
        return ""
    if code[0].isalpha():
        return code[0]
    return code[:2]


@dataclass
class Prediction:
    code: str
    description: str
    score: float


print("✅ Utility functions ready")

# %%
# -----------------------------
# Classifier
# -----------------------------
class OENACEClassifier:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        test_size: float = DEFAULT_TEST_SIZE,
        random_state: int = DEFAULT_RANDOM_STATE,
        top_k_example_neighbors: int = 10,
    ) -> None:
        self.model_name = model_name
        self.test_size = test_size
        self.random_state = random_state
        self.top_k_example_neighbors = top_k_example_neighbors
        self.use_e5_prefix = "e5" in model_name.lower()

        self.model: SentenceTransformer | None = None
        self.code_docs: pd.DataFrame | None = None
        self.code_embeddings: np.ndarray | None = None
        self.train_unt: pd.DataFrame | None = None
        self.test_unt: pd.DataFrame | None = None
        self.train_unt_embeddings: np.ndarray | None = None
        self.vectorizer: TfidfVectorizer | None = None
        self.code_tfidf = None
        self.code_to_desc: Dict[str, str] = {}
        self.codes: List[str] = []

    def load_data(
        self,
        cot_path: str | Path = "OENACE2025_DE_COT.csv",
        cal_path: str | Path = "OENACE2025_DE_CAL.csv",
        unt_path: str | Path = "OENACE2025_DE_UNT.csv",
    ) -> None:
        cot = read_pipe_csv(cot_path)
        cal = read_pipe_csv(cal_path)
        unt = read_pipe_csv(unt_path)

        cot["source"] = "cot"
        cal["source"] = "cal"
        unt["source"] = "unt"

        for df_ in (cot, cal, unt):
            df_["code"] = df_["code"].astype(str).str.strip()
            df_["text"] = df_["text"].fillna("").map(preprocess_text)

        self.code_to_desc = cot.groupby("code")["text"].first().to_dict()

        try:
            train_unt, test_unt = train_test_split(
                unt,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=unt["code"],
            )
        except ValueError:
            train_unt, test_unt = train_test_split(
                unt,
                test_size=self.test_size,
                random_state=self.random_state,
                shuffle=True,
            )

        self.train_unt = train_unt.reset_index(drop=True)
        self.test_unt = test_unt.reset_index(drop=True)

        docs_df = pd.concat([cot, cal, self.train_unt], ignore_index=True)
        grouped = (
            docs_df.groupby("code")["text"]
            .apply(lambda s: " ".join(dedupe_preserve_order(s.tolist())))
            .reset_index(name="text")
        )
        grouped["description"] = grouped["code"].map(self.code_to_desc).fillna("")
        grouped["section"] = grouped["code"].map(infer_section)

        self.code_docs = grouped
        self.codes = grouped["code"].tolist()

    def _ensure_model(self) -> None:
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

    def _prepare_passages(self, texts: List[str]) -> List[str]:
        if self.use_e5_prefix:
            return [f"passage: {t}" for t in texts]
        return texts

    def _prepare_queries(self, texts: List[str]) -> List[str]:
        if self.use_e5_prefix:
            return [f"query: {t}" for t in texts]
        return texts

    def build_index(self) -> None:
        if self.code_docs is None or self.train_unt is None:
            raise RuntimeError("Call load_data() before build_index().")

        self._ensure_model()

        code_texts = self.code_docs["text"].tolist()
        self.code_embeddings = self.model.encode(
            self._prepare_passages(code_texts),
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        train_texts = self.train_unt["text"].tolist()
        self.train_unt_embeddings = self.model.encode(
            self._prepare_passages(train_texts),
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self.code_tfidf = self.vectorizer.fit_transform(code_texts)

    def _dense_code_scores(self, query: str) -> Dict[str, float]:
        assert self.model is not None
        assert self.code_embeddings is not None
        assert self.code_docs is not None

        parts = smart_split(query)
        q_embs = self.model.encode(
            self._prepare_queries(parts),
            batch_size=16,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        sims_per_part = cosine_similarity(q_embs, self.code_embeddings)
        combined = 0.65 * sims_per_part.max(axis=0) + 0.35 * sims_per_part.mean(axis=0)
        combined = minmax_scale(combined)
        return dict(zip(self.codes, combined.tolist()))

    def _tfidf_code_scores(self, query: str) -> Dict[str, float]:
        assert self.vectorizer is not None
        assert self.code_tfidf is not None

        q = self.vectorizer.transform([preprocess_text(query)])
        sims = cosine_similarity(q, self.code_tfidf)[0]
        sims = minmax_scale(sims)
        return dict(zip(self.codes, sims.tolist()))

    def _example_neighbor_scores(self, query: str) -> Dict[str, float]:
        assert self.model is not None
        assert self.train_unt is not None
        assert self.train_unt_embeddings is not None

        q_emb = self.model.encode(
            self._prepare_queries([preprocess_text(query)]),
            batch_size=1,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        sims = cosine_similarity(q_emb, self.train_unt_embeddings)[0]

        if len(sims) == 0:
            return {}

        k = min(self.top_k_example_neighbors, len(sims))
        idx = np.argsort(sims)[-k:][::-1]

        scores = defaultdict(float)
        for rank, i in enumerate(idx, start=1):
            code = self.train_unt.iloc[int(i)]["code"]
            weight = 1.0 / math.log2(rank + 1.5)
            scores[code] += max(float(sims[int(i)]), 0.0) * weight

        if not scores:
            return {}

        max_score = max(scores.values())
        if max_score > 0:
            for code in list(scores.keys()):
                scores[code] /= max_score

        return dict(scores)

    def _rule_scores(self, query: str) -> Dict[str, float]:
        text = preprocess_text(query)
        words = set(re.findall(r"\b[a-z0-9_\-]{3,}\b", text.replace(" ", "_")))
        words.update(re.findall(r"\b[a-z0-9\-]{3,}\b", text))

        boosts = defaultdict(float)
        for token, rules in RULE_BOOSTS.items():
            if token in words or re.search(r"\b" + re.escape(token) + r"\b", text):
                for code, score in rules:
                    boosts[code] += score

        if boosts:
            max_score = max(boosts.values())
            if max_score > 0:
                for c in list(boosts.keys()):
                    boosts[c] /= max_score

        return dict(boosts)

    def predict_top_k(self, label: str, k: int = DEFAULT_TOP_K) -> Dict[str, object]:
        if self.code_docs is None:
            raise RuntimeError("Index not built. Call load_data() and build_index() first.")

        dense = self._dense_code_scores(label)
        tfidf = self._tfidf_code_scores(label)
        examples = self._example_neighbor_scores(label)
        rules = self._rule_scores(label)

        final_scores = {}
        for code in self.codes:
            final_scores[code] = (
                WEIGHTS["dense"] * dense.get(code, 0.0)
                + WEIGHTS["tfidf"] * tfidf.get(code, 0.0)
                + WEIGHTS["examples"] * examples.get(code, 0.0)
            )

        for code, boost in rules.items():
            final_scores[code] = final_scores.get(code, 0.0) + 0.08 * boost

        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        predictions = [
            {
                "code": code,
                "description": self.code_to_desc.get(code, ""),
                "score": float(score),
            }
            for code, score in ranked
        ]
        return {"label": label, "predictions": predictions}

    def evaluate(self, n: int | None = None, verbose_examples: int = 0) -> Dict[str, float]:
        if self.test_unt is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        eval_df = self.test_unt.copy()
        if n is not None:
            n = min(int(n), len(eval_df))
            eval_df = eval_df.sample(n=n, random_state=self.random_state).reset_index(drop=True)

        top1 = 0
        top3 = 0
        shown = 0

        for _, row in eval_df.iterrows():
            result = self.predict_top_k(row["text"], k=3)
            preds = [p["code"] for p in result["predictions"]]

            if preds and preds[0] == row["code"]:
                top1 += 1
            if row["code"] in preds:
                top3 += 1

            if shown < verbose_examples:
                shown += 1
                print("\n" + "=" * 80)
                print("TEXT:", row["text"])
                print("TRUE:", row["code"], self.code_to_desc.get(row["code"], ""))
                print("PRED:")
                for p in result["predictions"]:
                    print("  ", p)

        total = len(eval_df)
        return {
            "n": total,
            "accuracy_at_1": top1 / total if total else 0.0,
            "accuracy_at_3": top3 / total if total else 0.0,
        }

    def predict_dataframe(self, data: pd.DataFrame, label_column: str = "label") -> pd.DataFrame:
        if label_column not in data.columns:
            raise ValueError(f"Input file must contain column '{label_column}'.")

        results = []
        for label in data[label_column].fillna("").astype(str).tolist():
            pred = self.predict_top_k(label, k=3)["predictions"]
            row = {}
            for i, item in enumerate(pred, start=1):
                row[f"pred_{i}_code"] = item["code"]
                row[f"pred_{i}_description"] = item["description"]
                row[f"pred_{i}_score"] = float(item["score"])
            results.append(row)

        return pd.concat([data.reset_index(drop=True), pd.DataFrame(results)], axis=1)

    def predict_file(
        self,
        input_file: str | Path,
        output_file: str | Path,
        label_column: str = "label"
    ) -> None:
        input_file = Path(input_file)
        output_file = Path(output_file)

        suffix = input_file.suffix.lower()
        if suffix == ".xlsx":
            df = pd.read_excel(input_file, engine="openpyxl")
        elif suffix == ".csv":
            df = pd.read_csv(input_file)
        elif suffix == ".json":
            raw = json.loads(input_file.read_text(encoding="utf-8"))
            df = pd.DataFrame([raw]) if isinstance(raw, dict) else pd.DataFrame(raw)
        else:
            raise ValueError("Supported input formats: .xlsx, .csv, .json")

        out = self.predict_dataframe(df, label_column=label_column)

        out_suffix = output_file.suffix.lower()
        if out_suffix == ".xlsx":
            out.to_excel(output_file, index=False, engine="openpyxl")
        elif out_suffix == ".csv":
            out.to_csv(output_file, index=False)
        elif out_suffix == ".json":
            output_file.write_text(
                out.to_json(orient="records", force_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            raise ValueError("Supported output formats: .xlsx, .csv, .json")


print("✅ Class definition ready")

# %%
# -----------------------------
# Initialize classifier
# -----------------------------
clf = OENACEClassifier(
    model_name=DEFAULT_MODEL_NAME,
    test_size=DEFAULT_TEST_SIZE,
    random_state=DEFAULT_RANDOM_STATE,
)

print("✅ Classifier initialized")
print("Model:", clf.model_name)

# %%
# -----------------------------
# Load data
# -----------------------------
print("🚀 Loading data...")
clf.load_data(
    cot_path="OENACE2025_DE_COT.csv",
    cal_path="OENACE2025_DE_CAL.csv",
    unt_path="OENACE2025_DE_UNT.csv",
)

print(f"Train UNT rows: {len(clf.train_unt)}")
print(f"Test UNT rows:  {len(clf.test_unt)}")
print(f"Codes indexed:  {len(clf.code_docs)}")
clf.code_docs.head()

# %%
# -----------------------------
# Build index
# -----------------------------
print("🧠 Building index...")
start = time.time()
clf.build_index()
print(f"✅ Index ready in {time.time() - start:.2f}s")

# %%
# -----------------------------
# Single test predictions
# -----------------------------
examples = [
    "Hotel mit Restaurant und Spa",
    "Taxi service und Transport von Personen",
    "Food truck mit takeaway",
    "Wir betreiben einen Taxibetrieb und Krankenfahrten",
]

for e in examples:
    print("\n" + "=" * 80)
    print("INPUT:", e)
    result = clf.predict_top_k(e, k=3)
    for p in result["predictions"]:
        print(p)

# %%
# -----------------------------
# Predict one custom label
# -----------------------------
label = "Hotel with restaurant and spa services"
result = clf.predict_top_k(label, k=3)
result

# %%
# -----------------------------
# Evaluate on held-out test set
# -----------------------------
metrics = clf.evaluate(n=200, verbose_examples=3)
print(json.dumps(metrics, ensure_ascii=False, indent=2))


# %%
# -----------------------------
# Batch prediction from DataFrame
# -----------------------------
sample_df = pd.DataFrame(
    {
        "label": [
            "Hotel mit Restaurant und Spa",
            "Taxi und Krankenfahrten",
            "Food truck mit takeaway",
        ]
    }
)

predicted_df = clf.predict_dataframe(sample_df, label_column="label")
predicted_df

# %%
# -----------------------------
# Batch prediction from file
# -----------------------------
# Example:
# input.xlsx must contain a column named 'label'

# clf.predict_file(
#     input_file="input.xlsx",
#     output_file="predictions.xlsx",
#     label_column="label"
# )

print("✅ Uncomment the lines above to run file-based prediction")

# %%
# -----------------------------
# Save a few example predictions as JSON
# -----------------------------
demo_labels = pd.DataFrame(
    {
        "label": [
            "Hotel mit Restaurant und Spa",
            "Restaurant mit Mitnahme und Lieferservice",
            "Betrieb eines Taxidienstes inklusive Krankenfahrten",
        ]
    }
)

demo_predictions = clf.predict_dataframe(demo_labels)
Path("demo_predictions.json").write_text(
    demo_predictions.to_json(orient="records", force_ascii=False, indent=2),
    encoding="utf-8",
)

print("✅ Saved demo_predictions.json")
demo_predictions