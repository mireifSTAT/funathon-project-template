# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ÖNACE 2025 top-3 classifier (revised hackathon version)

What is improved vs. baseline:
- Safer per-class train/test split for rare classes
- Better description fallback (COT -> CAL -> UNT raw text)
- Preserves original human-readable descriptions
- Dense retrieval with multilingual embeddings (E5 by default)
- Sparse retrieval with TF-IDF (1-2 grams)
- Controlled example-neighbour voting from train UNT only
- Optional cross-encoder reranking on top candidates
- Softmax-based score normalization (more stable than min-max)
- Rule validation against actual indexed codes
- Lightweight score caching for repeated labels
- Batch prediction for JSON / CSV / Excel input
- Top-1 / Top-3 / Top-5 evaluation on held-out UNT test split
- Rich diagnostics for demo and evaluation
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import defaultdict
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# CrossEncoder is optional; we import it defensively
try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

print("✅ Imports loaded")

# %%
# -----------------------------
# Configuration
# -----------------------------
@dataclass
class OENACEConfig:
    model_name: str = "intfloat/multilingual-e5-small"
    reranker_name: Optional[str] = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    use_reranker: bool = True

    random_state: int = 42
    test_size: float = 0.20
    top_k: int = 3
    top_k_example_neighbors: int = 10
    candidate_pool_size: int = 15

    dense_weight: float = 0.50
    tfidf_weight: float = 0.20
    examples_weight: float = 0.15
    rerank_weight: float = 0.15

    # Rules are applied multiplicatively:
    # final_score *= (1 + rule_multiplier * rule_score)
    rule_multiplier: float = 0.08

    dense_temperature: float = 0.10
    tfidf_temperature: float = 0.08
    rerank_temperature: float = 0.12


RULE_BOOSTS = {
    # Accommodation
    "hotel": [("55101", 0.35)],
    "hotels": [("55101", 0.35)],
    "hotelpension": [("55101", 0.45), ("55102", 0.20)],
    "als hotel gefuehrt": [("55101", 0.55)],
    "beherbergung": [("55101", 0.20)],
    "pension": [("55102", 0.30)],
    "gasthof": [("55102", 0.30)],
    "gasthoefe": [("55102", 0.30)],

    # Food service
    "restaurant": [("56111", 0.30)],
    "restaurants": [("56111", 0.30)],
    "gaststaette": [("56111", 0.30)],
    "gaststaetten": [("56111", 0.30)],

    # Take-away / snack shop
    "imbiss": [("56112", 0.25)],
    "takeaway": [("56112", 0.20), ("56120", 0.10)],
    "mitnahme": [("56112", 0.20), ("56120", 0.10)],

    # Mobile food = 56120
    "foodtruck": [("56120", 0.55)],
    "food truck": [("56120", 0.55)],
    "imbisswagen": [("56120", 0.55)],
    "mobil": [("56120", 0.10)],

    # Wellness
    "spa": [("55101", 0.12)],
    "wellness": [("55101", 0.15)],
}

SYNONYMS = {
    "taxibetrieb": "taxi personenbefoerderung",
    "krankenfahrten": "taxi personenbefoerderung krankentransport",
    "krankenfahrt": "taxi personenbefoerderung krankentransport",

    "takeaway": "mitnahme imbiss",
    "to-go": "mitnahme imbiss",
    "to go": "mitnahme imbiss",

    "food truck": "foodtruck imbisswagen mobil",
    "foodtruck": "imbisswagen mobil",
    "imbisswagen": "foodtruck mobil",

    "gaststaette": "restaurant",
    "gaststaetten": "restaurants",

    "spa": "wellness",
    "beherbergung": "hotel unterkunft",

    "hotelpension": "hotel pension beherbergung",
    "als hotel gefuehrt": "hotel hotelbetrieb",
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
    """
    Robust reader for code|text files.

    Handles:
    - normal lines:          55101|Hotels
    - glued records:         53300|...bestellen55101|Hotels
    - continuation lines:    <no pipe> -> append to previous record text
    """
    path = Path(path)
    content = path.read_text(encoding="latin1")

    rows = []
    repaired_glued = []
    repaired_continuations = []

    code_pattern = r"[A-Z]?\d{5}\|"

    for lineno, raw in enumerate(content.splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue

        pipe_count = raw.count("|")

        # Case 1: normal record
        if pipe_count == 1:
            code, text = raw.split("|", 1)
            rows.append([code.strip(), text.strip()])
            continue

        # Case 2: continuation line (no separator) -> append to previous text
        if pipe_count == 0:
            if not rows:
                raise ValueError(f"Found continuation line before first record at line {lineno}: {raw[:200]}")
            rows[-1][1] = f"{rows[-1][1]} {raw}".strip()
            repaired_continuations.append((lineno, raw[:120]))
            continue

        # Case 3: glued multiple records in one line
        chunks = [c for c in re.split(rf"(?={code_pattern})", raw) if c.strip()]
        local_rows = []

        for chunk in chunks:
            if chunk.count("|") == 1:
                code, text = chunk.split("|", 1)
                local_rows.append([code.strip(), text.strip()])
            elif chunk.count("|") == 0 and local_rows:
                # continuation inside a glued chunk
                local_rows[-1][1] = f"{local_rows[-1][1]} {chunk}".strip()

        if local_rows:
            rows.extend(local_rows)
            repaired_glued.append((lineno, raw[:200]))
        else:
            raise ValueError(f"Could not parse line {lineno}: {raw[:200]}")

    if repaired_glued:
        print("⚠️ Repaired glued multi-record lines:")
        for lineno, preview in repaired_glued[:10]:
            print(f"  line {lineno}: {preview}")

    if repaired_continuations:
        print("⚠️ Repaired continuation lines:")
        for lineno, preview in repaired_continuations[:10]:
            print(f"  line {lineno}: {preview}")
        if len(repaired_continuations) > 10:
            print(f"  ... and {len(repaired_continuations) - 10} more")

    df = pd.DataFrame(rows, columns=["code", "text"])
    return df

def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = (
        text.replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
    )
    text = text.replace("&amp;", " und ")
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


def softmax_scale(scores: np.ndarray, temperature: float = 0.10) -> np.ndarray:
    if scores.size == 0:
        return scores
    x = np.asarray(scores, dtype=float)
    x = x - np.max(x)
    temperature = max(float(temperature), 1e-6)
    ex = np.exp(x / temperature)
    denom = ex.sum()
    if denom <= 0 or not np.isfinite(denom):
        return np.zeros_like(x, dtype=float)
    return ex / denom


def dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        value = str(value).strip()
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def infer_section(code: Optional[str]) -> str:
    code = str(code or "").strip()
    if not code:
        return ""
    if code[0].isalpha():
        return code[0]
    return code[:2]


def first_non_empty(series: pd.Series) -> str:
    for x in series:
        x = str(x).strip()
        if x:
            return x
    return ""


def safe_group_split(
    df: pd.DataFrame,
    code_col: str = "code",
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by class more safely:
    - singletons stay in train
    - each class keeps at least one training example
    """
    rng = np.random.default_rng(random_state)
    train_parts = []
    test_parts = []

    for _, group in df.groupby(code_col, sort=False):
        group = group.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        n = len(group)

        if n <= 1:
            train_parts.append(group)
            continue

        n_test = max(1, int(round(n * test_size)))
        n_test = min(n_test, n - 1)  # keep at least one in train

        test_idx = sorted(rng.choice(n, size=n_test, replace=False).tolist())
        test_parts.append(group.iloc[test_idx])
        train_parts.append(group.drop(index=test_idx))

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=df.columns)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=df.columns)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


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
    def __init__(self, config: Optional[OENACEConfig] = None) -> None:
        self.config = config or OENACEConfig()
        self.use_e5_prefix = "e5" in self.config.model_name.lower()

        self.model: Optional[SentenceTransformer] = None
        self.reranker = None

        self.code_docs: Optional[pd.DataFrame] = None
        self.code_embeddings: Optional[np.ndarray] = None

        self.train_unt: Optional[pd.DataFrame] = None
        self.test_unt: Optional[pd.DataFrame] = None
        self.train_unt_embeddings: Optional[np.ndarray] = None

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.code_tfidf = None

        self.code_to_desc: Dict[str, str] = {}
        self.codes: List[str] = []

        self._score_cache: Dict[str, pd.DataFrame] = {}

    def clear_cache(self) -> None:
        self._score_cache = {}

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

        # Preserve raw text for human-readable descriptions
        for df_ in (cot, cal, unt):
            df_["code"] = df_["code"].astype(str).str.strip()
            df_["raw_text"] = df_["text"].fillna("").astype(str).str.strip()
            df_["text"] = df_["raw_text"].map(preprocess_text)

        # Better description fallback: COT -> CAL -> UNT raw_text
        desc_source = pd.concat(
            [
                cot[["code", "raw_text"]].assign(desc_priority=0),
                cal[["code", "raw_text"]].assign(desc_priority=1),
                unt[["code", "raw_text"]].assign(desc_priority=2),
            ],
            ignore_index=True,
        ).sort_values(["code", "desc_priority"])

        self.code_to_desc = (
            desc_source.groupby("code")["raw_text"]
            .apply(first_non_empty)
            .to_dict()
        )

        # Safer split
        self.train_unt, self.test_unt = safe_group_split(
            unt,
            code_col="code",
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )

        # Build one enriched document per code from COT + CAL + train UNT
        docs_df = pd.concat([cot, cal, self.train_unt], ignore_index=True)
        grouped = (
            docs_df.groupby("code")["text"]
            .apply(lambda s: " ".join(dedupe_preserve_order(s.tolist())))
            .reset_index(name="text")
        )
        grouped["description"] = grouped["code"].map(self.code_to_desc).fillna("")
        grouped["section"] = grouped["code"].map(infer_section)

        self.code_docs = grouped.reset_index(drop=True)
        self.codes = self.code_docs["code"].tolist()

        self.clear_cache()

    def _ensure_model(self) -> None:
        if self.model is None:
            self.model = SentenceTransformer(self.config.model_name)

    def _ensure_reranker(self) -> None:
        if not self.config.use_reranker:
            return
        if self.reranker is not None:
            return
        if CrossEncoder is None:
            print("⚠️ CrossEncoder import not available. Continuing without reranker.")
            self.config.use_reranker = False
            return
        if not self.config.reranker_name:
            self.config.use_reranker = False
            return

        try:
            self.reranker = CrossEncoder(self.config.reranker_name)
        except Exception as e:
            print(f"⚠️ Could not load reranker '{self.config.reranker_name}'. Continuing without reranker.")
            print(f"   Reason: {e}")
            self.config.use_reranker = False
            self.reranker = None

    def _prepare_passages(self, texts: List[str]) -> List[str]:
        if self.use_e5_prefix:
            return [f"passage: {t}" for t in texts]
        return texts

    def _prepare_queries(self, texts: List[str]) -> List[str]:
        if self.use_e5_prefix:
            return [f"query: {t}" for t in texts]
        return texts

    def validate_rules(self) -> None:
        indexed_codes = set(self.codes)
        unknown = sorted(
            {
                code
                for rules in RULE_BOOSTS.values()
                for code, _ in rules
                if code not in indexed_codes
            }
        )
        if unknown:
            print("⚠️ RULE_BOOSTS contain codes not present in the index:")
            print("   ", unknown)
        else:
            print("✅ All rule codes exist in the current index")

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
        if len(train_texts) > 0:
            self.train_unt_embeddings = self.model.encode(
                self._prepare_passages(train_texts),
                batch_size=32,
                convert_to_numpy=True,
                show_progress_bar=True,
                normalize_embeddings=True,
            )
        else:
            self.train_unt_embeddings = np.zeros((0, self.code_embeddings.shape[1]), dtype=float)

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self.code_tfidf = self.vectorizer.fit_transform(code_texts)

        self.validate_rules()
        self.clear_cache()

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
        combined = softmax_scale(combined, temperature=self.config.dense_temperature)
        return dict(zip(self.codes, combined.tolist()))

    def _tfidf_code_scores(self, query: str) -> Dict[str, float]:
        assert self.vectorizer is not None
        assert self.code_tfidf is not None

        q = self.vectorizer.transform([preprocess_text(query)])
        sims = cosine_similarity(q, self.code_tfidf)[0]
        sims = softmax_scale(sims, temperature=self.config.tfidf_temperature)
        return dict(zip(self.codes, sims.tolist()))

    def _example_neighbor_scores(self, query: str) -> Dict[str, float]:
        assert self.model is not None
        assert self.train_unt is not None
        assert self.train_unt_embeddings is not None

        if len(self.train_unt) == 0 or self.train_unt_embeddings.shape[0] == 0:
            return {}

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

        k = min(self.config.top_k_example_neighbors, len(sims))
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

        # Phrase checks + token checks
        tokens = set(re.findall(r"\b[a-z0-9_\-]{3,}\b", text))
        boosts = defaultdict(float)

        for token, rules in RULE_BOOSTS.items():
            token_hit = False
            if token in tokens:
                token_hit = True
            elif re.search(r"\b" + re.escape(token) + r"\b", text):
                token_hit = True

            if token_hit:
                for code, score in rules:
                    boosts[code] += score

        if boosts:
            max_score = max(boosts.values())
            if max_score > 0:
                for c in list(boosts.keys()):
                    boosts[c] /= max_score

        return dict(boosts)

    def _get_candidate_codes(
        self,
        label: str,
        dense: Dict[str, float],
        tfidf: Dict[str, float],
        examples: Dict[str, float],
        rules: Dict[str, float],
        top_n: Optional[int] = None,
    ) -> List[str]:
        top_n = top_n or self.config.candidate_pool_size

        def top_codes(score_dict: Dict[str, float], n: int) -> List[str]:
            return [k for k, _ in sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:n]]

        candidates = []
        candidates += top_codes(dense, top_n)
        candidates += top_codes(tfidf, top_n)
        candidates += top_codes(examples, max(5, top_n // 2))
        candidates += top_codes(rules, max(3, top_n // 3))

        return dedupe_preserve_order(candidates)

    def _rerank_candidates(self, label: str, candidate_codes: List[str]) -> Dict[str, float]:
        if not self.config.use_reranker:
            return {}
        if not candidate_codes:
            return {}

        self._ensure_reranker()
        if self.reranker is None:
            return {}

        assert self.code_docs is not None
        code_to_text = dict(zip(self.code_docs["code"], self.code_docs["text"]))

        pairs = [(label, code_to_text.get(c, "")) for c in candidate_codes]
        try:
            scores = self.reranker.predict(pairs)
            scores = np.asarray(scores, dtype=float)
            scores = softmax_scale(scores, temperature=self.config.rerank_temperature)
            return dict(zip(candidate_codes, scores.tolist()))
        except Exception as e:
            print(f"⚠️ Reranking failed; continuing without reranker. Reason: {e}")
            return {}

    def score_all_codes(self, label: str) -> pd.DataFrame:
        if self.code_docs is None:
            raise RuntimeError("Index not built. Call load_data() and build_index() first.")

        cache_key = str(label)
        if cache_key in self._score_cache:
            return self._score_cache[cache_key].copy()

        dense = self._dense_code_scores(label)
        tfidf = self._tfidf_code_scores(label)
        examples = self._example_neighbor_scores(label)
        rules = self._rule_scores(label)

        candidate_codes = self._get_candidate_codes(label, dense, tfidf, examples, rules)
        rerank = self._rerank_candidates(label, candidate_codes)

        rows = []
        for code in self.codes:
            dense_score = float(dense.get(code, 0.0))
            tfidf_score = float(tfidf.get(code, 0.0))
            example_score = float(examples.get(code, 0.0))
            rerank_score = float(rerank.get(code, 0.0))
            rule_score = float(rules.get(code, 0.0))

            base_score = (
                self.config.dense_weight * dense_score
                + self.config.tfidf_weight * tfidf_score
                + self.config.examples_weight * example_score
                + self.config.rerank_weight * rerank_score
            )

            # Rules reinforce candidates instead of overpowering them
            rule_add = self.config.rule_multiplier
            rule_mult = 0.5 * self.config.rule_multiplier

            final_score = base_score + rule_add * rule_score
            final_score *= (1.0 + rule_mult * rule_score)

            rows.append(
                {
                    "code": code,
                    "description": self.code_to_desc.get(code, ""),
                    "dense_score": dense_score,
                    "tfidf_score": tfidf_score,
                    "example_score": example_score,
                    "rerank_score": rerank_score,
                    "rule_score": rule_score,
                    "base_score": base_score,
                    "final_score": final_score,
                    "section": infer_section(code),
                }
            )

        ranked = (
            pd.DataFrame(rows)
            .sort_values("final_score", ascending=False)
            .reset_index(drop=True)
        )
        ranked["rank"] = np.arange(1, len(ranked) + 1)

        self._score_cache[cache_key] = ranked.copy()
        return ranked

    def predict_top_k(self, label: str, k: Optional[int] = None) -> Dict[str, object]:
        k = k or self.config.top_k
        ranked = self.score_all_codes(label).head(k)

        predictions = []
        for _, row in ranked.iterrows():
            predictions.append(
                {
                    "code": row["code"],
                    "description": row["description"],
                    "score": float(row["final_score"]),
                    "dense_score": float(row["dense_score"]),
                    "tfidf_score": float(row["tfidf_score"]),
                    "example_score": float(row["example_score"]),
                    "rerank_score": float(row["rerank_score"]),
                    "rule_score": float(row["rule_score"]),
                }
            )

        return {"label": label, "predictions": predictions}

    def evaluate(self, n: Optional[int] = None, verbose_examples: int = 0, show_progress: bool = True) -> Dict[str, float]:
        summary, _, _, _, _ = self.evaluate_detailed(
            n=n,
            verbose_examples=verbose_examples,
            show_progress=show_progress,
        )
        return summary

    def evaluate_detailed(self, n: Optional[int] = None, verbose_examples: int = 0, show_progress: bool = True,):
        if self.test_unt is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        eval_df = self.test_unt.copy()
        if n is not None:
            n = min(int(n), len(eval_df))
            eval_df = eval_df.sample(n=n, random_state=self.config.random_state).reset_index(drop=True)

        rows = []
        shown = 0

        iterable = eval_df.iterrows()

        progress = None
        if show_progress and tqdm is not None:
            progress = tqdm(
                iterable,
                total=len(eval_df),
                desc="Evaluating",
                unit="sample",
            )
            iterable = progress
        elif show_progress:
            print(f"Evaluating {len(eval_df)} samples...")

        correct_1_running = 0
        correct_3_running = 0

        for idx, (_, row) in enumerate(iterable, start=1):
            label = row["text"]
            true_code = row["code"]

            ranked = self.score_all_codes(label)
            top5 = ranked.head(5).copy()
            true_match = ranked[ranked["code"] == true_code]

            true_rank = int(true_match["rank"].iloc[0]) if not true_match.empty else None
            true_score = float(true_match["final_score"].iloc[0]) if not true_match.empty else 0.0

            pred_codes = top5["code"].tolist()
            pred_scores = top5["final_score"].tolist()

            pred_1_code = pred_codes[0] if len(pred_codes) >= 1 else None
            pred_2_code = pred_codes[1] if len(pred_codes) >= 2 else None
            pred_3_code = pred_codes[2] if len(pred_codes) >= 3 else None

            pred_1_score = float(pred_scores[0]) if len(pred_scores) >= 1 else 0.0
            pred_2_score = float(pred_scores[1]) if len(pred_scores) >= 2 else 0.0
            pred_3_score = float(pred_scores[2]) if len(pred_scores) >= 3 else 0.0

            margin_12 = pred_1_score - pred_2_score if len(pred_scores) >= 2 else pred_1_score
            reciprocal_rank = 1.0 / true_rank if true_rank and true_rank > 0 else 0.0

            correct_at_1 = int(true_rank == 1)
            correct_at_3 = int(true_rank is not None and true_rank <= 3)
            correct_at_5 = int(true_rank is not None and true_rank <= 5)

            correct_1_running += correct_at_1
            correct_3_running += correct_at_3

            rows.append(
                {
                    "label": label,
                    "true_code": true_code,
                    "true_description": self.code_to_desc.get(true_code, ""),
                    "true_section": infer_section(true_code),
                    "true_rank": true_rank,
                    "true_score": true_score,
                    "reciprocal_rank": reciprocal_rank,

                    "pred_1_code": pred_1_code,
                    "pred_1_description": self.code_to_desc.get(pred_1_code, "") if pred_1_code else "",
                    "pred_1_score": pred_1_score,

                    "pred_2_code": pred_2_code,
                    "pred_2_description": self.code_to_desc.get(pred_2_code, "") if pred_2_code else "",
                    "pred_2_score": pred_2_score,

                    "pred_3_code": pred_3_code,
                    "pred_3_description": self.code_to_desc.get(pred_3_code, "") if pred_3_code else "",
                    "pred_3_score": pred_3_score,

                    "margin_12": margin_12,
                    "correct_at_1": correct_at_1,
                    "correct_at_3": correct_at_3,
                    "correct_at_5": correct_at_5,
                    "pred_1_section": infer_section(pred_1_code),
                    "same_section_as_top1": int(infer_section(true_code) == infer_section(pred_1_code)) if pred_1_code else 0,
                }
            )

            # Live progress info
            if progress is not None:
                progress.set_postfix(
                    acc1=f"{correct_1_running / idx:.2%}",
                    acc3=f"{correct_3_running / idx:.2%}",
                    refresh=False,
                )
            elif show_progress and idx % 25 == 0:
                print(
                    f"  processed {idx}/{len(eval_df)} | "
                    f"running Acc@1={correct_1_running / idx:.2%} | "
                    f"Acc@3={correct_3_running / idx:.2%}"
                )

            if shown < verbose_examples:
                shown += 1
                print("\n" + "=" * 100)
                print("TEXT:", label)
                print("TRUE:", true_code, self.code_to_desc.get(true_code, ""))
                print("TOP-5:")
                print(
                    top5[
                        [
                            "rank",
                            "code",
                            "description",
                            "final_score",
                            "base_score",
                            "dense_score",
                            "tfidf_score",
                            "example_score",
                            "rerank_score",
                            "rule_score",
                        ]
                    ].to_string(index=False)
                )

        if progress is not None:
            progress.close()

        details = pd.DataFrame(rows)

        if details.empty:
            summary = {
                "n": 0,
                "accuracy_at_1": 0.0,
                "accuracy_at_3": 0.0,
                "accuracy_at_5": 0.0,
                "mrr": 0.0,
                "mean_top1_score": 0.0,
                "mean_true_score": 0.0,
                "mean_margin_top1_top2": 0.0,
                "median_true_rank": 0.0,
                "same_section_as_top1_rate": 0.0,
            }
            return summary, details, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        summary = {
            "n": int(len(details)),
            "accuracy_at_1": float(details["correct_at_1"].mean()),
            "accuracy_at_3": float(details["correct_at_3"].mean()),
            "accuracy_at_5": float(details["correct_at_5"].mean()),
            "mrr": float(details["reciprocal_rank"].mean()),
            "mean_top1_score": float(details["pred_1_score"].mean()),
            "mean_true_score": float(details["true_score"].mean()),
            "mean_margin_top1_top2": float(details["margin_12"].mean()),
            "median_true_rank": float(details["true_rank"].median()),
            "same_section_as_top1_rate": float(details["same_section_as_top1"].mean()),
        }

        per_section = (
            details.groupby("true_section")
            .agg(
                n=("true_code", "count"),
                accuracy_at_1=("correct_at_1", "mean"),
                accuracy_at_3=("correct_at_3", "mean"),
                accuracy_at_5=("correct_at_5", "mean"),
                mrr=("reciprocal_rank", "mean"),
                mean_top1_score=("pred_1_score", "mean"),
            )
            .sort_values(["accuracy_at_3", "accuracy_at_1", "n"], ascending=[False, False, False])
            .reset_index()
        )

        confusions = (
            details[details["correct_at_1"] == 0]
            .groupby(["true_code", "pred_1_code"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .reset_index(drop=True)
        )
        if not confusions.empty:
            confusions["true_description"] = confusions["true_code"].map(self.code_to_desc).fillna("")
            confusions["pred_1_description"] = confusions["pred_1_code"].map(self.code_to_desc).fillna("")
            confusions = confusions[
                ["true_code", "true_description", "pred_1_code", "pred_1_description", "count"]
            ]

        thresholds = [0.20, 0.30, 0.40, 0.50, 0.60]
        threshold_rows = []
        total = len(details)

        for threshold in thresholds:
            subset = details[details["pred_1_score"] >= threshold]
            threshold_rows.append(
                {
                    "threshold": threshold,
                    "n": int(len(subset)),
                    "coverage": float(len(subset) / total if total else 0.0),
                    "precision_at_1": float(subset["correct_at_1"].mean()) if len(subset) else np.nan,
                    "precision_at_3": float(subset["correct_at_3"].mean()) if len(subset) else np.nan,
                }
            )
        threshold_table = pd.DataFrame(threshold_rows)

        return summary, details, per_section, confusions, threshold_table

    def quick_eval(self, n: int = 200) -> Dict[str, object]:
        summary, details, per_section, confusions, threshold_table = self.evaluate_detailed(n=n)
        return {
            "summary": summary,
            "per_section": per_section,
            "confusions": confusions.head(10),
            "thresholds": threshold_table,
            "details_preview": details.head(10),
        }

    def predict_dataframe(self, data: pd.DataFrame, label_column: str = "label") -> pd.DataFrame:
        if label_column not in data.columns:
            raise ValueError(f"Input file must contain column '{label_column}'.")

        rows = []
        for label in data[label_column].fillna("").astype(str).tolist():
            pred = self.predict_top_k(label, k=3)["predictions"]
            row = {}
            for i, item in enumerate(pred, start=1):
                row[f"pred_{i}_code"] = item["code"]
                row[f"pred_{i}_description"] = item["description"]
                row[f"pred_{i}_score"] = float(item["score"])
                row[f"pred_{i}_dense_score"] = float(item["dense_score"])
                row[f"pred_{i}_tfidf_score"] = float(item["tfidf_score"])
                row[f"pred_{i}_example_score"] = float(item["example_score"])
                row[f"pred_{i}_rerank_score"] = float(item["rerank_score"])
                row[f"pred_{i}_rule_score"] = float(item["rule_score"])
            rows.append(row)

        return pd.concat([data.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

    def predict_file(
        self,
        input_file: str | Path,
        output_file: str | Path,
        label_column: str = "label",
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

    def save_diagnostics(self, output_prefix: str = "evaluation") -> Dict[str, str]:
        summary, details, per_section, confusions, threshold_table = self.evaluate_detailed()

        json_path = f"{output_prefix}_summary.json"
        xlsx_path = f"{output_prefix}_diagnostics.xlsx"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            details.to_excel(writer, sheet_name="details", index=False)
            per_section.to_excel(writer, sheet_name="per_section", index=False)
            confusions.to_excel(writer, sheet_name="confusions", index=False)
            threshold_table.to_excel(writer, sheet_name="thresholds", index=False)

        return {"summary_json": json_path, "diagnostics_excel": xlsx_path}


print("✅ Class definition ready")

# %%
# -----------------------------
# Helper for a demo-friendly summary
# -----------------------------
def print_demo_summary(summary: dict, threshold_table: pd.DataFrame) -> None:
    print("🎤 DEMO SUMMARY")
    print("-" * 80)
    print(f"Evaluated samples:        {summary['n']}")
    print(f"Accuracy@1:              {summary['accuracy_at_1']:.2%}")
    print(f"Accuracy@3:              {summary['accuracy_at_3']:.2%}")
    print(f"Accuracy@5:              {summary['accuracy_at_5']:.2%}")
    print(f"MRR:                     {summary['mrr']:.3f}")
    print(f"Mean top-1 score:        {summary['mean_top1_score']:.3f}")
    print(f"Mean true score:         {summary['mean_true_score']:.3f}")
    print(f"Mean top1-top2 margin:   {summary['mean_margin_top1_top2']:.3f}")
    print(f"Median true rank:        {summary['median_true_rank']:.1f}")
    print(f"Same section as top-1:   {summary['same_section_as_top1_rate']:.2%}")
    print("\nConfidence / coverage:")
    print(threshold_table.to_string(index=False))


print("✅ Demo summary helper ready")

# %%
# -----------------------------
# Initialize classifier
# -----------------------------
config = OENACEConfig(
    model_name="intfloat/multilingual-e5-small",
    reranker_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    use_reranker=False,   # set False if you want pure retriever mode
    test_size=0.20,
    random_state=42,
    top_k=3,
    top_k_example_neighbors=10,
    candidate_pool_size=15,
    dense_weight=0.55,
    tfidf_weight=0.20,
    examples_weight=0.10,
    rerank_weight=0.05,
    rule_multiplier=0.10,
)

clf = OENACEClassifier(config=config)

print("✅ Classifier initialized")
print("Embedding model:", clf.config.model_name)
print("Reranker enabled:", clf.config.use_reranker)
if clf.config.use_reranker:
    print("Reranker model:", clf.config.reranker_name)

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
# Inspect full ranking for one label
# -----------------------------
label = "Hotel with restaurant and spa services"
full_ranking = clf.score_all_codes(label)
full_ranking.head(10)

# %%
# -----------------------------
# Predict one custom label
# -----------------------------
label = "Hotel with restaurant and spa services"
result = clf.predict_top_k(label, k=3)
result

# %%
# -----------------------------
# Evaluate on held-out test set (compact)
# -----------------------------
metrics = clf.evaluate(n=200, verbose_examples=3)
print(json.dumps(metrics, ensure_ascii=False, indent=2))

# %%
# -----------------------------
# Evaluate on held-out test set (detailed diagnostics)
# -----------------------------
summary, details, per_section, confusions, threshold_table = clf.evaluate_detailed(
    n=200,
    verbose_examples=2,
)
print(json.dumps(summary, ensure_ascii=False, indent=2))

# %%
# -----------------------------
# Print demo-friendly summary
# -----------------------------
print_demo_summary(summary, threshold_table)

# %%
# -----------------------------
# Most useful diagnostic tables
# -----------------------------
print("Top of details table:")
details.head(10)

# %%
print("Per-section performance:")
per_section

# %%
print("Most common top-1 confusions:")
confusions.head(15)

# %%
print("Confidence / coverage table:")
threshold_table

# %%
# -----------------------------
# Best and worst cases for demo
# -----------------------------
best_cases = (
    details[details["correct_at_1"] == 1]
    .sort_values(["pred_1_score", "margin_12"], ascending=False)
    .head(10)
)

best_cases[
    [
        "label",
        "true_code",
        "true_description",
        "pred_1_code",
        "pred_1_description",
        "pred_1_score",
        "margin_12",
    ]
]

# %%
# -----------------------------
# High-confidence wrong predictions
# -----------------------------
hard_failures = (
    details[details["correct_at_1"] == 0]
    .sort_values(["pred_1_score", "margin_12"], ascending=False)
    .head(10)
)

hard_failures[
    [
        "label",
        "true_code",
        "true_description",
        "pred_1_code",
        "pred_1_description",
        "pred_1_score",
        "pred_2_code",
        "pred_2_description",
        "pred_3_code",
        "pred_3_description",
        "true_rank",
    ]
]

# %%
# -----------------------------
# Cases where top-1 is wrong but top-3 is correct
# -----------------------------
rescued_by_top3 = (
    details[(details["correct_at_1"] == 0) & (details["correct_at_3"] == 1)]
    .sort_values("true_rank")
    .head(15)
)

rescued_by_top3[
    [
        "label",
        "true_code",
        "true_description",
        "pred_1_code",
        "pred_2_code",
        "pred_3_code",
        "true_rank",
    ]
]

# %%
# -----------------------------
# Quick evaluation bundle
# -----------------------------
quick = clf.quick_eval(n=200)
quick["summary"]

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
# Save example predictions as JSON
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

# %%
# -----------------------------
# Save full diagnostics bundle
# -----------------------------
diagnostic_files = clf.save_diagnostics(output_prefix="evaluation_demo")
diagnostic_files