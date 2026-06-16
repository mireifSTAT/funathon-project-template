# %%
import polars as pl
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sentence_transformers import SentenceTransformer

# %%
# 1. LOAD DATA
df = pl.read_parquet(
    "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet"
)

print(df.head())
print(f"Total rows: {len(df)}")

# %%
# 2. FILTER TO ÖNACE I + H
df = df.filter(
    pl.col("code").str.starts_with("I") |
    pl.col("code").str.starts_with("H")
)

print(f"Filtered rows (I + H): {len(df)}")
print(f"Unique codes: {df['code'].n_unique()}")

# %%
# 3. TRAIN / TEST SPLIT
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

X_train = train_df["label"].to_numpy()
y_train = train_df["code"].to_numpy()

X_test = test_df["label"].to_numpy()
y_test = test_df["code"].to_numpy()

print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

# %%
# 4. ENCODE LABELS
encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train)
y_test_enc = encoder.transform(y_test)

# %%
# 5. LOAD EMBEDDING MODEL (FAST & STRONG)
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Encoding training data...")
X_train_emb = model.encode(X_train, show_progress_bar=True)

print("Encoding test data...")
X_test_emb = model.encode(X_test, show_progress_bar=True)

# %%
# 6. TRAIN CLASSIFIER
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_emb, y_train_enc)

# %%
# 7. EVALUATE
y_pred = clf.predict(X_test_emb)
acc = accuracy_score(y_test_enc, y_pred)

print(f"Test accuracy: {acc:.4f}")

# %%
# 8. TOP-3 PREDICTIONS
def predict_top_k(texts, k=3):
    embeddings = model.encode(texts)
    probs = clf.predict_proba(embeddings)

    top_k_idx = np.argsort(probs, axis=1)[:, -k:][:, ::-1]

    top_k_codes = encoder.inverse_transform(top_k_idx.flatten()).reshape(top_k_idx.shape)
    top_k_scores = np.take_along_axis(probs, top_k_idx, axis=1)

    return top_k_codes, top_k_scores


# %%
# 9. TEST EXAMPLES
example_texts = X_test[:5]

top_k_codes, top_k_scores = predict_top_k(example_texts, k=3)

for i, text in enumerate(example_texts):
    print("\n" + "="*60)
    print(f"Text: {text}")

    for code, score in zip(top_k_codes[i], top_k_scores[i]):
        print(f"  {code} (score: {score:.3f})")

# %%