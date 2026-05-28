# %%
# Imported from file "supervised_own_model.py"
import polars as pl
from sklearn.model_selection import train_test_split

url = "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet"

df = pl.read_parquet(url)

random_state=42

# train (70%)
train_df, tmp_df = train_test_split(df, test_size=0.30, random_state=random_state)

# validation (15%) + test (15%) of tmp_df (leftover 30%)
val_df, test_df  = train_test_split(tmp_df, test_size=0.50, random_state=random_state)

# X = features, y = target
X_train, y_train = train_df["label"].to_numpy(), train_df["code"].to_numpy()
X_val, y_val = val_df["label"].to_numpy(), val_df["code"].to_numpy()
X_test, y_test = test_df["label"].to_numpy(), test_df["code"].to_numpy()

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# 7. Prediction and explainability

# %%
# Question 0 - Load the pretrained model from MLflow
# Based on: https://aiml4os.github.io/funathon-project2/1-ttc.html#question-0-load-the-pretrained-model-from-mlflow

import s3fs
from torchTextClassifiers import torchTextClassifiers

fs = s3fs.S3FileSystem(
    anon=True,  # public bucket
    endpoint_url="https://minio.lab.sspcloud.fr",
)

local_dir = "./mlflow-artifacts/"
fs.get(
    "projet-funathon/diffusion/mlflow-artifacts/",
    local_dir,
    recursive=True,
)
# Rebuild the torchTextClassifiers object from the downloaded files
ttc = torchTextClassifiers.load(local_dir)

ttc.pytorch_model.eval()

# %%
# Question 1 - Generate top-5 predictions with confidence scores
# Based on: https://aiml4os.github.io/funathon-project2/1-ttc.html#question-1-generate-top-5-predictions-with-confidence-scores

import random
import numpy as np

random_indices = random.sample(range(len(X_test)), 3)
example_texts = X_test[random_indices]
example_true_codes = y_test[random_indices]
print(example_texts)
top_k = 5

results = ttc.predict(example_texts, top_k=top_k, explain_with_captum=True)
for i, text in enumerate(example_texts):
    predicted_codes = [results["prediction"][i][k] for k in range(top_k)]
    confidence = [results["confidence"][i][k].item() for k in range(top_k)]
    print(f"\nText: {text}")
    print(f"  True code: {example_true_codes[i]}")
    for code, conf in zip(predicted_codes, confidence):
        print(f"  {code}  (confidence: {conf:.3f})")

# %%
# Question 2 - Visualise word attributions for the top prediction
# Based on: https://aiml4os.github.io/funathon-project2/1-ttc.html#question-2-visualise-word-attributions-for-the-top-prediction

from torchTextClassifiers.utilities.plot_explainability import (
    map_attributions_to_char, map_attributions_to_word,
    plot_attributions_at_char, plot_attributions_at_word, figshow,
)

text_idx = 0
top_k_idx = 0
text_sample = example_texts[text_idx]
offsets = results["offset_mapping"][text_idx]
word_ids = results["word_ids"][text_idx]
predicted_code = results["prediction"][text_idx][top_k_idx]

attributions = results["captum_attributions"][text_idx][top_k_idx]

words, word_attributions = map_attributions_to_word(
    attributions.unsqueeze(0), text_sample, word_ids, offsets
)
char_attributions = map_attributions_to_char(attributions.unsqueeze(0), offsets, text_sample)

titles = [f"Attributions for NACE code {predicted_code}"]

figshow(plot_attributions_at_char(
    text=text_sample, attributions_per_char=char_attributions, titles=titles,
)[0])

figshow(plot_attributions_at_word(
    text=text_sample, words=words.values(), attributions_per_word=word_attributions, titles=titles,
)[0])

# %%
# Question 3 - Evaluate accuracy on the test set
# Based on: https://aiml4os.github.io/funathon-project2/1-ttc.html#question-3-evaluate-accuracy-on-the-test-set

results_test = ttc.predict(X_test, top_k=1)
preds = results_test["prediction"].squeeze(1)
accuracy = (preds == y_test).mean()
print(f"Test accuracy: {accuracy:.4f} ({int(accuracy * len(y_test))}/{len(y_test)} correct)")
