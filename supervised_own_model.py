# 2. Load the data

# %%
# Question 1 - Import libraries and load environment variables
# Based on: https://aiml4os.github.io/funathon-project2/1-ttc.html#question-1-import-libraries-and-load-environment-variables

from dotenv import load_dotenv

load_dotenv(override=True)

# %%
# Question 2 - Load the dataset from s3
# Based on: https://aiml4os.github.io/funathon-project2/1-ttc.html#question-2-load-the-dataset-from-s3

import polars as pl

url = "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet"

df = pl.read_parquet(url)

print("Columns:")
print(df.columns)

print("\nSchema:")
print(df.schema)

print("\nFirst rows:")
print(df.head())

print("\nTotal observations:")
print(df.height)

# %%
# Question 3 - Count unique NACE codes
# Based on: https://aiml4os.github.io/funathon-project2/1-ttc.html#question-3-count-unique-nace-codes

n_classes = df.n_unique

print(f"Number of unique NACE codes: {n_classes}")

# 3. Split the data

# %%
# Question 1 - Split the dataset into train / validation / test sets
# Based on: https://aiml4os.github.io/funathon-project2/1-ttc.html#question-1-split-the-dataset-into-train-validation-test-sets

from sklearn.model_selection import train_test_split
import numpy as np

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

# %%
# Question 2 - Encode the labels
# Based on: https://aiml4os.github.io/funathon-project2/1-ttc.html#question-2-encode-the-labels

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

encoder.fit(train_df["code"].to_numpy())

all_codes  = set(df['code'])
train_codes = set(train_df['code'])
missing = all_codes - train_codes

if missing:
    print(f"WARNING: {len(missing)} code(s) missing from training set: {missing}")
else:
    print(f"OK — all {len(all_codes)} codes appear in the training set.")

# %%
# Question 3 - Prepare the labels to use them with ttc
# Based on: https://aiml4os.github.io/funathon-project2/1-ttc.html#question-3-prepare-the-labels-to-use-them-with-ttc

from torchTextClassifiers.value_encoder import ValueEncoder

value_encoder = ValueEncoder(label_encoder=encoder)

# 4. Applying a tokenizer

# %%
# Question 1 - Train the tokenizer and inspect a sample
# Based on: https://aiml4os.github.io/funathon-project2/1-ttc.html#question-1-train-the-tokenizer-and-inspect-a-sample

from torchTextClassifiers.tokenizers import WordPieceTokenizer

tokenizer = WordPieceTokenizer(
    vocab_size=5000,
    output_dim=10
)

tokenizer.train(X_train)

print("Output tensor size:", tokenizer.tokenize(X_train[0]).input_ids.shape)
print("Vocabulary size:", tokenizer.vocab_size)

# Look at an example of tokenization
print("Raw text:", X_train[0])
print(
    "Tokens id:",
    tokenizer.tokenize(X_train[0]).input_ids.squeeze(0)
)
print(
    "Tokens:",
    tokenizer.tokenizer.convert_ids_to_tokens(
        tokenizer.tokenize(X_train[0]).input_ids.squeeze(0)
    )
)
# 6. Training

# %%
# Question 1 - Create the classifier
# Based on: https://aiml4os.github.io/funathon-project2/1-ttc.html#question-1-create-the-classifier

from torchTextClassifiers import ModelConfig, torchTextClassifiers

embedding_dim = 96

model_config = ModelConfig(
    embedding_dim=embedding_dim,
    num_classes=n_classes,)

ttc = torchTextClassifiers(
    tokenizer=tokenizer,
    model_config=model_config,
    value_encoder=value_encoder,
)

# %%
# Question 2 - Prepare training
# Based on: https://aiml4os.github.io/funathon-project2/1-ttc.html#question-2-prepare-training

from torchTextClassifiers import TrainingConfig

training_config = TrainingConfig(
    num_epochs=1,
    lr=5 * 1e-4,
    batch_size=128,
    patience_early_stopping=5,
)

# %%
# Question 3 - Train on a small subsample
# Based on: https://aiml4os.github.io/funathon-project2/1-ttc.html#question-3-train-on-a-small-subsample

import mlflow

mlflow.set_experiment("funathon-2026-project2")
mlflow.pytorch.autolog()

with mlflow.start_run() as run:
    # This should take approximately 1-2mn
    ttc.train(
        X_train,
        y_train,
        training_config=training_config,
        X_val=X_val,
        y_val=y_val,
        verbose=True,
    )

    mlflow.log_artifacts(
        training_config.save_path,   # local folder produced by ttc.train()
        artifact_path="model_artifacts",
    )