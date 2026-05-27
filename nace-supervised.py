# %%
# If you need to change working directory (default is your interactive .py file location)
# import os
# os.chdir("<NEW_RELATIVE_LOCATION>")

# %% 2.1 Question 1 — Import libraries and load environment variables
import mlflow 
from dotenv import load_dotenv
import polars as plrs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchTextClassifiers.value_encoder import ValueEncoder
from torchTextClassifiers.tokenizers import WordPieceTokenizer
from torchTextClassifiers import torchTextClassifiers, ModelConfig, TrainingConfig


load_dotenv(override=True)

# %% 2.2 Question 2 — Load the dataset from s3
df = plrs.read_parquet('https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet')

print(df.head())
print(f"Total rows: {len(df)}")

# %% 2.3 Question 3 — Count unique NACE codes
n_classes = df["code"].n_unique()
print(f"Unique classes: {n_classes}")

# %% 3.1 Question 1 — Split the dataset into train / validation / test sets
train_df, tmp_df = train_test_split(df, test_size=0.30, random_state=42)
val_df, test_df = train_test_split(tmp_df, test_size=0.50, random_state=42)

X_train, y_train = train_df["label"].to_numpy(), train_df["code"].to_numpy()
X_val, y_val = val_df["label"].to_numpy(), val_df["code"].to_numpy()
X_test, y_test = test_df["label"].to_numpy(), test_df["code"].to_numpy()

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# %% 3.2 Question 2 — Encode the labels
label_encoder = LabelEncoder()
label_encoder.fit(y_train)

all_codes = set(df['code'])
train_codes = set(train_df['code'])
missing = all_codes - train_codes

if missing:
    print(f"WARNING: {len(missing)} code(s) missing from training set: {missing}")
else:
    print(f"OK — all {len(all_codes)} codes appear in the training set.")
# %% 3.3 Question 3 — Prepare the labels to use them with ttc
value_encoder = ValueEncoder(label_encoder)

# %% 4.1 Why tokenization?
tokenizer = WordPieceTokenizer(vocab_size=5000, output_dim=10)
tokenizer.train(X_train)

print("Output tensor size:", tokenizer.tokenize(X_train[0]).input_ids.shape)
print("Vocabulary size:", tokenizer.vocab_size)

# Look at an example of tokenization
print("Raw text", X_train[0])
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
# %% 6.1 Question 1 — Create the classifier
# 2. Configure model
model_config = ModelConfig(
    embedding_dim=128,
    num_classes=n_classes
)

# 3. Train
classifier = torchTextClassifiers(
    tokenizer=tokenizer,
    model_config=model_config,
    value_encoder=value_encoder)

training_config = TrainingConfig(
    num_epochs=1,
    batch_size=128,
    lr=1e-3,
    patience_early_stopping=5)

mlflow.set_experiment("funathon-2026-project2")
mlflow.pytorch.autolog()

with mlflow.start_run() as run:
    # This should take approximately 1-2mn
    classifier.train(
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

# %%
