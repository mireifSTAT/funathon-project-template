# 7. Prediction and explainability

# %%
# Question 0 - Load the pretrained model from MLflow
# Based on: https://aiml4os.github.io/funathon-project2/1-ttc.html#question-0-load-the-pretrained-model-from-mlflow

import s3fs

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

# TBD