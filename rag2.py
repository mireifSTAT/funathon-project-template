# %%
# Models
EMB_MODEL_NAME = "qwen3-embedding-8b"   # Embedding model
GEN_MODEL_NAME = "gemma4-26b-moe"          # Generative model

# Qdrant
COLLECTION_NAME = "nace-collection"
RETRIEVER_LIMIT = 5    # Number of candidates returned by the vector search

# Generation
TEMPERATURE = 0.1      # Low temperature → more deterministic, reproducible outputs

# Evaluation
SAMPLE_SIZE = 100       # Number of activities to evaluate (increase for more robust results)activity = "Installation, maintenance and repair of residential air conditioning systems for private customers"

response = client_llmlab.embeddings.create(
    model=EMB_MODEL_NAME,
    input=activity
)

search_embedding = response.data[0].embedding

print(f"Vector created of length: {len(search_embedding)}")
# %%
points = client_qdrant.query_points(
    collection_name=COLLECTION_NAME,
    query=search_embedding,
    limit=RETRIEVER_LIMIT,
)

descriptions_retrieved = []
codes_retrieved = []

for point in points.model_dump()["points"]:
    descriptions_retrieved.append(point["payload"]["text"])
    codes_retrieved.append(point["payload"]["code"])

print(
    f"✓ Vector search completed: {len(descriptions_retrieved)} codes and descriptions retrieved\n"
)
print("Check the first code retrieved ==============\n")
print(descriptions_retrieved[0])
# %%
SYSTEM_PROMPT = """\
You are an expert classifier for the NACE 2.1 nomenclature (Statistical Classification of Economic Activities in the European Community).

Given a company activity description and a short list of candidate NACE codes, your job is to pick the single most appropriate code from the candidates — or to declare the activity not codable if the description is too ambiguous.

Always reply with a valid JSON object matching the requested schema. No explanations, no extra text.
"""

USER_PROMPT_TEMPLATE = """\
## Activity to classify
{activity}

## Candidate NACE codes and their explanatory notes
{proposed_nace_descriptions}

## Rules
- Pick exactly one code from this list: [{proposed_nace_codes}]. Do not invent codes outside the list.
- If several activities are mentioned, only consider the first one.
- If the description is too vague to decide, return `nace_code: null` and `codable: false`.

## Output — valid JSON only
{{
  "nace_code": "<one code from the candidate list, or null>",
  "codable": <true | false>,
  "confidence": <float between 0.0 and 1.0>
}}
"""
# %%
from pydantic import BaseModel, Field
from typing import Optional

class NaceClassificationResult(BaseModel):
    nace_code: Optional[str] = Field(description="Chosen NACE code from the candidate list, or null")
    codable: bool = Field(description="False if the description is too vague to code")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")
import json

user_prompt = USER_PROMPT_TEMPLATE.format(
    activity=activity,
    proposed_nace_descriptions="## " + "\n\n## ".join(descriptions_retrieved),
    proposed_nace_codes=", ".join(codes_retrieved)
)

response = client_llmlab.chat.completions.parse(
    model=GEN_MODEL_NAME,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ],
    temperature=TEMPERATURE,
    response_format=NaceClassificationResult,
)

llm_response: NaceClassificationResult = response.choices[0].message.parsed
print(json.dumps(llm_response.model_dump(), indent=2))
# %%
import duckdb

con = duckdb.connect(database=":memory:")

con.execute("INSTALL httpfs;")
con.execute("LOAD httpfs;")

query_definition = f"""
SELECT *
FROM read_parquet(
  'https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet'
)
USING SAMPLE {SAMPLE_SIZE}
"""

annotations = (
    con.sql(query_definition)
    .to_df()
    .to_dict(orient="records")
)
print(f"Dataset loaded: {len(annotations)} rows")
print(f"Keys: {list(annotations[0].keys())}")
annotations[:2]
# %%
def run_rag_pipeline(activity: str) -> dict:
    """
    Run the full RAG pipeline for a single activity label.

    Parameters
    ----------
    activity : str
        Free-text economic activity label to be coded.

    Returns
    -------
    dict with keys:
        - nace_code (str | None) : predicted NACE code
        - codable (bool)        : True if the label could be coded
        - confidence (float)    : confidence score (0–1)
        - retrieved_codes (list): candidates returned by the retriever
    """
    # --- Step 1: Embedding ---
    emb_response = client_llmlab.embeddings.create(model=EMB_MODEL_NAME, input=activity)
    embedding = emb_response.data[0].embedding

    # --- Step 2: Retrieval ---
    points = client_qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=RETRIEVER_LIMIT,
    )
    descriptions_retrieved = []
    codes_retrieved = []
    for point in points.model_dump()["points"]:
        descriptions_retrieved.append(point["payload"]["text"])
        codes_retrieved.append(point["payload"]["code"])

    # --- Step 3: Prompt construction ---
    user_prompt = USER_PROMPT_TEMPLATE.format(
        activity=activity,
        proposed_nace_descriptions="## " + "\n\n## ".join(descriptions_retrieved),
        proposed_nace_codes=", ".join(codes_retrieved),
    )

    # --- Step 4: LLM inference ---
    gen_response = client_llmlab.chat.completions.parse(
        model=GEN_MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        response_format=NaceClassificationResult,
    )

    result = gen_response.choices[0].message.parsed.model_dump()
    # Keep retrieved candidates for retriever evaluation
    result["retrieved_codes"] = codes_retrieved

    return result
# %%
from tqdm import tqdm
import pandas as pd

records = []

for row in tqdm(annotations, total=len(annotations), desc="Coding"):
    activity_label = row["label"]
    true_code      = row["code"]

    try:
        pred = run_rag_pipeline(activity_label)
    except Exception as e:
        pred = {
            "nace_code":       None,
            "codable":        False,
            "confidence":     0.0,
            "retrieved_codes": []
        }
        print(f"⚠ Error for '{activity_label[:60]}...': {e}")

    records.append({
        "activity":        activity_label,
        "true_code":       true_code,
        "pred_code":       pred.get("nace_code"),
        "codable":         pred.get("codable", False),
        "confidence":      pred.get("confidence", 0.0),
        "retrieved_codes": pred.get("retrieved_codes", []),
    })

results = pd.DataFrame(records)
print(f"\n✓ Inference complete: {len(results)} activities processed")
results.head()
# %%
# Is the true code among the retriever's candidates?
results["retriever_hit"] = results.apply(
    lambda row: row["true_code"] in row["retrieved_codes"], axis=1
)

# Is the predicted code correct?
results["pipeline_correct"] = results["pred_code"] == results["true_code"]

# Did the LLM pick the right code, given that the retriever found it?
results["llm_correct_given_retriever"] = results.apply(
    lambda row: row["pipeline_correct"] if row["retriever_hit"] else None,
    axis=1
)
# %%
retriever_accuracy = results["retriever_hit"].mean()
print(f"Retriever@{RETRIEVER_LIMIT} accuracy: {retriever_accuracy:.1%}")
print(f"  → {results['retriever_hit'].sum()} / {len(results)} correctly retrieved")

retriever_success = results[results["retriever_hit"]]
llm_accuracy = retriever_success["pipeline_correct"].mean()

print(f"LLM accuracy (conditional on retriever): {llm_accuracy:.1%}")
print(f"  → {retriever_success['pipeline_correct'].sum()} / {len(retriever_success)} correctly coded by the LLM")
# %%
pipeline_accuracy = results["pipeline_correct"].mean()

print(f"Pipeline accuracy (end-to-end)          : {pipeline_accuracy:.1%}")
print(f"  → {results['pipeline_correct'].sum()} / {len(results)} correctly coded")
print()
print(f"Cross-check: Retriever@k × LLM = {retriever_accuracy:.3f} × {llm_accuracy:.3f} = {retriever_accuracy * llm_accuracy:.1%}")
# %%
n_total          = len(results)
n_retriever_miss = (~results["retriever_hit"]).sum()
n_llm_miss       = (results["retriever_hit"] & ~results["pipeline_correct"]).sum()
n_correct        = results["pipeline_correct"].sum()

print(
    "\n".join(
        [
            "=" * 52,
            "      DASHBOARD — RAG PIPELINE NACE 2.1",
            "=" * 52,
            f"  Activities processed        : {n_total:>6}",
            f"  Correctly coded             : {n_correct:>6}  ({pipeline_accuracy:.1%})",
            "",
            f"  Retriever@{RETRIEVER_LIMIT} accuracy        : {retriever_accuracy:>6.1%}",
            f"  LLM accuracy (conditional)  : {llm_accuracy:>6.1%}",
            f"  Pipeline accuracy           : {pipeline_accuracy:>6.1%}",
            "",
            f"  Retriever errors            : {n_retriever_miss:>6}  ({n_retriever_miss / n_total:.1%})",
            f"  LLM errors                  : {n_llm_miss:>6}  ({n_llm_miss / n_total:.1%})",
            "=" * 52,
        ]
    )
)
# %%
from plotnine import (
    ggplot, aes,
    geom_boxplot, geom_line, geom_point,
    scale_color_manual, scale_linetype_manual,
    labs, theme_minimal,
)


# --- Left: confidence distribution by correctness ---
results_plot = results.assign(
    correctness=results["pipeline_correct"].map({False: "Incorrect", True: "Correct"})
)

p1 = (
    ggplot(results_plot, aes(x="correctness", y="confidence"))
    + geom_boxplot()
    + labs(
        title="Confidence distribution by pipeline correctness",
        x="Prediction correct",
        y="Confidence score",
    )
    + theme_minimal()
)

# --- Right: precision and coverage vs confidence threshold ---
thresholds = [i / 10 for i in range(1, 10)]
rows = []
for t in thresholds:
    subset = results[results["confidence"] >= t]
    if len(subset) > 0:
        rows += [
            {"threshold": t, "metric": "Precision", "value": subset["pipeline_correct"].mean()},
            {"threshold": t, "metric": "Coverage",  "value": len(subset) / len(results)},
        ]

df_thresh = pd.DataFrame(rows)

p2 = (
    ggplot(df_thresh, aes(x="threshold", y="value", color="metric", linetype="metric"))
    + geom_line()
    + geom_point()
    + scale_color_manual(values={"Precision": "steelblue", "Coverage": "coral"})
    + scale_linetype_manual(values={"Precision": "solid", "Coverage": "dashed"})
    + labs(
        title="Precision and coverage vs. confidence threshold",
        x="Confidence threshold",
        y="Value",
        color="",
        linetype="",
    )
    + theme_minimal()
)

from IPython.display import display
display(p1)
display(p2)
# %%
