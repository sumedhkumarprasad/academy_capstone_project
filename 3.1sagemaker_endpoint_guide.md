# SageMaker Real-Time Endpoint — Complete Guide
### Build in Databricks → Deploy to AWS → Wire as Agent Tool

---

## Architecture Overview

```
Databricks Notebook
│
├─ STEP 1 ─ Train XGBoost (you already did this)
├─ STEP 2 ─ Package model + inference script → model.tar.gz
├─ STEP 3 ─ Upload model.tar.gz → S3
├─ STEP 4 ─ Register SageMaker Model (points to S3 + Docker image)
├─ STEP 5 ─ Create Endpoint Config (instance type, variant)
├─ STEP 6 ─ Create Endpoint (deploy — takes ~5 min)
├─ STEP 7 ─ Test endpoint with a real transaction row
└─ STEP 8 ─ Wire as invoke_fraud_model tool in the Agent
```

---

## STEP 0 — AWS Setup (paste at top of your Databricks notebook)

```python
import boto3, os, json, io
import pandas as pd
import numpy as np
import joblib, tarfile, time

STUDENT_NUM = "06"

os.environ["AWS_ACCESS_KEY_ID"]     = "AKIA6AK5B2HLV2E6FD6F"
os.environ["AWS_SECRET_ACCESS_KEY"] = "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"
os.environ["AWS_REGION"]            = "us-west-2"
os.environ["AWS_DEFAULT_REGION"]    = "us-west-2"

session          = boto3.Session(region_name="us-west-2")
sm_client        = session.client("sagemaker")
sm_runtime       = session.client("sagemaker-runtime")
s3_client        = session.client("s3")
sts              = session.client("sts")

ACCOUNT_ID       = sts.get_caller_identity()["Account"]
REGION           = "us-west-2"
BUCKET           = f"sagemaker-{REGION}-{ACCOUNT_ID}"   # default SageMaker bucket
S3_PREFIX        = f"student-{STUDENT_NUM}/fraud-model"
ROLE_ARN         = f"arn:aws:iam::{ACCOUNT_ID}:role/SageMakerExecutionRole"
ENDPOINT_NAME    = f"fraud-detector-student-{STUDENT_NUM}"

print(f"✅ Account : {ACCOUNT_ID}")
print(f"✅ Bucket  : s3://{BUCKET}/{S3_PREFIX}/")
print(f"✅ Endpoint: {ENDPOINT_NAME}")
```

> **Note on ROLE_ARN:** Your IAM role must have `AmazonSageMakerFullAccess` and
> `AmazonS3FullAccess` (or read access to the model bucket). Ask your instructor for
> the exact role ARN if the above default doesn't exist in your account.

---

## STEP 1 — Make Sure the S3 Bucket Exists

```python
# Create the bucket if it doesn't already exist
try:
    s3_client.head_bucket(Bucket=BUCKET)
    print(f"✅ Bucket s3://{BUCKET} already exists")
except:
    s3_client.create_bucket(
        Bucket=BUCKET,
        CreateBucketConfiguration={"LocationConstraint": REGION}
    )
    print(f"✅ Created bucket s3://{BUCKET}")
```

---

## STEP 2 — Write the Inference Script (inference.py)

SageMaker's XGBoost container calls four hook functions:
`model_fn`, `input_fn`, `predict_fn`, `output_fn`.
You must provide them so SageMaker knows how to load your model and
handle JSON payloads.

```python
# Write inference.py to local disk in Databricks (then package it)
INFERENCE_SCRIPT = '''
import os, json, io, joblib
import pandas as pd
import numpy as np

# ── These are the features your final model was trained on ──────────────────
# Replace this list with the output of your selected_features variable
# (the features that survived correlation filter + 80% Gini cut).
# Hardcoded here so the endpoint is self-contained.
SELECTED_FEATURES = []   # <-- paste your selected_features list here


def model_fn(model_dir):
    """Load the XGBoost model from the model directory SageMaker unpacks."""
    model_path = os.path.join(model_dir, "xgb_fraud_final.joblib")
    model = joblib.load(model_path)
    print(f"[model_fn] Model loaded from {model_path}")
    return model


def input_fn(request_body, request_content_type):
    """
    Deserialise the incoming request.
    Accepts:
      - application/json  : {"features": {col: value, ...}}
                            OR a flat dict {col: value, ...}
      - text/csv          : comma-separated values in SELECTED_FEATURES order
    """
    if request_content_type == "application/json":
        payload = json.loads(request_body)
        # Support both {"features": {...}} and flat {...}
        if "features" in payload:
            payload = payload["features"]
        df = pd.DataFrame([payload])
    elif request_content_type == "text/csv":
        df = pd.read_csv(io.StringIO(request_body), header=None)
        df.columns = SELECTED_FEATURES
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

    # Reorder / align columns to match training order
    df = df.reindex(columns=SELECTED_FEATURES, fill_value=0)
    return df


def predict_fn(input_data, model):
    """
    Run inference. Returns both the binary prediction and the fraud probability.
    """
    prob  = model.predict_proba(input_data)[:, 1]   # fraud probability
    pred  = (prob >= 0.5).astype(int)               # binary label
    return {"fraud_probability": float(prob[0]),
            "is_fraud_prediction": int(pred[0])}


def output_fn(prediction, accept):
    """Serialise the output dict to JSON."""
    return json.dumps(prediction), "application/json"
'''

# Write to disk
with open("/tmp/inference.py", "w") as f:
    f.write(INFERENCE_SCRIPT)
print("✅ inference.py written to /tmp/inference.py")
```

> **Important:** Before packaging, replace `SELECTED_FEATURES = []` in the script
> with the actual list from your training notebook:
> ```python
> # In your training notebook, after the Gini selection step:
> print(selected_features)   # copy this output into inference.py
> ```

---

## STEP 3 — Package Model + Inference Script → model.tar.gz

SageMaker's built-in XGBoost container expects this exact layout inside the tar:
```
model.tar.gz
├── xgb_fraud_final.joblib   ← your trained model
└── code/
    └── inference.py          ← your serving hooks
```

```python
MODEL_TAR_PATH = "/tmp/model.tar.gz"

with tarfile.open(MODEL_TAR_PATH, "w:gz") as tar:
    # Add the serialised model
    tar.add("/tmp/xgb_fraud_final.joblib", arcname="xgb_fraud_final.joblib")
    # Add the inference script under code/
    tar.add("/tmp/inference.py", arcname="code/inference.py")

print(f"✅ Packaged model → {MODEL_TAR_PATH}")

# Verify contents
with tarfile.open(MODEL_TAR_PATH, "r:gz") as tar:
    print("Contents:", tar.getnames())
```

> **Tip — save the model file first** (if you haven't already from the training
> section of the main prompt):
> ```python
> import joblib
> joblib.dump(final_model, "/tmp/xgb_fraud_final.joblib")
> print("✅ Model serialised")
> ```

---

## STEP 4 — Upload model.tar.gz to S3

```python
S3_MODEL_KEY = f"{S3_PREFIX}/model.tar.gz"

s3_client.upload_file(MODEL_TAR_PATH, BUCKET, S3_MODEL_KEY)
MODEL_S3_URI = f"s3://{BUCKET}/{S3_MODEL_KEY}"

print(f"✅ Model uploaded → {MODEL_S3_URI}")
```

---

## STEP 5 — Register a SageMaker Model

This tells SageMaker **which Docker image** to use and **where the model artifact is**.
Use the AWS-managed XGBoost container — no Docker work needed on your end.

```python
# AWS-managed XGBoost 1.7 container for us-west-2
# Full list: https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-containers-frameworks-table.html
XGB_IMAGE_URI = (
    "246618743249.dkr.ecr.us-west-2.amazonaws.com/"
    "sagemaker-xgboost:1.7-1"
)

MODEL_NAME = f"fraud-xgb-student-{STUDENT_NUM}-{int(time.time())}"

response = sm_client.create_model(
    ModelName        = MODEL_NAME,
    ExecutionRoleArn = ROLE_ARN,
    PrimaryContainer = {
        "Image":        XGB_IMAGE_URI,
        "ModelDataUrl": MODEL_S3_URI,
        "Environment":  {
            "SAGEMAKER_PROGRAM":          "inference.py",   # entry point inside code/
            "SAGEMAKER_SUBMIT_DIRECTORY": MODEL_S3_URI,     # where to find code/
        },
    },
)

print(f"✅ SageMaker Model created: {MODEL_NAME}")
print(f"   ARN: {response['ModelArn']}")
```

---

## STEP 6 — Create an Endpoint Configuration

```python
ENDPOINT_CONFIG_NAME = f"{ENDPOINT_NAME}-config"

sm_client.create_endpoint_config(
    EndpointConfigName = ENDPOINT_CONFIG_NAME,
    ProductionVariants  = [
        {
            "VariantName":          "primary",
            "ModelName":            MODEL_NAME,
            "InitialInstanceCount": 1,
            "InstanceType":         "ml.m5.large",   # cheapest general-purpose option
            "InitialVariantWeight": 1.0,
        }
    ],
)

print(f"✅ Endpoint config created: {ENDPOINT_CONFIG_NAME}")
```

**Instance type guidance for this capstone:**

| Instance | vCPU | RAM | Cost (approx) | Use when |
|---|---|---|---|---|
| `ml.t2.medium` | 2 | 4 GB | $0.056/hr | Dev/test only — may OOM on large payloads |
| `ml.m5.large` | 2 | 8 GB | $0.115/hr | ✅ Recommended for this project |
| `ml.m5.xlarge` | 4 | 16 GB | $0.23/hr | If you add heavy pre-processing |

---

## STEP 7 — Deploy the Endpoint

```python
sm_client.create_endpoint(
    EndpointName       = ENDPOINT_NAME,
    EndpointConfigName = ENDPOINT_CONFIG_NAME,
)

print(f"⏳ Deploying endpoint: {ENDPOINT_NAME}")
print("   This takes ~5 minutes. Polling status...")

# Poll until InService
while True:
    status = sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)["EndpointStatus"]
    print(f"   Status: {status}", end="\r")
    if status == "InService":
        print(f"\n✅ Endpoint is LIVE: {ENDPOINT_NAME}")
        break
    elif status == "Failed":
        reason = sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)["FailureReason"]
        raise RuntimeError(f"❌ Endpoint creation FAILED: {reason}")
    time.sleep(30)
```

---

## STEP 8 — Test the Endpoint With a Real Transaction Row

Pull one row from `fad_transactions.csv`, run it through the same
preprocessing pipeline used during training, then call the endpoint.

```python
# ── Load one sample transaction ───────────────────────────────────────────────
sample_raw = pd.read_csv("fad_transactions.csv", nrows=1)
txn_id     = sample_raw["transaction_id"].iloc[0]
print(f"Testing with transaction: {txn_id}")

# ── Apply the same preprocessing you did in training ─────────────────────────
# (This assumes you have preprocess_row() — see helper below)
sample_features = preprocess_for_inference(sample_raw)   # returns a dict

# ── Call the endpoint ─────────────────────────────────────────────────────────
payload  = json.dumps({"features": sample_features})

response = sm_runtime.invoke_endpoint(
    EndpointName = ENDPOINT_NAME,
    ContentType  = "application/json",
    Body         = payload,
)

result = json.loads(response["Body"].read().decode("utf-8"))
print(f"\n=== Endpoint Response ===")
print(f"  Transaction ID     : {txn_id}")
print(f"  Fraud Probability  : {result['fraud_probability']:.4f}")
print(f"  Is Fraud Prediction: {result['is_fraud_prediction']}")
```

---

## STEP 9 — Preprocessing Helper for Inference

You must apply **exactly the same transformations** at inference time as you did
during training. Wrap them in a reusable function.

```python
import pandas as pd
import numpy as np

# ── Constants from training ───────────────────────────────────────────────────
HIGH_RISK_MCC    = {7995, 6051, 4829, 6010, 6011, 5912}

OHE_COLS = [
    "card_prsn_cd", "entry_mode_ind", "keyed_swiped_ind",
    "ecom_in", "cvv2_cvc2_otcm_cd", "addr_vrfc_otcm_cd",
    "score_type_cd", "device_model_cd", "mrch_cntry_cd",
    "cust_credit_score_band", "cust_risk_tier",
    "ch_credit_score_band", "fraud_type_cd",
]

DROP_COLS = [
    "transaction_id", "account_num", "cust_customer_id", "ch_customer_id",
    "txn_month", "month", "partition_date",
    "label_type_cd", "label_type_desc",
    "gross_fraud_amt", "net_fraud_amt", "chargeback_amt",
    "chargeback_cnt", "loss_type_desc",
    "mrch_nm", "merch_city_nm", "ip_address_ipv4_id", "risk_reason_cd",
    "cust_profile_summary", "cust_occupation", "ch_credit_profile_note",
    "cust_segment",
]

# Load lookup tables once at notebook start
customers_df = pd.read_csv("customers.csv")
cred_hist_df = pd.read_csv("customer_credit_history.csv")
macro_df     = pd.read_csv("macro_context.csv")
macro_df["month"] = pd.to_datetime(macro_df["month"])


def preprocess_for_inference(raw_df: pd.DataFrame) -> dict:
    """
    Given a raw transaction DataFrame (1+ rows from fad_transactions),
    apply the full pipeline and return a dict of features ready for the endpoint.

    Returns a dict keyed by SELECTED_FEATURES — one entry per transaction.
    Call once per transaction inside invoke_fraud_model.
    """
    df = raw_df.copy()

    # 1. Parse timestamp & join key
    df["transaction_ts"] = pd.to_datetime(df["transaction_ts"])
    df["txn_month"]      = df["transaction_ts"].dt.to_period("M").dt.to_timestamp()

    # 2. Join customer profile
    df = df.merge(
        customers_df.add_prefix("cust_"),
        left_on="account_num", right_on="cust_customer_id", how="left"
    )

    # 3. Join credit history
    df = df.merge(
        cred_hist_df.add_prefix("ch_"),
        left_on="account_num", right_on="ch_customer_id", how="left"
    )

    # 4. Join macro context
    df = df.merge(macro_df, left_on="txn_month", right_on="month", how="left")

    # 5. Drop identifier / leakage columns
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # 6. Fill nulls
    df["device_model_cd"] = df["device_model_cd"].fillna("UNKNOWN")
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median() if len(df) > 1 else 0)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("NONE")

    # 7. Feature engineering (mirror training exactly)
    df["hour_of_day"]           = df["transaction_ts"].dt.hour
    df["day_of_week"]           = df["transaction_ts"].dt.dayofweek
    df["is_weekend"]            = (df["day_of_week"] >= 5).astype(int)
    df["is_night_txn"]          = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 5)).astype(int)
    df["amt_vs_monthly_avg"]    = df["tran_amt"] / (df.get("cust_avg_monthly_spend", pd.Series([1])).replace(0, np.nan) + 1)
    df["velocity_to_credit"]    = df["total_velocity_amt"] / (df["crdt_line_amt"].replace(0, np.nan) + 1)
    df["fraud_score_delta"]     = df["new_fraud_score"] - df["old_fraud_score"]
    df["fraud_score_ratio"]     = df["new_fraud_score"] / (df["old_fraud_score"].replace(0, np.nan) + 1)
    df["credit_stress"]         = df["perc_cred_limt_utlz_pct"] * (df["nmbr_days_dlnq_cnt"] + 1)
    df["zip_mismatch"]          = (df["card_zip_cd"] != df["merch_zip_cd"]).astype(int)
    df["is_cross_border"]       = (df["mrch_cntry_cd"] != "US").astype(int)
    df["is_high_risk_mcc"]      = df["merch_cat_code_cd"].isin(HIGH_RISK_MCC).astype(int)
    df["cash_velocity_ratio"]   = df["cash_velocity_amt"] / (df["total_velocity_amt"].replace(0, np.nan) + 1)
    df["high_unemployment"]     = (df.get("unemployment_rate", pd.Series([0])) > 4.5).astype(int)

    # 8. One-Hot Encoding (must match training columns exactly)
    ohe_present = [c for c in OHE_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=ohe_present, drop_first=True, dtype=int)

    # 9. Align to SELECTED_FEATURES — add missing OHE cols as 0, drop extras
    for col in SELECTED_FEATURES:
        if col not in df.columns:
            df[col] = 0
    df = df[SELECTED_FEATURES]

    # Return as a dict (one row)
    return df.iloc[0].to_dict()
```

---

## STEP 10 — Wire as `invoke_fraud_model` Agent Tool

This is the function the Bedrock Agent calls when it picks the `invoke_fraud_model` tool.

```python
def invoke_fraud_model(transaction_id: str, raw_transaction_df: pd.DataFrame) -> dict:
    """
    Agent tool: score one transaction against the SageMaker endpoint.

    Parameters
    ----------
    transaction_id     : the transaction_id string (e.g. 'txn_00049897')
    raw_transaction_df : a 1-row DataFrame from fad_transactions for this txn

    Returns
    -------
    dict with keys:
        transaction_id       : str
        fraud_probability    : float  (0.0 – 1.0)
        is_fraud_prediction  : int    (0 or 1)
        risk_label           : str    ("HIGH" / "MEDIUM" / "LOW")
        endpoint_name        : str
    """
    # 1. Preprocess
    features = preprocess_for_inference(raw_transaction_df)

    # 2. Call SageMaker endpoint
    payload  = json.dumps({"features": features})
    response = sm_runtime.invoke_endpoint(
        EndpointName = ENDPOINT_NAME,
        ContentType  = "application/json",
        Body         = payload,
    )
    result = json.loads(response["Body"].read().decode("utf-8"))

    # 3. Derive risk label from probability
    prob = result["fraud_probability"]
    if prob >= 0.70:
        risk_label = "HIGH"
    elif prob >= 0.40:
        risk_label = "MEDIUM"
    else:
        risk_label = "LOW"

    return {
        "transaction_id":      transaction_id,
        "fraud_probability":   round(prob, 4),
        "is_fraud_prediction": result["is_fraud_prediction"],
        "risk_label":          risk_label,
        "endpoint_name":       ENDPOINT_NAME,
    }


# ── Quick smoke test ──────────────────────────────────────────────────────────
sample_df = pd.read_csv("fad_transactions.csv", nrows=1)
txn_id    = sample_df["transaction_id"].iloc[0]

tool_output = invoke_fraud_model(txn_id, sample_df)
print(json.dumps(tool_output, indent=2))
```

Expected output:
```json
{
  "transaction_id": "txn_00049897",
  "fraud_probability": 0.0821,
  "is_fraud_prediction": 0,
  "risk_label": "LOW",
  "endpoint_name": "fraud-detector-student-06"
}
```

---

## STEP 11 — Integrate with Bedrock Agent (LangGraph style)

If you're using **LangGraph** (recommended), wrap `invoke_fraud_model` as a
LangChain tool and add it to the agent's tool list:

```python
from langchain_core.tools import tool

@tool
def invoke_fraud_model_tool(transaction_id: str) -> str:
    """
    Score a single card transaction for fraud risk.
    Call this first for every transaction in the batch.

    Parameters
    ----------
    transaction_id : the transaction_id from fad_transactions
                     (format: txn_XXXXXXXX)

    Returns
    -------
    JSON string with fraud_probability, is_fraud_prediction, and risk_label.
    """
    # Look up the raw transaction row
    txn_df = fad_txn_df[fad_txn_df["transaction_id"] == transaction_id]
    if txn_df.empty:
        return json.dumps({"error": f"transaction_id {transaction_id} not found"})

    result = invoke_fraud_model(transaction_id, txn_df)
    return json.dumps(result)


# Add to your agent tools list alongside the other three tools
tools = [
    invoke_fraud_model_tool,
    retrieve_rules_tool,
    get_customer_profile_tool,
    generate_report_tool,
]
```

If you're using **Bedrock Agents** (action groups), the function schema for this tool is:

```python
function_schema = {
    "functions": [
        {
            "name": "invoke_fraud_model",
            "description": (
                "Score a card transaction for fraud risk using a trained XGBoost model. "
                "Returns fraud_probability (0-1), binary prediction, and risk_label "
                "(HIGH/MEDIUM/LOW). Call this first for every transaction."
            ),
            "parameters": {
                "transaction_id": {
                    "type": "string",
                    "description": "The transaction_id from fad_transactions (e.g. txn_00049897)",
                    "required": True,
                }
            },
        }
    ]
}
```

---

## STEP 12 — Endpoint Lifecycle Management

```python
# ── Check endpoint status ─────────────────────────────────────────────────────
def check_endpoint():
    info = sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
    print(f"Status       : {info['EndpointStatus']}")
    print(f"Created      : {info['CreationTime']}")
    print(f"Last modified: {info['LastModifiedTime']}")
    return info

check_endpoint()

# ── Update endpoint (e.g. after retraining) ───────────────────────────────────
def update_endpoint(new_model_s3_uri):
    """Upload a retrained model and swap it in with zero downtime."""
    # 1. Register new model version
    new_model_name = f"fraud-xgb-student-{STUDENT_NUM}-{int(time.time())}"
    sm_client.create_model(
        ModelName        = new_model_name,
        ExecutionRoleArn = ROLE_ARN,
        PrimaryContainer = {
            "Image":        XGB_IMAGE_URI,
            "ModelDataUrl": new_model_s3_uri,
            "Environment":  {
                "SAGEMAKER_PROGRAM":          "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": new_model_s3_uri,
            },
        },
    )
    # 2. New config pointing to new model
    new_config_name = f"{ENDPOINT_NAME}-config-{int(time.time())}"
    sm_client.create_endpoint_config(
        EndpointConfigName = new_config_name,
        ProductionVariants  = [{
            "VariantName": "primary", "ModelName": new_model_name,
            "InitialInstanceCount": 1, "InstanceType": "ml.m5.large",
            "InitialVariantWeight": 1.0,
        }],
    )
    # 3. Rolling update — endpoint stays live during swap
    sm_client.update_endpoint(
        EndpointName       = ENDPOINT_NAME,
        EndpointConfigName = new_config_name,
    )
    print(f"⏳ Updating endpoint to {new_model_name}...")

# ── DELETE endpoint when done (saves cost) ────────────────────────────────────
def delete_endpoint():
    sm_client.delete_endpoint(EndpointName=ENDPOINT_NAME)
    print(f"🗑️  Endpoint {ENDPOINT_NAME} deleted")

# Uncomment when you're done with the capstone:
# delete_endpoint()
```

---

## Common Errors & Fixes

| Error | Cause | Fix |
|---|---|---|
| `ValidationException: Could not find role` | Wrong `ROLE_ARN` | Get exact ARN from IAM console or ask instructor |
| `ModelError: An error occurred while requesting the model` | Crash inside `inference.py` | Check CloudWatch Logs `/aws/sagemaker/Endpoints/{ENDPOINT_NAME}` |
| `KeyError: 'some_feature'` at inference | `SELECTED_FEATURES` list in `inference.py` doesn't match training | Re-paste the exact `selected_features` list into `inference.py` |
| `EndpointStatus: Failed` | Container OOM or IAM permission issue | Check `FailureReason` via `describe_endpoint()`; increase instance size |
| `NoSuchBucket` | S3 bucket doesn't exist | Run STEP 1 to create it |
| `AccessDeniedException` on S3 upload | IAM role lacks S3 write | Confirm `AmazonS3FullAccess` is attached to your role |

---

## Files Generated by This Guide

```
/tmp/
├── xgb_fraud_final.joblib      ← serialised trained model
├── inference.py                ← SageMaker serving hooks
└── model.tar.gz                ← what gets uploaded to S3
                                   contains: xgb_fraud_final.joblib
                                             code/inference.py
```

S3 layout:
```
s3://sagemaker-us-west-2-{ACCOUNT_ID}/
└── student-06/
    └── fraud-model/
        └── model.tar.gz
```
