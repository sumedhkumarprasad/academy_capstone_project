# Capstone 1: Fraud Detection Agent with RAG

**Format:** hackathon (light guidance, you drive). **Time:** ~6 hours.
**Environment:** Azure Databricks notebooks calling AWS (datacouch, us-west-2) via boto3.

---

## The objective

Build an **agentic fraud investigation system**. The pipeline:

1. Read a batch of card transactions from a Delta table.
2. Score each one with a fraud classifier **you train and deploy** on SageMaker.
3. For the **high-risk** ones, hand off to an **agent** that decides which tools to call:
   - look up the **fraud rules** that match (RAG over a Bedrock Knowledge Base),
   - pull the **customer profile** for context,
   - write a structured **investigation report**.
4. Produce one report per flagged transaction, plus an audit trail of the agent's reasoning.

The agent picks its own tool sequence. That is the point. A fixed script is not an agent.

**Minimum to "done":**
- A SageMaker real-time endpoint serving a fraud model you trained.
- A Bedrock Knowledge Base built from the fraud-rules corpus, retrieval validated.
- An agent loop (Bedrock Agents or LangGraph) wiring classifier + KB + customer lookup as tools.
- A structured report for each high-risk transaction in a sample batch.

**Stretch:** use the MWAA Airflow environment (see bottom) to orchestrate the batch run as a DAG.

---

## Setup: the standard auth cell

Paste this at the top of your notebook (replace `STUDENT_NUM` with your 2-digit number).
Full details in `ENVIRONMENT_SETUP.md`.

```python
import boto3, os
STUDENT_NUM = "01"  # <-- your number

scope = f"aws-course-creds-{STUDENT_NUM}"
os.environ["AWS_ACCESS_KEY_ID"]     = dbutils.secrets.get(scope, "aws-access-key-id")
os.environ["AWS_SECRET_ACCESS_KEY"] = dbutils.secrets.get(scope, "aws-secret-access-key")
os.environ["AWS_REGION"] = os.environ["AWS_DEFAULT_REGION"] = "us-west-2"

session           = boto3.Session(region_name="us-west-2")
sagemaker_client  = session.client("sagemaker")
sagemaker_runtime = session.client("sagemaker-runtime")
bedrock_runtime   = session.client("bedrock-runtime")
bedrock_agent     = session.client("bedrock-agent")
bedrock_agent_rt  = session.client("bedrock-agent-runtime")
print("Caller:", session.client("sts").get_caller_identity()["Arn"])
```

Model to use for the agent: `us.anthropic.claude-sonnet-4-5-20250929-v1:0` (via `converse`).
Embeddings for the KB: `amazon.titan-embed-text-v2:0`. Your IAM already covers Bedrock,
SageMaker, CloudWatch, and S3 writes to the course buckets - no extra setup needed.

---

## The data

Everything lives in Unity Catalog under `bread_academy.course_data`. Read it with Spark.

### 1. Transactions: `bread_academy.course_data.fad_transactions` (50,000 rows)

One row per card authorization. ~3% are confirmed fraud (`label_type_cd = 1`). This is a
curated, synthetic version of Bread Financial's real Fiserv FAD feed - the column names are
the real ones. Key columns:

| Column | Meaning |
|---|---|
| `transaction_id` | PK, `txn_00000001` ... |
| `account_num` | FK to customers, `acct_0000001` ... |
| `transaction_ts` | timestamp |
| `tran_amt` | amount USD |
| `merch_cat_code_cd` | 4-digit MCC (e.g. 7995 gambling, 6051 crypto) |
| `mrch_nm`, `merch_city_nm` | merchant name and city |
| `card_prsn_cd` | Y = card present, N = card-not-present |
| `entry_mode_ind` | chip / swipe / contactless / ecom / manual / token |
| `mrch_cntry_cd` | ISO country (US dominant, ~18% foreign) |
| `new_fraud_score` | issuer fraud score 0-999 |
| `total_velocity_amt`, `cash_velocity_amt`, `hour_24_cnt` | 24h velocity features |
| `cvv2_cvc2_otcm_cd`, `addr_vrfc_otcm_cd` | CVV / AVS outcomes |
| `device_model_cd`, `ip_address_ipv4_id` | device / IP |
| `label_type_cd` | **ground-truth label: 0 or 1** (your training target) |

```python
df = spark.read.table("bread_academy.course_data.fad_transactions")
df.groupBy("label_type_cd").count().show()           # ~3% fraud
pdf = df.toPandas()                                  # for XGBoost / sklearn
```

There is also `bread_academy.course_data.ft_fraud_cases` (1,500 rows): one row per
**confirmed-fraud** transaction with case-level fields (`gross_fraud_amt`, `net_fraud_amt`,
`chargeback_amt`, `loss_dt`, `external_status_desc`, `case_narrative`). Joins to transactions
on `transaction_id` / `account_num`. Useful context for your investigation reports.

### 2. Customers: `bread_academy.course_data.customers` (5,000 rows)

One row per `account_num`. This is what your agent's `get_customer_profile` tool reads.
`risk_tier` is derived from each account's real fraud history, so high-risk customers
genuinely have fraud in the transaction table.

| Column | Meaning |
|---|---|
| `customer_id` | = `account_num` (join key) |
| `account_tenure_months`, `avg_monthly_spend`, `home_zip` | profile basics |
| `credit_score_band` | poor / fair / good / excellent |
| `risk_tier` | low / medium / high |
| `delinquency_flag` | 0 / 1 |
| `occupation`, `segment`, `profile_summary` | LLM-written, human-readable context |

```python
cust = spark.read.table("bread_academy.course_data.customers")
# look up one customer for the agent tool
cust.filter("customer_id = 'acct_0000042'").toPandas().iloc[0].to_dict()
```

### 3. Fraud rules: `fraud_rules/` (32 markdown docs) - this is your RAG corpus

In this folder, next to this file. ~32 documents, each with YAML frontmatter
(`rule_id`, `category`, `severity`, `source`) and a plain-English rule with worked examples.
Categories: **AML thresholds, velocity rules, MCC risk, geographic risk, card-not-present
heuristics, device anomalies**. The thresholds in these docs are wired to the actual data
(same MCCs, same high-risk country codes, same columns) - so retrieval genuinely matches
what you see in `fad_transactions`.

---

## Building the Bedrock Knowledge Base (the RAG part)

You ingest the `fraud_rules/` docs into a Bedrock Knowledge Base, then your agent retrieves
from it. Two paths:

**Option A - reuse the pre-built KB (fastest).** A fraud-rules KB already exists in the
account. Get its id from the shared scope and start querying immediately:

```python
KB_ID = dbutils.secrets.get("aws-course-shared", "knowledge-base-id")  # bread-academy-fraud-kb

resp = bedrock_agent_rt.retrieve(
    knowledgeBaseId=KB_ID,
    retrievalQuery={"text": "high velocity card-not-present from a high-risk country"},
)
for r in resp["retrievalResults"]:
    print(r["content"]["text"][:200])
```

**Option B - build your own KB (more of the learning).** This is the recommended hackathon
path if you want the full RAG experience:

1. Upload `fraud_rules/*.md` to your S3 area under the course bucket:
   ```python
   import glob
   BUCKET = dbutils.secrets.get("aws-course-shared", "course-s3-bucket")  # bread-academy-shared
   for f in glob.glob("fraud_rules/*.md"):
       s3 = session.client("s3")
       s3.upload_file(f, BUCKET, f"student-{STUDENT_NUM}/fraud_rules/{f.split('/')[-1]}")
   ```
2. In the Bedrock console (Knowledge Bases), create a KB:
   - Data source = your S3 prefix above.
   - Embeddings model = `amazon.titan-embed-text-v2:0`.
   - Vector store = quick-create (OpenSearch Serverless).
   - Choose a chunking strategy (try fixed-size ~300 tokens, or semantic) and **sync**.
3. Validate retrieval with `bedrock_agent_rt.retrieve(...)` as above before wiring it to the agent.

Either way: your `retrieve_rules` tool is a thin wrapper around `bedrock_agent_rt.retrieve`.

---

## The agent

Single agent loop. Expose four tools and let the model choose the sequence:

| Tool | Does |
|---|---|
| `invoke_fraud_model` | call your SageMaker endpoint -> risk score |
| `retrieve_rules` | `bedrock_agent_rt.retrieve` against the KB |
| `get_customer_profile` | read one row from `customers` |
| `generate_report` | structured investigation report (JSON) |

Build it with **Bedrock Agents** (action groups) or **LangGraph** (`langchain-aws`,
`langgraph` are pre-installed on the cluster). Drive the model with the Claude Sonnet 4.5
inference profile.

**Hint - Bedrock Agents action-group shape (if you go the Agents route).** Each tool is
an action group backed by an OpenAPI-ish function schema; the agent calls back with a
`returnControl` event you fulfil. The fiddly part is the schema + the invoke loop:

```python
ba = boto3.Session(region_name="us-west-2").client("bedrock-agent")
# one action group per tool; functionSchema describes the callable
ba.create_agent_action_group(
    agentId=AGENT_ID, agentVersion="DRAFT", actionGroupName="invoke_fraud_model",
    actionGroupExecutor={"customControl": "RETURN_CONTROL"},   # you run the tool yourself
    functionSchema={"functions": [{
        "name": "invoke_fraud_model",
        "parameters": {"transaction_id": {"type": "string", "required": True}}}]})
# at runtime: bedrock-agent-runtime.invoke_agent(...) streams a returnControl event
# with the chosen function + args; you execute it and post the result back. (LangGraph
# avoids this dance - tools are just python callables on nodes.)
```

**Hint - CloudWatch audit trail.** Emit each tool call as a structured log line to your
own log group so the reasoning is auditable:

```python
import json, time
logs = boto3.Session(region_name="us-west-2").client("logs")
GROUP = f"/bread-academy/capstone1/student-{STUDENT_NUM}"
# create_log_group / create_log_stream once (ignore ResourceAlreadyExists), then:
logs.put_log_events(logGroupName=GROUP, logStreamName=run_id, logEvents=[{
    "timestamp": int(time.time()*1000),
    "message": json.dumps({"step": "retrieve_rules", "txn": txn_id,
                           "args": args, "result_summary": summary})}])
```

---

## Airflow at your disposal (optional orchestration, from Weeks 21-22)

You have a live MWAA (managed Airflow) environment:

- **Web UI:** https://b6ee3526-eec1-4266-891c-c3218a2d8231.c24.airflow.us-west-2.on.aws/home
- **Environment name:** `bread-academy-airflow` (us-west-2)

**How MWAA reads your DAGs (recap from Weeks 21-22):** the environment syncs DAG code from an
S3 bucket every ~30 seconds. You upload your DAG `.py` to:

```
s3://bread-academy-airflow-dags/dags/student_<NN>/<your_dag>.py
```

Use a **per-student `dag_id`** like `capstone1_<STUDENT_ID>` so it does not collide with other
students. Upload, wait ~30s for MWAA to register it, then trigger from the UI (or the
`airflow:CreateCliToken` / REST API pattern you used in Week 22):

```python
s3 = session.client("s3")
DAGS_BUCKET = "bread-academy-airflow-dags"
DAG_PREFIX  = f"dags/student_{STUDENT_NUM}"
s3.put_object(Bucket=DAGS_BUCKET, Key=f"{DAG_PREFIX}/capstone1.py",
              Body=open("capstone1_dag.py","rb").read())
# then poll GET /dags/{dag_id} until 200, and POST /dags/{dag_id}/dagRuns to trigger
```

**Idea for the stretch goal:** wrap the batch pipeline (score -> filter high-risk ->
investigate -> write reports) as a DAG: one task scores the batch, a branch sends only
high-risk transactions to an investigation task, a final task writes reports to S3 or a
`student_work` Delta table.

---

## Deliverables checklist

- [ ] Trained fraud model deployed to a SageMaker endpoint (you evaluated it on a holdout).
- [ ] Bedrock Knowledge Base over `fraud_rules/`, retrieval validated on sample queries.
- [ ] Agent loop wiring the 4 tools; it chooses its own sequence.
- [ ] Structured investigation report per high-risk transaction in a sample batch.
- [ ] Audit trail (CloudWatch) of the agent's reasoning.
- [ ] (Stretch) An Airflow DAG that runs the batch end to end.

Write results to `bread_academy.student_work.*` (your read/write schema) or to your
`student-<NN>/` prefix in the course S3 bucket. Have fun - this is a hackathon, not an exam.
