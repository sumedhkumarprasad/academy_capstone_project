# Bedrock Knowledge Base — Complete Beginner Guide
### Option B: Build Your Own KB from fraud_rules/ → Wire as `retrieve_rules` Agent Tool

---

## What You Are Building

```
fraud_rules/*.md  (32 markdown files already in S3)
        │
        ▼
  AWS Bedrock KB                         Your Databricks Notebook
  ┌─────────────────────────┐            ┌──────────────────────────────┐
  │  Data Source            │            │  retrieve_rules tool         │
  │  └─ S3 prefix           │◄──────────►│  bedrock_agent_rt.retrieve() │
  │                         │            │  ↓                           │
  │  Embeddings             │            │  Returns matching fraud rules │
  │  └─ Titan Embed v2      │            │  for the agent to reason with│
  │                         │            └──────────────────────────────┘
  │  Vector Store           │
  │  └─ OpenSearch          │
  │     Serverless          │
  └─────────────────────────┘
```

**What a Knowledge Base does:** It takes your 32 fraud-rule markdown files, splits them
into small chunks (~300 tokens each), converts each chunk into a vector embedding using
Amazon Titan, stores those vectors in OpenSearch Serverless, and lets you query them
with plain English. The query "card-not-present transaction from a high-risk country"
returns the most semantically similar fraud rules — even if those exact words don't appear.

---

## Prerequisites Checklist

Before starting, confirm all of these:

- [ ] You have already uploaded `fraud_rules/*.md` to S3 at
      `s3://<your-bucket>/student-06/fraud_rules/` (capstone says you already did this ✅)
- [ ] Your IAM credentials have permissions for: Bedrock, S3, OpenSearch Serverless
- [ ] You are working in AWS region **us-west-2** (all course resources are there)
- [ ] Your `STUDENT_NUM = "06"` throughout

---

## PART 1 — AWS Console Steps (One-Time Setup, ~10 Minutes)

You do PART 1 once in the AWS web console. After that everything is code in Databricks.

---

### CONSOLE STEP 1 — Open Amazon Bedrock

1. Open your browser and go to: **https://us-west-2.console.aws.amazon.com/bedrock**
2. Make sure the region selector (top-right corner) shows **US West (Oregon) us-west-2**

   > If it shows a different region, click it and select **US West (Oregon)**.

3. In the left sidebar, scroll down and click **"Knowledge bases"**

   You will see either an empty list or the shared course KB. Ignore the shared one —
   you are creating your own.

---

### CONSOLE STEP 2 — Create the Knowledge Base

1. Click the orange **"Create knowledge base"** button (top right)

2. **Knowledge base details** screen:

   | Field | What to enter |
   |---|---|
   | Knowledge base name | `fraud-kb-student-06` |
   | Description | `Fraud detection rules corpus for capstone student 06` |
   | IAM permissions | Select **"Use an existing service role"** → pick `AmazonBedrockExecutionRoleForKnowledgeBase` or your course role |

   > If you don't see an existing role, select **"Create and use a new service role"**
   > and AWS will create one automatically. This is safe for the course.

3. Click **"Next"**

---

### CONSOLE STEP 3 — Configure the Data Source (your S3 folder)

1. **Data source name:** `fraud-rules-s3`

2. **Data source type:** Select **Amazon S3** (should already be selected)

3. **S3 URI:** Enter the exact S3 path where your fraud rules are:
   ```
   s3://bread-academy-shared/student-06/fraud_rules/
   ```
   > Click **"Browse S3"** if you want to navigate to it visually and confirm the
   > files are there before proceeding.

4. **Data source authentication:** Leave as default (uses the IAM role you selected)

5. **Chunking and parsing strategy** — this is important:

   Click **"Edit chunking and parsing configuration"**

   Select **"Fixed-size chunking"** and set:
   | Setting | Value |
   |---|---|
   | Max tokens per chunk | `300` |
   | Overlap percentage | `20` |

   > **Why 300 tokens?** Each fraud rule document is ~500-800 words. At 300 tokens
   > per chunk with 20% overlap, each chunk is about one rule or one sub-section,
   > which is the right granularity for retrieval. Too large = imprecise retrieval.
   > Too small = chunks lack enough context.

   > **Why 20% overlap?** Overlap ensures that a rule that spans a chunk boundary
   > isn't split in a way that loses its meaning.

6. Click **"Next"**

---

### CONSOLE STEP 4 — Configure the Embeddings Model

1. **Embeddings model** screen — click **"Browse"** or use the dropdown

2. Select: **Amazon → Titan Embeddings V2 - Text**
   - Model ID shown: `amazon.titan-embed-text-v2:0`
   - Dimensions: 1024 (default — leave it)

   > This is the model specified in `capstone1.md`. It converts each text chunk
   > into a 1024-dimensional vector that captures semantic meaning.

3. Click **"Next"**

---

### CONSOLE STEP 5 — Configure the Vector Store

1. Select **"Quick create a new vector store"**

   This tells Bedrock to automatically create an **OpenSearch Serverless** collection.
   You don't need to configure anything — Bedrock manages the entire vector database.

   > **What is OpenSearch Serverless?** It's AWS's managed vector database. Think of it
   > as the "filing cabinet" that stores all the embeddings so they can be searched
   > quickly. "Serverless" means AWS scales it automatically — you pay per query,
   > not per running instance.

2. Leave all other settings as default.

3. Click **"Next"**

---

### CONSOLE STEP 6 — Review and Create

1. Review the summary:
   - Knowledge base name: `fraud-kb-student-06`
   - Data source: your S3 prefix
   - Embeddings: `amazon.titan-embed-text-v2:0`
   - Vector store: OpenSearch Serverless (auto-created)

2. Click **"Create knowledge base"**

3. Wait ~2-3 minutes. You will see a spinner and then a green **"Knowledge base created"**
   banner at the top.

4. **IMPORTANT — Copy your KB ID now:**
   After creation, you will be taken to the KB detail page. At the top you will see:
   ```
   Knowledge base ID:  XXXXXXXXXX
   ```
   Copy this value. It looks like: `ABCDE12345`
   You will paste it into your Databricks notebook as `KB_ID`.

---

### CONSOLE STEP 7 — Sync the Data Source (Ingest the Documents)

Creating the KB doesn't automatically read your S3 files. You must trigger a **sync**.

1. On your KB detail page, click the **"Data sources"** tab

2. You will see `fraud-rules-s3` listed. Click on it.

3. Click the **"Sync"** button (top right of the data source panel)

4. The status will change to **"Syncing"** — this takes 2-5 minutes for 32 files.

5. Wait until status shows **"Available"** with a green checkmark.

   > **What happens during sync?**
   > Bedrock reads each `.md` file from S3 → splits into 300-token chunks →
   > calls Titan Embeddings on each chunk → stores vectors in OpenSearch Serverless.
   > For 32 files (~10,000 words total) this produces roughly 100-150 chunks.

6. Click **"Sync history"** tab to see the log:
   - Documents added: should show 32
   - Documents failed: should show 0
   - If any files failed, check the error message — usually it's a permission issue
     on the S3 bucket.

---

## PART 2 — Databricks Notebook Code

Now that the KB exists and is synced, everything below runs in your Databricks notebook.

---

### NOTEBOOK STEP 1 — AWS Setup

```python
import boto3, os, json

STUDENT_NUM = "06"

os.environ["AWS_ACCESS_KEY_ID"]     = "AKIA6AK5B2HLV2E6FD6F"
os.environ["AWS_SECRET_ACCESS_KEY"] = "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"
os.environ["AWS_REGION"]            = "us-west-2"
os.environ["AWS_DEFAULT_REGION"]    = "us-west-2"

session         = boto3.Session(region_name="us-west-2")
bedrock_agent   = session.client("bedrock-agent")
bedrock_agent_rt = session.client("bedrock-agent-runtime")

print("✅ Caller:", session.client("sts").get_caller_identity()["Arn"])
```

---

### NOTEBOOK STEP 2 — Set Your KB ID

Paste the KB ID you copied from the console in CONSOLE STEP 6.

```python
# Paste your Knowledge Base ID from the Bedrock console here
KB_ID = "ABCDE12345"   # <-- replace with your actual KB ID

print(f"✅ Using Knowledge Base: {KB_ID}")
```

---

### NOTEBOOK STEP 3 — Verify the KB Exists and Is Active

```python
def check_kb_status(kb_id):
    """Check that the KB is Active and the data source is synced."""
    kb_info = bedrock_agent.get_knowledge_base(knowledgeBaseId=kb_id)
    kb      = kb_info["knowledgeBase"]

    print(f"Name   : {kb['name']}")
    print(f"Status : {kb['status']}")         # should be ACTIVE
    print(f"Created: {kb['createdAt']}")

    # List data sources
    ds_list = bedrock_agent.list_data_sources(knowledgeBaseId=kb_id)
    for ds in ds_list["dataSourceSummaries"]:
        print(f"\nData Source: {ds['name']}  |  Status: {ds['status']}")

    if kb["status"] != "ACTIVE":
        print("⚠️  KB is not ACTIVE yet — wait a minute and re-run this cell")
    else:
        print("\n✅ KB is ready to query")

check_kb_status(KB_ID)
```

Expected output:
```
Name   : fraud-kb-student-06
Status : ACTIVE
Created: 2025-06-15 10:32:44+00:00

Data Source: fraud-rules-s3  |  Status: AVAILABLE
✅ KB is ready to query
```

---

### NOTEBOOK STEP 4 — Validate Retrieval With Test Queries

Before wiring the KB to the agent, confirm it actually returns relevant fraud rules.

```python
def query_kb(kb_id, query_text, n_results=3):
    """
    Query the Bedrock Knowledge Base and print the top matching chunks.

    Parameters
    ----------
    kb_id       : your Knowledge Base ID string
    query_text  : plain English question or phrase
    n_results   : how many chunks to retrieve (default 3; agent will use 3-5)
    """
    response = bedrock_agent_rt.retrieve(
        knowledgeBaseId = kb_id,
        retrievalQuery  = {"text": query_text},
        retrievalConfiguration = {
            "vectorSearchConfiguration": {
                "numberOfResults": n_results,
            }
        },
    )

    results = response["retrievalResults"]
    print(f"Query : '{query_text}'")
    print(f"Found : {len(results)} results\n")

    for i, r in enumerate(results, 1):
        score    = r.get("score", "N/A")
        text     = r["content"]["text"]
        location = r.get("location", {}).get("s3Location", {}).get("uri", "unknown")
        print(f"── Result {i}  (score: {score:.4f}) ────────────────────────")
        print(f"   Source : {location}")
        print(f"   Excerpt: {text[:300]}...")
        print()

    return results


# ── Test Query 1: Velocity rules ──────────────────────────────────────────────
query_kb(KB_ID,
         "high velocity card-not-present transaction from a high-risk country",
         n_results=3)

# ── Test Query 2: MCC risk ────────────────────────────────────────────────────
query_kb(KB_ID,
         "gambling merchant category code 7995 fraud risk",
         n_results=3)

# ── Test Query 3: AML threshold ───────────────────────────────────────────────
query_kb(KB_ID,
         "AML structuring threshold cash transaction suspicious",
         n_results=3)

# ── Test Query 4: Device anomaly ──────────────────────────────────────────────
query_kb(KB_ID,
         "device fingerprint anomaly new device first transaction",
         n_results=3)

# ── Test Query 5: CVV / AVS failure ──────────────────────────────────────────
query_kb(KB_ID,
         "CVV mismatch address verification failure card not present",
         n_results=3)
```

**What to look for in the results:**
- Score > 0.5 means a semantically strong match
- The excerpt should contain fraud rule language relevant to your query
- The source URI should point to one of your `fraud_rules/*.md` files

If you see scores near 0.0 or totally unrelated text, the sync may not have completed —
go back to the console and re-run the sync.

---

### NOTEBOOK STEP 5 — Trigger a Re-Sync Programmatically (Optional)

If you update any fraud rule files in S3 later, you can re-sync from code
without going back to the console.

```python
def resync_knowledge_base(kb_id):
    """Trigger a data source sync and poll until complete."""
    # Get the data source ID
    ds_list   = bedrock_agent.list_data_sources(knowledgeBaseId=kb_id)
    ds_id     = ds_list["dataSourceSummaries"][0]["dataSourceId"]

    # Start sync job
    job_resp  = bedrock_agent.start_ingestion_job(
        knowledgeBaseId = kb_id,
        dataSourceId    = ds_id,
    )
    job_id = job_resp["ingestionJob"]["ingestionJobId"]
    print(f"⏳ Sync started. Job ID: {job_id}")

    # Poll until complete
    import time
    while True:
        status_resp = bedrock_agent.get_ingestion_job(
            knowledgeBaseId = kb_id,
            dataSourceId    = ds_id,
            ingestionJobId  = job_id,
        )
        job    = status_resp["ingestionJob"]
        status = job["status"]
        stats  = job.get("statistics", {})
        print(f"   Status: {status}  |  "
              f"Scanned: {stats.get('numberOfDocumentsScanned', 0)}  |  "
              f"Indexed: {stats.get('numberOfNewDocumentsIndexed', 0)}  |  "
              f"Failed : {stats.get('numberOfDocumentsFailed', 0)}",
              end="\r")

        if status in ("COMPLETE", "FAILED"):
            print()
            break
        time.sleep(10)

    if status == "COMPLETE":
        print(f"✅ Sync complete — {stats.get('numberOfNewDocumentsIndexed', 0)} chunks indexed")
    else:
        print(f"❌ Sync FAILED — check the console for details")

# Only call this if you need to re-sync after updating files in S3
# resync_knowledge_base(KB_ID)
```

---

### NOTEBOOK STEP 6 — The `retrieve_rules` Agent Tool

This is the thin wrapper the capstone asks you to build. The agent calls this
whenever it needs to look up which fraud rules apply to a transaction.

```python
def retrieve_rules(query: str, n_results: int = 5) -> dict:
    """
    Agent tool: retrieve the most relevant fraud rules for a given situation.

    The agent calls this after invoke_fraud_model flags a transaction as
    high-risk. It uses the transaction's key attributes as the query.

    Parameters
    ----------
    query     : Plain-English description of the suspicious transaction.
                The agent will construct this from the transaction fields.
                Example: "card-not-present ecom transaction MCC 7995 gambling
                          from Nigeria, CVV mismatch, high velocity"
    n_results : Number of rule chunks to retrieve (default 5)

    Returns
    -------
    dict with keys:
        query          : the original query string
        rules_found    : int — number of results returned
        rules          : list of dicts, each with:
                           rule_text  : the retrieved chunk text
                           score      : relevance score (0-1)
                           source_doc : S3 URI of the source .md file
        rules_summary  : single concatenated string for easy LLM consumption
    """
    response = bedrock_agent_rt.retrieve(
        knowledgeBaseId = KB_ID,
        retrievalQuery  = {"text": query},
        retrievalConfiguration = {
            "vectorSearchConfiguration": {
                "numberOfResults": n_results,
            }
        },
    )

    rules = []
    for r in response["retrievalResults"]:
        rules.append({
            "rule_text":   r["content"]["text"],
            "score":       round(r.get("score", 0.0), 4),
            "source_doc":  r.get("location", {})
                            .get("s3Location", {})
                            .get("uri", "unknown"),
        })

    # Build a single string the LLM can read easily
    rules_summary = "\n\n---\n\n".join(
        f"[Rule {i+1} | score={r['score']} | source={r['source_doc'].split('/')[-1]}]\n"
        f"{r['rule_text']}"
        for i, r in enumerate(rules)
    )

    return {
        "query":         query,
        "rules_found":   len(rules),
        "rules":         rules,
        "rules_summary": rules_summary,
    }


# ── Smoke test the tool ───────────────────────────────────────────────────────
result = retrieve_rules(
    "card-not-present ecommerce transaction from high-risk country "
    "CVV mismatch high velocity gambling MCC 7995"
)

print(f"Rules found: {result['rules_found']}\n")
print(result["rules_summary"])
```

---

### NOTEBOOK STEP 7 — Build a Smart Query from Transaction Fields

The agent needs to construct a meaningful query from the transaction data.
Here's how to auto-generate one:

```python
import pandas as pd

def build_rules_query(transaction_row: dict, fraud_score: float) -> str:
    """
    Build a plain-English query for retrieve_rules from raw transaction fields.
    This is what the agent uses to formulate its KB lookup.

    Parameters
    ----------
    transaction_row : dict of raw fad_transactions fields for one transaction
    fraud_score     : fraud_probability from invoke_fraud_model (0-1)

    Returns
    -------
    A descriptive query string
    """
    parts = []

    # Card presence
    card_prsn = transaction_row.get("card_prsn_cd", "")
    if card_prsn == "N":
        parts.append("card-not-present")
    else:
        parts.append("card-present")

    # Entry mode
    entry = transaction_row.get("entry_mode_ind", "")
    if entry:
        parts.append(f"{entry} entry mode")

    # Ecommerce flag
    if transaction_row.get("ecom_in") == "Y":
        parts.append("ecommerce transaction")

    # Amount
    amt = transaction_row.get("tran_amt", 0)
    if amt > 500:
        parts.append(f"high-value transaction ${amt:.0f}")
    elif amt > 0:
        parts.append(f"transaction amount ${amt:.0f}")

    # MCC
    mcc = transaction_row.get("merch_cat_code_cd")
    MCC_NAMES = {
        7995: "gambling MCC 7995",
        6051: "cryptocurrency MCC 6051",
        4829: "wire transfer MCC 4829",
        6010: "cash advance MCC 6010",
        6011: "ATM cash MCC 6011",
    }
    if mcc in MCC_NAMES:
        parts.append(f"high-risk {MCC_NAMES[mcc]}")
    elif mcc:
        parts.append(f"MCC {mcc}")

    # Country
    country = transaction_row.get("mrch_cntry_cd", "US")
    if country != "US":
        parts.append(f"foreign merchant country {country}")

    # CVV outcome
    cvv = transaction_row.get("cvv2_cvc2_otcm_cd", "")
    if cvv in ("N", "M"):
        parts.append("CVV verification failure")

    # AVS outcome
    avs = transaction_row.get("addr_vrfc_otcm_cd", "")
    if avs == "N":
        parts.append("AVS address mismatch")

    # Velocity
    vel = transaction_row.get("total_velocity_amt", 0)
    cnt = transaction_row.get("hour_24_cnt", 0)
    if vel > 1000 or cnt > 5:
        parts.append(f"high velocity ${vel:.0f} in 24h / {cnt} transactions")

    # Fraud score context
    if fraud_score >= 0.70:
        parts.append("very high model fraud score")
    elif fraud_score >= 0.40:
        parts.append("elevated fraud score")

    query = " ".join(parts)
    return query


# ── Test it with a sample transaction ────────────────────────────────────────
sample = pd.read_csv("fad_transactions.csv", nrows=100)
# Pick a transaction where card is not present for a better demo
cnp_txn = sample[sample["card_prsn_cd"] == "N"].iloc[0].to_dict()

auto_query = build_rules_query(cnp_txn, fraud_score=0.75)
print(f"Auto-generated query:\n  '{auto_query}'\n")

# Now retrieve rules with this auto-generated query
rules_result = retrieve_rules(auto_query, n_results=5)
print(f"Matched {rules_result['rules_found']} rules:\n")
print(rules_result["rules_summary"])
```

---

### NOTEBOOK STEP 8 — LangGraph Tool Wrapper

If you are using LangGraph for the agent (recommended), wrap `retrieve_rules`
as a LangChain tool:

```python
from langchain_core.tools import tool

@tool
def retrieve_rules_tool(query: str) -> str:
    """
    Look up fraud detection rules that are relevant to a suspicious transaction.

    Use this tool after invoke_fraud_model returns a HIGH or MEDIUM risk score.
    Formulate a query describing the suspicious aspects of the transaction:
    - card presence (card-present vs card-not-present)
    - entry mode (chip, ecom, swipe, etc.)
    - merchant category (MCC code and name)
    - merchant country (especially if foreign)
    - CVV / AVS outcomes
    - velocity (number of transactions or amount in 24h)

    Parameters
    ----------
    query : Plain-English description of the suspicious transaction context.
            Example: 'card-not-present ecommerce gambling MCC 7995 Nigeria
                      CVV mismatch high velocity $2000 in 24h'

    Returns
    -------
    JSON string containing matched fraud rules with relevance scores.
    """
    result = retrieve_rules(query, n_results=5)

    # Return the human-readable summary for the LLM
    return json.dumps({
        "rules_found":   result["rules_found"],
        "rules_summary": result["rules_summary"],
    })


# Add to your agent tool list
tools = [
    invoke_fraud_model_tool,   # from sagemaker_endpoint_guide.md
    retrieve_rules_tool,       # ← this one
    get_customer_profile_tool,
    generate_report_tool,
]
```

---

## Troubleshooting Reference

| Problem | Likely Cause | Fix |
|---|---|---|
| KB status is `CREATING` for >5 min | Normal for first creation | Wait — OpenSearch Serverless provisioning takes time |
| Data source status `FAILED` | IAM role lacks S3 read access | Add `s3:GetObject` and `s3:ListBucket` permissions to your Bedrock execution role |
| Sync shows 0 documents indexed | Wrong S3 URI or empty folder | Confirm your S3 path contains `.md` files via the S3 console |
| `retrieve()` returns empty results | Sync not complete | Check sync history in console; re-run sync |
| Scores all below 0.3 | Chunking too aggressive or wrong files | Try semantic chunking; verify files are fraud rules, not empty |
| `AccessDeniedException` on `retrieve()` | Bedrock model access not enabled | In Bedrock console → Model access → enable Titan Embeddings V2 |
| `ResourceNotFoundException: KB not found` | Wrong `KB_ID` or wrong region | Double-check KB_ID from console; confirm region is `us-west-2` |

---

## Enabling Bedrock Model Access (If You Get AccessDeniedException)

1. Go to **https://us-west-2.console.aws.amazon.com/bedrock**
2. Left sidebar → **"Model access"**
3. Click **"Manage model access"**
4. Find and check:
   - **Amazon → Titan Embeddings V2 - Text** (`amazon.titan-embed-text-v2:0`)
   - **Anthropic → Claude Sonnet** (for the agent later)
5. Click **"Save changes"**
6. Wait 1-2 minutes for access to activate, then retry.

---

## Full File Listing — What This Guide Produces

After completing both parts, your setup is:

```
AWS (us-west-2)
├── S3
│   └── bread-academy-shared/student-06/fraud_rules/*.md  (32 files — already uploaded)
│
├── Bedrock Knowledge Base
│   └── fraud-kb-student-06
│       ├── Data Source: fraud-rules-s3 → S3 prefix above
│       ├── Embeddings : amazon.titan-embed-text-v2:0
│       └── Vector Store: OpenSearch Serverless (auto-created)
│
└── Your KB_ID string (e.g. "ABCDE12345")

Databricks Notebook
├── query_kb()             — debug/test retrieval manually
├── retrieve_rules()       — the actual agent tool
├── build_rules_query()    — auto-builds query from transaction fields
└── retrieve_rules_tool    — LangChain @tool wrapper for LangGraph agent
```
