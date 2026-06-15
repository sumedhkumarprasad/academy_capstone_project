# Bedrock Agent — Complete Guide
### Four Tools · Action Groups · Invoke Loop · CloudWatch Observability

---

## Platform Key

Every step in this guide is clearly labelled with where you run it:

```
┌─────────────────────────────────────────────────────────┐
│  🌐  AWS CONSOLE   — browser UI at console.aws.amazon.com │
│  🐍  DATABRICKS    — Python cell in your Databricks       │
│                      notebook (calls AWS via boto3)        │
└─────────────────────────────────────────────────────────┘
```

> **Rule of thumb:** You use the AWS Console only for things that need
> a one-time manual click (enabling model access, verifying resources).
> Everything else — creating the agent, action groups, running tools,
> reading logs — is Python code in Databricks.

---

## Architecture: What You Are Building

```
  AWS CONSOLE (browser — one-time setup)
  └─ Enable Claude Sonnet 4.5 model access

  DATABRICKS NOTEBOOK (all Python cells below)
  │
  │  SETUP — run once per capstone session
  │  ├─ STEP 1  ─ AWS clients + constants
  │  ├─ STEP 2  ─ Create Bedrock Agent (the "brain")
  │  ├─ STEP 3  ─ Attach Action Group 1: invoke_fraud_model
  │  ├─ STEP 4  ─ Attach Action Group 2: retrieve_rules
  │  ├─ STEP 5  ─ Attach Action Group 3: get_customer_profile
  │  ├─ STEP 6  ─ Attach Action Group 4: generate_report
  │  ├─ STEP 7  ─ Prepare agent → DRAFT alias
  │  └─ STEP 8  ─ CloudWatch log group setup
  │
  │  RUNTIME — run for every transaction batch
  │  ├─ STEP 9  ─ Python implementations of all 4 tools
  │  ├─ STEP 10 ─ The invoke loop
  │  ├─ STEP 11 ─ Run the full batch + write reports to S3
  │  └─ STEP 12 ─ Validate & view results
  │
  │  MAINTENANCE — run as needed
  │  └─ STEP 13 ─ Agent lifecycle management
  │
  AWS SERVICES (called silently from Databricks via boto3)
  ├─ Amazon Bedrock        — hosts the agent + KB
  ├─ Amazon SageMaker      — hosts the XGBoost endpoint
  ├─ Amazon S3             — stores investigation reports
  └─ Amazon CloudWatch     — stores reasoning audit trail
```

**How Bedrock Agents work (one paragraph for beginners):**
You define an agent with a system prompt and give it a list of "action groups" — each
action group is one tool. At runtime you send the agent a transaction. Claude Sonnet
reads the transaction, decides which tool to call first, and emits a `returnControl`
event with the tool name and arguments. Your Python code in Databricks catches that
event, runs the real tool (e.g. calls the SageMaker endpoint), and posts the result
back to the agent. The agent then decides what to do next (call another tool, or write
the final answer). This loop continues until the agent is satisfied.

---

## Prerequisites Checklist

Before running any cell, confirm ALL of these:

- [ ] SageMaker endpoint is `InService` — verified in `sagemaker_endpoint_guide.md`
- [ ] Bedrock KB is `ACTIVE` and synced — verified in `bedrock_kb_guide.md`
- [ ] `ENDPOINT_NAME`, `KB_ID`, `SELECTED_FEATURES` are already defined in your notebook
- [ ] `customers.csv` and `fad_transactions.csv` are at your Databricks working path
- [ ] Claude Sonnet 4.5 model access enabled in Bedrock console (see PRE-STEP below)

---

## PRE-STEP — Enable Claude Sonnet 4.5 Model Access

```
Platform : 🌐 AWS CONSOLE
URL      : https://us-west-2.console.aws.amazon.com/bedrock
Time     : ~2 minutes (one-time only)
```

**Why:** Bedrock requires you to explicitly opt-in to each model before you can
use it — even programmatically. You must do this in the browser; there is no
boto3 API for it.

**Steps:**
1. Open the URL above in your browser
2. Confirm top-right region selector shows **US West (Oregon) us-west-2**
3. In the left sidebar → click **"Model access"**
4. Click the orange **"Manage model access"** button (top right)
5. Find and tick:
   - **Anthropic → Claude Sonnet** (look for `claude-sonnet-4-5` or the latest Sonnet)
   - **Amazon → Titan Embeddings V2 - Text** (if not already enabled from the KB guide)
6. Click **"Save changes"** at the bottom
7. Wait 1–2 minutes. The status column should change to **"Access granted"**

✅ You only do this once. Move to STEP 1.

---

## STEP 1 — AWS Setup & Constants

```
Platform : 🐍 DATABRICKS NOTEBOOK
Cell     : Run this FIRST in every new session — it initialises all AWS clients
           and sets the constants used by every step below.
```

```python
import boto3, os, json, time, uuid, io
import pandas as pd
import numpy as np
import traceback as tb

# ── Student credentials ───────────────────────────────────────────────────────
STUDENT_NUM = "06"

os.environ["AWS_ACCESS_KEY_ID"]     = "AKIA6AK5B2HLV2E6FD6F"
os.environ["AWS_SECRET_ACCESS_KEY"] = "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"
os.environ["AWS_REGION"]            = "us-west-2"
os.environ["AWS_DEFAULT_REGION"]    = "us-west-2"

session          = boto3.Session(region_name="us-west-2")

# ── boto3 clients — each talks to a different AWS service ────────────────────
bedrock_agent    = session.client("bedrock-agent")          # CREATE: agents, action groups
bedrock_agent_rt = session.client("bedrock-agent-runtime")  # INVOKE: agent, KB retrieve
sm_runtime       = session.client("sagemaker-runtime")      # CALL: SageMaker endpoint
s3_client        = session.client("s3")                     # READ/WRITE: S3 reports
logs_client      = session.client("logs")                   # WRITE: CloudWatch log events
sts              = session.client("sts")

ACCOUNT_ID    = sts.get_caller_identity()["Account"]
REGION        = "us-west-2"

# ── Resource IDs — paste values from your previous guides here ────────────────
ENDPOINT_NAME = f"fraud-detector-student-{STUDENT_NUM}"   # from sagemaker_endpoint_guide.md
KB_ID         = "ABCDE12345"                              # from bedrock_kb_guide.md ← PASTE YOURS

# ── Agent identity ────────────────────────────────────────────────────────────
AGENT_NAME    = f"fraud-investigation-agent-student-{STUDENT_NUM}"

# IAM role that Bedrock uses to call other AWS services on your behalf.
# Find it: AWS Console → IAM → Roles → search "BedrockExecution" or ask instructor.
AGENT_ROLE    = f"arn:aws:iam::{ACCOUNT_ID}:role/AmazonBedrockExecutionRoleForAgents"

# Claude Sonnet 4.5 inference profile (cross-region, as specified in capstone1.md)
MODEL_ID      = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

# ── Output destinations ───────────────────────────────────────────────────────
REPORT_BUCKET = f"sagemaker-{REGION}-{ACCOUNT_ID}"         # S3 bucket for JSON/MD reports
REPORT_PREFIX = f"student-{STUDENT_NUM}/investigation-reports"
LOG_GROUP     = f"/bread-academy/capstone1/student-{STUDENT_NUM}"  # CloudWatch log group

print(f"✅ Account      : {ACCOUNT_ID}")
print(f"✅ Endpoint     : {ENDPOINT_NAME}")
print(f"✅ KB ID        : {KB_ID}")
print(f"✅ Agent name   : {AGENT_NAME}")
print(f"✅ Model        : {MODEL_ID}")
print(f"✅ Report bucket: s3://{REPORT_BUCKET}/{REPORT_PREFIX}/")
print(f"✅ Log group    : {LOG_GROUP}")
```

> **AGENT_ROLE tip:** If the auto-built ARN above doesn't exist, open
> AWS Console → IAM → Roles → search "Bedrock" and copy the ARN of any role
> with "BedrockExecution" in its name. Paste it as the literal string.

---

## STEP 2 — Create the Bedrock Agent

```
Platform : 🐍 DATABRICKS NOTEBOOK
Action   : Creates the agent in Amazon Bedrock via boto3.
           No AWS Console action needed — boto3 does it all.
Run once : Yes — skip this cell if AGENT_ID is already set from a previous session.
```

```python
# ── System prompt: tells Claude its role and how to use the four tools ────────
AGENT_SYSTEM_PROMPT = """
You are an expert fraud investigator at Bread Financial. Your job is to analyze
card transactions and produce structured investigation reports for compliance review.

For each transaction you receive, follow this reasoning process:

1. SCORE: Always call invoke_fraud_model first to get the fraud risk score.

2. INVESTIGATE (only if score >= 0.40):
   - Call retrieve_rules with a plain-English description of the suspicious
     transaction attributes (card presence, MCC, country, CVV/AVS outcome,
     velocity, amount).
   - Call get_customer_profile to understand the cardholder's risk history.
   - You may call these in any order based on what you learn.

3. REPORT: Always call generate_report last to produce the structured JSON output.
   Include: transaction_id, fraud_probability, risk_label, matched_rules,
   customer_risk_summary, recommendation, and reasoning_summary.

If the fraud score is below 0.40 (LOW risk), skip to generate_report immediately
with a brief "no investigation needed" conclusion.

Be concise. Summarize tool outputs rather than repeating them verbatim.
Always finish with generate_report.
""".strip()

# ── Call Bedrock to create the agent ──────────────────────────────────────────
create_resp = bedrock_agent.create_agent(
    agentName               = AGENT_NAME,
    foundationModel         = MODEL_ID,
    agentResourceRoleArn    = AGENT_ROLE,
    description             = f"Fraud investigation agent — capstone student {STUDENT_NUM}",
    instruction             = AGENT_SYSTEM_PROMPT,
    idleSessionTTLInSeconds = 1800,   # 30-min session timeout per transaction
)

AGENT_ID = create_resp["agent"]["agentId"]
print(f"✅ Agent created in Amazon Bedrock!")
print(f"   Agent ID   : {AGENT_ID}   ← SAVE THIS")
print(f"   Agent name : {AGENT_NAME}")
print(f"   Status     : {create_resp['agent']['agentStatus']}")

# Wait a moment before adding action groups
time.sleep(5)
```

> **📋 Save AGENT_ID** — copy it to a text file. If you close the notebook
> you'll need it to avoid creating duplicate agents.
>
> **Verify in AWS Console (optional):**
> Bedrock → Agents → you should see `fraud-investigation-agent-student-06` listed.

---

## STEP 3 — Action Group 1: `invoke_fraud_model`

```
Platform : 🐍 DATABRICKS NOTEBOOK
Action   : Registers the tool SCHEMA with Bedrock so Claude knows it exists.
           Does NOT run the actual tool — that happens at runtime in STEP 10.
```

**What this does:** Tells the agent "you have a tool called `invoke_fraud_model`
that takes a `transaction_id` string and returns a fraud risk score." The agent
will decide when to call it.

```python
ag1_resp = bedrock_agent.create_agent_action_group(
    agentId          = AGENT_ID,
    agentVersion     = "DRAFT",
    actionGroupName  = "invoke_fraud_model",
    description      = (
        "Score a card transaction for fraud risk using the XGBoost model "
        "deployed on Amazon SageMaker. Returns fraud_probability (0–1), "
        "binary prediction, and risk_label (HIGH / MEDIUM / LOW). "
        "Call this FIRST for every transaction."
    ),
    # RETURN_CONTROL = the agent asks YOU (Databricks) to run the tool,
    # then you post the result back. No Lambda function needed.
    actionGroupExecutor = {"customControl": "RETURN_CONTROL"},
    functionSchema = {
        "functions": [
            {
                "name": "invoke_fraud_model",
                "description": (
                    "Call the SageMaker XGBoost endpoint to get a fraud probability "
                    "score for one card transaction. Always call this tool first."
                ),
                "parameters": {
                    "transaction_id": {
                        "type": "string",
                        "description": (
                            "Unique transaction identifier from fad_transactions. "
                            "Format: txn_XXXXXXXX  e.g. 'txn_00049897'"
                        ),
                        "required": True,
                    }
                },
            }
        ]
    },
)

print(f"✅ Action Group 1 registered: invoke_fraud_model")
print(f"   State: {ag1_resp['agentActionGroup']['actionGroupState']}")
```

> **Verify in AWS Console (optional):**
> Bedrock → Agents → click your agent → "Action groups" tab →
> you should see `invoke_fraud_model` listed with state ENABLED.

---

## STEP 4 — Action Group 2: `retrieve_rules`

```
Platform : 🐍 DATABRICKS NOTEBOOK
Action   : Registers the retrieve_rules tool schema with Bedrock.
           At runtime, this triggers a query to your Bedrock Knowledge Base.
```

```python
ag2_resp = bedrock_agent.create_agent_action_group(
    agentId          = AGENT_ID,
    agentVersion     = "DRAFT",
    actionGroupName  = "retrieve_rules",
    description      = (
        "Query the Bedrock Knowledge Base to retrieve fraud detection rules "
        "that match the suspicious attributes of a transaction. "
        "Use after invoke_fraud_model returns a score >= 0.40."
    ),
    actionGroupExecutor = {"customControl": "RETURN_CONTROL"},
    functionSchema = {
        "functions": [
            {
                "name": "retrieve_rules",
                "description": (
                    "Search the fraud-rules knowledge base with a plain-English "
                    "description of why this transaction looks suspicious. "
                    "Returns the most relevant fraud rule chunks with scores."
                ),
                "parameters": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Plain-English description of the suspicious transaction. "
                            "Include: card presence (card-present / card-not-present), "
                            "entry mode (chip / ecom / swipe), MCC and merchant type, "
                            "merchant country if foreign, CVV/AVS outcomes, "
                            "velocity (24h amount and count), transaction amount. "
                            "Example: 'card-not-present ecom gambling MCC 7995 "
                            "Nigeria CVV mismatch high velocity $2000 in 24h'"
                        ),
                        "required": True,
                    },
                    "n_results": {
                        "type": "integer",
                        "description": (
                            "Number of rule chunks to retrieve. Default 5. "
                            "Use 3 for a quick lookup, 7 for deep investigation."
                        ),
                        "required": False,
                    },
                },
            }
        ]
    },
)

print(f"✅ Action Group 2 registered: retrieve_rules")
print(f"   State: {ag2_resp['agentActionGroup']['actionGroupState']}")
```

---

## STEP 5 — Action Group 3: `get_customer_profile`

```
Platform : 🐍 DATABRICKS NOTEBOOK
Action   : Registers the get_customer_profile tool schema with Bedrock.
           At runtime, this reads one row from customers.csv in Databricks.
```

```python
ag3_resp = bedrock_agent.create_agent_action_group(
    agentId          = AGENT_ID,
    agentVersion     = "DRAFT",
    actionGroupName  = "get_customer_profile",
    description      = (
        "Retrieve the full cardholder profile for a given account number. "
        "Returns tenure, credit score band, risk tier, delinquency flag, "
        "average monthly spend, occupation, segment, and profile summary."
    ),
    actionGroupExecutor = {"customControl": "RETURN_CONTROL"},
    functionSchema = {
        "functions": [
            {
                "name": "get_customer_profile",
                "description": (
                    "Look up a cardholder's profile from the customers table. "
                    "Call this alongside retrieve_rules when fraud score is HIGH "
                    "or MEDIUM to understand the customer's history and risk tier."
                ),
                "parameters": {
                    "account_num": {
                        "type": "string",
                        "description": (
                            "The account number from fad_transactions. "
                            "Format: acct_XXXXXXX  e.g. 'acct_0002002'. "
                            "Same as customer_id in the customers table."
                        ),
                        "required": True,
                    }
                },
            }
        ]
    },
)

print(f"✅ Action Group 3 registered: get_customer_profile")
print(f"   State: {ag3_resp['agentActionGroup']['actionGroupState']}")
```

---

## STEP 6 — Action Group 4: `generate_report`

```
Platform : 🐍 DATABRICKS NOTEBOOK
Action   : Registers the generate_report tool schema with Bedrock.
           At runtime, this builds a JSON + Markdown report and saves it to S3.
```

```python
ag4_resp = bedrock_agent.create_agent_action_group(
    agentId          = AGENT_ID,
    agentVersion     = "DRAFT",
    actionGroupName  = "generate_report",
    description      = (
        "Generate the final structured JSON investigation report for a transaction. "
        "Always call this LAST — after scoring and (if high-risk) after retrieving "
        "rules and customer profile."
    ),
    actionGroupExecutor = {"customControl": "RETURN_CONTROL"},
    functionSchema = {
        "functions": [
            {
                "name": "generate_report",
                "description": (
                    "Produce a structured compliance investigation report in JSON format. "
                    "Synthesises fraud score, matched rules, and customer profile "
                    "into a report with a clear recommendation."
                ),
                "parameters": {
                    "transaction_id": {
                        "type": "string",
                        "description": "The transaction ID being investigated.",
                        "required": True,
                    },
                    "fraud_probability": {
                        "type": "number",
                        "description": "Fraud probability (0.0–1.0) from invoke_fraud_model.",
                        "required": True,
                    },
                    "risk_label": {
                        "type": "string",
                        "description": "HIGH, MEDIUM, or LOW — from invoke_fraud_model.",
                        "required": True,
                    },
                    "matched_rules": {
                        "type": "string",
                        "description": (
                            "Summary of fraud rules matched by retrieve_rules. "
                            "Pass empty string if score was LOW and no rules retrieved."
                        ),
                        "required": True,
                    },
                    "customer_risk_summary": {
                        "type": "string",
                        "description": (
                            "One-paragraph summary from get_customer_profile. "
                            "Pass empty string if score was LOW."
                        ),
                        "required": True,
                    },
                    "recommendation": {
                        "type": "string",
                        "description": (
                            "One of: BLOCK / REVIEW / MONITOR / APPROVE. "
                            "BLOCK  = very high confidence fraud (score >= 0.80). "
                            "REVIEW = HIGH or MEDIUM risk — needs human analyst. "
                            "MONITOR = borderline — watch for pattern. "
                            "APPROVE = LOW risk — genuine transaction."
                        ),
                        "required": True,
                    },
                    "reasoning_summary": {
                        "type": "string",
                        "description": (
                            "2–4 sentences explaining the recommendation. "
                            "Reference specific rules triggered and customer factors."
                        ),
                        "required": True,
                    },
                },
            }
        ]
    },
)

print(f"✅ Action Group 4 registered: generate_report")
print(f"   State: {ag4_resp['agentActionGroup']['actionGroupState']}")
```

---

## STEP 7 — Prepare the Agent (Compile + Create Alias)

```
Platform : 🐍 DATABRICKS NOTEBOOK
Action   : Compiles all 4 action groups into a deployable agent version,
           then creates a named alias to invoke at runtime.
Run once : Yes — re-run only if you change the system prompt or action groups.
Wait     : ~1–2 minutes while status polls from PREPARING → PREPARED.
```

```python
# ── Trigger preparation ───────────────────────────────────────────────────────
print("⏳ Preparing agent — compiling all 4 action groups...")
bedrock_agent.prepare_agent(agentId=AGENT_ID)

# ── Poll until PREPARED ───────────────────────────────────────────────────────
while True:
    status = bedrock_agent.get_agent(agentId=AGENT_ID)["agent"]["agentStatus"]
    print(f"   Status: {status}", end="\r")
    if status == "PREPARED":
        print(f"\n✅ Agent is PREPARED and ready")
        break
    elif status == "FAILED":
        detail = bedrock_agent.get_agent(agentId=AGENT_ID)["agent"]
        raise RuntimeError(f"❌ Preparation FAILED.\n{detail}")
    time.sleep(5)

# ── Create a named alias pointing to the DRAFT version ───────────────────────
alias_resp = bedrock_agent.create_agent_alias(
    agentId              = AGENT_ID,
    agentAliasName       = f"v1-student-{STUDENT_NUM}",
    description          = "Capstone alias — points to DRAFT version",
    routingConfiguration = [{"agentVersion": "DRAFT"}],
)

AGENT_ALIAS_ID = alias_resp["agentAlias"]["agentAliasId"]

print(f"\n✅ Agent alias created!")
print(f"\n📋 SAVE THESE TWO VALUES — needed every time you invoke the agent:")
print(f'   AGENT_ID       = "{AGENT_ID}"')
print(f'   AGENT_ALIAS_ID = "{AGENT_ALIAS_ID}"')
```

> **Verify in AWS Console (optional):**
> Bedrock → Agents → click your agent → status shows **Prepared**.
> Click "Aliases" tab → you should see `v1-student-06` listed.

---

## STEP 8 — CloudWatch Observability Setup

```
Platform : 🐍 DATABRICKS NOTEBOOK
Action   : Creates a CloudWatch log group and helper functions that emit a
           structured JSON log line after every tool call.
           The logs appear in AWS CloudWatch Logs (browser) after you run STEP 10.
```

```python
def setup_cloudwatch():
    """Create the log group. Safe to call multiple times — ignores ResourceAlreadyExists."""
    try:
        logs_client.create_log_group(logGroupName=LOG_GROUP)
        print(f"✅ Created CloudWatch log group : {LOG_GROUP}")
    except logs_client.exceptions.ResourceAlreadyExistsException:
        print(f"✅ CloudWatch log group exists  : {LOG_GROUP}")

setup_cloudwatch()


def create_log_stream(run_id: str):
    """
    Create one log stream per batch run.
    run_id becomes the stream name — makes it easy to find a run's logs.
    """
    try:
        logs_client.create_log_stream(logGroupName=LOG_GROUP, logStreamName=run_id)
    except logs_client.exceptions.ResourceAlreadyExistsException:
        pass  # reuse existing stream


def emit_log(run_id: str, event_dict: dict):
    """
    Write one structured JSON line to CloudWatch.
    Call after every tool invocation for a full audit trail.

    Parameters
    ----------
    run_id     : str — the log stream name for this batch run
    event_dict : dict — must contain at minimum: step, txn_id
    """
    try:
        streams = logs_client.describe_log_streams(
            logGroupName        = LOG_GROUP,
            logStreamNamePrefix = run_id,
            limit               = 1,
        )["logStreams"]

        log_event = {
            "timestamp": int(time.time() * 1000),    # milliseconds epoch
            "message":   json.dumps(event_dict, default=str),
        }
        kwargs = {
            "logGroupName":  LOG_GROUP,
            "logStreamName": run_id,
            "logEvents":     [log_event],
        }
        # CloudWatch needs a sequence token for streams that already have events
        if streams and streams[0].get("uploadSequenceToken"):
            kwargs["sequenceToken"] = streams[0]["uploadSequenceToken"]

        logs_client.put_log_events(**kwargs)

    except Exception as e:
        # Non-fatal — don't let a logging failure crash the investigation
        print(f"⚠️  CloudWatch emit failed (non-fatal): {e}")


# ── Quick smoke test ──────────────────────────────────────────────────────────
_test_stream = f"setup-test-{uuid.uuid4().hex[:8]}"
create_log_stream(_test_stream)
emit_log(_test_stream, {
    "step":           "setup_test",
    "txn_id":         "txn_00000001",
    "result_summary": "CloudWatch logging is working",
})
print(f"✅ Test log written. Verify in AWS Console:")
print(f"   CloudWatch → Log groups → {LOG_GROUP} → stream: {_test_stream}")
```

> **View logs in AWS Console:**
> CloudWatch → Log groups → `/bread-academy/capstone1/student-06`
> → click any stream → you will see one JSON line per tool call.

---

## STEP 9 — Python Implementations of All 4 Tools

```
Platform : 🐍 DATABRICKS NOTEBOOK
Action   : Defines the four Python functions that actually DO the work.
           These run in Databricks — NOT in AWS Lambda.
           Bedrock agent calls these indirectly via the returnControl event
           you catch in STEP 10.
```

```python
# ── Load data tables once at notebook start ───────────────────────────────────
# These stay in Databricks memory; the agent queries them via get_customer_profile
customers_df = pd.read_csv("customers.csv").set_index("customer_id")
fad_txn_df   = pd.read_csv("fad_transactions.csv").set_index("transaction_id")

print(f"✅ Loaded customers    : {len(customers_df):,} rows")
print(f"✅ Loaded transactions : {len(fad_txn_df):,} rows")


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 1: invoke_fraud_model
# Runs in  : DATABRICKS (calls Amazon SageMaker endpoint in AWS)
# Data flow: Databricks → SageMaker endpoint → fraud score → Databricks → Agent
# ══════════════════════════════════════════════════════════════════════════════
def invoke_fraud_model(transaction_id: str) -> dict:
    """
    Preprocess one transaction row and call the SageMaker XGBoost endpoint.
    Returns fraud_probability, binary prediction, risk_label, and raw fields
    the agent uses to formulate its retrieve_rules query.
    """
    if transaction_id not in fad_txn_df.index:
        return {"error": f"transaction_id '{transaction_id}' not found"}

    raw_row = fad_txn_df.loc[[transaction_id]].reset_index()

    # preprocess_for_inference() comes from sagemaker_endpoint_guide.md
    # It applies the same joins + feature engineering + OHE as training
    features = preprocess_for_inference(raw_row)

    # Call the live SageMaker endpoint in AWS (us-west-2)
    payload  = json.dumps({"features": features})
    response = sm_runtime.invoke_endpoint(
        EndpointName = ENDPOINT_NAME,
        ContentType  = "application/json",
        Body         = payload,
    )
    result = json.loads(response["Body"].read().decode("utf-8"))

    prob = result["fraud_probability"]
    pred = result["is_fraud_prediction"]

    if prob >= 0.70:
        risk_label = "HIGH"
    elif prob >= 0.40:
        risk_label = "MEDIUM"
    else:
        risk_label = "LOW"

    raw = fad_txn_df.loc[transaction_id]
    return {
        "transaction_id":      transaction_id,
        "fraud_probability":   round(prob, 4),
        "is_fraud_prediction": int(pred),
        "risk_label":          risk_label,
        # Pass raw fields so agent can build an informed retrieve_rules query
        "tran_amt":            float(raw.get("tran_amt", 0)),
        "card_prsn_cd":        str(raw.get("card_prsn_cd", "")),
        "entry_mode_ind":      str(raw.get("entry_mode_ind", "")),
        "merch_cat_code_cd":   int(raw.get("merch_cat_code_cd", 0)),
        "mrch_cntry_cd":       str(raw.get("mrch_cntry_cd", "")),
        "cvv2_cvc2_otcm_cd":   str(raw.get("cvv2_cvc2_otcm_cd", "")),
        "addr_vrfc_otcm_cd":   str(raw.get("addr_vrfc_otcm_cd", "")),
        "total_velocity_amt":  float(raw.get("total_velocity_amt", 0)),
        "hour_24_cnt":         int(raw.get("hour_24_cnt", 0)),
        "account_num":         str(raw.get("account_num", "")),
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2: retrieve_rules
# Runs in  : DATABRICKS (calls Amazon Bedrock Knowledge Base in AWS)
# Data flow: Databricks → Bedrock KB (OpenSearch) → matching rules → Databricks → Agent
# ══════════════════════════════════════════════════════════════════════════════
def retrieve_rules(query: str, n_results: int = 5) -> dict:
    """
    Query the Bedrock Knowledge Base with a plain-English description of
    the suspicious transaction. Returns the top matching fraud rule chunks.
    Thin wrapper around bedrock_agent_rt.retrieve() as specified in capstone1.md.
    """
    response = bedrock_agent_rt.retrieve(
        knowledgeBaseId = KB_ID,
        retrievalQuery  = {"text": query},
        retrievalConfiguration = {
            "vectorSearchConfiguration": {"numberOfResults": int(n_results)}
        },
    )

    rules = []
    for r in response["retrievalResults"]:
        rules.append({
            "rule_text":  r["content"]["text"],
            "score":      round(r.get("score", 0.0), 4),
            "source_doc": r.get("location", {})
                           .get("s3Location", {})
                           .get("uri", "unknown").split("/")[-1],
        })

    rules_summary = "\n\n---\n\n".join(
        f"[Rule {i+1} | score={r['score']} | source={r['source_doc']}]\n{r['rule_text']}"
        for i, r in enumerate(rules)
    )

    return {
        "query":         query,
        "rules_found":   len(rules),
        "rules_summary": rules_summary,   # agent reads this
        "rules":         rules,           # structured list for debugging
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 3: get_customer_profile
# Runs in  : DATABRICKS (reads customers.csv already loaded into memory)
# Data flow: Databricks memory (customers_df) → profile dict → Agent
# ══════════════════════════════════════════════════════════════════════════════
def get_customer_profile(account_num: str) -> dict:
    """
    Look up one cardholder from the customers table.
    customers.customer_id == fad_transactions.account_num (same value, different name).
    """
    if account_num not in customers_df.index:
        return {"error": f"account_num '{account_num}' not found in customers table"}

    row = customers_df.loc[account_num]
    return {
        "account_num":            account_num,
        "account_tenure_months":  int(row.get("account_tenure_months", 0)),
        "avg_monthly_spend":      float(row.get("avg_monthly_spend", 0)),
        "home_zip":               str(row.get("home_zip", "")),
        "credit_score_band":      str(row.get("credit_score_band", "")),
        "risk_tier":              str(row.get("risk_tier", "")),
        "delinquency_flag":       int(row.get("delinquency_flag", 0)),
        "occupation":             str(row.get("occupation", "")),
        "segment":                str(row.get("segment", "")),
        "profile_summary":        str(row.get("profile_summary", "")),
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 4: generate_report
# Runs in  : DATABRICKS — builds report dict, then saves JSON + Markdown to S3
# Data flow: Agent args → Databricks → report dict → Amazon S3 (two files)
# ══════════════════════════════════════════════════════════════════════════════
def generate_report(
    transaction_id:        str,
    fraud_probability:     float,
    risk_label:            str,
    matched_rules:         str,
    customer_risk_summary: str,
    recommendation:        str,
    reasoning_summary:     str,
) -> dict:
    """
    Build the structured JSON investigation report and save both JSON and
    Markdown versions to S3. Returns the report dict for the agent to confirm.
    """
    report = {
        "report_id":              f"RPT-{transaction_id}-{int(time.time())}",
        "generated_at":           time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "transaction_id":         transaction_id,
        "fraud_probability":      round(float(fraud_probability), 4),
        "risk_label":             risk_label,
        "recommendation":         recommendation,
        "matched_rules":          matched_rules,
        "customer_risk_summary":  customer_risk_summary,
        "reasoning_summary":      reasoning_summary,
        "investigator":           AGENT_NAME,
    }

    # ── Save JSON to S3 ───────────────────────────────────────────────────────
    json_key = f"{REPORT_PREFIX}/{transaction_id}/report.json"
    s3_client.put_object(
        Bucket      = REPORT_BUCKET,
        Key         = json_key,
        Body        = json.dumps(report, indent=2),
        ContentType = "application/json",
    )

    # ── Save Markdown to S3 for human reviewers ───────────────────────────────
    md_content = f"""# Fraud Investigation Report
**Report ID:** {report['report_id']}
**Generated:** {report['generated_at']}

## Transaction Summary
| Field | Value |
|---|---|
| Transaction ID | `{transaction_id}` |
| Fraud Probability | {fraud_probability:.4f} |
| Risk Label | **{risk_label}** |
| Recommendation | **{recommendation}** |

## Reasoning
{reasoning_summary}

## Matched Fraud Rules
{matched_rules if matched_rules else "_No rules triggered (LOW risk)_"}

## Customer Risk Summary
{customer_risk_summary if customer_risk_summary else "_Not retrieved (LOW risk)_"}
"""
    md_key = f"{REPORT_PREFIX}/{transaction_id}/report.md"
    s3_client.put_object(
        Bucket      = REPORT_BUCKET,
        Key         = md_key,
        Body        = md_content,
        ContentType = "text/markdown",
    )

    report["s3_json"] = f"s3://{REPORT_BUCKET}/{json_key}"
    report["s3_md"]   = f"s3://{REPORT_BUCKET}/{md_key}"
    return report


# ── Tool dispatch table: agent tool name → Python function ────────────────────
TOOL_DISPATCH = {
    "invoke_fraud_model":   invoke_fraud_model,
    "retrieve_rules":       retrieve_rules,
    "get_customer_profile": get_customer_profile,
    "generate_report":      generate_report,
}

print("✅ All 4 tool functions defined in Databricks")
print("   Dispatch table:", list(TOOL_DISPATCH.keys()))
```

---

## STEP 10 — The Agent Invoke Loop

```
Platform  : 🐍 DATABRICKS NOTEBOOK
Action    : Sends a transaction to the Bedrock agent, catches returnControl
            events, dispatches Python tools locally, and posts results back.
Data flow :
  Databricks → Bedrock (invoke_agent) → returnControl event
       → Databricks runs tool → posts result back → Bedrock reasons again
       → … repeats until agent returns final chunk
  Every tool call is also logged to AWS CloudWatch from Databricks.
```

```python
def run_fraud_investigation(transaction_id: str, run_id: str) -> dict:
    """
    Full investigation loop for one transaction.

    Steps:
      1. Send transaction_id to the Bedrock agent
      2. Agent emits returnControl → tool name + args
      3. Databricks runs the tool and posts result back
      4. Repeat until agent emits a final 'chunk' response
      5. Log every tool call to CloudWatch

    Parameters
    ----------
    transaction_id : str — the transaction to investigate
    run_id         : str — CloudWatch stream name for this batch

    Returns
    -------
    dict with transaction_id, final_agent_response, report, tool_calls_made
    """
    print(f"\n{'='*65}")
    print(f"🔍 Investigating: {transaction_id}")
    print(f"{'='*65}")

    # Unique session keeps this transaction separate from others
    session_id = f"session-{transaction_id}-{int(time.time())}"

    initial_message = (
        f"Investigate this card transaction for fraud:\n"
        f"transaction_id = {transaction_id}\n\n"
        f"Follow your investigation protocol: score first, investigate if "
        f"high-risk, then generate a report."
    )

    investigation_report = None
    final_response       = None
    tool_call_count      = 0

    # ── First call to Bedrock ─────────────────────────────────────────────────
    response = bedrock_agent_rt.invoke_agent(
        agentId      = AGENT_ID,
        agentAliasId = AGENT_ALIAS_ID,
        sessionId    = session_id,
        inputText    = initial_message,
        enableTrace  = True,   # captures reasoning steps in the stream
    )

    # ── Main loop: process events until agent is done ─────────────────────────
    while True:
        tool_name     = None
        tool_args     = {}
        invocation_id = None
        completion    = []

        for event in response["completion"]:

            # ── Agent wants to call a tool ────────────────────────────────────
            if "returnControl" in event:
                rc            = event["returnControl"]
                invocation_id = rc["invocationId"]
                func_input    = rc["invocationInputs"][0].get(
                                    "functionInvocationInput", {})
                tool_name     = func_input.get("function", "")

                # Parse parameters — cast numbers where needed
                for p in func_input.get("parameters", []):
                    val = p.get("value", "")
                    if p["name"] == "fraud_probability":
                        val = float(val)
                    elif p["name"] == "n_results":
                        val = int(val) if val else 5
                    tool_args[p["name"]] = val

                print(f"\n  🔧 Agent → {tool_name}({tool_args})")
                break   # exit event loop to dispatch the tool

            # ── Agent reasoning trace ─────────────────────────────────────────
            if "trace" in event:
                orch = (event["trace"]
                        .get("trace", {})
                        .get("orchestrationTrace", {}))
                thinking = orch.get("rationale", {}).get("text", "")
                if thinking:
                    print(f"\n  💭 {thinking[:180]}...")
                    emit_log(run_id, {
                        "step":     "agent_reasoning",
                        "txn_id":   transaction_id,
                        "thinking": thinking,
                    })

            # ── Agent is done — final answer ──────────────────────────────────
            if "chunk" in event:
                completion.append(
                    event["chunk"].get("bytes", b"").decode("utf-8")
                )

        # ── Final chunk received — agent is done ──────────────────────────────
        if completion:
            final_response = "".join(completion)
            print(f"\n  ✅ Agent complete — {tool_call_count} tool calls made")
            print(f"     Response preview: {final_response[:200]}...")
            break

        # ── Dispatch the tool Databricks-side ─────────────────────────────────
        if tool_name:
            tool_call_count += 1

            try:
                if tool_name not in TOOL_DISPATCH:
                    tool_result = {"error": f"Unknown tool: {tool_name}"}
                else:
                    # ← This is where the real work happens in Databricks:
                    #   SageMaker call, KB retrieval, CSV lookup, S3 write
                    tool_result = TOOL_DISPATCH[tool_name](**tool_args)

                if tool_name == "generate_report":
                    investigation_report = tool_result

                result_str = json.dumps(tool_result, default=str)
                print(f"  📤 Result: {result_str[:200]}...")

                # Log to CloudWatch (in AWS) from Databricks
                emit_log(run_id, {
                    "step":           tool_name,
                    "txn_id":         transaction_id,
                    "tool_call_num":  tool_call_count,
                    "args":           tool_args,
                    "result_summary": result_str[:500],
                })

            except Exception as e:
                tool_result = {"error": str(e), "traceback": tb.format_exc()[:400]}
                result_str  = json.dumps(tool_result)
                print(f"  ❌ Tool error: {e}")
                emit_log(run_id, {
                    "step":  f"{tool_name}_ERROR",
                    "txn_id": transaction_id,
                    "error":  str(e),
                })

            # ── Post the Databricks tool result back to Bedrock ───────────────
            response = bedrock_agent_rt.invoke_agent(
                agentId      = AGENT_ID,
                agentAliasId = AGENT_ALIAS_ID,
                sessionId    = session_id,
                inputText    = "",
                enableTrace  = True,
                sessionState = {
                    "invocationId": invocation_id,
                    "returnControlInvocationResults": [
                        {
                            "functionResult": {
                                "actionGroup":  tool_name,
                                "function":     tool_name,
                                "responseBody": {
                                    "application/json": {"body": result_str}
                                },
                            }
                        }
                    ],
                },
            )

    return {
        "transaction_id":       transaction_id,
        "final_agent_response": final_response,
        "report":               investigation_report,
        "tool_calls_made":      tool_call_count,
    }


print("✅ invoke_agent loop defined in Databricks")
```

---

## STEP 11 — Run the Full Batch

```
Platform  : 🐍 DATABRICKS NOTEBOOK
Action    : Calls run_fraud_investigation() for each transaction_id in the batch.
            Progress prints to the Databricks cell output.
            Reports are saved to Amazon S3 and traces to Amazon CloudWatch.
```

```python
def run_batch_investigation(transaction_ids: list, run_id: str = None) -> list:
    """
    Investigate a batch of transactions one by one.

    Parameters
    ----------
    transaction_ids : list of transaction_id strings from fad_transactions
    run_id          : CloudWatch stream name — auto-generated if not provided

    Returns
    -------
    list of result dicts (one per transaction)
    """
    if run_id is None:
        run_id = f"batch-{int(time.time())}-{uuid.uuid4().hex[:8]}"

    create_log_stream(run_id)

    emit_log(run_id, {
        "step":       "batch_start",
        "run_id":     run_id,
        "batch_size": len(transaction_ids),
        "txn_ids":    transaction_ids,
    })

    results = []
    for i, txn_id in enumerate(transaction_ids, 1):
        print(f"\n[{i}/{len(transaction_ids)}] Starting: {txn_id}")
        try:
            result = run_fraud_investigation(txn_id, run_id)
            results.append(result)
        except Exception as e:
            print(f"❌ Failed for {txn_id}: {e}")
            results.append({"transaction_id": txn_id, "error": str(e), "report": None})
        time.sleep(1)   # avoid Bedrock throttling

    emit_log(run_id, {
        "step":              "batch_complete",
        "run_id":            run_id,
        "total_processed":   len(results),
        "reports_generated": sum(1 for r in results if r.get("report")),
    })

    print(f"\n{'='*65}")
    print(f"✅ Batch complete — {len(results)} transactions processed")
    print(f"   Reports in S3    : s3://{REPORT_BUCKET}/{REPORT_PREFIX}/")
    print(f"   CloudWatch stream: {LOG_GROUP}  →  {run_id}")
    return results


# ── Run on 5 sample transactions ──────────────────────────────────────────────
sample_txn_ids = fad_txn_df.index[:5].tolist()
print("Transactions to investigate:")
for t in sample_txn_ids:
    print(f"  {t}")

batch_results = run_batch_investigation(sample_txn_ids)

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n=== INVESTIGATION SUMMARY ===")
for r in batch_results:
    rpt = r.get("report", {}) or {}
    print(f"  {r['transaction_id']:20s} | "
          f"Risk: {rpt.get('risk_label','?'):6s} | "
          f"Score: {rpt.get('fraud_probability', '?')!s:6s} | "
          f"Action: {rpt.get('recommendation','?')}")
```

---

## STEP 12 — Validate & View Results

```
Platform  : 🐍 DATABRICKS NOTEBOOK (boto3 reads from S3)
            🌐 AWS CONSOLE         (visual check of S3 files and CloudWatch logs)
```

**Read reports from Python (Databricks):**

```python
def list_reports() -> list:
    """List all report files in S3 under your student prefix."""
    paginator = s3_client.get_paginator("list_objects_v2")
    pages     = paginator.paginate(Bucket=REPORT_BUCKET, Prefix=REPORT_PREFIX)
    return [obj["Key"] for page in pages for obj in page.get("Contents", [])]

report_files = list_reports()
print(f"Reports saved in S3 ({len(report_files)} files):")
for f in report_files:
    print(f"  s3://{REPORT_BUCKET}/{f}")


def read_report(transaction_id: str) -> dict:
    """Download and parse one JSON report from S3."""
    key = f"{REPORT_PREFIX}/{transaction_id}/report.json"
    obj = s3_client.get_object(Bucket=REPORT_BUCKET, Key=key)
    return json.loads(obj["Body"].read())

# Print the first report
if batch_results and batch_results[0].get("report"):
    first_txn = batch_results[0]["transaction_id"]
    report    = read_report(first_txn)
    print(f"\n=== Report: {first_txn} ===")
    print(json.dumps(report, indent=2))
```

**Verify in AWS Console (optional visual check):**

```
S3 reports  : AWS Console → S3 → sagemaker-us-west-2-{ACCOUNT_ID}
              → student-06/investigation-reports/ → txn_XXXXXXXX/
              → report.json  (structured data)
              → report.md    (human-readable)

CloudWatch  : AWS Console → CloudWatch → Log groups
              → /bread-academy/capstone1/student-06
              → click your run_id stream
              → one JSON line per tool call + batch_start / batch_complete
```

---

## STEP 13 — Agent Lifecycle Management

```
Platform : 🐍 DATABRICKS NOTEBOOK
Action   : Utility functions to check, update, or clean up the agent.
           Run as needed — not part of the main flow.
```

```python
# ── Check current agent status ────────────────────────────────────────────────
def check_agent_status():
    info = bedrock_agent.get_agent(agentId=AGENT_ID)["agent"]
    print(f"Agent ID   : {AGENT_ID}")
    print(f"Alias ID   : {AGENT_ALIAS_ID}")
    print(f"Status     : {info['agentStatus']}")   # should be PREPARED
    print(f"Model      : {info['foundationModel']}")
    print(f"Created    : {info['createdAt']}")
    return info

check_agent_status()


# ── Verify all 4 action groups are attached ───────────────────────────────────
def list_action_groups():
    groups = bedrock_agent.list_agent_action_groups(
        agentId=AGENT_ID, agentVersion="DRAFT"
    )["actionGroupSummaries"]
    print(f"Action Groups ({len(groups)}) — should be 4:")
    for g in groups:
        print(f"  {g['actionGroupName']:30s}  {g['actionGroupState']}")

list_action_groups()


# ── Update the system prompt and re-prepare ───────────────────────────────────
def update_instructions(new_prompt: str):
    """
    Change the agent's system prompt then re-prepare.
    Run this if you want to tune the agent's reasoning behaviour.
    Takes ~1 minute.
    """
    bedrock_agent.update_agent(
        agentId              = AGENT_ID,
        agentName            = AGENT_NAME,
        foundationModel      = MODEL_ID,
        agentResourceRoleArn = AGENT_ROLE,
        instruction          = new_prompt,
    )
    bedrock_agent.prepare_agent(agentId=AGENT_ID)
    print("✅ Agent instructions updated and re-prepared")


# ── Delete agent when capstone is complete ────────────────────────────────────
def delete_agent():
    """
    Clean up to avoid orphaned AWS resources.
    Deletes the alias first (required), then the agent.
    """
    bedrock_agent.delete_agent_alias(agentId=AGENT_ID, agentAliasId=AGENT_ALIAS_ID)
    bedrock_agent.delete_agent(agentId=AGENT_ID, skipResourceInUseCheck=True)
    print(f"🗑️  Agent {AGENT_ID} and alias {AGENT_ALIAS_ID} deleted")

# Uncomment only when you are completely done with the capstone:
# delete_agent()
```

---

## Common Errors & Fixes

| Error | Where it appears | Cause | Fix |
|---|---|---|---|
| `agentStatus: FAILED` after prepare | Databricks output | IAM role missing Bedrock trust policy | Add `bedrock.amazonaws.com` as trusted principal to `AGENT_ROLE` |
| `returnControl` event never arrives | Databricks invoke loop | `customControl` not set | Confirm `actionGroupExecutor={"customControl": "RETURN_CONTROL"}` on all 4 groups |
| `AccessDeniedException` on `invoke_agent` | Databricks output | Claude Sonnet not enabled | AWS Console → Bedrock → Model access → enable Claude Sonnet 4.5 |
| `KeyError` in TOOL_DISPATCH | Databricks output | Agent used a different tool name | Print `tool_name`; ensure it matches `functionSchema.functions[0].name` exactly |
| `ThrottlingException` | Databricks output | Too many concurrent requests | Add `time.sleep(1)` between transactions in batch loop |
| Report missing in S3 | AWS Console → S3 | `generate_report` not called | Check CloudWatch trace — agent may have stopped; tighten system prompt |
| Logs not appearing | AWS Console → CloudWatch | Wrong `LOG_GROUP` or stream | Re-run STEP 8; verify `LOG_GROUP` matches exactly |

---

## Full Platform Map — Summary

```
STEP     PLATFORM     WHAT HAPPENS
─────────────────────────────────────────────────────────────────────
PRE      🌐 Console   Enable Claude Sonnet 4.5 model access (once)
1        🐍 Databricks boto3 clients + constants defined
2        🐍 Databricks bedrock_agent.create_agent() → AGENT_ID created in AWS
3        🐍 Databricks create_agent_action_group(invoke_fraud_model) registered in AWS
4        🐍 Databricks create_agent_action_group(retrieve_rules) registered in AWS
5        🐍 Databricks create_agent_action_group(get_customer_profile) registered in AWS
6        🐍 Databricks create_agent_action_group(generate_report) registered in AWS
7        🐍 Databricks prepare_agent() + create_agent_alias() → AGENT_ALIAS_ID in AWS
8        🐍 Databricks CloudWatch log group created in AWS via boto3
9        🐍 Databricks Python tool functions defined (run locally in Databricks)
10       🐍 Databricks invoke_agent loop: streams events from Bedrock, dispatches
                       tools in Databricks, posts results back to Bedrock
11       🐍 Databricks batch loop → tools call SageMaker + Bedrock KB + S3
12a      🐍 Databricks list_reports() / read_report() → boto3 reads from S3
12b      🌐 Console   S3 browser: view report.json and report.md files
12c      🌐 Console   CloudWatch browser: view reasoning trace per tool call
13       🐍 Databricks check / update / delete agent via boto3
```
