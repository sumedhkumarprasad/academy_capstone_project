# Capstone: Fraud Detection XGBoost Classifier on AWS SageMaker
### Complete Step-by-Step Prompt — Grounded in Actual Dataset Schemas

---

## SECTION 0 — AWS SETUP

```python
import boto3, os, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

STUDENT_NUM = "06"

os.environ["AWS_ACCESS_KEY_ID"]     = "AKIA6AK5B2HLV2E6FD6F"
os.environ["AWS_SECRET_ACCESS_KEY"] = "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"
os.environ["AWS_REGION"] = os.environ["AWS_DEFAULT_REGION"] = "us-west-2"

session           = boto3.Session(region_name="us-west-2")
sagemaker_client  = session.client("sagemaker")
sagemaker_runtime = session.client("sagemaker-runtime")
bedrock_runtime   = session.client("bedrock-runtime")
bedrock_agent     = session.client("bedrock-agent")
bedrock_agent_rt  = session.client("bedrock-agent-runtime")

print("✅ Caller:", session.client("sts").get_caller_identity()["Arn"])
```

> **Important:** Use ONLY the local CSV files provided. Do NOT connect to Databricks,
> Unity Catalog, or any external data source.

---

## SECTION 1 — DATA LOADING & SCHEMA DISCOVERY

Load all six CSV files from the local folder and print a full schema report for each.

### Files and their roles

| File | Rows | Cols | Role |
|---|---|---|---|
| `fad_transactions.csv` | 50,000 | 34 | **Primary fact table** — one row per card authorization. Contains the fraud label. |
| `customers.csv` | 5,000 | 10 | Customer profile table — one row per account. |
| `customer_credit_history.csv` | 10,000 | 7 | Credit bureau history — one row per customer. |
| `ft_fraud_cases.csv` | 1,500 | 15 | Confirmed fraud case details — one row per confirmed fraud transaction. |
| `fraud_transactions.csv` | ~45,000 | 11 | Supplementary transaction table with simplified schema and engineered features. |
| `macro_context.csv` | 36 | 4 | Monthly macroeconomic indicators — one row per month. |

### Schema reference

**fad_transactions.csv** (Primary table — use this as your base)
```
transaction_id          str     PK  — txn_00000001 format
account_num             str     FK  → customers.customer_id
transaction_ts          str     timestamp ISO 8601 UTC
tran_amt                float   transaction amount USD
tran_cd                 int     transaction type code
merch_cat_code_cd       int     4-digit MCC (7995=gambling, 6051=crypto)
mrch_nm                 str     merchant name
merch_city_nm           str     merchant city
card_prsn_cd            str     Y=card present, N=card-not-present
entry_mode_ind          str     chip/swipe/contactless/ecom/manual/token
keyed_swiped_ind        str     keyed vs swiped indicator
mrch_cntry_cd           str     ISO country (71 unique; US dominant ~82%)
merch_zip_cd            int     merchant zip code
card_zip_cd             int     cardholder zip code
ecom_in                 str     Y/N — ecommerce indicator
device_model_cd         str     device model (0.1% null — impute "UNKNOWN")
ip_address_ipv4_id      str     IP address
old_fraud_score         int     prior fraud score 0-999
new_fraud_score         int     current fraud score 0-999
score_type_cd           str     VAA/FALCON/MC_DI/EMS — scoring model used
total_velocity_amt      float   24-hour total spend velocity
cash_velocity_amt       float   24-hour cash spend velocity
hour_24_cnt             int     transaction count in last 24 hours
cvv2_cvc2_otcm_cd       str     CVV outcome: P/U/N/M
addr_vrfc_otcm_cd       str     AVS outcome: N/A/Z/Y/U
avail_credit_amt        float   available credit USD
crdt_line_amt           float   total credit line USD
perc_cred_limt_utlz_pct float  % credit utilization
nmbr_days_dlnq_cnt      int     number of days delinquent
time_on_books_cnt       int     account age in days
risk_reason_cd          str     human-readable risk reason
label_type_cd           int     TARGET: 0=genuine, 1=FRAUD
label_type_desc         str     "GENUINE" or "FRAUD" (text version of label)
partition_date          str     date partition
```

**customers.csv**
```
customer_id             str     PK  — acct_0000001 format (joins to fad_transactions.account_num)
account_tenure_months   int     months since account opened
avg_monthly_spend       float   average monthly spend USD
home_zip                int     home zip code
credit_score_band       str     poor/fair/good/excellent
risk_tier               str     low/medium/high
delinquency_flag        int     0/1
occupation              str     free text
segment                 str     prime/subprime/etc.
profile_summary         str     LLM-written text — DROP for modeling
```

**customer_credit_history.csv**
```
customer_id             str     FK → customers.customer_id (1 row per customer)
credit_score_band       str     poor/fair/good/excellent
utilization_pct         float   credit utilization ratio
delinquency_count_12mo  int     delinquencies in last 12 months
account_age_months      int     age of oldest account
default_label           int     0/1 — ever defaulted
credit_profile_note     str     text description — DROP for modeling
```

**ft_fraud_cases.csv** (confirmed fraud details)
```
ft_case_id              str     PK
transaction_id          str     FK → fad_transactions.transaction_id
account_num             str     FK → fad_transactions.account_num
loss_dt                 str     date of loss
reported_dt             str     date reported
fraud_type_cd           str     fraud type code (FPF, LST, etc.)
gross_fraud_amt         float   gross fraud amount USD
merchant_credit_amt     float   merchant credit received
net_fraud_amt           float   net fraud loss USD
chargeback_amt          float   chargeback amount
chargeback_cnt          int     number of chargebacks
total_transaction_cnt   int     total txns in case
external_status_desc    str     case status text
case_narrative          str     free text narrative — DROP for modeling
loss_type_desc          str     Recovered/Net/etc.
```

**macro_context.csv**
```
month                   str     YYYY-MM-DD (first of each month, 36 months)
unemployment_rate       float   US unemployment rate
fed_funds_rate          float   Federal funds rate
consumer_confidence_idx float   consumer confidence index
```

### Loading code
```python
fad_txn    = pd.read_csv("fad_transactions.csv")
customers  = pd.read_csv("customers.csv")
cred_hist  = pd.read_csv("customer_credit_history.csv")
fraud_cases = pd.read_csv("ft_fraud_cases.csv")
macro      = pd.read_csv("macro_context.csv")

# Print schema for each table
for name, df in [("fad_transactions", fad_txn), ("customers", customers),
                  ("credit_history", cred_hist), ("fraud_cases", fraud_cases),
                  ("macro_context", macro)]:
    print(f"\n{'='*60}")
    print(f"TABLE: {name}  |  Shape: {df.shape}")
    print(df.dtypes.to_frame("dtype").assign(
        pct_null=(df.isnull().mean()*100).round(2),
        n_unique=df.nunique()
    ).to_string())
    print(df.head(3).to_string())
```

---

## SECTION 2 — TABLE JOINS

Join all tables into one modeling dataset. Perform LEFT JOINs on `fad_transactions`
as the base so you keep all 50,000 transactions.

### Join map
```
fad_transactions  (base, 50k rows)
    LEFT JOIN  customers            ON  fad_transactions.account_num = customers.customer_id
    LEFT JOIN  customer_credit_hist ON  fad_transactions.account_num = customer_credit_history.customer_id
    LEFT JOIN  ft_fraud_cases       ON  fad_transactions.transaction_id = ft_fraud_cases.transaction_id
    LEFT JOIN  macro_context        ON  DATE_TRUNC(fad_transactions.transaction_ts, 'month') = macro_context.month
```

### Join code
```python
# Parse timestamps first
fad_txn["transaction_ts"] = pd.to_datetime(fad_txn["transaction_ts"])
fad_txn["txn_month"] = fad_txn["transaction_ts"].dt.to_period("M").dt.to_timestamp()
macro["month"] = pd.to_datetime(macro["month"])

# Perform joins
df = (fad_txn
      .merge(customers.add_prefix("cust_"), left_on="account_num",
             right_on="cust_customer_id", how="left")
      .merge(cred_hist.add_prefix("ch_"), left_on="account_num",
             right_on="ch_customer_id", how="left")
      .merge(fraud_cases[["transaction_id","fraud_type_cd","gross_fraud_amt",
                           "net_fraud_amt","chargeback_amt","chargeback_cnt",
                           "loss_type_desc"]],
             on="transaction_id", how="left")
      .merge(macro, left_on="txn_month", right_on="month", how="left")
)

# Verify — shape should still be 50,000 rows
print(f"Shape after joins: {df.shape}")
assert df.shape[0] == 50000, "Row count changed — check for fan-out joins!"
print("Label distribution:\n", df["label_type_cd"].value_counts())
```

---

## SECTION 3 — TARGET LABEL

```python
# label_type_cd == 1 means FRAUD
df["is_fraud"] = (df["label_type_cd"] == 1).astype(int)

fraud_rate = df["is_fraud"].mean()
print(f"Fraud rate: {fraud_rate:.2%}")  # ~3%

# Class imbalance ratio — needed later for scale_pos_weight
neg = (df["is_fraud"] == 0).sum()
pos = (df["is_fraud"] == 1).sum()
scale_pos_weight = neg / pos
print(f"scale_pos_weight = {scale_pos_weight:.2f}")
```

---

## SECTION 4 — EXPLORATORY DATA ANALYSIS (EDA)

Perform the following plots and statistics. Save each figure to disk.

```python
fig_dir = "figures/"
import os; os.makedirs(fig_dir, exist_ok=True)
```

**4.1 Class distribution**
```python
df["is_fraud"].value_counts(normalize=True).plot(kind="bar", title="Fraud vs Genuine")
plt.savefig(f"{fig_dir}01_class_distribution.png", bbox_inches="tight"); plt.show()
```

**4.2 Missing values heatmap**
```python
null_pct = df.isnull().mean().sort_values(ascending=False)
null_pct[null_pct > 0].plot(kind="bar", title="% Missing by Column")
plt.savefig(f"{fig_dir}02_missing_values.png", bbox_inches="tight"); plt.show()
```

**4.3 Transaction amount by fraud label**
```python
df.boxplot(column="tran_amt", by="is_fraud")
plt.suptitle(""); plt.title("Transaction Amount by Fraud Label")
plt.savefig(f"{fig_dir}03_amount_by_label.png", bbox_inches="tight"); plt.show()
```

**4.4 Fraud rate by key categoricals**
```python
for col in ["card_prsn_cd", "entry_mode_ind", "cvv2_cvc2_otcm_cd",
            "addr_vrfc_otcm_cd", "ecom_in", "score_type_cd"]:
    fraud_by_col = df.groupby(col)["is_fraud"].mean().sort_values(ascending=False)
    fraud_by_col.plot(kind="bar", title=f"Fraud Rate by {col}")
    plt.ylabel("Fraud Rate"); plt.tight_layout()
    plt.savefig(f"{fig_dir}04_fraud_rate_{col}.png", bbox_inches="tight"); plt.show()
```

**4.5 Top 20 high-risk MCC codes**
```python
top_mcc = df.groupby("merch_cat_code_cd")["is_fraud"].mean().sort_values(ascending=False).head(20)
top_mcc.plot(kind="bar", title="Fraud Rate by MCC (Top 20)")
plt.savefig(f"{fig_dir}05_fraud_rate_mcc.png", bbox_inches="tight"); plt.show()
```

**4.6 Fraud rate by merchant country (Top 30)**
```python
country_fraud = df.groupby("mrch_cntry_cd")["is_fraud"].mean().sort_values(ascending=False).head(30)
country_fraud.plot(kind="bar", figsize=(14,4), title="Fraud Rate by Country (Top 30)")
plt.savefig(f"{fig_dir}06_fraud_rate_country.png", bbox_inches="tight"); plt.show()
```

**4.7 Fraud score distributions**
```python
for col in ["new_fraud_score", "old_fraud_score", "total_velocity_amt", "perc_cred_limt_utlz_pct"]:
    df.groupby("is_fraud")[col].plot(kind="hist", bins=50, alpha=0.6,
                                     legend=True, title=f"{col} by Fraud Label")
    plt.savefig(f"{fig_dir}07_{col}_dist.png", bbox_inches="tight"); plt.show()
```

**4.8 Numeric correlation heatmap**
```python
num_cols = df.select_dtypes(include="number").columns.tolist()
corr = df[num_cols].corr()
plt.figure(figsize=(18, 14))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
plt.title("Correlation Heatmap — All Numeric Features")
plt.savefig(f"{fig_dir}08_correlation_heatmap.png", bbox_inches="tight"); plt.show()
```

---

## SECTION 5 — PREPROCESSING

**5.1 Drop identifier / leakage / text columns**
```python
DROP_COLS = [
    # Direct identifiers
    "transaction_id", "account_num", "cust_customer_id", "ch_customer_id",
    "txn_month", "month", "partition_date",
    # Leakage: label text and fraud case amounts (only exist when fraud=1)
    "label_type_cd", "label_type_desc",
    "gross_fraud_amt", "net_fraud_amt", "chargeback_amt", "chargeback_cnt", "loss_type_desc",
    # Free text / LLM-generated — not usable as ML features
    "mrch_nm", "merch_city_nm", "ip_address_ipv4_id", "risk_reason_cd",
    "cust_profile_summary", "cust_occupation", "ch_credit_profile_note",
    # Redundant keys
    "cust_segment"
]

df_model = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
print(f"Shape after dropping identifiers: {df_model.shape}")
print("Remaining columns:", df_model.columns.tolist())
```

**5.2 Handle missing values**
```python
# device_model_cd has 0.1% nulls — fill with UNKNOWN
df_model["device_model_cd"] = df_model["device_model_cd"].fillna("UNKNOWN")

# Numeric nulls (from left joins) — fill with median
num_cols = df_model.select_dtypes(include="number").columns.tolist()
num_cols = [c for c in num_cols if c != "is_fraud"]
for col in num_cols:
    df_model[col].fillna(df_model[col].median(), inplace=True)

# Categorical nulls from left joins (fraud_type_cd etc.) — fill with NONE
cat_cols = df_model.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    df_model[col].fillna("NONE", inplace=True)

print("Nulls remaining:", df_model.isnull().sum().sum())
```

**5.3 One-Hot Encoding**
```python
# Identify categorical columns for OHE
OHE_COLS = [
    "card_prsn_cd", "entry_mode_ind", "keyed_swiped_ind",
    "ecom_in", "cvv2_cvc2_otcm_cd", "addr_vrfc_otcm_cd",
    "score_type_cd", "device_model_cd", "mrch_cntry_cd",
    "cust_credit_score_band", "cust_risk_tier",
    "ch_credit_score_band", "fraud_type_cd"
]
OHE_COLS = [c for c in OHE_COLS if c in df_model.columns]

df_model = pd.get_dummies(df_model, columns=OHE_COLS, drop_first=True, dtype=int)
print(f"Shape after OHE: {df_model.shape}")
```

---

## SECTION 6 — FEATURE ENGINEERING

Create derived features grounded in fraud domain knowledge.

```python
# -- Timestamp-based features
df_model["hour_of_day"]  = pd.to_datetime(df["transaction_ts"]).dt.hour
df_model["day_of_week"]  = pd.to_datetime(df["transaction_ts"]).dt.dayofweek
df_model["is_weekend"]   = (df_model["day_of_week"] >= 5).astype(int)
df_model["is_night_txn"] = ((df_model["hour_of_day"] >= 22) | (df_model["hour_of_day"] <= 5)).astype(int)

# -- Amount anomaly features
df_model["amt_vs_monthly_avg"]     = df["tran_amt"] / (df["cust_avg_monthly_spend"].replace(0, np.nan) + 1)
df_model["velocity_to_credit"]     = df["total_velocity_amt"] / (df["crdt_line_amt"].replace(0, np.nan) + 1)

# -- Score change signal
df_model["fraud_score_delta"]      = df["new_fraud_score"] - df["old_fraud_score"]
df_model["fraud_score_ratio"]      = df["new_fraud_score"] / (df["old_fraud_score"].replace(0, np.nan) + 1)

# -- Credit stress
df_model["credit_stress"]          = df["perc_cred_limt_utlz_pct"] * (df["nmbr_days_dlnq_cnt"] + 1)

# -- Zip mismatch flag (card zip != merchant zip)
df_model["zip_mismatch"]           = (df["card_zip_cd"] != df["merch_zip_cd"]).astype(int)

# -- Cross-border flag
df_model["is_cross_border"]        = (df["mrch_cntry_cd"] != "US").astype(int)

# -- High-risk MCC flag (gambling=7995, crypto=6051, wire=4829)
HIGH_RISK_MCC = {7995, 6051, 4829, 6010, 6011, 5912}
df_model["is_high_risk_mcc"]       = df["merch_cat_code_cd"].isin(HIGH_RISK_MCC).astype(int)

# -- Cash velocity ratio
df_model["cash_velocity_ratio"]    = df["cash_velocity_amt"] / (df["total_velocity_amt"].replace(0, np.nan) + 1)

# -- Macro context — high-unemployment flag
df_model["high_unemployment"]      = (df["unemployment_rate"] > df["unemployment_rate"].median()).astype(int)

print("Feature engineering complete. New shape:", df_model.shape)
```

---

## SECTION 7 — TRAIN / VALIDATION / TEST SPLIT (60 / 20 / 20)

```python
from sklearn.model_selection import train_test_split

X = df_model.drop(columns=["is_fraud"])
y = df_model["is_fraud"]

# First split: 60% train, 40% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, stratify=y, random_state=42
)

# Second split: 50% of temp = 20% each of valid + test
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

print(f"Train : {X_train.shape[0]} rows | Fraud rate: {y_train.mean():.2%}")
print(f"Valid : {X_valid.shape[0]} rows | Fraud rate: {y_valid.mean():.2%}")
print(f"Test  : {X_test.shape[0]} rows  | Fraud rate: {y_test.mean():.2%}")
```

---

## SECTION 8 — BASELINE XGBOOST MODEL (All Features)

Train with default hyperparameters. Use `scale_pos_weight` to address the ~3% fraud rate.

```python
import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

baseline_model = xgb.XGBClassifier(
    n_estimators=200,
    scale_pos_weight=scale_pos_weight,   # neg/pos ratio ~32.3
    eval_metric="auc",
    early_stopping_rounds=20,
    random_state=42,
    use_label_encoder=False
)

baseline_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    verbose=50
)

# AUC on all three splits
for split_name, X_s, y_s in [("TRAIN", X_train, y_train),
                               ("VALID", X_valid, y_valid),
                               ("TEST",  X_test,  y_test)]:
    prob = baseline_model.predict_proba(X_s)[:, 1]
    auc  = roc_auc_score(y_s, prob)
    print(f"{split_name} AUC: {auc:.4f}")
    print(classification_report(y_s, baseline_model.predict(X_s)))
    print(confusion_matrix(y_s, baseline_model.predict(X_s)))
```

### 8.1 ROC Curve
```python
from sklearn.metrics import RocCurveDisplay

fig, ax = plt.subplots(figsize=(8, 6))
for split_name, X_s, y_s in [("Train", X_train, y_train),
                               ("Valid", X_valid, y_valid),
                               ("Test",  X_test,  y_test)]:
    prob = baseline_model.predict_proba(X_s)[:, 1]
    RocCurveDisplay.from_predictions(y_s, prob, name=split_name, ax=ax)
ax.set_title("Baseline XGBoost — ROC Curve")
plt.savefig(f"{fig_dir}09_baseline_roc.png", bbox_inches="tight"); plt.show()
```

### 8.2 Lift & Gain Charts — Baseline Model (Train / Valid / Test)

Define the reusable helper once, then call it for every split of every model.

```python
# ── HELPER: compute decile statistics ────────────────────────────────────────
def compute_decile_stats(y_true, y_prob):
    """
    Rank observations by predicted probability (high → low), assign deciles,
    then compute per-decile lift and cumulative gain.

    Returns a DataFrame with columns:
        decile | actual_fraud | total | fraud_rate | lift | cum_gain_pct
    """
    df_r = pd.DataFrame({"y_true": np.array(y_true), "y_prob": np.array(y_prob)})
    df_r = df_r.sort_values("y_prob", ascending=False).reset_index(drop=True)

    n           = len(df_r)
    total_fraud = df_r["y_true"].sum()
    overall_rate = df_r["y_true"].mean()

    # pd.qcut gives equal-frequency deciles (handles ties gracefully)
    df_r["decile"] = pd.qcut(df_r.index, q=10, labels=range(1, 11)).astype(int)

    stats = (df_r.groupby("decile", observed=True)
               .agg(actual_fraud=("y_true", "sum"),
                    total=("y_true", "count"))
               .reset_index())
    stats["fraud_rate"]    = stats["actual_fraud"] / stats["total"]
    stats["lift"]          = stats["fraud_rate"] / overall_rate
    stats["cum_fraud"]     = stats["actual_fraud"].cumsum()
    stats["cum_gain_pct"]  = stats["cum_fraud"] / total_fraud * 100
    return stats


# ── HELPER: single Lift + Gain figure for one split ──────────────────────────
def plot_lift_gain(y_true, y_prob, title_prefix, save_prefix, color="steelblue"):
    """
    Produce a side-by-side Lift Chart and Cumulative Gain Chart for one
    (model, split) combination. Saves PNG and returns decile_stats DataFrame.

    Parameters
    ----------
    y_true       : array-like of true binary labels
    y_prob       : array-like of predicted fraud probabilities
    title_prefix : string shown in both subplot titles
    save_prefix  : filename prefix under fig_dir  (e.g. '10_baseline_train')
    color        : bar / line colour
    """
    stats = compute_decile_stats(y_true, y_prob)
    auc   = roc_auc_score(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{title_prefix}  |  AUC = {auc:.4f}", fontsize=13, fontweight="bold")

    # — Lift Chart ——————————————————————————————————————————————
    bars = axes[0].bar(stats["decile"], stats["lift"],
                       color=color, edgecolor="white", alpha=0.85)
    axes[0].axhline(1.0, color="crimson", linestyle="--", linewidth=1.5,
                    label="Random baseline (lift = 1)")
    # annotate lift values on top of each bar
    for bar, lift_val in zip(bars, stats["lift"]):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.05,
                     f"{lift_val:.2f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_xlabel("Decile (1 = highest predicted risk)", fontsize=10)
    axes[0].set_ylabel("Lift", fontsize=10)
    axes[0].set_xticks(range(1, 11))
    axes[0].set_title(f"{title_prefix}\nLift Chart", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(bottom=0)

    # — Cumulative Gain Chart ———————————————————————————————————
    deciles     = [0] + stats["decile"].tolist()
    cum_gains   = [0] + stats["cum_gain_pct"].tolist()
    axes[1].plot(deciles, cum_gains, marker="o", color=color,
                 linewidth=2, markersize=5, label="Model")
    axes[1].plot([0, 10], [0, 100], "r--", linewidth=1.5, label="Random baseline")
    axes[1].fill_between(deciles, cum_gains,
                         np.interp(deciles, [0, 10], [0, 100]),
                         alpha=0.12, color=color, label="Lift area")
    # annotate cumulative % at each decile
    for d, g in zip(deciles[1:], cum_gains[1:]):
        axes[1].annotate(f"{g:.0f}%", (d, g),
                         textcoords="offset points", xytext=(4, -10),
                         fontsize=7, color=color)
    axes[1].set_xlabel("Decile (1 = highest predicted risk)", fontsize=10)
    axes[1].set_ylabel("Cumulative % of Fraud Captured", fontsize=10)
    axes[1].set_xticks(range(0, 11))
    axes[1].set_yticks(range(0, 110, 10))
    axes[1].set_title(f"{title_prefix}\nCumulative Gain Chart", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{fig_dir}{save_prefix}_lift_gain.png", bbox_inches="tight", dpi=150)
    plt.show()
    print(stats.to_string(index=False))
    return stats


# ── HELPER: overlay Gain curves for Train / Valid / Test on one chart ─────────
def plot_gain_overlay(splits_dict, title, save_prefix):
    """
    Draw all three cumulative gain curves on a single axes for easy comparison.

    Parameters
    ----------
    splits_dict : {"Train": (y_true, y_prob), "Valid": ..., "Test": ...}
    title       : chart title string
    save_prefix : filename prefix under fig_dir
    """
    COLORS = {"Train": "steelblue", "Valid": "darkorange", "Test": "seagreen"}

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 10], [0, 100], "k--", linewidth=1.2,
            label="Random baseline", zorder=1)

    for split_name, (y_true, y_prob) in splits_dict.items():
        stats = compute_decile_stats(y_true, y_prob)
        auc   = roc_auc_score(y_true, y_prob)
        deciles   = [0] + stats["decile"].tolist()
        cum_gains = [0] + stats["cum_gain_pct"].tolist()
        ax.plot(deciles, cum_gains, marker="o", markersize=4,
                color=COLORS[split_name], linewidth=2,
                label=f"{split_name}  (AUC={auc:.4f})", zorder=2)

    ax.set_xlabel("Decile (1 = highest predicted risk)", fontsize=11)
    ax.set_ylabel("Cumulative % of Fraud Captured", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(range(0, 11)); ax.set_yticks(range(0, 110, 10))
    ax.legend(fontsize=10); ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}{save_prefix}_gain_overlay.png", bbox_inches="tight", dpi=150)
    plt.show()


# ── HELPER: overlay Lift bars (grouped) for Train / Valid / Test ──────────────
def plot_lift_overlay(splits_dict, title, save_prefix):
    """
    Grouped bar chart: one cluster of 3 bars per decile (Train / Valid / Test).
    """
    COLORS = {"Train": "steelblue", "Valid": "darkorange", "Test": "seagreen"}
    n_splits = len(splits_dict)
    bar_width = 0.25
    deciles = np.arange(1, 11)

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (split_name, (y_true, y_prob)) in enumerate(splits_dict.items()):
        stats = compute_decile_stats(y_true, y_prob)
        offset = (i - n_splits / 2 + 0.5) * bar_width
        ax.bar(deciles + offset, stats["lift"],
               width=bar_width, color=COLORS[split_name],
               alpha=0.85, edgecolor="white", label=split_name)

    ax.axhline(1.0, color="crimson", linestyle="--", linewidth=1.5,
               label="Random baseline (lift = 1)")
    ax.set_xlabel("Decile (1 = highest predicted risk)", fontsize=11)
    ax.set_ylabel("Lift", fontsize=11)
    ax.set_xticks(deciles); ax.set_ylim(bottom=0)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}{save_prefix}_lift_overlay.png", bbox_inches="tight", dpi=150)
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE MODEL — individual Lift & Gain for each split
# ═══════════════════════════════════════════════════════════════════════════════
SPLIT_COLORS = {"Train": "steelblue", "Valid": "darkorange", "Test": "seagreen"}
SPLIT_FILE_IDS = {"Train": "10a", "Valid": "10b", "Test": "10c"}

baseline_splits = {
    "Train": (y_train, baseline_model.predict_proba(X_train)[:, 1]),
    "Valid": (y_valid, baseline_model.predict_proba(X_valid)[:, 1]),
    "Test":  (y_test,  baseline_model.predict_proba(X_test)[:, 1]),
}

print("=" * 65)
print("BASELINE MODEL — Lift & Gain Charts (Train / Valid / Test)")
print("=" * 65)

for split_name, (y_s, prob_s) in baseline_splits.items():
    plot_lift_gain(
        y_true       = y_s,
        y_prob       = prob_s,
        title_prefix = f"Baseline XGBoost — {split_name} Set",
        save_prefix  = f"{SPLIT_FILE_IDS[split_name]}_baseline_{split_name.lower()}",
        color        = SPLIT_COLORS[split_name],
    )

# Overlay: all three splits on one Gain chart and one grouped Lift chart
plot_gain_overlay(
    splits_dict = baseline_splits,
    title       = "Baseline XGBoost — Cumulative Gain Overlay (Train / Valid / Test)",
    save_prefix = "10d_baseline_all_splits",
)
plot_lift_overlay(
    splits_dict = baseline_splits,
    title       = "Baseline XGBoost — Lift Overlay (Train / Valid / Test)",
    save_prefix = "10e_baseline_lift_overlay",
)
```

---

## SECTION 9 — FEATURE CORRELATION REDUCTION (> 0.90 threshold)

```python
# Compute correlation on training set only
corr_matrix = X_train.corr().abs()

# Upper triangle to avoid duplicates
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Get feature importances from baseline model
feat_imp = pd.Series(baseline_model.feature_importances_, index=X_train.columns)

# For each correlated pair, drop the LESS important feature
cols_to_drop = set()
for col in upper.columns:
    correlated_features = upper.index[upper[col] > 0.90].tolist()
    for corr_feat in correlated_features:
        # Drop whichever has lower Gini importance
        if feat_imp.get(col, 0) < feat_imp.get(corr_feat, 0):
            cols_to_drop.add(col)
        else:
            cols_to_drop.add(corr_feat)

print(f"Features dropped due to correlation > 0.90: {len(cols_to_drop)}")
print(sorted(cols_to_drop))

X_train_uncorr = X_train.drop(columns=list(cols_to_drop))
X_valid_uncorr = X_valid.drop(columns=list(cols_to_drop))
X_test_uncorr  = X_test.drop(columns=list(cols_to_drop))
print(f"Features remaining after correlation filter: {X_train_uncorr.shape[1]}")
```

---

## SECTION 10 — GINI IMPORTANCE SELECTION (Cumulative ≤ 80%)

```python
# Re-train on uncorrelated features to get fresh importances
model_uncorr = xgb.XGBClassifier(
    n_estimators=200, scale_pos_weight=scale_pos_weight,
    eval_metric="auc", early_stopping_rounds=20,
    random_state=42, use_label_encoder=False
)
model_uncorr.fit(
    X_train_uncorr, y_train,
    eval_set=[(X_valid_uncorr, y_valid)],
    verbose=False
)

# Rank features by importance
importance_df = pd.DataFrame({
    "feature": X_train_uncorr.columns,
    "importance": model_uncorr.feature_importances_
}).sort_values("importance", ascending=False).reset_index(drop=True)

importance_df["cumulative_importance"] = importance_df["importance"].cumsum()
importance_df["cumulative_pct"] = (
    importance_df["cumulative_importance"] / importance_df["importance"].sum() * 100
)

# Select features with cumulative Gini importance up to 80%
selected_features = importance_df[importance_df["cumulative_pct"] <= 80]["feature"].tolist()
print(f"Features selected (cumulative Gini ≤ 80%): {len(selected_features)}")
print(selected_features)

# Plot
fig, ax1 = plt.subplots(figsize=(16, 6))
top_n = importance_df.head(50)
ax1.bar(range(len(top_n)), top_n["importance"], color="steelblue", alpha=0.7)
ax2 = ax1.twinx()
ax2.plot(range(len(top_n)), top_n["cumulative_pct"], color="red", marker="o", ms=3)
ax2.axhline(80, color="orange", linestyle="--", label="80% threshold")
ax1.set_xticks(range(len(top_n)))
ax1.set_xticklabels(top_n["feature"], rotation=90, fontsize=7)
ax1.set_ylabel("Gini Importance"); ax2.set_ylabel("Cumulative %")
ax1.set_title("Feature Importance with Cumulative Gini (Top 50)")
ax2.legend(); plt.tight_layout()
plt.savefig(f"{fig_dir}11_gini_cumulative.png", bbox_inches="tight"); plt.show()

# Subset datasets to selected features
X_train_sel = X_train_uncorr[selected_features]
X_valid_sel = X_valid_uncorr[selected_features]
X_test_sel  = X_test_uncorr[selected_features]
```

---

## SECTION 11 — OPTUNA HYPERPARAMETER TUNING

Tune XGBoost on the selected features. Objective: maximize validation AUC.

```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 1000),
        "max_depth":         trial.suggest_int("max_depth", 3, 10),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
        "gamma":             trial.suggest_float("gamma", 0, 5),
        "reg_alpha":         trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda":        trial.suggest_float("reg_lambda", 0, 5),
        "scale_pos_weight":  scale_pos_weight,
        "eval_metric":       "auc",
        "early_stopping_rounds": 50,
        "random_state":      42,
        "use_label_encoder": False,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train_sel, y_train,
        eval_set=[(X_valid_sel, y_valid)],
        verbose=False
    )
    prob = model.predict_proba(X_valid_sel)[:, 1]
    return roc_auc_score(y_valid, prob)

study = optuna.create_study(direction="maximize", study_name="xgb_fraud_tuning")
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"\nBest Validation AUC: {study.best_value:.4f}")
print(f"Best Params: {study.best_params}")

# Optuna optimization history plot
optuna.visualization.matplotlib.plot_optimization_history(study)
plt.savefig(f"{fig_dir}12_optuna_history.png", bbox_inches="tight"); plt.show()
```

---

## SECTION 12 — FINAL XGBOOST MODEL

Train the final model using the best Optuna parameters on selected features.

```python
best_params = study.best_params
best_params.update({
    "scale_pos_weight": scale_pos_weight,
    "eval_metric": "auc",
    "early_stopping_rounds": 50,
    "random_state": 42,
    "use_label_encoder": False
})

final_model = xgb.XGBClassifier(**best_params)
final_model.fit(
    X_train_sel, y_train,
    eval_set=[(X_train_sel, y_train), (X_valid_sel, y_valid)],
    verbose=50
)

# Evaluate on all splits
print("\n=== FINAL MODEL EVALUATION ===")
for split_name, X_s, y_s in [("TRAIN", X_train_sel, y_train),
                               ("VALID", X_valid_sel, y_valid),
                               ("TEST",  X_test_sel,  y_test)]:
    prob = final_model.predict_proba(X_s)[:, 1]
    auc  = roc_auc_score(y_s, prob)
    print(f"\n{split_name} AUC: {auc:.4f}")
    print(classification_report(y_s, final_model.predict(X_s)))
    print(confusion_matrix(y_s, final_model.predict(X_s)))
```

### 12.1 Final ROC Curve
```python
fig, ax = plt.subplots(figsize=(8, 6))
for split_name, X_s, y_s in [("Train", X_train_sel, y_train),
                               ("Valid", X_valid_sel, y_valid),
                               ("Test",  X_test_sel,  y_test)]:
    prob = final_model.predict_proba(X_s)[:, 1]
    RocCurveDisplay.from_predictions(y_s, prob, name=split_name, ax=ax)
ax.set_title("Final XGBoost (Optuna Tuned) — ROC Curve")
plt.savefig(f"{fig_dir}13_final_roc.png", bbox_inches="tight"); plt.show()
```

### 12.2 Final Model — Lift & Gain Charts (Train / Valid / Test)

```python
# ═══════════════════════════════════════════════════════════════════════════════
# FINAL (OPTUNA-TUNED) MODEL — individual Lift & Gain for each split
# ═══════════════════════════════════════════════════════════════════════════════
FINAL_SPLIT_FILE_IDS = {"Train": "14a", "Valid": "14b", "Test": "14c"}

final_splits = {
    "Train": (y_train, final_model.predict_proba(X_train_sel)[:, 1]),
    "Valid": (y_valid, final_model.predict_proba(X_valid_sel)[:, 1]),
    "Test":  (y_test,  final_model.predict_proba(X_test_sel)[:, 1]),
}

print("=" * 65)
print("FINAL TUNED MODEL — Lift & Gain Charts (Train / Valid / Test)")
print("=" * 65)

for split_name, (y_s, prob_s) in final_splits.items():
    plot_lift_gain(
        y_true       = y_s,
        y_prob       = prob_s,
        title_prefix = f"Final XGBoost (Optuna) — {split_name} Set",
        save_prefix  = f"{FINAL_SPLIT_FILE_IDS[split_name]}_final_{split_name.lower()}",
        color        = SPLIT_COLORS[split_name],
    )

# Overlay: all three splits on one Gain chart and one grouped Lift chart
plot_gain_overlay(
    splits_dict = final_splits,
    title       = "Final XGBoost (Optuna) — Cumulative Gain Overlay (Train / Valid / Test)",
    save_prefix = "14d_final_all_splits",
)
plot_lift_overlay(
    splits_dict = final_splits,
    title       = "Final XGBoost (Optuna) — Lift Overlay (Train / Valid / Test)",
    save_prefix = "14e_final_lift_overlay",
)
```

### 12.3 Side-by-Side Comparison: Baseline vs Final (Test Set only)

```python
# ── Head-to-head on the test set ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Baseline vs Final XGBoost — Test Set Comparison",
             fontsize=14, fontweight="bold")

base_stats  = compute_decile_stats(y_test, baseline_model.predict_proba(X_test)[:, 1])
final_stats = compute_decile_stats(y_test, final_model.predict_proba(X_test_sel)[:, 1])

# — Lift comparison ——————————————————————————————————————————
bar_w = 0.35
dec   = np.arange(1, 11)
axes[0].bar(dec - bar_w/2, base_stats["lift"],
            width=bar_w, color="steelblue", alpha=0.85, label="Baseline", edgecolor="white")
axes[0].bar(dec + bar_w/2, final_stats["lift"],
            width=bar_w, color="seagreen",  alpha=0.85, label="Final (Optuna)", edgecolor="white")
axes[0].axhline(1.0, color="crimson", linestyle="--", linewidth=1.5, label="Random baseline")
axes[0].set_xlabel("Decile (1 = highest predicted risk)", fontsize=10)
axes[0].set_ylabel("Lift", fontsize=10)
axes[0].set_title("Lift Chart — Baseline vs Final (Test)", fontsize=11)
axes[0].set_xticks(dec); axes[0].set_ylim(bottom=0)
axes[0].legend(fontsize=9)

# — Gain comparison ——————————————————————————————————————————
base_auc  = roc_auc_score(y_test, baseline_model.predict_proba(X_test)[:, 1])
final_auc = roc_auc_score(y_test, final_model.predict_proba(X_test_sel)[:, 1])

axes[1].plot([0] + base_stats["decile"].tolist(),
             [0] + base_stats["cum_gain_pct"].tolist(),
             marker="o", color="steelblue", linewidth=2, markersize=5,
             label=f"Baseline  (AUC={base_auc:.4f})")
axes[1].plot([0] + final_stats["decile"].tolist(),
             [0] + final_stats["cum_gain_pct"].tolist(),
             marker="s", color="seagreen", linewidth=2, markersize=5,
             label=f"Final (Optuna)  (AUC={final_auc:.4f})")
axes[1].plot([0, 10], [0, 100], "k--", linewidth=1.2, label="Random baseline")
axes[1].set_xlabel("Decile (1 = highest predicted risk)", fontsize=10)
axes[1].set_ylabel("Cumulative % of Fraud Captured", fontsize=10)
axes[1].set_title("Cumulative Gain — Baseline vs Final (Test)", fontsize=11)
axes[1].set_xticks(range(0, 11)); axes[1].set_yticks(range(0, 110, 10))
axes[1].legend(fontsize=9); axes[1].grid(axis="y", linestyle=":", alpha=0.5)

plt.tight_layout()
plt.savefig(f"{fig_dir}15_baseline_vs_final_test.png", bbox_inches="tight", dpi=150)
plt.show()

# ── Print decile tables side by side for inspection ──────────────────────────
print("\n=== BASELINE  — Test Set Decile Stats ===")
print(base_stats.to_string(index=False))
print("\n=== FINAL (OPTUNA) — Test Set Decile Stats ===")
print(final_stats.to_string(index=False))
```

---

## SECTION 13 — SUMMARY COMPARISON TABLE

```python
# Collect AUC for each model stage
results = []
for model_name, model, X_tr, X_va, X_te in [
    ("Baseline (all features)",   baseline_model, X_train,       X_valid,       X_test),
    ("After corr. reduction",     model_uncorr,   X_train_uncorr, X_valid_uncorr, X_test_uncorr),
    ("Final (Optuna + Gini 80%)", final_model,    X_train_sel,   X_valid_sel,   X_test_sel),
]:
    results.append({
        "Model": model_name,
        "# Features": X_tr.shape[1],
        "Train AUC": round(roc_auc_score(y_train, model.predict_proba(X_tr)[:, 1]), 4),
        "Valid AUC": round(roc_auc_score(y_valid, model.predict_proba(X_va)[:, 1]), 4),
        "Test AUC":  round(roc_auc_score(y_test,  model.predict_proba(X_te)[:, 1]), 4),
    })

summary = pd.DataFrame(results)
print("\n=== MODEL COMPARISON SUMMARY ===")
print(summary.to_string(index=False))
summary.to_csv("model_summary.csv", index=False)
```

---

## SECTION 14 — SAVE MODEL ARTIFACTS FOR SAGEMAKER DEPLOYMENT

```python
import joblib, tarfile

# Save the final model
joblib.dump(final_model, "xgb_fraud_final.joblib")

# Save the selected feature list
with open("selected_features.txt", "w") as f:
    f.write("\n".join(selected_features))

# Package as tar.gz for SageMaker
with tarfile.open("model.tar.gz", "w:gz") as tar:
    tar.add("xgb_fraud_final.joblib")

# Upload to S3 (SageMaker needs this)
import boto3
s3 = session.client("s3")
BUCKET = f"sagemaker-us-west-2-{session.client('sts').get_caller_identity()['Account']}"
S3_KEY = f"student-{STUDENT_NUM}/fraud-model/model.tar.gz"

s3.upload_file("model.tar.gz", BUCKET, S3_KEY)
print(f"✅ Model uploaded to s3://{BUCKET}/{S3_KEY}")
MODEL_S3_URI = f"s3://{BUCKET}/{S3_KEY}"
```

---

## NOTES

- Use `random_state=42` everywhere for reproducibility.
- All plots must have titles, axis labels, and legends. Save as PNG to `figures/`.
- `label_type_cd == 1` in `fad_transactions` is the **FRAUD** class → `is_fraud = 1`.
- `fraud_transactions.csv` uses a different ID format (`txn_030407`) than `fad_transactions.csv`
  (`txn_00049897`) — these are separate datasets; **do not join them** unless keys match.
- `ft_fraud_cases.csv` joins to `fad_transactions` on `transaction_id` and `account_num`.
- `macro_context.csv` joins on the year-month of the transaction timestamp.
- After all joins, assert `df.shape[0] == 50000` to catch accidental fan-out.
- Features derived from `ft_fraud_cases` (gross_fraud_amt, net_fraud_amt, chargeback fields)
  **must be dropped** before modeling — they are only populated for known fraud rows and
  would cause target leakage.
