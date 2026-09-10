# AppleSupport AI — Customer Support Take-Home Project

An end-to-end AI customer-support agent built using historical Apple Support conversations from the **Customer Support on Twitter (TWCS)** dataset.

The system classifies incoming customer messages, retrieves historically similar Apple Support conversations as evidence, generates a grounded draft reply, and decides whether the case should be **AUTO-HANDLED** or **ESCALATED TO A HUMAN**.

---

## Problem

Customer-support teams receive large volumes of repetitive requests. A useful support agent should be able to:

1. Understand the customer's intent.
2. Find relevant historical support interactions.
3. Draft a response grounded in previous support behavior.
4. Avoid confidently handling risky or ambiguous cases.
5. Escalate cases that require human attention.
6. Provide an auditable record of the decision and supporting evidence.

This project implements that workflow for **AppleSupport**.

---

## Dataset

The project uses the Kaggle **Customer Support on Twitter (TWCS)** dataset.

The dataset contains customer and brand-side Twitter conversations with fields including:

* `tweet_id`
* `author_id`
* `inbound`
* `created_at`
* `text`
* `response_tweet_id`
* `in_response_to_tweet_id`

The raw dataset is approximately **516 MB** and is intentionally not committed to this repository.

Download:

[Kaggle — Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)

Place the downloaded dataset at:

```text
dataset/twcs/twcs.csv
```

---

# Setup

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Ollama

Ollama is used for:

* generating grounded draft replies
* optional LLM-based evaluation
* assisting golden-set annotation

The intent classifier itself does **not** depend on Ollama.

Install Ollama from:

https://ollama.com

Then:

```bash
ollama pull qwen2.5:7b
ollama serve
```

---

# Run Order

The data pipeline can be reproduced using the following steps.

## Step 1 — Audit the raw dataset

```bash
python data_analysis.py
```

Outputs:

```text
reports/twcs/
```

This performs basic dataset-quality checks and conversation reconstruction analysis.

---

## Step 2 — Select Target Brand

```bash
python brand_selection.py
```

Outputs:

```text
reports/brand_selection/
```

Selected brand:

**AppleSupport**

The selection is documented in:

```text
reports/brand_selection/brand_selection.md
```

---

## Step 3 — Extract AppleSupport Conversations

```bash
python extract_apple_support.py
```

Outputs:

```text
data/processed/apple_support_train.csv
data/processed/apple_support_golden.csv
```

The extracted pools contain:

* Training pool: **12,011 examples**
* Golden pool: **1,353 examples**

The two pools are **conversation-disjoint and tweet-disjoint**.

No tweet ID or conversation ID appears in both pools.

The golden pool is never used to create training labels.

---

## Step 4 — Assign Weak Intent Labels

```bash
python label_data.py
```

Outputs:

```text
data/labeled_training.csv
data/labeled_high_confidence.csv
data/label_review.csv
data/taxonomy.json
reports/intent_discovery.json
```

The high-confidence subset can be generated using:

```bash
python label_data.py --high-confidence-threshold 0.90
```

### Important

`rule_confidence` is a **keyword rule-score ratio**, not a calibrated probability.

The labels generated at this stage are **weak labels** and should not be treated as human ground truth.

---

# Intent Taxonomy

The system uses 11 intents:

| Intent                     | Definition                                                               |
| -------------------------- | ------------------------------------------------------------------------ |
| `account_access`           | Apple ID, iCloud, password, and login issues                             |
| `billing_payment`          | Charges, invoices, payment errors                                        |
| `app_software_issue`       | Software/app failures, updates, crashes, freezes, and performance issues |
| `device_issue`             | Physical hardware, battery, screen damage, and device failures           |
| `subscription`             | Subscription access, renewal, and cancellation                           |
| `refund_return`            | Refunds, returns, and money-back requests                                |
| `order_purchase`           | Orders, purchases, shipping, and delivery                                |
| `connectivity_issue`       | Wi-Fi, Bluetooth, cellular, and network issues                           |
| `general_information`      | How-to questions and general product information                         |
| `complaint`                | Explicit complaints and strongly negative feedback                       |
| `insufficient_information` | Bare mentions, URL-only messages, and very short/underspecified requests |

A device name alone does not determine `app_software_issue`.

When both device and software signals are present, `app_software_issue` takes priority over `device_issue`.

---

# Step 5 — Golden Evaluation Set

Generate the golden-set template:

```bash
python generate_golden.py
```

This produces:

```text
data/golden_set.csv
```

The finalized golden set contains **208 examples** and is stratified to include difficult and less frequent intents.

The following fields require human review:

* `intent`
* `should_escalate`
* `expected_resolution`
* `difficulty`
* `evidence_conversation_id`

`annotate_golden.py` can use Ollama to provide annotation suggestions, but these suggestions are **not ground truth**. Final evaluation labels should be human-reviewed.

**Golden-set examples must never be used for model training.**

---

# Step 6 — Train Models

Train using all weak labels:

```bash
python model.py --labels data/labeled_training.csv
```

Alternatively, train using only high-confidence weak labels:

```bash
python model.py --labels data/labeled_high_confidence.csv --high-confidence-only
```

Generated artifacts:

```text
models/majority.joblib
models/tfidf_logistic.joblib
models/embedding_logistic.joblib
```

The primary classifier is:

**TF-IDF + Logistic Regression**

The majority classifier provides a trivial baseline, while the embedding-based model provides an additional comparison.

---

# Step 7 — Evaluation

After finalizing the golden-set annotations:

```bash
python evaluation.py \
  --labels data/labeled_training.csv \
  --golden data/golden_set.csv
```

## LLM Judge

With Ollama running:

```bash
python evaluation.py \
  --labels data/labeled_training.csv \
  --golden data/golden_set.csv \
  --run-llm-judge
```

The LLM judge evaluates generated responses on:

* Groundedness
* Helpfulness
* Safety
* Overall quality

Each dimension is scored from **1–5**.

Per-response scores are written to:

```text
reports/llm_judge_scores.csv
```

## Human–LLM Agreement

A fixed subset of approximately 50 responses can also be scored manually.

Human scores are stored in:

```text
reports/llm_judge_human_scores.csv
```

Agreement can then be calculated using:

```bash
python evaluation.py \
  --labels data/labeled_training.csv \
  --golden data/golden_set.csv \
  --judge-scores reports/llm_judge_scores.csv \
  --human-scores reports/llm_judge_human_scores.csv
```

Final evaluation output:

```text
reports/evaluation.json
```

---

# Evaluation Methodology

| Benchmark                           | What it measures                                 | Interpretation                  |
| ----------------------------------- | ------------------------------------------------ | ------------------------------- |
| Weak-label test accuracy/F1         | Reproduction of keyword-generated labels         | Secondary                       |
| Golden-set Macro-F1                 | Intent classification on reviewed examples       | Primary                         |
| Retrieval Recall@K                  | Ability to retrieve relevant historical evidence | Requires evidence annotation    |
| Escalation recall / false-auto rate | Safety of escalation policy                      | Requires escalation annotations |
| LLM judge scores                    | Reply quality, grounding, helpfulness, safety    | Requires Ollama                 |
| Human–LLM agreement                 | Reliability of automated judging                 | Requires human scoring          |

## Baseline Results

On a conversation-grouped weak-label holdout:

**TF-IDF + Logistic Regression**

* Accuracy: **84.18%**
* Macro-F1: **75.38%**

On a stricter temporal 80/20 split:

* Accuracy: **81.89%**
* Macro-F1: **72.41%**

These are **secondary benchmarks** because their evaluation labels are generated by the same weak-label rules used to construct the training labels.

They therefore measure **rule reproduction**, not true real-world intent accuracy.

---

# What is misleading about my headline number?

The golden-set score is **not a production success rate**.

The golden set contains only 208 examples and was intentionally stratified toward difficult and less frequent cases. Its class distribution therefore does not represent live customer traffic.

The bulk training labels are also generated through weak supervision rather than human annotation.

Therefore:

* Weak-label performance mainly measures how well the model reproduces the labeling rules.
* Golden-set performance is more meaningful because it uses reviewed examples.
* Neither number should be interpreted as an estimate of real-world customer resolution rate.

Accuracy is also potentially misleading because the intent distribution is highly imbalanced, with `app_software_issue` representing the majority class.

For this reason, **Macro-F1 is treated as the primary classification metric** because it gives equal weight to each intent.

---

# Top 5 Failure Modes

## 1. `app_software_issue` vs `device_issue` bleed

Messages mentioning a device alongside a software term can be ambiguous.

For example:

```text
My iPhone won't update.
```

is software-related, while:

```text
My iPhone screen is physically broken.
```

is a device issue.

The classifier can struggle when the message simply says:

```text
My iPhone is broken.
```

**Hypothesis:** TF-IDF lacks sufficient semantic context to reliably distinguish hardware from software problems.

---

## 2. `complaint` vs issue-type classification

Strongly negative messages can contain enough technical vocabulary to be classified as an issue rather than a complaint.

For example:

```text
My AirPods are absolute garbage and keep disconnecting.
```

**Hypothesis:** complaint is partly defined by tone and dissatisfaction, which bag-of-words features do not capture reliably.

---

## 3. `general_information` vs `app_software_issue`

How-to questions can be lexically similar to software issue reports.

For example:

```text
How do I turn on dark mode?
```

and:

```text
Dark mode stopped working.
```

are different intents but share similar vocabulary.

Weak-label boundaries may also introduce noise between these classes.

---

## 4. Insufficient-information boundary

Very short or underspecified messages can be difficult to classify.

Examples:

```text
Help
```

```text
iPhone
```

```text
It's not working.
```

These should generally trigger clarification rather than a highly specific troubleshooting response.

**Hypothesis:** the classifier relies heavily on lexical evidence and has limited conversational context.

---

## 5. Low-frequency intent collapse

Some intents have very few golden examples, particularly:

* `order_purchase`
* `subscription`
* `refund_return`

A single error can therefore cause a large change in per-class F1.

Low-frequency intents can also have higher operational risk when they involve billing, refunds, or escalation.

---

# Project Decision Log

1. Selected AppleSupport because it provides high conversation volume, diversity, and actionable historical responses.
2. Reconstructed conversations before creating examples to avoid row-level leakage.
3. Split training and golden data by conversation and tweet IDs rather than randomly by row.
4. Kept weak labels separate from human-finalized golden labels.
5. Chose TF-IDF + Logistic Regression as the primary baseline before adding model complexity.
6. Treated `insufficient_information` as a real evaluation class rather than automatically discarding it.
7. Treated historical responses as evidence rather than proof that an issue was resolved.
8. Used escalation as a safety decision, with false-auto rate as an important metric.
9. Avoided claiming retrieval quality until golden evidence IDs are annotated.
10. Avoided claiming LLM-judge reliability until human agreement is measured.
11. Named `rule_confidence` explicitly as a heuristic score rather than a calibrated probability.
12. Used Macro-F1 rather than accuracy as the primary classification metric because of class imbalance.
13. Made `app_software_issue` take priority over `device_issue` when both software and device signals are present.
14. Used Ollama to accelerate annotation and generate responses while keeping final evaluation labels human-reviewed.
15. Used temporal evaluation to test robustness against changes in language and conversation patterns over time.

---

# Current Limitations and Next Week

The current prototype has several limitations:

* Bulk training labels are weakly supervised and may contain labeling noise.
* The golden set is relatively small.
* Low-frequency intents have limited evaluation coverage.
* Retrieval evaluation depends on correctly annotated historical evidence.
* The classifier scores are not calibrated probabilities.
* The reply generator depends on a local Ollama model.
* Escalation thresholds are heuristic and should be tuned using a finalized golden set.
* The historical dataset represents Twitter support interactions and may not perfectly represent modern support traffic.

### Next week

1. Complete human review of the golden set.
2. Populate `evidence_conversation_id` for golden examples.
3. Run and report Retrieval Recall@1/3/5.
4. Human-score a fixed 50-response subset.
5. Calculate human–LLM judge agreement.
6. Tune escalation thresholds against finalized annotations.
7. Review the largest confusion pairs.
8. Improve handling of ambiguous and insufficient-information messages.
9. Add richer conversation-history features to classification and response generation.
10. Evaluate additional lightweight classifiers before considering larger models.

---

# Model Artifacts

Generated model artifacts are stored in:

```text
models/
```

Current artifacts:

```text
majority.joblib
tfidf_logistic.joblib
embedding_logistic.joblib
```

The primary production/demo classifier is:

```text
TF-IDF + Logistic Regression
```

To regenerate the artifacts from scratch, delete the existing `.joblib` files and rerun Step 6.

---

# Web Application

The project includes a full-stack web interface for interacting with the support agent.

## Architecture

```text
React + Vite
     │
     ▼
Flask REST API
     │
     ▼
Support Agent
 ┌───┼───────────────┐
 ▼   ▼               ▼
ML Model  Historical Retrieval  Ollama
 │          │                    │
 └──────────┴────────────────────┘
              │
              ▼
           SQLite
```

---

# End-to-End Flow

1. Customer sends a message through the React interface.
2. Flask receives and persists the message.
3. TF-IDF + Logistic Regression predicts the intent.
4. Historical AppleSupport conversations are retrieved as evidence.
5. The escalation policy decides **AUTO-HANDLE** or **ESCALATE TO HUMAN**.
6. Ollama generates a draft response using the available evidence.
7. The UI displays the intent, model score, evidence, generated response, and escalation decision.
8. Conversation and decision information are persisted in SQLite for auditability.

---

# Quick Start — Web Application

## Terminal 1 — Ollama

```bash
ollama serve
```

If the model has not already been downloaded:

```bash
ollama pull qwen2.5:7b
```

---

## Terminal 2 — Backend

From the project root:

```bash
pip install -r requirements.txt
cd backend
python app.py
```

The Flask API starts at:

```text
http://localhost:5000
```

---

## Terminal 3 — Frontend

From the project root:

```bash
cd frontend
npm install
npm run dev
```

Vite normally starts the development server at:

```text
http://localhost:5173
```

If the configured Vite port differs, use the URL printed by `npm run dev`.

---

# API Endpoints

| Method | Endpoint                       | Description                                        |
| ------ | ------------------------------ | -------------------------------------------------- |
| POST   | `/api/conversations`           | Start a new conversation                           |
| GET    | `/api/conversations`           | List conversations                                 |
| GET    | `/api/conversations/<id>`      | Get a conversation and its messages                |
| POST   | `/api/conversations/<id>/chat` | Send a customer message and receive an AI response |
| PATCH  | `/api/conversations/<id>`      | Update conversation status                         |
| GET    | `/api/stats`                   | Dashboard statistics                               |
| GET    | `/api/taxonomy`                | Intent taxonomy                                    |
| GET    | `/api/health`                  | Health check                                       |

---

# What the UI Shows

* **Chat interface** — send customer messages and receive AI-generated responses.
* **Intent classification** — detected intent and model score.
* **Escalation decision** — AUTO-HANDLE vs ESCALATE TO HUMAN.
* **Escalation reasoning** — why the policy made the decision.
* **Historical evidence** — matching historical Apple Support conversations.
* **Conversation history** — persistent conversation sidebar.
* **Dashboard statistics** — conversation and escalation statistics.

---

# Manual End-to-End Testing

The full-stack application was manually tested with **25 customer messages** covering:

* connectivity issues
* device problems
* software/update issues
* account access
* billing
* subscriptions
* refunds
* orders
* general information
* complaints
* ambiguous requests
* insufficient-information messages
* security-related cases
* escalation scenarios

The tests were used to verify:

* intent classification
* response generation
* historical evidence retrieval/display
* escalation decisions
* conversation persistence
* frontend/backend integration

The purpose of these tests was functional validation rather than claiming that every generated response was correct.

---

# Database

For this prototype, SQLite provides lightweight persistent storage.

The application stores:

* conversations
* customer messages
* timestamps
* predicted intent
* model score
* escalation decision
* escalation reason
* historical evidence match
* evidence similarity
* generated reply

This provides an audit trail for each analyzed customer interaction.

For production deployment, this layer could be replaced by PostgreSQL, with Redis used for caching and high-throughput ephemeral state.

---

# Reproducing the Submitted Demo

The repository includes the trained model artifacts and processed data required to run the web application.

For the fastest demo path:

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Ollama

```bash
ollama serve
```

### 3. Start the backend

```bash
cd backend
python app.py
```

### 4. Start the frontend in another terminal

```bash
cd frontend
npm install
npm run dev
```

Then open the URL printed by Vite, normally:

```text
http://localhost:5173
```

The complete data-generation and evaluation pipeline is documented above if a full reproduction from the raw TWCS dataset is required.

---

# Repository Structure

```text
.
├── backend/
│   ├── app.py
│   ├── database.py
│   └── ...
│
├── frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.js
│
├── data/
│   ├── processed/
│   ├── golden_set.csv
│   ├── labeled_training.csv
│   └── taxonomy.json
│
├── dataset/
│   └── twcs/
│       └── twcs.csv          # not committed
│
├── models/
│   ├── majority.joblib
│   ├── tfidf_logistic.joblib
│   └── embedding_logistic.joblib
│
├── reports/
│   ├── evaluation.json
│   ├── intent_discovery.json
│   └── ...
│
├── agent.py
├── data_analysis.py
├── brand_selection.py
├── extract_apple_support.py
├── label_data.py
├── generate_golden.py
├── annotate_golden.py
├── model.py
├── evaluation.py
├── requirements.txt
└── README.md
```

---

# Key Takeaway

The project intentionally separates:

```text
Weak supervision
      ↓
Model training
      ↓
Historical evidence retrieval
      ↓
Escalation policy
      ↓
Grounded response generation
      ↓
Human-reviewed evaluation
```

The goal is not simply to maximize a classifier accuracy number, but to demonstrate an **auditable customer-support decision system** that can identify intent, use historical evidence, generate useful responses, and avoid automatically handling risky or insufficiently specified cases.
