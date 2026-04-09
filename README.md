# 🧬 Agentic Protein Structure Classifier — Gradio App

> **GPT-4o · LangGraph · Random Forest · SMOTE · Gradio**

An autonomous AI agent that loads a PDB-derived protein dataset, engineers features,
trains a Random Forest classifier, evaluates its performance, and produces a full
scientific report — all driven by GPT-4o function calling inside a LangGraph pipeline.
The trained model is saved to disk and served via a Gradio prediction UI.

---

## How It Works

This project has **two separate scripts** with distinct roles:

| Script | Role | When to run |
|--------|------|-------------|
| `agent_core.py` | Trains the model, saves artefacts to `model/` | **Once**, before using the app |
| `app.py` | Loads the saved model, serves a prediction UI | Every time you want to predict |

```
agent_core.py  →  model/rf_model.joblib
                  model/label_encoder.joblib
                  model/feature_columns.json
                  model/training_report.txt
                  model/feature_importance.csv
                  model/tool_call_log.csv
                       ↓
app.py         →  loads model/ → serves Gradio prediction UI
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  LangGraph StateGraph  (agent_core.py)                          │
│                                                                 │
│  load_data                                                      │
│      ↓                                                          │
│  filter_top_classes   (keep top-20 protein classes)             │
│      ↓                                                          │
│  feature_engineering  (5 physicochemical + 27 AA seq features)  │
│      ↓                                                          │
│  split_data           (80 / 10 / 10  stratified)                │
│      ↓                                                          │
│  apply_smote          (cap 6 000/class, k_neighbors=3)          │
│      ↓                                                          │
│  ┌──────────────────────────────────────────────────┐           │
│  │  AGENT NODE  (GPT-4o + tool loop)                │           │
│  │  ├─ train_random_forest()  → saves model/        │           │
│  │  ├─ evaluate_model(split)                        │           │
│  │  ├─ compute_feature_importance(top_n)            │           │
│  │  ├─ run_cross_validation(cv_folds)               │           │
│  │  ├─ inspect_class_distribution()                 │           │
│  │  └─ generate_final_report()  → saves reports     │           │
│  └──────────────────────────────────────────────────┘           │
│      ↓                                                          │
│  END                                                            │
└─────────────────────────────────────────────────────────────────┘
```

**Fixed Random Forest config:**
```
n_estimators=500  max_features="sqrt"  max_depth=None
min_samples_split=2  min_samples_leaf=1  bootstrap=True
class_weight="balanced_subsample"  random_state=50
```

---

## File Structure

```
protein_classifier_app/
│
├── agent_core.py       ← Training pipeline (run this first)
├── app.py              ← Prediction UI (run this after training)
├── requirements.txt    ← Python dependencies
├── .env.example        ← API key template (copy → .env)
├── README.md           ← This file
│
└── model/              ← Created automatically after training
    ├── rf_model.joblib
    ├── label_encoder.joblib
    ├── feature_columns.json
    ├── training_report.txt
    ├── feature_importance.csv
    └── tool_call_log.csv
```

---

## Setup

### 1 — Clone / copy the files

```bash
cd protein_classifier_app
```

### 2 — Create a Python virtual environment

```bash
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Set your OpenAI API key

**Option A — .env file (recommended):**
```bash
cp .env.example .env
# Edit .env and fill in your key:
#   OPENAI_API_KEY=sk-...
```

**Option B — environment variable:**
```bash
# macOS / Linux:
export OPENAI_API_KEY="sk-..."
# Windows CMD:
set OPENAI_API_KEY=sk-...
# Windows PowerShell:
$env:OPENAI_API_KEY="sk-..."
```

---

## Step 1 — Train the model

Run `agent_core.py` once with your training CSV. This uses GPT-4o to autonomously
train, evaluate, and report on the Random Forest — then saves all artefacts to `model/`.

```bash
python agent_core.py --csv path/to/data_ai.csv
```

You will see live output as the agent works through each step:
```
✓ Loaded 150,000 rows × 12 cols
✓ Filtered to top 20 classes → 120,000 rows
✓ Features → X(120000, 32)
✓ Train=(96000, 32) Val=(12000, 32) Test=(12000, 32)
✓ SMOTE → 110,000 training samples
🤖 Agent iteration 1/12…
🌲 Training Random Forest (500 trees)…
✅ Trained — OOB: 0.8712  (94.3s)
💾 Model artefacts saved → model/
📊 Evaluating on validation…
...
💾 All artefacts → model/
```

**Training only needs to be done once.** The saved `model/` folder can be reused
indefinitely — copy it to any machine running this app.

> ⚠️ Training requires a valid OpenAI API key with sufficient credits.
> Use `OPENAI_MODEL = "gpt-4o-mini"` in `agent_core.py` for lower cost.

---

## Step 2 — Launch the prediction app

Once `model/` exists, start the Gradio UI:

```bash
python app.py
```

Open your browser at: **http://localhost:7860**

The app loads the saved model at startup — no API key needed, no retraining.

---

## Reproducing on a New Machine

To run this on another machine without retraining:

```bash
# 1. Copy the project files AND the model/ folder to the new machine
scp -r protein_classifier_app/ user@newmachine:~/

# 2. On the new machine — install dependencies
cd protein_classifier_app
pip install -r requirements.txt

# 3. Launch the prediction app directly (no training needed)
python app.py
```

> The `model/` folder contains everything needed for prediction.
> `agent_core.py` and an OpenAI key are only required if you want to retrain.

---

## Required CSV Format (for training)

| Column | Type | Description |
|--------|------|-------------|
| `classification` | string | Target protein class label |
| `chain_sequences` | string (dict) | `{"A": "MKTAY...", "B": "ACDEF..."}` |
| `resolution` | float | X-ray resolution (Å) |
| `structure_molecular_weight` | float | Molecular weight (Da) |
| `density_matthews` | float | Matthews coefficient |
| `density_percent_sol` | float | Solvent content (%) |
| `ph_value` | float | Crystallisation pH |

The pipeline automatically:
- Filters to the **top 20** most frequent classes
- Extracts **27 amino acid sequence features** from `chain_sequences`
- Handles missing values (fills with 0)

---

## Prediction UI

Provide whatever protein data you have — **nothing is required**.
The more fields filled in, the more accurate the prediction.

| Input | Features unlocked |
|-------|-------------------|
| `chain_sequences` | 27 sequence features computed automatically |
| Physicochemical fields | Up to 5 additional numeric features |

**Output:**
- Top predicted protein class with probability
- Confidence label (High / Medium / Low)
- Data completeness indicator
- Top-5 class probability chart

---

## Configuration

Edit constants at the top of `agent_core.py`:

```python
TOP_N_CLASSES   = 20       # classes to include in training
RANDOM_STATE    = 50       # reproducibility seed
MAX_AGENT_ITERS = 12       # safety cap on GPT-4o tool loop
OPENAI_MODEL    = "gpt-4o" # swap to "gpt-4o-mini" for lower cost
```

> Increasing `TOP_N_CLASSES` allows rarer protein classes (e.g. OXYGEN TRANSPORT)
> to be included in training, at the cost of longer training time.

---

## Public Sharing

```python
# In app.py, change:
demo.launch(share=True)
```

Gradio generates a `https://....gradio.live` URL valid for 72 hours.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Model not loaded` error in app | Run `python agent_core.py --csv data.csv` first to generate `model/` |
| `AuthenticationError` | Check `OPENAI_API_KEY` is set and has credits |
| `RateLimitError: insufficient_quota` | Add billing credits at platform.openai.com or switch to `gpt-4o-mini` |
| `CSV missing columns` | Ensure CSV has `classification` and `chain_sequences` columns |
| `SMOTE failed` | Need at least 4 samples per class — use a larger dataset |
| Class not predicted | That class may not be in top-20 — increase `TOP_N_CLASSES` and retrain |
| `Port 7860 in use` | Change `server_port=7861` in `app.py` |
