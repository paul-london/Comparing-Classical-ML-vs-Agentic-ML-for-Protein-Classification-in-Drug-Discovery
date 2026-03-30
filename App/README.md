# 🧬 Agentic Protein Structure Classifier — Gradio App

> **GPT-4o · LangGraph · Random Forest · SMOTE · Gradio**

An autonomous AI agent that loads a PDB-derived protein dataset, engineers features,
trains a Random Forest classifier, evaluates its performance, and produces a full
scientific report — all driven by GPT-4o function calling inside a LangGraph pipeline,
with a polished Gradio web UI for interactive use.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  LangGraph StateGraph                                            │
│                                                                  │
│  load_data                                                       │
│      ↓                                                           │
│  filter_top_classes   (keep top-20 protein classes)             │
│      ↓                                                           │
│  feature_engineering  (5 physicochemical + 27 AA seq features)  │
│      ↓                                                           │
│  split_data           (80 / 10 / 10  stratified)                │
│      ↓                                                           │
│  apply_smote          (cap 6 000/class, k_neighbors=3)          │
│      ↓                                                           │
│  ┌──────────────────────────────────────────────────┐           │
│  │  AGENT NODE  (GPT-4o + tool loop)                │           │
│  │                                                  │           │
│  │  ├─ train_random_forest()   ← fixed RF params    │           │
│  │  ├─ evaluate_model(split)                        │           │
│  │  ├─ compute_feature_importance(top_n)            │           │
│  │  ├─ run_cross_validation(cv_folds)               │           │
│  │  ├─ inspect_class_distribution()                 │           │
│  │  └─ generate_final_report(...)  ← ends loop      │           │
│  └──────────────────────────────────────────────────┘           │
│      ↓                                                           │
│  END                                                             │
└─────────────────────────────────────────────────────────────────┘
```

**Fixed Random Forest config** (matches original pipeline exactly):
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
├── app.py              ← Gradio UI (entry point)
├── agent_core.py       ← LangGraph pipeline + GPT-4o agent + tools
├── requirements.txt    ← Python dependencies
├── .env.example        ← API key template (copy → .env)
└── README.md           ← This file
```

---

## Setup

### 1 — Clone / copy the files

```bash
# All files should be in the same directory
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
# Edit .env and add your key:
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

**Option C — Gradio UI:**
Enter your key directly in the API Key field in the browser.

### 5 — Launch the app

```bash
python app.py
```

Open your browser at: **http://localhost:7860**

---

## Required CSV Format

Your dataset must contain these columns:

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

## UI Walkthrough

### Input Panel (left sidebar)
| Control | Purpose |
|---------|---------|
| **Upload CSV** | Drag-and-drop your CSV file |
| **File path** | Alternative: type an absolute path |
| **API Key** | Optional override (blanks use `.env`) |
| **▶ Run** | Starts the full pipeline |

### Live Agent Log
Streams every pipeline step in real time:
- `✓` nodes — data loading, filtering, feature engineering, SMOTE
- `🤖` — agent iterations
- `🧠` — GPT-4o reasoning snippets
- `🌲 / 📊 / 📌 / 🔄` — tool executions

### Results Tabs

| Tab | Contents |
|-----|---------|
| **⚡ Metrics** | Gauge charts + numeric values for Val/Test Accuracy and Macro-F1 |
| **📌 Feature Importance** | Horizontal bar chart of top-15 RF feature importances |
| **📋 Class Distribution** | Grouped bar chart of class counts across all three splits |
| **📄 Classification Reports** | Full sklearn classification reports for Val and Test |
| **⬇ Downloads** | Full report (.txt), feature importance (.csv), tool call log (.csv) |

---

## Programmatic Usage

You can also call the pipeline directly from Python:

```python
from agent_core import run_pipeline

def my_logger(msg):
    print(msg)

final_state = run_pipeline(
    csv_path="path/to/data_ai.csv",
    log_callback=my_logger,
)

print(final_state["summary"])
print(f"Test accuracy: {final_state['test_accuracy']}")
final_state["feature_importance"].to_csv("fi.csv")
```

---

## Configuration

Edit the constants at the top of `agent_core.py`:

```python
TOP_N_CLASSES   = 20      # how many protein classes to keep
RANDOM_STATE    = 50      # reproducibility seed
MAX_AGENT_ITERS = 12      # safety cap on the GPT-4o tool loop
OPENAI_MODEL    = "gpt-4o"  # swap to "gpt-4o-mini" for lower cost
```

---

## Public Sharing (Gradio tunnel)

To get a temporary public URL (useful for demos):

```python
# In app.py, change:
demo.launch(share=True)
```

Gradio will print a `https://....gradio.live` URL valid for 72 hours.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `AuthenticationError` | Check `OPENAI_API_KEY` is set correctly |
| `Load failed: ...` | Verify the CSV path and column names |
| `SMOTE failed` | Ensure at least 4 samples per class in training split |
| `Port 7860 in use` | Change `server_port=7861` in `app.py` |
| Slow training | `n_jobs=-1` uses all CPU cores; close other heavy processes |
