"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   AGENTIC PROTEIN STRUCTURE CLASSIFIER  — Tool-Augmented LangGraph Agent   ║
║   Uses OpenAI GPT-4o as the reasoning backbone with function calling        ║
║   and a LangGraph StateGraph for pipeline orchestration.                   ║
║                                                                            ║
║   Random Forest config is FIXED to match the original pipeline:            ║
║     n_estimators=500, max_features="sqrt", max_depth=None,                 ║
║     min_samples_split=2, min_samples_leaf=1, bootstrap=True,               ║
║     class_weight="balanced_subsample", random_state=50                     ║
║   SMOTE cap: 6 000 per class, k_neighbors=3                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

Architecture overview
─────────────────────
  ┌─────────────────────────────────────────────────────────────┐
  │  LangGraph StateGraph                                        │
  │  load_data → filter → feature_eng → split → smote           │
  │      → [AGENT NODE: plan + tool loop]                        │
  │           ├─ tool: train_random_forest()      ← fixed params │
  │           ├─ tool: evaluate_model(split)                     │
  │           ├─ tool: compute_feature_importance(top_n)         │
  │           ├─ tool: run_cross_validation(cv_folds)            │
  │           ├─ tool: inspect_class_distribution()              │
  │           └─ tool: generate_final_report(...)                │
  │      → END                                                   │
  └─────────────────────────────────────────────────────────────┘

  OpenAI API differences vs Anthropic:
  ─────────────────────────────────────
  • Client  : openai.OpenAI()  (requires OPENAI_API_KEY env var)
  • Call    : client.chat.completions.create(model=..., messages=..., tools=...)
  • Tools   : list of {"type":"function","function":{"name":...,"description":...,"parameters":...}}
              (parameters uses JSON Schema, same as Anthropic's input_schema)
  • Response: response.choices[0].message
              .content        → reasoning text (str | None)
              .tool_calls     → list of ChatCompletionMessageToolCall | None
              .finish_reason  → "stop" | "tool_calls" | "length"
  • Tool result message:
              {"role":"tool", "tool_call_id": tc.id, "content": result_str}
  • System prompt goes as first message: {"role":"system","content":...}
    (OpenAI doesn't have a separate system= parameter in the same way —
     it lives inside the messages list)

  The agent node:
    1. Sends full messages list to GPT-4o
    2. GPT-4o replies with reasoning text and/or tool_calls
    3. Python executes each tool call
    4. Results appended as role="tool" messages → GPT-4o reasons again
    5. Loop until GPT-4o calls generate_final_report
"""

import ast
import json
import operator
import time
import warnings
from collections import Counter
from typing import Annotated, Any, Dict, List, Optional, TypedDict

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from langgraph.graph import END, StateGraph
from openai import OpenAI
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
#  LOAD ENVIRONMENT VARIABLES FROM .env
# ──────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()   # reads .env from the same directory as this script

# ──────────────────────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────────────────────FILE_PATH = r"G:\Visual WWW\Python\1000_python_workspace_new\tampa_bay_biotech_2026_04_01\csv\data_ai.csv"
TARGET_COL = "classification"
TOP_N_CLASSES = 20
RANDOM_STATE = 50
MAX_AGENT_ITERATIONS = 12          # safety cap on the tool loop
OPENAI_MODEL = "gpt-4o"            # swap to "gpt-4o-mini" for lower cost/latency

# ── Fixed Random Forest config (identical to original pipeline) ──────────────
RF_PARAMS = {
    "n_estimators":      500,
    "max_depth":         None,
    "min_samples_split": 2,
    "min_samples_leaf":  1,
    "max_features":      "sqrt",
    "bootstrap":         True,
    "oob_score":         True,
    "class_weight":      "balanced_subsample",
    "n_jobs":            -1,
    "random_state":      RANDOM_STATE,
}

# ──────────────────────────────────────────────────────────────
#  AMINO ACID HELPERS
# ──────────────────────────────────────────────────────────────
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AMINO_SET = set(AMINO_ACIDS)


def parse_chain_sequences(chain_str: str) -> Dict[str, str]:
    try:
        obj = ast.literal_eval(chain_str)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def clean_sequence(seq: str) -> str:
    return "".join(ch for ch in str(seq).upper() if ch in AMINO_SET)


def extract_sequence_features(chain_str: str) -> Dict[str, float]:
    chain_dict = parse_chain_sequences(chain_str)
    base = {
        "num_chains": 0, "total_seq_len": 0, "avg_chain_len": 0.0,
        "max_chain_len": 0.0, "min_chain_len": 0.0, "std_chain_len": 0.0,
        "unique_aa_count": 0,
        **{f"aa_freq_{aa}": 0.0 for aa in AMINO_ACIDS},
    }
    if not chain_dict:
        return base
    seqs = [clean_sequence(s) for s in chain_dict.values()]
    seqs = [s for s in seqs if s]
    if not seqs:
        return base
    lengths = [len(s) for s in seqs]
    full = "".join(seqs)
    total = len(full)
    counts = Counter(full)
    return {
        "num_chains": len(seqs),
        "total_seq_len": total,
        "avg_chain_len": float(np.mean(lengths)),
        "max_chain_len": float(np.max(lengths)),
        "min_chain_len": float(np.min(lengths)),
        "std_chain_len": float(np.std(lengths)),
        "unique_aa_count": len(set(full)),
        **{f"aa_freq_{aa}": counts.get(aa, 0) / total for aa in AMINO_ACIDS},
    }


# ---------------------------------------------------------------
#  STATE  -  plain dict so LangGraph merges all keys correctly
#  across every node.  TypedDict with total=False causes newer
#  versions of LangGraph to treat node returns as partial updates
#  and silently drop keys set by earlier nodes (e.g. y_train_smote
#  disappears before agent_node runs).  A plain dict avoids this.
# ---------------------------------------------------------------
ProteinState = dict


# ──────────────────────────────────────────────────────────────
#  TOOL DEFINITIONS  (OpenAI function-calling schema)
#
#  OpenAI format:
#    {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
#  "parameters" is a JSON Schema object identical to Anthropic's "input_schema".
# ──────────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "train_random_forest",
            "description": (
                "Train the Random Forest classifier on the SMOTE-resampled training data "
                "using the fixed production configuration: n_estimators=500, max_features='sqrt', "
                "max_depth=None, min_samples_split=2, min_samples_leaf=1, bootstrap=True, "
                "class_weight='balanced_subsample'. No hyperparameter inputs accepted — "
                "call this tool with an empty object {}. Returns OOB score and training time."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_model",
            "description": (
                "Evaluate the currently trained model on a specific split. "
                "Returns accuracy, macro-F1, and a full classification report."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "split": {
                        "type": "string",
                        "enum": ["validation", "test"],
                        "description": "Which split to evaluate on",
                    },
                },
                "required": ["split"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_feature_importance",
            "description": "Return the top-N most important features from the trained model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "top_n": {
                        "type": "integer",
                        "description": "How many top features to return (default 15)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_cross_validation",
            "description": (
                "Run k-fold cross-validation on the SMOTE training data using the same fixed "
                "RF configuration (500 trees, sqrt features, balanced_subsample). "
                "Only cv_folds is configurable. Returns mean accuracy, std, and per-fold scores."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cv_folds": {
                        "type": "integer",
                        "description": "Number of CV folds (default 5)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_class_distribution",
            "description": "Return class counts in train (post-SMOTE), validation, and test splits.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_final_report",
            "description": (
                "Compile the complete analysis report with all metrics, feature importance, "
                "reasoning chain, and agent decisions. This ENDS the agent loop."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "executive_summary": {
                        "type": "string",
                        "description": "Your 3-5 sentence scientific summary of findings",
                    },
                    "recommendations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of actionable recommendations",
                    },
                },
                "required": ["executive_summary", "recommendations"],
            },
        },
    },
]


# ──────────────────────────────────────────────────────────────
#  TOOL EXECUTOR
# ──────────────────────────────────────────────────────────────
class ToolExecutor:
    """Bridges GPT-4o function_call results to Python ML operations on the live state."""

    def __init__(self, state: ProteinState):
        self.state = state
        self.done = False           # set True when generate_final_report is called

    def execute(self, tool_name: str, tool_input: Dict) -> str:
        dispatch = {
            "train_random_forest":        self._train_rf,
            "evaluate_model":             self._evaluate,
            "compute_feature_importance": self._feature_importance,
            "run_cross_validation":       self._cross_validate,
            "inspect_class_distribution": self._class_distribution,
            "generate_final_report":      self._final_report,
        }
        fn = dispatch.get(tool_name)
        if fn is None:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        try:
            return fn(tool_input)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    # ── train ────────────────────────────────────────────────
    def _train_rf(self, inp: Dict) -> str:
        """Train with the fixed production RF config. inp is intentionally ignored."""
        t0 = time.time()
        model = RandomForestClassifier(**RF_PARAMS)
        model.fit(self.state["X_train_smote"], self.state["y_train_smote"])
        elapsed = round(time.time() - t0, 2)

        self.state["model"] = model
        fi = pd.DataFrame({
            "feature":    self.state["X"].columns,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        self.state["feature_importance"] = fi

        reported_params = {k: v for k, v in RF_PARAMS.items() if k != "random_state"}
        reported_params["random_state"] = RANDOM_STATE

        return json.dumps({
            "status":          "trained",
            "config":          "fixed — original pipeline configuration",
            "params":          reported_params,
            "oob_score":       round(model.oob_score_, 4),
            "training_time_s": elapsed,
            "n_features":      self.state["X"].shape[1],
            "train_samples":   self.state["X_train_smote"].shape[0],
        })

    # ── evaluate ─────────────────────────────────────────────
    def _evaluate(self, inp: Dict) -> str:
        split = inp["split"]
        model = self.state.get("model")
        if model is None:
            return json.dumps({"error": "No model trained yet. Call train_random_forest first."})

        X, y = (
            (self.state["X_val"], self.state["y_val"])
            if split == "validation"
            else (self.state["X_test"], self.state["y_test"])
        )

        y_pred = model.predict(X)
        acc    = round(accuracy_score(y, y_pred), 4)
        f1_mac = round(f1_score(y, y_pred, average="macro", zero_division=0), 4)
        cm     = confusion_matrix(y, y_pred)
        report = classification_report(
            y, y_pred,
            target_names=self.state["label_encoder"].classes_,
            zero_division=0,
        )

        if split == "validation":
            self.state["val_accuracy"] = acc
            self.state["val_f1"] = f1_mac
            self.state["val_classification_report"] = report
            self.state["val_confusion_matrix"] = cm
        else:
            self.state["test_accuracy"] = acc
            self.state["test_f1"] = f1_mac
            self.state["test_classification_report"] = report
            self.state["test_confusion_matrix"] = cm

        return json.dumps({
            "split":    split,
            "accuracy": acc,
            "macro_f1": f1_mac,
            "report":   report,
        })

    # ── feature importance ───────────────────────────────────
    def _feature_importance(self, inp: Dict) -> str:
        top_n = inp.get("top_n", 15)
        fi = self.state.get("feature_importance")
        if fi is None:
            return json.dumps({"error": "Train the model first."})
        return fi.head(top_n).to_json(orient="records")

    # ── cross-validation ─────────────────────────────────────
    def _cross_validate(self, inp: Dict) -> str:
        cv_folds = inp.get("cv_folds", 5)
        cv_params = {**RF_PARAMS, "n_estimators": 200, "oob_score": False}
        clf = RandomForestClassifier(**cv_params)
        scores = cross_val_score(
            clf, self.state["X_train_smote"], self.state["y_train_smote"],
            cv=cv_folds, scoring="accuracy", n_jobs=-1,
        )
        return json.dumps({
            "cv_folds":      cv_folds,
            "rf_config":     "fixed (200 trees for speed, same all other params)",
            "mean_accuracy": round(float(scores.mean()), 4),
            "std_accuracy":  round(float(scores.std()), 4),
            "fold_scores":   [round(s, 4) for s in scores.tolist()],
        })

    # ── class distribution ───────────────────────────────────
    def _class_distribution(self, _: Dict) -> str:
        le = self.state["label_encoder"]
        def counts(arr):
            vc = pd.Series(arr).value_counts().sort_index()
            return {le.classes_[k]: int(v) for k, v in vc.items()}
        return json.dumps({
            "train_smote": counts(self.state["y_train_smote"]),
            "validation":  counts(self.state["y_val"]),
            "test":        counts(self.state["y_test"]),
        })

    # ── final report ─────────────────────────────────────────
    def _final_report(self, inp: Dict) -> str:
        exec_summary    = inp.get("executive_summary", "")
        recommendations = inp.get("recommendations", [])

        lines = [
            "=" * 80,
            "AGENTIC PROTEIN CLASSIFICATION — FINAL REPORT",
            "=" * 80,
            "",
            "EXECUTIVE SUMMARY",
            "-" * 40,
            exec_summary,
            "",
            "PERFORMANCE METRICS",
            "-" * 40,
            f"  Validation Accuracy : {self.state.get('val_accuracy', 'N/A')}",
            f"  Validation Macro-F1 : {self.state.get('val_f1', 'N/A')}",
            f"  Test Accuracy       : {self.state.get('test_accuracy', 'N/A')}",
            f"  Test Macro-F1       : {self.state.get('test_f1', 'N/A')}",
            "",
            "TOP 15 FEATURES",
            "-" * 40,
        ]
        fi = self.state.get("feature_importance")
        if fi is not None:
            lines.append(fi.head(15).to_string(index=False))
        lines += [
            "",
            "RECOMMENDATIONS",
            "-" * 40,
            *[f"  • {r}" for r in recommendations],
            "",
            "AGENT REASONING CHAIN",
            "-" * 40,
            *self.state.get("agent_reasoning", []),
            "",
            "VALIDATION CLASSIFICATION REPORT",
            "-" * 40,
            self.state.get("val_classification_report", "N/A"),
            "",
            "TEST CLASSIFICATION REPORT",
            "-" * 40,
            self.state.get("test_classification_report", "N/A"),
        ]

        self.state["summary"] = "\n".join(lines)
        self.done = True
        return json.dumps({"status": "report_generated", "message": "Agent loop complete."})


# ──────────────────────────────────────────────────────────────
#  PIPELINE NODES  (pre-agent, identical to Anthropic version)
# ──────────────────────────────────────────────────────────────
def load_data_node(state: ProteinState) -> ProteinState:
    try:
        df = pd.read_csv(r"G:\Visual WWW\Python\1000_python_workspace_new\tampa_bay_biotech_2026_04_01\csv\data_ai.csv")
        state["df"] = df
        state["status"] = f"✓ Loaded {df.shape[0]:,} rows × {df.shape[1]} cols"
        print(state["status"])
    except Exception as e:
        state["error"] = f"Load failed: {e}"
    return state


def filter_top_classes_node(state: ProteinState) -> ProteinState:
    try:
        df = state["df"].copy()
        top = df[TARGET_COL].value_counts().head(TOP_N_CLASSES).index.tolist()
        filtered = df[df[TARGET_COL].isin(top)].copy()
        state["df_filtered"] = filtered
        state["top_classes"] = top
        state["status"] = f"✓ Filtered to top {TOP_N_CLASSES} classes → {filtered.shape[0]:,} rows"
        print(state["status"])
    except Exception as e:
        state["error"] = f"Filter failed: {e}"
    return state


def feature_engineering_node(state: ProteinState) -> ProteinState:
    try:
        df = state["df_filtered"].copy()
        numeric_cols = [
            "resolution", "structure_molecular_weight",
            "density_matthews", "density_percent_sol", "ph_value",
        ]
        seq_feats = pd.DataFrame(
            df["chain_sequences"].fillna("{}").apply(extract_sequence_features).tolist()
        )
        X = pd.concat([
            df[numeric_cols].reset_index(drop=True),
            seq_feats.reset_index(drop=True),
        ], axis=1).apply(pd.to_numeric, errors="coerce").fillna(0)

        le = LabelEncoder()
        y  = pd.Series(le.fit_transform(df[TARGET_COL].reset_index(drop=True)))

        state["X"] = X
        state["y"] = y
        state["label_encoder"] = le
        state["status"] = f"✓ Features built → X{X.shape}"
        print(state["status"])
    except Exception as e:
        state["error"] = f"Feature eng failed: {e}"
    return state


def split_data_node(state: ProteinState) -> ProteinState:
    try:
        X, y = state["X"], state["y"].values
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
        X_val, X_te, y_val, y_te = train_test_split(
            X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RANDOM_STATE)
        state.update(X_train=X_tr, X_val=X_val, X_test=X_te,
                     y_train=y_tr, y_val=y_val, y_test=y_te)
        state["status"] = f"✓ Train={X_tr.shape} | Val={X_val.shape} | Test={X_te.shape}"
        print(state["status"])
    except Exception as e:
        state["error"] = f"Split failed: {e}"
    return state


def smote_node(state: ProteinState) -> ProteinState:
    try:
        y_tr = state["y_train"]
        counts = pd.Series(y_tr).value_counts()
        cap = min(6000, counts.max())
        strat = {cls: cap for cls, cnt in counts.items() if cnt < cap}
        sm = SMOTE(sampling_strategy=strat, random_state=RANDOM_STATE, k_neighbors=3)
        Xr, yr = sm.fit_resample(state["X_train"], y_tr)
        state["X_train_smote"] = Xr
        state["y_train_smote"] = yr
        state["status"] = f"✓ SMOTE → {Xr.shape[0]:,} training samples"
        print(state["status"])
    except Exception as e:
        state["error"] = f"SMOTE failed: {e}"
    return state


# ──────────────────────────────────────────────────────────────
#  SYSTEM PROMPT  (injected as first message with role="system")
# ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert computational biologist and machine learning engineer.
Your task is to analyse and classify protein structures from the Protein Data Bank (PDB).

The dataset has already been loaded, filtered to the top 20 classification classes,
feature-engineered (5 numeric physicochemical features + 27 amino acid sequence features),
split 80/10/10 with stratification, and SMOTE-resampled on the training set (cap 6,000/class).

IMPORTANT — MODEL CONFIGURATION IS FIXED:
The Random Forest uses the exact production configuration from the original pipeline:
  n_estimators=500, max_features="sqrt", max_depth=None,
  min_samples_split=2, min_samples_leaf=1, bootstrap=True,
  class_weight="balanced_subsample", random_state=50
Do NOT attempt to change hyperparameters. The train_random_forest tool accepts no inputs.

YOUR WORKFLOW:
1. Train the model first (call train_random_forest with {}).
2. Evaluate on validation split (call evaluate_model with split="validation").
3. Evaluate on test split (call evaluate_model with split="test").
4. Inspect feature importance (call compute_feature_importance) — reason about which
   protein properties (sequence composition, physicochemical traits) drive classification.
5. Optionally run cross-validation to check stability across folds.
6. Optionally inspect class distribution to understand per-class performance.
7. Call generate_final_report with a scientific executive summary and actionable
   recommendations (e.g. additional features, biological interpretation, next steps).

REASONING GUIDELINES:
- Think step-by-step before each tool call. State what you expect and why.
- After each evaluation, interpret the metrics biologically — what does macro-F1 mean
  for a 20-class protein classification problem? Which subtypes are hardest to distinguish?
- Feature importance: connect ML findings back to structural biology (e.g. why does
  total_seq_len or aa_freq_C matter for certain protein families?).
- Be concise and scientific. Avoid padding. End each reasoning block with a clear
  statement of which tool you are calling next and why.
"""


# ──────────────────────────────────────────────────────────────
#  AGENT NODE  (GPT-4o + tool loop)
# ──────────────────────────────────────────────────────────────
def agent_node(state: ProteinState) -> ProteinState:
    """
    Core agentic loop — OpenAI edition:

      GPT-4o reasons → finish_reason="tool_calls" → Python executes each
      tool call → results appended as role="tool" messages → GPT-4o reasons again.
      Loop until GPT-4o calls generate_final_report (executor.done = True)
      or finish_reason="stop" (natural end without tool call).

    Key OpenAI-specific details:
      • System prompt is the FIRST message with role="system".
      • Assistant message must be appended exactly as returned (response.choices[0].message)
        converted to a plain dict so it serialises across iterations.
      • Tool results use role="tool" (not "user") with tool_call_id matching tc.id.
      • tool_input comes from tc.function.arguments as a JSON string → parse with json.loads.
    """
    client   = OpenAI()           # reads OPENAI_API_KEY from environment
    executor = ToolExecutor(state)

    messages: List[Dict]   = state.get("agent_messages", [])
    reasoning_log: List[str] = state.get("agent_reasoning", [])
    tool_log: List[Dict]   = state.get("tool_call_log", [])
    iterations             = state.get("iterations", 0)

    # ── Initialise conversation ───────────────────────────────
    if not messages:
        # System prompt goes as the first message (OpenAI convention)
        messages.append({"role": "system", "content": SYSTEM_PROMPT})

        # Dataset briefing as the first user turn
        class_counts = pd.Series(state["y_train_smote"]).value_counts()
        briefing = (
            f"Dataset briefing:\n"
            f"  • Target classes (top {TOP_N_CLASSES}): {', '.join(state['top_classes'])}\n"
            f"  • Feature dimensions: {state['X'].shape[1]} features\n"
            f"    (5 physicochemical: resolution, mol_weight, density_matthews,\n"
            f"     density_percent_sol, ph_value — plus 27 amino acid sequence features)\n"
            f"  • Training samples (post-SMOTE): {state['X_train_smote'].shape[0]:,}\n"
            f"  • Validation samples: {state['X_val'].shape[0]:,}\n"
            f"  • Test samples: {state['X_test'].shape[0]:,}\n"
            f"  • SMOTE class range in train: {class_counts.min()}–{class_counts.max()} per class\n\n"
            f"Model configuration is FIXED (original pipeline RF params). "
            f"Start by calling train_random_forest with an empty input {{}}."
        )
        messages.append({"role": "user", "content": briefing})

    # ── TOOL LOOP ─────────────────────────────────────────────
    while not executor.done and iterations < MAX_AGENT_ITERATIONS:
        iterations += 1
        print(f"\n{'─'*60}")
        print(f"  Agent iteration {iterations}/{MAX_AGENT_ITERATIONS}")
        print(f"{'─'*60}")

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",       # let GPT decide when to call tools
            max_tokens=4096,
        )

        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # ── Capture reasoning text ────────────────────────────
        if msg.content:
            step_label = f"[Iteration {iterations}] {msg.content}"
            reasoning_log.append(step_label)
            print(f"\n🧠 Agent reasoning:\n{msg.content}\n")

        # ── Append assistant turn to history ──────────────────
        # Convert to plain dict for JSON-serialisable state storage.
        # OpenAI's message object may contain tool_calls; we must preserve them.
        assistant_dict: Dict = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_dict["tool_calls"] = [
                {
                    "id":       tc.id,
                    "type":     "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments,   # raw JSON string
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_dict)

        # ── Natural stop (no tool calls) ──────────────────────
        if finish_reason == "stop" or not msg.tool_calls:
            print("ℹ GPT-4o reached stop without calling generate_final_report.")
            break

        # ── Execute tool calls ────────────────────────────────
        # GPT-4o can request multiple tool calls in a single turn.
        # Execute all of them and append each as a separate role="tool" message.
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            try:
                tool_input = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_input = {}

            print(f"🔧 Tool call: {tool_name}({json.dumps(tool_input, indent=2)})")

            result_str = executor.execute(tool_name, tool_input)
            try:
                result_pretty = json.dumps(json.loads(result_str), indent=2)
            except Exception:
                result_pretty = result_str

            print(f"📊 Result: {result_pretty[:600]}{'...' if len(result_pretty) > 600 else ''}")

            tool_log.append({
                "iteration": iterations,
                "tool":      tool_name,
                "input":     tool_input,
                "result":    result_str,
            })

            # OpenAI requires one role="tool" message per tool call, matched by tool_call_id
            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result_str,
            })

    state["agent_messages"]  = messages
    state["agent_reasoning"] = reasoning_log
    state["tool_call_log"]   = tool_log
    state["iterations"]      = iterations
    state["status"]          = f"Agent completed in {iterations} iterations"
    return state


# ──────────────────────────────────────────────────────────────
#  BUILD LANGGRAPH
# ──────────────────────────────────────────────────────────────
workflow = StateGraph(ProteinState)

for name, fn in [
    ("load_data",           load_data_node),
    ("filter_top_classes",  filter_top_classes_node),
    ("feature_engineering", feature_engineering_node),
    ("split_data",          split_data_node),
    ("apply_smote",         smote_node),
    ("agent",               agent_node),
]:
    workflow.add_node(name, fn)

workflow.set_entry_point("load_data")
workflow.add_edge("load_data",           "filter_top_classes")
workflow.add_edge("filter_top_classes",  "feature_engineering")
workflow.add_edge("feature_engineering", "split_data")
workflow.add_edge("split_data",          "apply_smote")
workflow.add_edge("apply_smote",         "agent")
workflow.add_edge("agent",               END)

app = workflow.compile()


# ──────────────────────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═" * 80)
    print("  AGENTIC PROTEIN STRUCTURE CLASSIFIER — OpenAI Edition")
    print(f"  Powered by {OPENAI_MODEL} + LangGraph + Random Forest")
    print("═" * 80 + "\n")

    initial: ProteinState = {
        "status":          "Starting",
        "error":           None,
        "agent_messages":  [],
        "agent_reasoning": [],
        "tool_call_log":   [],
        "iterations":      0,
    }

    final = app.invoke(initial)

    if final.get("error"):
        print(f"\n❌ WORKFLOW FAILED:\n{final['error']}")
    else:
        print("\n" + "═" * 80)
        print(final.get("summary", "No summary generated"))
        print("═" * 80)

        # Save artefacts
        if final.get("feature_importance") is not None:
            final["feature_importance"].to_csv("protein_feature_importance.csv", index=False)
            print("\n✓ Feature importance → protein_feature_importance.csv")

        tool_log_df = pd.DataFrame(final.get("tool_call_log", []))
        if not tool_log_df.empty:
            tool_log_df.to_csv("agent_tool_call_log.csv", index=False)
            print("✓ Tool call log       → agent_tool_call_log.csv")

        reasoning = final.get("agent_reasoning", [])
        if reasoning:
            with open("agent_reasoning_chain.txt", "w", encoding="utf-8") as f:
                f.write("\n\n".join(reasoning))
            print("✓ Reasoning chain     → agent_reasoning_chain.txt")