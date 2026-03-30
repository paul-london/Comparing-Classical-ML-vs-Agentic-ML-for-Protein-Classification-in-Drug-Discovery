"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  agent_core.py  —  Agentic Protein Structure Classifier                     ║
║  LangGraph StateGraph  +  OpenAI GPT-4o function calling                   ║
║  Random Forest (fixed production config)  +  SMOTE resampling              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Architecture
────────────
  load_data → filter_top_classes → feature_engineering → split_data → smote
      → [AGENT NODE: GPT-4o + tool loop]
           ├─ train_random_forest()
           ├─ evaluate_model(split)
           ├─ compute_feature_importance(top_n)
           ├─ run_cross_validation(cv_folds)
           ├─ inspect_class_distribution()
           └─ generate_final_report(...)
      → END
"""

import ast
import json
import operator
import time
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from langgraph.graph import END, StateGraph
from openai import OpenAI
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
#  GLOBAL CONFIG
# ──────────────────────────────────────────────────────────────
TARGET_COL         = "classification"
TOP_N_CLASSES      = 20
RANDOM_STATE       = 50
MAX_AGENT_ITERS    = 12
OPENAI_MODEL       = "gpt-4o"

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
AMINO_SET   = set(AMINO_ACIDS)


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
        "num_chains":       0,
        "total_seq_len":    0,
        "avg_chain_len":    0.0,
        "max_chain_len":    0.0,
        "min_chain_len":    0.0,
        "std_chain_len":    0.0,
        "unique_aa_count":  0,
        **{f"aa_freq_{aa}": 0.0 for aa in AMINO_ACIDS},
    }
    if not chain_dict:
        return base
    seqs = [clean_sequence(s) for s in chain_dict.values()]
    seqs = [s for s in seqs if s]
    if not seqs:
        return base
    lengths = [len(s) for s in seqs]
    full    = "".join(seqs)
    total   = len(full)
    counts  = Counter(full)
    return {
        "num_chains":      len(seqs),
        "total_seq_len":   total,
        "avg_chain_len":   float(np.mean(lengths)),
        "max_chain_len":   float(np.max(lengths)),
        "min_chain_len":   float(np.min(lengths)),
        "std_chain_len":   float(np.std(lengths)),
        "unique_aa_count": len(set(full)),
        **{f"aa_freq_{aa}": counts.get(aa, 0) / total for aa in AMINO_ACIDS},
    }


# ──────────────────────────────────────────────────────────────
#  STATE  —  plain dict (LangGraph merges all keys)
# ──────────────────────────────────────────────────────────────
ProteinState = dict


# ──────────────────────────────────────────────────────────────
#  TOOL DEFINITIONS  (OpenAI function-calling schema)
# ──────────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "train_random_forest",
            "description": (
                "Train the Random Forest classifier on the SMOTE-resampled training data "
                "using the fixed production config: n_estimators=500, max_features='sqrt', "
                "max_depth=None, min_samples_split=2, min_samples_leaf=1, bootstrap=True, "
                "class_weight='balanced_subsample'. Call with empty object {}."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_model",
            "description": "Evaluate the trained model on a split. Returns accuracy, macro-F1, and classification report.",
            "parameters": {
                "type": "object",
                "properties": {
                    "split": {
                        "type": "string",
                        "enum": ["validation", "test"],
                        "description": "Which split to evaluate on",
                    }
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
                    "top_n": {"type": "integer", "description": "How many top features (default 15)"}
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
                "Run k-fold cross-validation on SMOTE training data with the fixed RF config "
                "(200 trees for speed). Returns mean accuracy, std, and per-fold scores."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cv_folds": {"type": "integer", "description": "Number of CV folds (default 5)"}
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
            "parameters": {"type": "object", "properties": {}, "required": []},
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
                        "description": "3-5 sentence scientific summary of findings",
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
    """Bridges GPT-4o function_call results to Python ML operations on live state."""

    def __init__(self, state: ProteinState, log_callback=None):
        self.state        = state
        self.done         = False
        self.log_callback = log_callback or (lambda msg: None)

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

    def _train_rf(self, inp: Dict) -> str:
        self.log_callback("🌲 Training Random Forest (500 trees)…")
        t0    = time.time()
        model = RandomForestClassifier(**RF_PARAMS)
        model.fit(self.state["X_train_smote"], self.state["y_train_smote"])
        elapsed = round(time.time() - t0, 2)

        self.state["model"] = model
        fi = pd.DataFrame({
            "feature":    self.state["X"].columns,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        self.state["feature_importance"] = fi

        self.log_callback(f"✅ RF trained — OOB: {model.oob_score_:.4f}  ({elapsed}s)")
        return json.dumps({
            "status":          "trained",
            "config":          "fixed — original pipeline configuration",
            "oob_score":       round(model.oob_score_, 4),
            "training_time_s": elapsed,
            "n_features":      self.state["X"].shape[1],
            "train_samples":   self.state["X_train_smote"].shape[0],
        })

    def _evaluate(self, inp: Dict) -> str:
        split = inp["split"]
        model = self.state.get("model")
        if model is None:
            return json.dumps({"error": "No model trained yet. Call train_random_forest first."})

        self.log_callback(f"📊 Evaluating on {split} split…")
        X, y = (
            (self.state["X_val"],  self.state["y_val"])
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
            self.state.update(
                val_accuracy=acc, val_f1=f1_mac,
                val_classification_report=report, val_confusion_matrix=cm,
            )
        else:
            self.state.update(
                test_accuracy=acc, test_f1=f1_mac,
                test_classification_report=report, test_confusion_matrix=cm,
            )

        self.log_callback(f"✅ {split.capitalize()}: Acc={acc}  Macro-F1={f1_mac}")
        return json.dumps({"split": split, "accuracy": acc, "macro_f1": f1_mac, "report": report})

    def _feature_importance(self, inp: Dict) -> str:
        top_n = inp.get("top_n", 15)
        fi    = self.state.get("feature_importance")
        if fi is None:
            return json.dumps({"error": "Train the model first."})
        self.log_callback(f"📌 Computing top-{top_n} feature importances…")
        return fi.head(top_n).to_json(orient="records")

    def _cross_validate(self, inp: Dict) -> str:
        cv_folds = inp.get("cv_folds", 5)
        self.log_callback(f"🔄 Running {cv_folds}-fold cross-validation…")
        cv_params = {**RF_PARAMS, "n_estimators": 200, "oob_score": False}
        clf       = RandomForestClassifier(**cv_params)
        scores    = cross_val_score(
            clf, self.state["X_train_smote"], self.state["y_train_smote"],
            cv=cv_folds, scoring="accuracy", n_jobs=-1,
        )
        self.log_callback(f"✅ CV mean accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
        return json.dumps({
            "cv_folds":      cv_folds,
            "rf_config":     "fixed (200 trees for speed)",
            "mean_accuracy": round(float(scores.mean()), 4),
            "std_accuracy":  round(float(scores.std()), 4),
            "fold_scores":   [round(s, 4) for s in scores.tolist()],
        })

    def _class_distribution(self, _: Dict) -> str:
        self.log_callback("📋 Inspecting class distribution…")
        le = self.state["label_encoder"]
        def counts(arr):
            vc = pd.Series(arr).value_counts().sort_index()
            return {le.classes_[k]: int(v) for k, v in vc.items()}
        return json.dumps({
            "train_smote": counts(self.state["y_train_smote"]),
            "validation":  counts(self.state["y_val"]),
            "test":        counts(self.state["y_test"]),
        })

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
        self.log_callback("📝 Final report generated — agent loop complete.")
        return json.dumps({"status": "report_generated", "message": "Agent loop complete."})


# ──────────────────────────────────────────────────────────────
#  SYSTEM PROMPT
# ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert computational biologist and machine learning engineer.
Your task is to analyse and classify protein structures from the Protein Data Bank (PDB).

The dataset has already been loaded, filtered to the top 20 classification classes,
feature-engineered (5 physicochemical features + 27 amino acid sequence features),
split 80/10/10 with stratification, and SMOTE-resampled (cap 6,000/class).

IMPORTANT — MODEL CONFIGURATION IS FIXED:
The Random Forest uses the exact production configuration:
  n_estimators=500, max_features="sqrt", max_depth=None,
  min_samples_split=2, min_samples_leaf=1, bootstrap=True,
  class_weight="balanced_subsample", random_state=50
Do NOT attempt to change hyperparameters.

YOUR WORKFLOW:
1. Train the model first (train_random_forest with {}).
2. Evaluate on validation split.
3. Evaluate on test split.
4. Inspect feature importance — reason about which protein properties drive classification.
5. Optionally run cross-validation to check stability.
6. Optionally inspect class distribution.
7. Call generate_final_report with a scientific summary and actionable recommendations.

REASONING GUIDELINES:
- Think step-by-step before each tool call.
- Interpret metrics biologically — what does macro-F1 mean for a 20-class protein problem?
- Connect feature importance back to structural biology.
- Be concise and scientific.
"""


# ──────────────────────────────────────────────────────────────
#  LANGGRAPH PIPELINE NODES
# ──────────────────────────────────────────────────────────────
def load_data_node(state: ProteinState) -> ProteinState:
    """Load CSV from the path provided in state['csv_path']."""
    try:
        path = state.get("csv_path", "")
        df   = pd.read_csv(path)
        state["df"]     = df
        state["status"] = f"✓ Loaded {df.shape[0]:,} rows × {df.shape[1]} cols"
        state["pipeline_log"] = state.get("pipeline_log", []) + [state["status"]]
    except Exception as e:
        state["error"] = f"Load failed: {e}"
    return state


def filter_top_classes_node(state: ProteinState) -> ProteinState:
    try:
        df  = state["df"].copy()
        top = df[TARGET_COL].value_counts().head(TOP_N_CLASSES).index.tolist()
        filtered = df[df[TARGET_COL].isin(top)].copy()
        state["df_filtered"] = filtered
        state["top_classes"] = top
        state["status"]      = f"✓ Filtered to top {TOP_N_CLASSES} classes → {filtered.shape[0]:,} rows"
        state["pipeline_log"] = state.get("pipeline_log", []) + [state["status"]]
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

        state["X"]             = X
        state["y"]             = y
        state["label_encoder"] = le
        state["status"]        = f"✓ Features built → X{X.shape}"
        state["pipeline_log"]  = state.get("pipeline_log", []) + [state["status"]]
    except Exception as e:
        state["error"] = f"Feature engineering failed: {e}"
    return state


def split_data_node(state: ProteinState) -> ProteinState:
    try:
        X, y = state["X"], state["y"].values
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
        X_val, X_te, y_val, y_te = train_test_split(
            X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RANDOM_STATE)
        state.update(
            X_train=X_tr, X_val=X_val, X_test=X_te,
            y_train=y_tr, y_val=y_val,  y_test=y_te,
        )
        state["status"]       = f"✓ Train={X_tr.shape} | Val={X_val.shape} | Test={X_te.shape}"
        state["pipeline_log"] = state.get("pipeline_log", []) + [state["status"]]
    except Exception as e:
        state["error"] = f"Split failed: {e}"
    return state


def smote_node(state: ProteinState) -> ProteinState:
    try:
        y_tr   = state["y_train"]
        counts = pd.Series(y_tr).value_counts()
        cap    = min(6000, counts.max())
        strat  = {cls: cap for cls, cnt in counts.items() if cnt < cap}
        sm     = SMOTE(sampling_strategy=strat, random_state=RANDOM_STATE, k_neighbors=3)
        Xr, yr = sm.fit_resample(state["X_train"], y_tr)
        state["X_train_smote"] = Xr
        state["y_train_smote"] = yr
        state["status"]        = f"✓ SMOTE → {Xr.shape[0]:,} training samples"
        state["pipeline_log"]  = state.get("pipeline_log", []) + [state["status"]]
    except Exception as e:
        state["error"] = f"SMOTE failed: {e}"
    return state


def make_agent_node(log_callback=None):
    """Factory that returns an agent_node closure with an optional live-log callback."""

    def agent_node(state: ProteinState) -> ProteinState:
        client   = OpenAI()
        executor = ToolExecutor(state, log_callback=log_callback or (lambda m: None))

        messages:      List[Dict] = state.get("agent_messages", [])
        reasoning_log: List[str]  = state.get("agent_reasoning", [])
        tool_log:      List[Dict] = state.get("tool_call_log", [])
        iterations:    int        = state.get("iterations", 0)

        if not messages:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
            class_counts = pd.Series(state["y_train_smote"]).value_counts()
            briefing = (
                f"Dataset briefing:\n"
                f"  • Target classes (top {TOP_N_CLASSES}): {', '.join(state['top_classes'])}\n"
                f"  • Feature dimensions: {state['X'].shape[1]} features\n"
                f"  • Training samples (post-SMOTE): {state['X_train_smote'].shape[0]:,}\n"
                f"  • Validation samples: {state['X_val'].shape[0]:,}\n"
                f"  • Test samples: {state['X_test'].shape[0]:,}\n"
                f"  • SMOTE class range: {class_counts.min()}–{class_counts.max()} per class\n\n"
                f"Model configuration is FIXED. Start by calling train_random_forest with {{}}."
            )
            messages.append({"role": "user", "content": briefing})

        while not executor.done and iterations < MAX_AGENT_ITERS:
            iterations += 1
            if log_callback:
                log_callback(f"🤖 Agent iteration {iterations}/{MAX_AGENT_ITERS}…")

            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=4096,
            )
            msg           = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            if msg.content:
                step_label = f"[Iteration {iterations}] {msg.content}"
                reasoning_log.append(step_label)
                if log_callback:
                    log_callback(f"🧠 {msg.content[:200]}{'…' if len(msg.content) > 200 else ''}")

            # Append assistant turn (plain dict, JSON-serialisable)
            assistant_dict: Dict = {"role": "assistant", "content": msg.content}
            if msg.tool_calls:
                assistant_dict["tool_calls"] = [
                    {
                        "id":       tc.id,
                        "type":     "function",
                        "function": {
                            "name":      tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_dict)

            if finish_reason == "stop" or not msg.tool_calls:
                break

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_input = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_input = {}

                result_str = executor.execute(tool_name, tool_input)
                tool_log.append({
                    "iteration": iterations,
                    "tool":      tool_name,
                    "input":     tool_input,
                    "result":    result_str,
                })
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

    return agent_node


# ──────────────────────────────────────────────────────────────
#  BUILD LANGGRAPH  (default, no live callback)
# ──────────────────────────────────────────────────────────────
def build_graph(log_callback=None) -> StateGraph:
    workflow = StateGraph(ProteinState)
    for name, fn in [
        ("load_data",           load_data_node),
        ("filter_top_classes",  filter_top_classes_node),
        ("feature_engineering", feature_engineering_node),
        ("split_data",          split_data_node),
        ("apply_smote",         smote_node),
        ("agent",               make_agent_node(log_callback)),
    ]:
        workflow.add_node(name, fn)

    workflow.set_entry_point("load_data")
    workflow.add_edge("load_data",           "filter_top_classes")
    workflow.add_edge("filter_top_classes",  "feature_engineering")
    workflow.add_edge("feature_engineering", "split_data")
    workflow.add_edge("split_data",          "apply_smote")
    workflow.add_edge("apply_smote",         "agent")
    workflow.add_edge("agent",               END)
    return workflow.compile()


# ──────────────────────────────────────────────────────────────
#  CONVENIENCE RUNNER
# ──────────────────────────────────────────────────────────────
def run_pipeline(csv_path: str, log_callback=None) -> ProteinState:
    """
    Execute the full LangGraph pipeline and return the final state.

    Args:
        csv_path:     Path to the CSV file (must contain 'classification'
                      and 'chain_sequences' columns).
        log_callback: Optional callable(msg: str) for live progress messages.

    Returns:
        Final ProteinState dict with all results, metrics, and the summary report.
    """
    app = build_graph(log_callback)

    initial: ProteinState = {
        "csv_path":        csv_path,
        "status":          "Starting",
        "error":           None,
        "agent_messages":  [],
        "agent_reasoning": [],
        "tool_call_log":   [],
        "pipeline_log":    [],
        "iterations":      0,
    }

    return app.invoke(initial)
