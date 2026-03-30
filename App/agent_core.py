"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  agent_core.py  —  Agentic Protein Structure Classifier  (TRAINING ONLY)   ║
║                                                                             ║
║  Run once to train and save the model:                                      ║
║      python agent_core.py --csv path/to/data_ai.csv                        ║
║                                                                             ║
║  Saves to model/ directory:                                                 ║
║      rf_model.joblib         ← trained Random Forest                       ║
║      label_encoder.joblib    ← fitted LabelEncoder                         ║
║      feature_columns.json    ← ordered feature names for prediction        ║
║      training_report.txt     ← full agent report                           ║
║      feature_importance.csv  ← ranked importances                          ║
║      tool_call_log.csv       ← agent tool call history                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import ast
import json
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
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
load_dotenv()

# ──────────────────────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────────────────────
TARGET_COL      = "classification"
TOP_N_CLASSES   = 20
RANDOM_STATE    = 50
MAX_AGENT_ITERS = 12
OPENAI_MODEL    = "gpt-4o"
MODEL_DIR       = Path("model")

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
#  FEATURE HELPERS  (shared with app.py predictor)
# ──────────────────────────────────────────────────────────────
AMINO_ACIDS  = list("ACDEFGHIKLMNPQRSTVWY")
AMINO_SET    = set(AMINO_ACIDS)
NUMERIC_COLS = [
    "resolution",
    "structure_molecular_weight",
    "density_matthews",
    "density_percent_sol",
    "ph_value",
]


def parse_chain_sequences(chain_str: str) -> Dict[str, str]:
    try:
        obj = ast.literal_eval(str(chain_str))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def clean_sequence(seq: str) -> str:
    return "".join(ch for ch in str(seq).upper() if ch in AMINO_SET)


def extract_sequence_features(chain_str: str) -> Dict[str, float]:
    """Derive 27 numeric features from a chain_sequences string."""
    chain_dict = parse_chain_sequences(chain_str)
    base = {
        "num_chains":      0,   "total_seq_len":   0,
        "avg_chain_len":   0.0, "max_chain_len":   0.0,
        "min_chain_len":   0.0, "std_chain_len":   0.0,
        "unique_aa_count": 0,
        **{f"aa_freq_{aa}": 0.0 for aa in AMINO_ACIDS},
    }
    if not chain_dict:
        return base
    seqs = [clean_sequence(s) for s in chain_dict.values() if clean_sequence(s)]
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
#  STATE
# ──────────────────────────────────────────────────────────────
ProteinState = dict


# ──────────────────────────────────────────────────────────────
#  TOOL DEFINITIONS
# ──────────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "train_random_forest",
            "description": (
                "Train the Random Forest on SMOTE-resampled data with fixed production config. "
                "Saves rf_model.joblib, label_encoder.joblib, and feature_columns.json to disk. "
                "Call with empty object {}."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_model",
            "description": "Evaluate trained model on validation or test split.",
            "parameters": {
                "type": "object",
                "properties": {
                    "split": {"type": "string", "enum": ["validation", "test"]}
                },
                "required": ["split"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_feature_importance",
            "description": "Return the top-N most important features.",
            "parameters": {
                "type": "object",
                "properties": {
                    "top_n": {"type": "integer", "description": "Default 15"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_cross_validation",
            "description": "K-fold CV on SMOTE training data (200 trees for speed).",
            "parameters": {
                "type": "object",
                "properties": {
                    "cv_folds": {"type": "integer", "description": "Default 5"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_class_distribution",
            "description": "Return class counts in train/val/test splits.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_final_report",
            "description": (
                "Compile and save the full analysis report. "
                "Saves training_report.txt, feature_importance.csv, tool_call_log.csv. "
                "ENDS the agent loop."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "executive_summary": {"type": "string"},
                    "recommendations":  {"type": "array", "items": {"type": "string"}},
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
    def __init__(self, state: ProteinState, log_callback=None):
        self.state        = state
        self.done         = False
        self.log_callback = log_callback or (lambda m: None)

    def execute(self, name: str, inp: Dict) -> str:
        fn = {
            "train_random_forest":        self._train_rf,
            "evaluate_model":             self._evaluate,
            "compute_feature_importance": self._feature_importance,
            "run_cross_validation":       self._cross_validate,
            "inspect_class_distribution": self._class_distribution,
            "generate_final_report":      self._final_report,
        }.get(name)
        if fn is None:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            return fn(inp)
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ── train + save model ───────────────────────────────────
    def _train_rf(self, _: Dict) -> str:
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

        # Save immediately so app.py can load without retraining
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(model,                        MODEL_DIR / "rf_model.joblib")
        joblib.dump(self.state["label_encoder"],  MODEL_DIR / "label_encoder.joblib")
        json.dump(
            list(self.state["X"].columns),
            open(MODEL_DIR / "feature_columns.json", "w"),
        )
        self.log_callback(f"✅ Trained — OOB: {model.oob_score_:.4f}  ({elapsed}s)")
        self.log_callback(f"💾 Model artefacts saved → {MODEL_DIR}/")
        return json.dumps({
            "status":          "trained",
            "oob_score":       round(model.oob_score_, 4),
            "training_time_s": elapsed,
            "n_features":      self.state["X"].shape[1],
            "train_samples":   self.state["X_train_smote"].shape[0],
            "saved_to":        str(MODEL_DIR),
        })

    # ── evaluate ─────────────────────────────────────────────
    def _evaluate(self, inp: Dict) -> str:
        split = inp["split"]
        model = self.state.get("model")
        if model is None:
            return json.dumps({"error": "Train first."})
        self.log_callback(f"📊 Evaluating on {split}…")
        X, y = (
            (self.state["X_val"], self.state["y_val"]) if split == "validation"
            else (self.state["X_test"], self.state["y_test"])
        )
        yp     = model.predict(X)
        acc    = round(accuracy_score(y, yp), 4)
        f1_mac = round(f1_score(y, yp, average="macro", zero_division=0), 4)
        report = classification_report(
            y, yp, target_names=self.state["label_encoder"].classes_, zero_division=0)
        if split == "validation":
            self.state.update(val_accuracy=acc, val_f1=f1_mac,
                              val_classification_report=report,
                              val_confusion_matrix=confusion_matrix(y, yp))
        else:
            self.state.update(test_accuracy=acc, test_f1=f1_mac,
                              test_classification_report=report,
                              test_confusion_matrix=confusion_matrix(y, yp))
        self.log_callback(f"✅ {split.capitalize()}: Acc={acc}  F1={f1_mac}")
        return json.dumps({"split": split, "accuracy": acc, "macro_f1": f1_mac, "report": report})

    # ── feature importance ───────────────────────────────────
    def _feature_importance(self, inp: Dict) -> str:
        top_n = inp.get("top_n", 15)
        fi    = self.state.get("feature_importance")
        if fi is None:
            return json.dumps({"error": "Train first."})
        self.log_callback(f"📌 Top-{top_n} importances computed")
        return fi.head(top_n).to_json(orient="records")

    # ── cross-validation ─────────────────────────────────────
    def _cross_validate(self, inp: Dict) -> str:
        cv_folds = inp.get("cv_folds", 5)
        self.log_callback(f"🔄 {cv_folds}-fold cross-validation…")
        clf    = RandomForestClassifier(**{**RF_PARAMS, "n_estimators": 200, "oob_score": False})
        scores = cross_val_score(clf, self.state["X_train_smote"],
                                 self.state["y_train_smote"], cv=cv_folds, n_jobs=-1)
        self.log_callback(f"✅ CV mean: {scores.mean():.4f} ± {scores.std():.4f}")
        return json.dumps({
            "cv_folds":      cv_folds,
            "mean_accuracy": round(float(scores.mean()), 4),
            "std_accuracy":  round(float(scores.std()), 4),
            "fold_scores":   [round(s, 4) for s in scores.tolist()],
        })

    # ── class distribution ───────────────────────────────────
    def _class_distribution(self, _: Dict) -> str:
        self.log_callback("📋 Class distribution…")
        le = self.state["label_encoder"]
        def counts(arr):
            vc = pd.Series(arr).value_counts().sort_index()
            return {le.classes_[k]: int(v) for k, v in vc.items()}
        return json.dumps({
            "train_smote": counts(self.state["y_train_smote"]),
            "validation":  counts(self.state["y_val"]),
            "test":        counts(self.state["y_test"]),
        })

    # ── final report + save all artefacts ───────────────────
    def _final_report(self, inp: Dict) -> str:
        summary_lines = [
            "=" * 80, "AGENTIC PROTEIN CLASSIFICATION — FINAL REPORT", "=" * 80, "",
            "EXECUTIVE SUMMARY", "-" * 40, inp.get("executive_summary", ""), "",
            "PERFORMANCE METRICS", "-" * 40,
            f"  Validation Accuracy : {self.state.get('val_accuracy', 'N/A')}",
            f"  Validation Macro-F1 : {self.state.get('val_f1', 'N/A')}",
            f"  Test Accuracy       : {self.state.get('test_accuracy', 'N/A')}",
            f"  Test Macro-F1       : {self.state.get('test_f1', 'N/A')}", "",
            "TOP 15 FEATURES", "-" * 40,
        ]
        fi = self.state.get("feature_importance")
        if fi is not None:
            summary_lines.append(fi.head(15).to_string(index=False))
        summary_lines += [
            "", "RECOMMENDATIONS", "-" * 40,
            *[f"  • {r}" for r in inp.get("recommendations", [])],
            "", "AGENT REASONING CHAIN", "-" * 40,
            *self.state.get("agent_reasoning", []),
            "", "VALIDATION CLASSIFICATION REPORT", "-" * 40,
            self.state.get("val_classification_report", "N/A"),
            "", "TEST CLASSIFICATION REPORT", "-" * 40,
            self.state.get("test_classification_report", "N/A"),
        ]
        summary = "\n".join(summary_lines)
        self.state["summary"] = summary

        MODEL_DIR.mkdir(exist_ok=True)
        (MODEL_DIR / "training_report.txt").write_text(summary, encoding="utf-8")
        if fi is not None:
            fi.to_csv(MODEL_DIR / "feature_importance.csv", index=False)
        tl = pd.DataFrame(self.state.get("tool_call_log", []))
        if not tl.empty:
            tl.to_csv(MODEL_DIR / "tool_call_log.csv", index=False)

        self.done = True
        self.log_callback("📝 Report generated and saved.")
        self.log_callback(f"💾 All artefacts → {MODEL_DIR}/")
        return json.dumps({"status": "report_generated"})


# ──────────────────────────────────────────────────────────────
#  SYSTEM PROMPT
# ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert computational biologist and machine learning engineer.
Your task is to train and evaluate a protein structure classifier on PDB data.

The dataset has been loaded, filtered to top 20 classes, feature-engineered
(5 physicochemical + 27 amino acid sequence features), split 80/10/10,
and SMOTE-resampled (cap 6,000/class).

FIXED RF CONFIG: n_estimators=500, max_features="sqrt", max_depth=None,
min_samples_split=2, min_samples_leaf=1, bootstrap=True,
class_weight="balanced_subsample", random_state=50.

WORKFLOW:
1. train_random_forest {} — trains and saves model to disk
2. evaluate_model validation
3. evaluate_model test
4. compute_feature_importance
5. (optional) run_cross_validation
6. (optional) inspect_class_distribution
7. generate_final_report — saves all artefacts, ends loop

Be concise, scientific, and reason step-by-step before each tool call.
"""


# ──────────────────────────────────────────────────────────────
#  PIPELINE NODES
# ──────────────────────────────────────────────────────────────
def _failed(state: ProteinState) -> bool:
    return bool(state.get("error"))


def load_data_node(state: ProteinState) -> ProteinState:
    try:
        df = pd.read_csv(state["csv_path"])
        missing = {"classification", "chain_sequences"} - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")
        state["df"]           = df
        state["status"]       = f"✓ Loaded {df.shape[0]:,} rows × {df.shape[1]} cols"
        state["pipeline_log"] = state.get("pipeline_log", []) + [state["status"]]
    except Exception as e:
        state["error"] = f"load_data: {e}"
    return state


def filter_top_classes_node(state: ProteinState) -> ProteinState:
    if _failed(state): return state
    try:
        df  = state["df"]
        top = df[TARGET_COL].value_counts().head(TOP_N_CLASSES).index.tolist()
        state["df_filtered"]  = df[df[TARGET_COL].isin(top)].copy()
        state["top_classes"]  = top
        state["status"]       = f"✓ Top {TOP_N_CLASSES} classes → {state['df_filtered'].shape[0]:,} rows"
        state["pipeline_log"] = state.get("pipeline_log", []) + [state["status"]]
    except Exception as e:
        state["error"] = f"filter: {e}"
    return state


def feature_engineering_node(state: ProteinState) -> ProteinState:
    if _failed(state): return state
    try:
        df           = state["df_filtered"]
        numeric_cols = [c for c in NUMERIC_COLS if c in df.columns]
        seq_feats    = pd.DataFrame(
            df["chain_sequences"].fillna("{}").apply(extract_sequence_features).tolist()
        )
        X = pd.concat([
            df[numeric_cols].reset_index(drop=True),
            seq_feats.reset_index(drop=True),
        ], axis=1).apply(pd.to_numeric, errors="coerce").fillna(0)
        le = LabelEncoder()
        y  = pd.Series(le.fit_transform(df[TARGET_COL].reset_index(drop=True)))
        state.update(X=X, y=y, label_encoder=le)
        state["status"]       = f"✓ Features → X{X.shape}"
        state["pipeline_log"] = state.get("pipeline_log", []) + [state["status"]]
    except Exception as e:
        state["error"] = f"feature_engineering: {e}"
    return state


def split_data_node(state: ProteinState) -> ProteinState:
    if _failed(state): return state
    try:
        X, y = state["X"], state["y"].values
        Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.20,
                                                  stratify=y, random_state=RANDOM_STATE)
        Xv, Xte, yv, yte = train_test_split(Xtmp, ytmp, test_size=0.50,
                                              stratify=ytmp, random_state=RANDOM_STATE)
        state.update(X_train=Xtr, X_val=Xv, X_test=Xte,
                     y_train=ytr, y_val=yv,  y_test=yte)
        state["status"]       = f"✓ Train={Xtr.shape} Val={Xv.shape} Test={Xte.shape}"
        state["pipeline_log"] = state.get("pipeline_log", []) + [state["status"]]
    except Exception as e:
        state["error"] = f"split: {e}"
    return state


def smote_node(state: ProteinState) -> ProteinState:
    if _failed(state): return state
    try:
        ytr    = state["y_train"]
        counts = pd.Series(ytr).value_counts()
        cap    = min(6000, counts.max())
        strat  = {c: cap for c, n in counts.items() if n < cap}
        Xr, yr = SMOTE(sampling_strategy=strat,
                        random_state=RANDOM_STATE, k_neighbors=3).fit_resample(
                            state["X_train"], ytr)
        state["X_train_smote"] = Xr
        state["y_train_smote"] = yr
        state["status"]        = f"✓ SMOTE → {Xr.shape[0]:,} training samples"
        state["pipeline_log"]  = state.get("pipeline_log", []) + [state["status"]]
    except Exception as e:
        state["error"] = f"smote: {e}"
    return state


def make_agent_node(log_callback=None):
    def agent_node(state: ProteinState) -> ProteinState:
        if _failed(state): return state
        client   = OpenAI()
        executor = ToolExecutor(state, log_callback=log_callback or (lambda m: None))
        messages:  List[Dict] = state.get("agent_messages", [])
        reasoning: List[str]  = state.get("agent_reasoning", [])
        tool_log:  List[Dict] = state.get("tool_call_log", [])
        iters:     int        = state.get("iterations", 0)

        if not messages:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
            cc = pd.Series(state["y_train_smote"]).value_counts()
            messages.append({"role": "user", "content": (
                f"Classes (top {TOP_N_CLASSES}): {', '.join(state['top_classes'])}\n"
                f"Features: {state['X'].shape[1]}  |  "
                f"Train (post-SMOTE): {state['X_train_smote'].shape[0]:,}  |  "
                f"Val: {state['X_val'].shape[0]:,}  |  Test: {state['X_test'].shape[0]:,}\n"
                f"SMOTE range: {cc.min()}–{cc.max()} per class\n\n"
                f"Start by calling train_random_forest with {{}}."
            )})

        while not executor.done and iters < MAX_AGENT_ITERS:
            iters += 1
            if log_callback:
                log_callback(f"🤖 Agent iteration {iters}/{MAX_AGENT_ITERS}…")

            resp  = client.chat.completions.create(
                model=OPENAI_MODEL, messages=messages,
                tools=TOOLS, tool_choice="auto", max_tokens=4096,
            )
            msg    = resp.choices[0].message
            reason = resp.choices[0].finish_reason

            if msg.content:
                reasoning.append(f"[{iters}] {msg.content}")
                if log_callback:
                    log_callback(f"🧠 {msg.content[:200]}{'…' if len(msg.content)>200 else ''}")

            adict: Dict = {"role": "assistant", "content": msg.content}
            if msg.tool_calls:
                adict["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ]
            messages.append(adict)

            if reason == "stop" or not msg.tool_calls:
                break

            for tc in msg.tool_calls:
                try:
                    ti = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    ti = {}
                result = executor.execute(tc.function.name, ti)
                tool_log.append({"iteration": iters, "tool": tc.function.name,
                                  "input": ti, "result": result})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        state.update(agent_messages=messages, agent_reasoning=reasoning,
                     tool_call_log=tool_log, iterations=iters,
                     status=f"Agent done in {iters} iterations")
        return state
    return agent_node


# ──────────────────────────────────────────────────────────────
#  GRAPH
# ──────────────────────────────────────────────────────────────
def _route(state: ProteinState):
    return END if state.get("error") else "next"


def build_graph(log_callback=None):
    wf = StateGraph(ProteinState)
    nodes = [
        ("load_data",           load_data_node),
        ("filter_top_classes",  filter_top_classes_node),
        ("feature_engineering", feature_engineering_node),
        ("split_data",          split_data_node),
        ("apply_smote",         smote_node),
        ("agent",               make_agent_node(log_callback)),
    ]
    for name, fn in nodes:
        wf.add_node(name, fn)
    wf.set_entry_point("load_data")
    transitions = [
        ("load_data",          "filter_top_classes"),
        ("filter_top_classes", "feature_engineering"),
        ("feature_engineering","split_data"),
        ("split_data",         "apply_smote"),
        ("apply_smote",        "agent"),
    ]
    for src, dst in transitions:
        wf.add_conditional_edges(src, _route, {"next": dst, END: END})
    wf.add_edge("agent", END)
    return wf.compile()


def run_pipeline(csv_path: str, log_callback=None) -> ProteinState:
    """Train the full pipeline and save all model artefacts to model/."""
    return build_graph(log_callback).invoke({
        "csv_path":        csv_path,
        "status":          "Starting",
        "error":           None,
        "agent_messages":  [],
        "agent_reasoning": [],
        "tool_call_log":   [],
        "pipeline_log":    [],
        "iterations":      0,
    })


# ──────────────────────────────────────────────────────────────
#  CLI  —  python agent_core.py --csv data_ai.csv
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train protein classifier agent")
    parser.add_argument("--csv", required=True, help="Path to training CSV")
    args = parser.parse_args()

    print("\n" + "═" * 70)
    print("  AGENTIC PROTEIN CLASSIFIER — Training Mode")
    print(f"  Model: {OPENAI_MODEL}  |  Output: {MODEL_DIR}/")
    print("═" * 70 + "\n")

    final = run_pipeline(args.csv, log_callback=print)

    if final.get("error"):
        print(f"\n❌ FAILED: {final['error']}")
    else:
        print("\n" + "═" * 70)
        print("✅ Training complete. Saved artefacts:")
        for f in sorted(MODEL_DIR.glob("*")):
            print(f"   {f}")
        print("═" * 70)
