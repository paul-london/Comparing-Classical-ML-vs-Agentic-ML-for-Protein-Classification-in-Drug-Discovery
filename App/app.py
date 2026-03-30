"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  app.py  —  Gradio frontend for the Agentic Protein Structure Classifier    ║
║                                                                             ║
║  Features:                                                                  ║
║    • CSV upload or path input                                               ║
║    • Live agent log streamed to the UI                                      ║
║    • Results dashboard: metrics cards, feature importance bar chart,        ║
║      confusion matrix, classification reports, full agent reasoning chain   ║
║    • Download buttons for artefacts                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import json
import tempfile
import threading
import queue
import traceback
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv

from agent_core import run_pipeline

load_dotenv()

# ──────────────────────────────────────────────────────────────
#  COLOUR PALETTE  (dark scientific / bioinformatics vibe)
# ──────────────────────────────────────────────────────────────
DEEP_NAVY   = "#0B1929"
CARD_BG     = "#0F2238"
ACCENT_TEAL = "#00D4B4"
ACCENT_BLUE = "#1E90FF"
TEXT_LIGHT  = "#E8F4F8"
TEXT_DIM    = "#7A9AB5"
SUCCESS     = "#00C896"
WARNING     = "#FFB347"
DANGER      = "#FF6B6B"

CUSTOM_CSS = f"""
/* ── Root & body ─────────────────────────── */
:root {{
    --deep-navy:   {DEEP_NAVY};
    --card-bg:     {CARD_BG};
    --accent-teal: {ACCENT_TEAL};
    --accent-blue: {ACCENT_BLUE};
    --text-light:  {TEXT_LIGHT};
    --text-dim:    {TEXT_DIM};
}}

body, .gradio-container {{
    background: var(--deep-navy) !important;
    font-family: 'IBM Plex Mono', 'Courier New', monospace;
    color: var(--text-light) !important;
}}

/* ── Header banner ──────────────────────── */
#header-banner {{
    background: linear-gradient(135deg, #0B1929 0%, #0F3460 50%, #0B1929 100%);
    border: 1px solid {ACCENT_TEAL}33;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}}
#header-banner::before {{
    content: '';
    position: absolute;
    inset: 0;
    background: repeating-linear-gradient(
        90deg,
        transparent,
        transparent 60px,
        {ACCENT_TEAL}08 60px,
        {ACCENT_TEAL}08 61px
    );
}}

/* ── Cards ──────────────────────────────── */
.metric-card {{
    background: var(--card-bg);
    border: 1px solid {ACCENT_TEAL}33;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}}

/* ── Buttons ────────────────────────────── */
#run-btn {{
    background: linear-gradient(135deg, {ACCENT_TEAL}, {ACCENT_BLUE}) !important;
    color: {DEEP_NAVY} !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 0.8rem 2rem !important;
    letter-spacing: 0.05em !important;
    transition: opacity 0.2s !important;
}}
#run-btn:hover {{ opacity: 0.85 !important; }}

#dl-report-btn, #dl-fi-btn, #dl-log-btn {{
    background: {CARD_BG} !important;
    color: {ACCENT_TEAL} !important;
    border: 1px solid {ACCENT_TEAL}55 !important;
    border-radius: 6px !important;
}}

/* ── Inputs & textboxes ─────────────────── */
.gr-textbox textarea, .gr-textbox input {{
    background: {CARD_BG} !important;
    color: var(--text-light) !important;
    border-color: {ACCENT_TEAL}44 !important;
    font-family: inherit !important;
}}

/* ── Tab labels ─────────────────────────── */
.tab-nav button {{
    color: var(--text-dim) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: 0.04em !important;
}}
.tab-nav button.selected {{
    color: {ACCENT_TEAL} !important;
    border-bottom-color: {ACCENT_TEAL} !important;
}}

/* ── Log area ───────────────────────────── */
#agent-log textarea {{
    background: #060F18 !important;
    color: {ACCENT_TEAL} !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    border-color: {ACCENT_TEAL}33 !important;
    line-height: 1.6 !important;
}}

/* ── Section labels ─────────────────────── */
.section-label {{
    color: {TEXT_DIM};
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}}
"""


# ──────────────────────────────────────────────────────────────
#  CHART BUILDERS
# ──────────────────────────────────────────────────────────────
_plotly_layout = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,34,56,0.9)",
    font=dict(color=TEXT_LIGHT, family="IBM Plex Mono, monospace"),
    margin=dict(l=10, r=10, t=40, b=10),
)


def build_feature_importance_chart(fi_df: pd.DataFrame) -> go.Figure:
    top15 = fi_df.head(15).sort_values("importance")
    fig = go.Figure(go.Bar(
        x=top15["importance"],
        y=top15["feature"],
        orientation="h",
        marker=dict(
            color=top15["importance"],
            colorscale=[[0, ACCENT_BLUE], [1, ACCENT_TEAL]],
            showscale=False,
        ),
    ))
    fig.update_layout(
        title="Top-15 Feature Importances",
        xaxis_title="Importance",
        **_plotly_layout,
    )
    return fig


def build_metrics_gauge(val_acc, val_f1, test_acc, test_f1) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=4,
        specs=[[{"type": "indicator"}] * 4],
        subplot_titles=["Val Accuracy", "Val Macro-F1", "Test Accuracy", "Test Macro-F1"],
    )
    for col, (title, value) in enumerate([
        ("Val Acc", val_acc), ("Val F1", val_f1),
        ("Test Acc", test_acc), ("Test F1", test_f1),
    ], start=1):
        if isinstance(value, float):
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=value,
                    gauge=dict(
                        axis=dict(range=[0, 1]),
                        bar=dict(color=ACCENT_TEAL),
                        bgcolor=CARD_BG,
                        borderwidth=1,
                        bordercolor=ACCENT_TEAL,
                        steps=[
                            dict(range=[0, 0.5], color="#1a2a3a"),
                            dict(range=[0.5, 0.75], color="#1a3040"),
                            dict(range=[0.75, 1], color="#1a3a50"),
                        ],
                    ),
                    number=dict(font=dict(size=28, color=ACCENT_TEAL)),
                ),
                row=1, col=col,
            )
    fig.update_layout(height=240, **_plotly_layout)
    return fig


def build_class_dist_chart(class_dist: dict) -> go.Figure:
    """Bar chart comparing train/val/test class counts."""
    if not class_dist:
        return go.Figure()
    train = class_dist.get("train_smote", {})
    classes = sorted(train.keys())
    fig = go.Figure()
    splits = [
        ("train_smote", ACCENT_TEAL),
        ("validation",  ACCENT_BLUE),
        ("test",        WARNING),
    ]
    for key, color in splits:
        counts = class_dist.get(key, {})
        fig.add_trace(go.Bar(
            name=key,
            x=classes,
            y=[counts.get(c, 0) for c in classes],
            marker_color=color,
        ))
    fig.update_layout(
        barmode="group",
        title="Class Distribution across Splits",
        xaxis_tickangle=-40,
        **_plotly_layout,
    )
    return fig


# ──────────────────────────────────────────────────────────────
#  DOWNLOAD HELPERS
# ──────────────────────────────────────────────────────────────
def _tmp_write(content: str, suffix: str) -> str:
    """Write content to a named temp file and return the path."""
    tf = tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, encoding="utf-8"
    )
    tf.write(content)
    tf.flush()
    return tf.name


# ──────────────────────────────────────────────────────────────
#  MAIN PIPELINE RUNNER  (called by Gradio)
# ──────────────────────────────────────────────────────────────
def run_classifier(csv_file, csv_path_str, openai_key, progress=gr.Progress()):
    """
    Entry point for the Gradio button click.
    Yields a sequence of tuples that update every output component.
    """

    # ── 0. Resolve CSV path ───────────────────────────────────
    if csv_file is not None:
        csv_path = csv_file.name
    elif csv_path_str and csv_path_str.strip():
        csv_path = csv_path_str.strip()
    else:
        yield (
            "❌ Please upload a CSV file or enter a file path.",
            None, None, None, None,
            "N/A", "N/A", "N/A", "N/A",
            "", "", None, None, None,
        )
        return

    if not Path(csv_path).exists():
        yield (
            f"❌ File not found: {csv_path}",
            None, None, None, None,
            "N/A", "N/A", "N/A", "N/A",
            "", "", None, None, None,
        )
        return

    # ── 1. Set API key if provided ────────────────────────────
    if openai_key and openai_key.strip():
        os.environ["OPENAI_API_KEY"] = openai_key.strip()

    if not os.environ.get("OPENAI_API_KEY"):
        yield (
            "❌ OPENAI_API_KEY is not set. Add it in the settings panel or your .env file.",
            None, None, None, None,
            "N/A", "N/A", "N/A", "N/A",
            "", "", None, None, None,
        )
        return

    # ── 2. Set up live logging via a thread-safe queue ────────
    log_queue: queue.Queue = queue.Queue()
    log_lines: list        = ["🚀 Pipeline started…"]
    result_holder          = {}

    def log_cb(msg: str):
        log_queue.put(msg)

    def pipeline_thread():
        try:
            final = run_pipeline(csv_path, log_callback=log_cb)
            result_holder["state"] = final
        except Exception:
            result_holder["error"] = traceback.format_exc()
        finally:
            log_queue.put("__DONE__")

    thread = threading.Thread(target=pipeline_thread, daemon=True)
    thread.start()

    # ── 3. Stream log while pipeline runs ─────────────────────
    empty_fig = go.Figure().update_layout(**_plotly_layout)

    while True:
        try:
            msg = log_queue.get(timeout=60)
        except queue.Empty:
            break
        if msg == "__DONE__":
            break
        log_lines.append(msg)
        yield (
            "\n".join(log_lines),
            empty_fig, empty_fig, empty_fig, empty_fig,
            "…", "…", "…", "…",
            "", "",
            None, None, None,
        )

    thread.join(timeout=5)

    # ── 4. Handle errors ──────────────────────────────────────
    if "error" in result_holder:
        err_msg = f"❌ Pipeline failed:\n\n{result_holder['error']}"
        log_lines.append(err_msg)
        yield (
            "\n".join(log_lines),
            empty_fig, empty_fig, empty_fig, empty_fig,
            "ERROR", "ERROR", "ERROR", "ERROR",
            "", "",
            None, None, None,
        )
        return

    # ── 5. Unpack final state ─────────────────────────────────
    state = result_holder["state"]

    if state.get("error"):
        log_lines.append(f"❌ {state['error']}")
        yield (
            "\n".join(log_lines),
            empty_fig, empty_fig, empty_fig, empty_fig,
            "ERROR", "ERROR", "ERROR", "ERROR",
            "", "",
            None, None, None,
        )
        return

    # ── 6. Build charts ───────────────────────────────────────
    fi_df   = state.get("feature_importance")
    fi_chart = build_feature_importance_chart(fi_df) if fi_df is not None else empty_fig

    val_acc  = state.get("val_accuracy",  "N/A")
    val_f1   = state.get("val_f1",        "N/A")
    test_acc = state.get("test_accuracy", "N/A")
    test_f1  = state.get("test_f1",       "N/A")

    gauges = build_metrics_gauge(
        val_acc  if isinstance(val_acc,  float) else 0.0,
        val_f1   if isinstance(val_f1,   float) else 0.0,
        test_acc if isinstance(test_acc, float) else 0.0,
        test_f1  if isinstance(test_f1,  float) else 0.0,
    )

    # Class distribution from tool log
    class_dist_chart = empty_fig
    for entry in reversed(state.get("tool_call_log", [])):
        if entry["tool"] == "inspect_class_distribution":
            try:
                class_dist_chart = build_class_dist_chart(json.loads(entry["result"]))
            except Exception:
                pass
            break

    # ── 7. Text outputs ───────────────────────────────────────
    val_report  = state.get("val_classification_report",  "Not available")
    test_report = state.get("test_classification_report", "Not available")
    summary     = state.get("summary", "No summary generated")
    reasoning   = "\n\n".join(state.get("agent_reasoning", []))

    # ── 8. Downloadable files ─────────────────────────────────
    report_path = _tmp_write(summary, "_protein_report.txt")

    fi_csv_path = None
    if fi_df is not None:
        fi_csv_path = _tmp_write(fi_df.to_csv(index=False), "_feature_importance.csv")

    tool_log_df = pd.DataFrame(state.get("tool_call_log", []))
    tool_log_path = None
    if not tool_log_df.empty:
        tool_log_path = _tmp_write(tool_log_df.to_csv(index=False), "_tool_log.csv")

    log_lines.append("✅ All done! Switch to the Results tabs.")

    yield (
        "\n".join(log_lines),         # agent_log
        fi_chart,                      # fi_chart
        gauges,                        # gauges_chart
        class_dist_chart,              # dist_chart
        empty_fig,                     # (reserved)
        f"{val_acc}",                  # val_acc_txt
        f"{val_f1}",                   # val_f1_txt
        f"{test_acc}",                 # test_acc_txt
        f"{test_f1}",                  # test_f1_txt
        val_report,                    # val_report_txt
        test_report,                   # test_report_txt
        report_path,                   # dl_report
        fi_csv_path,                   # dl_fi
        tool_log_path,                 # dl_log
    )


# ──────────────────────────────────────────────────────────────
#  GRADIO UI
# ──────────────────────────────────────────────────────────────
with gr.Blocks(
    title="🧬 Agentic Protein Structure Classifier",
    css=CUSTOM_CSS,
    theme=gr.themes.Base(
        primary_hue="teal",
        secondary_hue="blue",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("IBM Plex Mono"),
    ),
) as demo:

    # ── Header ────────────────────────────────────────────────
    gr.HTML("""
    <div id="header-banner">
      <div style="position:relative;z-index:1;">
        <div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.5rem;">
          <span style="font-size:2.2rem;">🧬</span>
          <div>
            <h1 style="margin:0;font-size:1.6rem;color:#00D4B4;letter-spacing:0.05em;
                        font-family:'IBM Plex Mono',monospace;font-weight:700;">
              AGENTIC PROTEIN STRUCTURE CLASSIFIER
            </h1>
            <p style="margin:0.25rem 0 0 0;color:#7A9AB5;font-size:0.82rem;letter-spacing:0.1em;">
              GPT-4o  ·  LangGraph  ·  Random Forest  ·  SMOTE  ·  PDB Dataset
            </p>
          </div>
        </div>
        <p style="margin:0.8rem 0 0 0;color:#B0CDD8;font-size:0.88rem;max-width:680px;line-height:1.6;">
          Upload your PDB-derived CSV and watch the AI agent autonomously train,
          evaluate, and interpret a Random Forest protein structure classifier —
          reasoning step-by-step like an expert computational biologist.
        </p>
      </div>
    </div>
    """)

    # ── Input Panel ───────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=1, min_width=320):
            gr.HTML('<div class="section-label">📁 Data Input</div>')
            csv_upload = gr.File(
                label="Upload CSV file",
                file_types=[".csv"],
                height=100,
            )
            csv_path_input = gr.Textbox(
                label="— or — enter file path",
                placeholder="/path/to/data_ai.csv",
            )

            gr.HTML('<div class="section-label" style="margin-top:1rem;">🔑 API Key</div>')
            api_key_input = gr.Textbox(
                label="OpenAI API Key (overrides .env)",
                placeholder="sk-…  (leave blank if set in .env)",
                type="password",
            )

            gr.HTML("""
            <div style="background:#0F2238;border:1px solid #00D4B455;border-radius:8px;
                        padding:1rem;margin-top:1rem;font-size:0.78rem;color:#7A9AB5;line-height:1.7;">
              <strong style="color:#00D4B4;">Required CSV columns:</strong><br>
              &nbsp;• <code>classification</code> — target label<br>
              &nbsp;• <code>chain_sequences</code> — dict of chain→sequence<br>
              &nbsp;• <code>resolution</code><br>
              &nbsp;• <code>structure_molecular_weight</code><br>
              &nbsp;• <code>density_matthews</code><br>
              &nbsp;• <code>density_percent_sol</code><br>
              &nbsp;• <code>ph_value</code>
            </div>
            """)

            run_btn = gr.Button(
                "▶  Run Classifier Agent",
                elem_id="run-btn",
            )

        # ── Live Log ─────────────────────────────────────────
        with gr.Column(scale=2):
            gr.HTML('<div class="section-label">🖥 Live Agent Log</div>')
            agent_log = gr.Textbox(
                label="",
                elem_id="agent-log",
                lines=28,
                max_lines=28,
                interactive=False,
                placeholder="Pipeline output will stream here once you click Run…",
            )

    gr.HTML("<hr style='border-color:#00D4B422;margin:1.5rem 0;'>")

    # ── Results Tabs ──────────────────────────────────────────
    gr.HTML('<div class="section-label">📊 Results Dashboard</div>')

    with gr.Tabs():

        # ── Tab 1: Metric Gauges ──────────────────────────────
        with gr.Tab("⚡ Metrics"):
            gauges_chart = gr.Plot(label="Performance Gauges")
            with gr.Row():
                val_acc_txt  = gr.Textbox(label="Validation Accuracy",  interactive=False, value="—")
                val_f1_txt   = gr.Textbox(label="Validation Macro-F1",  interactive=False, value="—")
                test_acc_txt = gr.Textbox(label="Test Accuracy",         interactive=False, value="—")
                test_f1_txt  = gr.Textbox(label="Test Macro-F1",         interactive=False, value="—")

        # ── Tab 2: Feature Importance ─────────────────────────
        with gr.Tab("📌 Feature Importance"):
            fi_chart = gr.Plot(label="Top-15 Feature Importances")

        # ── Tab 3: Class Distribution ─────────────────────────
        with gr.Tab("📋 Class Distribution"):
            dist_chart = gr.Plot(label="Class Counts per Split")

        # ── Tab 4: Classification Reports ────────────────────
        with gr.Tab("📄 Classification Reports"):
            with gr.Row():
                with gr.Column():
                    gr.HTML('<div class="section-label">Validation Set</div>')
                    val_report_txt = gr.Textbox(
                        label="", lines=25, interactive=False,
                        placeholder="Validation classification report will appear here…",
                        elem_classes=["mono-text"],
                    )
                with gr.Column():
                    gr.HTML('<div class="section-label">Test Set</div>')
                    test_report_txt = gr.Textbox(
                        label="", lines=25, interactive=False,
                        placeholder="Test classification report will appear here…",
                        elem_classes=["mono-text"],
                    )

        # ── Tab 5: Downloads ──────────────────────────────────
        with gr.Tab("⬇ Downloads"):
            gr.HTML('<div style="color:#7A9AB5;margin-bottom:1rem;font-size:0.85rem;">'
                    'Generated artefacts become available after the pipeline completes.'
                    '</div>')
            with gr.Row():
                dl_report = gr.File(label="📝 Full Report (.txt)",     elem_id="dl-report-btn")
                dl_fi     = gr.File(label="📊 Feature Importance (.csv)", elem_id="dl-fi-btn")
                dl_log    = gr.File(label="🔧 Tool Call Log (.csv)",   elem_id="dl-log-btn")

    # ── Wire up ───────────────────────────────────────────────
    # All outputs in the exact order yielded by run_classifier
    outputs = [
        agent_log,
        fi_chart,
        gauges_chart,
        dist_chart,
        gr.Plot(visible=False),   # reserved slot (keeps yield tuple length stable)
        val_acc_txt,
        val_f1_txt,
        test_acc_txt,
        test_f1_txt,
        val_report_txt,
        test_report_txt,
        dl_report,
        dl_fi,
        dl_log,
    ]

    run_btn.click(
        fn=run_classifier,
        inputs=[csv_upload, csv_path_input, api_key_input],
        outputs=outputs,
    )

    # ── Footer ────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center;margin-top:2rem;padding-top:1rem;
                border-top:1px solid #00D4B422;color:#3A5A6A;font-size:0.72rem;
                letter-spacing:0.08em;">
      AGENTIC PROTEIN STRUCTURE CLASSIFIER  ·  GPT-4o + LangGraph + Random Forest
    </div>
    """)


# ──────────────────────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,          # set True for a public Gradio tunnel link
        show_error=True,
    )
