"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  app.py  —  Protein Structure Classifier  (PREDICTION UI)                  ║
║                                                                             ║
║  Loads the pre-trained model from model/ and predicts protein class         ║
║  from whatever data the user provides.  Nothing is required —               ║
║  the more fields filled in, the more confident the prediction.              ║
║                                                                             ║
║  Run:  python app.py                                                        ║
║  Prerequisites: model/ directory must exist (run agent_core.py first)      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from agent_core import (
    AMINO_ACIDS, NUMERIC_COLS,
    extract_sequence_features,
)

# ──────────────────────────────────────────────────────────────
#  MODEL LOADING
# ──────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).parent / "model"

def load_model():
    """Load model artefacts.  Returns (model, label_encoder, feature_columns) or raises."""
    missing = [f for f in ["rf_model.joblib", "label_encoder.joblib", "feature_columns.json"]
               if not (MODEL_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing model artefacts: {missing}\n"
            f"Run  python agent_core.py --csv your_data.csv  first."
        )
    model    = joblib.load(MODEL_DIR / "rf_model.joblib")
    le       = joblib.load(MODEL_DIR / "label_encoder.joblib")
    feat_cols = json.load(open(MODEL_DIR / "feature_columns.json"))
    return model, le, feat_cols

try:
    MODEL, LE, FEATURE_COLS = load_model()
    MODEL_READY = True
    MODEL_ERROR = None
except Exception as e:
    MODEL, LE, FEATURE_COLS = None, None, []
    MODEL_READY = False
    MODEL_ERROR = str(e)

TOP_N_PRED = 5   # how many top classes to show

# ──────────────────────────────────────────────────────────────
#  COLOUR PALETTE
# ──────────────────────────────────────────────────────────────
DEEP_NAVY   = "#0B1929"
CARD_BG     = "#0F2238"
ACCENT_TEAL = "#00D4B4"
ACCENT_BLUE = "#1E90FF"
TEXT_LIGHT  = "#E8F4F8"
TEXT_DIM    = "#7A9AB5"
SUCCESS     = "#00C896"
WARNING     = "#FFB347"

CUSTOM_CSS = f"""
body, .gradio-container {{
    background: {DEEP_NAVY} !important;
    font-family: 'IBM Plex Mono', 'Courier New', monospace;
    color: {TEXT_LIGHT} !important;
}}
#header {{
    background: linear-gradient(135deg, #0B1929 0%, #0F3460 60%, #0B1929 100%);
    border: 1px solid {ACCENT_TEAL}33;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
}}
#header::before {{
    content:''; position:absolute; inset:0;
    background: repeating-linear-gradient(90deg, transparent, transparent 60px,
        {ACCENT_TEAL}08 60px, {ACCENT_TEAL}08 61px);
}}
#predict-btn {{
    background: linear-gradient(135deg, {ACCENT_TEAL}, {ACCENT_BLUE}) !important;
    color: {DEEP_NAVY} !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border-radius: 8px !important;
    border: none !important;
    letter-spacing: 0.05em !important;
    transition: opacity 0.2s !important;
    width: 100% !important;
    padding: 0.9rem !important;
}}
#predict-btn:hover {{ opacity: 0.85 !important; }}
.gr-textbox textarea, .gr-textbox input,
.gr-number input, .gr-slider input {{
    background: {CARD_BG} !important;
    color: {TEXT_LIGHT} !important;
    border-color: {ACCENT_TEAL}44 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}}
.gr-number label, .gr-textbox label, .gr-slider label {{
    color: {TEXT_DIM} !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
}}
.section-title {{
    color: {ACCENT_TEAL};
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    border-bottom: 1px solid {ACCENT_TEAL}33;
    padding-bottom: 0.4rem;
    margin-bottom: 0.8rem;
    margin-top: 1.2rem;
}}
.completeness-bar {{
    height: 6px;
    border-radius: 3px;
    background: linear-gradient(90deg, {ACCENT_TEAL}, {ACCENT_BLUE});
    transition: width 0.3s ease;
}}
.tab-nav button {{ color: {TEXT_DIM} !important; font-family: 'IBM Plex Mono' !important; }}
.tab-nav button.selected {{ color: {ACCENT_TEAL} !important; border-bottom-color: {ACCENT_TEAL} !important; }}
"""

# ──────────────────────────────────────────────────────────────
#  FEATURE BUILDER  —  assembles a 1-row DataFrame from inputs
# ──────────────────────────────────────────────────────────────
def build_feature_row(
    chain_sequences: str,
    resolution:      float | None,
    mol_weight:      float | None,
    density_matt:    float | None,
    density_sol:     float | None,
    ph_value:        float | None,
) -> tuple[pd.DataFrame, int, int, list[str]]:
    """
    Returns:
        feature_df      — 1-row DataFrame aligned to FEATURE_COLS
        filled_count    — how many feature groups the user provided
        total_groups    — total possible groups (for completeness %)
        provided_labels — human-readable list of what was provided
    """
    row       = {col: 0.0 for col in FEATURE_COLS}
    provided  = []
    filled    = 0
    total_grp = 2   # physicochemical + sequence

    # ── Physicochemical ───────────────────────────────────────
    phys_map = {
        "resolution":                 resolution,
        "structure_molecular_weight": mol_weight,
        "density_matthews":           density_matt,
        "density_percent_sol":        density_sol,
        "ph_value":                   ph_value,
    }
    phys_provided = []
    for col, val in phys_map.items():
        if val is not None and col in row:
            row[col] = float(val)
            phys_provided.append(col)
    if phys_provided:
        provided.append(f"Physicochemical ({len(phys_provided)}/5 fields)")
        filled += len(phys_provided) / 5   # partial credit

    # ── Sequence features ─────────────────────────────────────
    chain_str = (chain_sequences or "").strip()
    if chain_str:
        seq_feats = extract_sequence_features(chain_str)
        for k, v in seq_feats.items():
            if k in row:
                row[k] = v
        total_seq = sum(v != 0 for v in seq_feats.values())
        if total_seq > 0:
            provided.append(f"Sequence features (27 computed from chain_sequences)")
            filled += 1

    feature_df = pd.DataFrame([row])[FEATURE_COLS]
    return feature_df, filled, total_grp, provided


def completeness_pct(filled: float, total: int) -> int:
    return min(100, int(100 * filled / total))


# ──────────────────────────────────────────────────────────────
#  PREDICTION CHART
# ──────────────────────────────────────────────────────────────
_layout = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=f"rgba(15,34,56,0.9)",
    font=dict(color=TEXT_LIGHT, family="IBM Plex Mono, monospace"),
    margin=dict(l=10, r=10, t=40, b=10),
)

def build_prediction_chart(classes: list[str], probs: list[float]) -> go.Figure:
    colours = [
        f"rgba(0,212,180,{0.9 - i * 0.15})" for i in range(len(classes))
    ]
    fig = go.Figure(go.Bar(
        x=probs,
        y=classes,
        orientation="h",
        marker_color=colours,
        text=[f"{p:.1%}" for p in probs],
        textposition="outside",
        textfont=dict(color=TEXT_LIGHT),
    ))
    fig.update_layout(
        title=f"Top-{len(classes)} Predicted Classes",
        xaxis=dict(tickformat=".0%", range=[0, max(probs) * 1.25]),
        **_layout,
    )
    return fig


# ──────────────────────────────────────────────────────────────
#  MAIN PREDICT FUNCTION
# ──────────────────────────────────────────────────────────────
def predict(
    chain_sequences, resolution, mol_weight,
    density_matt, density_sol, ph_value,
):
    empty_fig = go.Figure().update_layout(**_layout)

    if not MODEL_READY:
        return (
            f"❌ Model not loaded.\n\n{MODEL_ERROR}",
            "", "", empty_fig,
        )

    # Convert blank number inputs (Gradio sends None for empty)
    def to_float(v):
        try:
            return float(v) if v not in (None, "", "None") else None
        except Exception:
            return None

    resolution   = to_float(resolution)
    mol_weight   = to_float(mol_weight)
    density_matt = to_float(density_matt)
    density_sol  = to_float(density_sol)
    ph_value     = to_float(ph_value)

    # Check at least something was provided
    has_seq  = bool((chain_sequences or "").strip())
    has_phys = any(v is not None for v in [resolution, mol_weight, density_matt, density_sol, ph_value])

    if not has_seq and not has_phys:
        return (
            "⚠️ Please provide at least one input field to generate a prediction.",
            "", "", empty_fig,
        )

    # Build feature row
    feature_df, filled, total_grp, provided_labels = build_feature_row(
        chain_sequences, resolution, mol_weight, density_matt, density_sol, ph_value
    )
    pct = completeness_pct(filled, total_grp)

    # Predict
    proba       = MODEL.predict_proba(feature_df)[0]
    top_idx     = np.argsort(proba)[::-1][:TOP_N_PRED]
    top_classes = [LE.classes_[i] for i in top_idx]
    top_probs   = [proba[i]       for i in top_idx]

    top1_class = top_classes[0]
    top1_prob  = top_probs[0]

    # Confidence label
    if top1_prob >= 0.60:
        conf_label = "High"
        conf_color = SUCCESS
    elif top1_prob >= 0.35:
        conf_label = "Medium"
        conf_color = WARNING
    else:
        conf_label = "Low"
        conf_color = "#FF6B6B"

    # Completeness warning
    warn = ""
    if pct < 40:
        warn = (
            "\n\n⚠️  Low data completeness — prediction is based on limited features. "
            "Providing chain_sequences and physicochemical values will improve accuracy."
        )

    # Summary text
    summary = (
        f"🔬 PREDICTION RESULT\n"
        f"{'─' * 40}\n"
        f"  Top Class   : {top1_class}\n"
        f"  Probability : {top1_prob:.1%}\n"
        f"  Confidence  : {conf_label}\n"
        f"{'─' * 40}\n"
        f"  Data completeness : {pct}%\n"
        f"  Inputs used       :\n"
        + "\n".join(f"    • {l}" for l in provided_labels)
        + warn
    )

    # Runner-up table
    runner_up_md = "| Rank | Class | Probability |\n|------|-------|-------------|\n"
    for i, (cl, pr) in enumerate(zip(top_classes, top_probs), 1):
        runner_up_md += f"| {i} | {cl} | {pr:.1%} |\n"

    chart = build_prediction_chart(list(reversed(top_classes)), list(reversed(top_probs)))

    return summary, runner_up_md, f"{pct}%", chart


# ──────────────────────────────────────────────────────────────
#  GRADIO UI
# ──────────────────────────────────────────────────────────────
with gr.Blocks(
    title="🧬 Protein Classifier — Prediction",
    css=CUSTOM_CSS,
    theme=gr.themes.Base(
        primary_hue="teal",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("IBM Plex Mono"),
    ),
) as demo:

    # ── Header ────────────────────────────────────────────────
    gr.HTML(f"""
    <div id="header">
      <div style="position:relative;z-index:1;">
        <div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.5rem;">
          <span style="font-size:2.2rem;">🧬</span>
          <div>
            <h1 style="margin:0;font-size:1.55rem;color:{ACCENT_TEAL};
                        font-family:'IBM Plex Mono',monospace;font-weight:700;
                        letter-spacing:0.04em;">
              PROTEIN STRUCTURE CLASSIFIER
            </h1>
            <p style="margin:0.2rem 0 0;color:{TEXT_DIM};font-size:0.8rem;letter-spacing:0.1em;">
              Random Forest · PDB Dataset · Agentic Training Pipeline
            </p>
          </div>
        </div>
        <p style="margin:0.8rem 0 0;color:#B0CDD8;font-size:0.86rem;max-width:660px;line-height:1.65;">
          Provide whatever protein data you have — nothing is required.
          The more fields you fill in, the more accurate the prediction.
          Chain sequences unlock 27 additional computed features automatically.
        </p>
        {"" if MODEL_READY else f'<div style="margin-top:1rem;padding:0.8rem 1rem;background:#2a0a0a;border:1px solid #ff4444;border-radius:6px;color:#ff8888;font-size:0.8rem;">⚠️ Model not loaded — {MODEL_ERROR}</div>'}
      </div>
    </div>
    """)

    with gr.Row(equal_height=False):

        # ── LEFT: Input Panel ──────────────────────────────────
        with gr.Column(scale=1, min_width=340):

            gr.HTML('<div class="section-title">🧪 Sequence Data</div>')
            chain_sequences = gr.Textbox(
                label='chain_sequences  (dict format)',
                placeholder='{"A": "MKTAYIAKQRQISFVK...", "B": "ACDEFGHIK..."}',
                lines=4,
                info="Paste the chain sequences dict. All 27 sequence features are computed automatically.",
            )

            gr.HTML('<div class="section-title">⚗️ Physicochemical Properties</div>')
            gr.HTML(f'<p style="color:{TEXT_DIM};font-size:0.76rem;margin-bottom:0.6rem;">'
                    f'All fields optional — leave blank if unknown.</p>')

            resolution = gr.Number(
                label="Resolution  (Å)",
                value=None, minimum=0, maximum=10, step=0.01,
                info="X-ray crystallography resolution",
            )
            mol_weight = gr.Number(
                label="Structure Molecular Weight  (Da)",
                value=None, minimum=0, step=100,
                info="Molecular weight of the structure",
            )
            density_matt = gr.Number(
                label="Density Matthews",
                value=None, minimum=0, maximum=10, step=0.01,
                info="Matthews coefficient (Å³/Da)",
            )
            density_sol = gr.Number(
                label="Density Percent Solvent  (%)",
                value=None, minimum=0, maximum=100, step=0.1,
                info="Solvent content percentage",
            )
            ph_value = gr.Number(
                label="pH Value",
                value=None, minimum=0, maximum=14, step=0.1,
                info="Crystallisation pH",
            )

            gr.HTML("<br>")
            predict_btn = gr.Button("🔬  Predict Protein Class", elem_id="predict-btn")

            gr.HTML(f"""
            <div style="margin-top:1.2rem;background:{CARD_BG};border:1px solid {ACCENT_TEAL}33;
                        border-radius:8px;padding:1rem;font-size:0.74rem;color:{TEXT_DIM};line-height:1.8;">
              <strong style="color:{ACCENT_TEAL};">Feature groups used by model:</strong><br>
              &nbsp;📐 Physicochemical (5 fields above)<br>
              &nbsp;🧬 Sequence stats: num_chains, lengths, unique AA count<br>
              &nbsp;🔤 AA frequencies: aa_freq_A … aa_freq_Y (20 features)<br>
              <br>
              <strong style="color:{ACCENT_TEAL};">Model:</strong> Random Forest · 500 trees<br>
              <strong style="color:{ACCENT_TEAL};">Classes:</strong> Top-20 PDB protein types
            </div>
            """)

        # ── RIGHT: Results Panel ───────────────────────────────
        with gr.Column(scale=2):

            with gr.Row():
                completeness_out = gr.Textbox(
                    label="Data Completeness",
                    value="—",
                    interactive=False,
                    scale=1,
                )

            summary_out = gr.Textbox(
                label="Prediction Summary",
                lines=12,
                interactive=False,
                placeholder="Prediction will appear here after you click Predict…",
            )

            with gr.Tabs():
                with gr.Tab("📊 Probability Chart"):
                    chart_out = gr.Plot(label="")

                with gr.Tab("📋 All Top Classes"):
                    table_out = gr.Markdown(
                        value="_Top-5 predictions will appear here…_"
                    )

    # ── Wire up ───────────────────────────────────────────────
    inputs  = [chain_sequences, resolution, mol_weight,
               density_matt, density_sol, ph_value]
    outputs = [summary_out, table_out, completeness_out, chart_out]

    predict_btn.click(fn=predict, inputs=inputs, outputs=outputs)

    # Also predict on Enter in the sequence box
    chain_sequences.submit(fn=predict, inputs=inputs, outputs=outputs)

    # ── Footer ────────────────────────────────────────────────
    gr.HTML(f"""
    <div style="text-align:center;margin-top:2rem;padding-top:1rem;
                border-top:1px solid {ACCENT_TEAL}22;
                color:#3A5A6A;font-size:0.7rem;letter-spacing:0.08em;">
      PROTEIN STRUCTURE CLASSIFIER  ·  Pre-trained Random Forest  ·  PDB Dataset
    </div>
    """)


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )
