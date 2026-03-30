import gradio as gr
from agent import run_agent

# ── Example sequences for the demo dropdown ───────────────────────────────────
EXAMPLES = [
    [
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGEDEDtokenaaagdslntlaleefvdlqnakasegplkvsiyrignklqsrhpqelplmmelqhvdsktlatlmttiagavasgkntdmaaigaalssiqsalanfggkalnaaaggeqkegkkeaakkvqdllkqrdsfdddklrrqlkknakglklefdqsevtfhsgkkdllqkynageafkqnlkqhrsedaallqyqktlkdieatsgndvifgvhsqedrqenalqqrmkqlkqasghiekkqlesygrqkikelfdmdqlkqtdsygrqkikelfdm",
        "This appears to be a large multidomain protein — classify it."
    ],
    [
        "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRDLYDDDDKDPSSEFIAMATKFNHSTFKKAINEAISDQFNKELPTKLLTRLSQKGKSGKGLPVKWFNTNIGELKSKDSFEIWPRQSSKELVEKKFRELLNRLQLNLPNLIENEQSLATNAENLDKQLKQAQVKLKVLASNGEQLEQLNKQLNILQNQLKENQENVTDVQKQLQKENLKENITKINKDLKQKNIQLQNQLKENQENVTDVQ",
        ""
    ],
    [
        "ACDEFGHIKLMNPQRSTVWY",
        "Short test sequence with all amino acids."
    ]
]

# ── Gradio UI layout ──────────────────────────────────────────────────────────

def classify(sequence: str, context: str) -> str:
    """Wrapper called by Gradio — validates input then runs the agent."""
    sequence = sequence.strip().upper()
    # Strip FASTA header if present
    if sequence.startswith(">"):
        lines = sequence.splitlines()
        sequence = "".join(l.strip() for l in lines[1:])

    if len(sequence) < 10:
        return "⚠️ Please enter a protein sequence of at least 10 amino acids."

    invalid = set(sequence) - set("ACDEFGHIKLMNPQRSTVWYBZXU")
    if invalid:
        return f"⚠️ Sequence contains invalid characters: {', '.join(sorted(invalid))}"

    return run_agent(sequence, context)


with gr.Blocks(title="🧬 Protein Classification Agent", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 🧬 Protein Classification Agent
        Paste a protein sequence in **single-letter FASTA format** and the AI agent will:
        - Compute physicochemical properties (MW, pI, GRAVY, instability index…)
        - Query **UniProt** for similar annotated proteins
        - Scan for functional **motifs** (signal peptides, TM helices, zinc fingers…)
        - Synthesize all evidence into a **classification report**
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            seq_input = gr.Textbox(
                label="Protein Sequence (single-letter or FASTA)",
                placeholder=">MyProtein\nMKTAYIAKQR...",
                lines=8,
            )
            ctx_input = gr.Textbox(
                label="Optional Context (organism, known function, source, etc.)",
                placeholder="e.g. Human protein, suspected kinase, found in mitochondria",
                lines=2,
            )
            classify_btn = gr.Button("🔬 Classify Protein", variant="primary")

        with gr.Column(scale=3):
            output = gr.Markdown(label="Classification Report")

    gr.Examples(
        examples=EXAMPLES,
        inputs=[seq_input, ctx_input],
        label="Example Sequences"
    )

    classify_btn.click(
        fn=classify,
        inputs=[seq_input, ctx_input],
        outputs=output,
    )

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",   # accessible on your local network
        server_port=7860,
        share=False,              # set True to get a public Gradio link
    )