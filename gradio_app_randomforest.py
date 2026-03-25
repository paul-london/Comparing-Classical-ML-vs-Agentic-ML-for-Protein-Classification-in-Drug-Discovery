
# -----------------------------
# 1. Load model
# -----------------------------
import pickle

with open("saved_models/RandomForest_optuna_3-20-26.pkl", "rb") as f:
    model = pickle.load(f)
    
with open("saved_models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -----------------------------
# 2. Preprocess data
# -----------------------------
import numpy as np

def preprocess(sequence, resolution, mw, density, percent_sol, ph):
    length = len(sequence)
    if length == 0:
        raise ValueError("Protein sequence cannot be empty.")
    
    # Biochemical fractions
    AROMATIC = "FYW"
    NONPOLAR = "GAVLIMP"
    POLAR = "STCNQ"
    ACIDIC = "DE"
    BASIC = "KRH"
    AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

    frac_aromatic = sum(sequence.count(aa) for aa in AROMATIC) / length
    frac_nonpolar = sum(sequence.count(aa) for aa in NONPOLAR) / length
    frac_polar = sum(sequence.count(aa) for aa in POLAR) / length
    frac_acidic = sum(sequence.count(aa) for aa in ACIDIC) / length
    frac_basic = sum(sequence.count(aa) for aa in BASIC) / length

    aa_fracs = [sequence.count(aa) / length for aa in AMINO_ACIDS]

    # Combine raw features in the same order as training
    raw_features = [
        resolution,
        mw,
        density,
        percent_sol,
        ph,
        length,
        frac_aromatic,
        frac_nonpolar,
        frac_polar,
        frac_acidic,
        frac_basic,
        *aa_fracs
    ]

    raw_features = np.array(raw_features)  # shape (n_features,) - sequence length wasn't scaled during training, so we keep it as is
    
    # Scale only the first 5 numeric features
    raw_features[:5] = scaler.transform([raw_features[:5]])[0]
    
    # The rest (length + fractions) are already scaled/normalized if needed
    features_scaled = raw_features.reshape(1, -1)  # shape (1, n_features)
    
    return features_scaled

# -----------------------------
# 3. Create AI agent
# -----------------------------
def ai_agent(sequence, resolution, mw, density, percent_sol, ph):
    try:
        features = preprocess(sequence, resolution, mw, density, percent_sol, ph)
        
        # Predict class
        prediction = model.predict(features)[0]
        
        # Predict probabilities for all classes
        classes = model.classes_
        proba_values = model.predict_proba(features)[0]
        proba_dict = {cls: f"{prob:.2f}" for cls, prob in zip(classes, proba_values)}
        proba_str = "\n".join([f"{cls}: {prob}" for cls, prob in proba_dict.items()])
        
        return f"""
🧬 Prediction: {prediction}
📊 Confidence (max class): {max(proba_values):.2f}

Class Probabilities:
{proba_str}

Length: {len(sequence)}
Resolution: {resolution}
pH: {ph}
Molecular Weight: {mw}
Density: {density}
Percent Solvent: {percent_sol}
"""
    except Exception as e:
        return f"Error: {str(e)}"

# -----------------------------
# 4. Gradio UI
# -----------------------------
import gradio as gr

demo = gr.Interface(
    fn=ai_agent,
    inputs=[
        gr.Textbox(label="Protein Sequence"),
        gr.Number(label="Resolution"),
        gr.Number(label="Molecular Weight"),
        gr.Number(label="Density (Matthews)"),
        gr.Number(label="Percent Soluble"),
        gr.Number(label="pH Value")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Protein Classification Agent",
    description="Enter protein sequence and experimental conditions."
)

demo.launch(share=True)