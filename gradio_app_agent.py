# -----------------------------
# 1. Agent
# -----------------------------
from transformers import pipeline

chatbot = pipeline("text-generation", model="gpt2")

def ai_agent(user_input):
    prompt = user_input + ":\n"

    output = chatbot(
        prompt,
        max_length=100,
        temperature=0,
        top_p=0.9,
        do_sample=True
    )

    return output[0]['generated_text']


# -----------------------------
# 2. Chat wrapper
# -----------------------------
def chat_agent(message, history):
    response = ai_agent(message)
    return response


# -----------------------------
# 3. Gradio Chat UI
# -----------------------------
import gradio as gr

demo = gr.ChatInterface(
    fn=chat_agent,
    title="Protein Classification Agent",
    description="Provide protein data for classification."
)

demo.launch(share=True)