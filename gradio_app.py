# 1. Import agent
from transformers import pipeline

chatbot = pipeline("text-generation", model="gpt2")

def ai_agent(user_input):
    # Ensure user_input is a string
    prompt = user_input + ":\n"  # combine prompt elements if needed
    
    output = chatbot(prompt,
                     max_length=100,
                     temperature=0,    # lower for less randomness
                     top_p=0.9,        # nucleus sampling
                     do_sample=True)   # sampling mode
    return output[0]['generated_text']

# 2. Create Gradio Interface
import gradio as gr

iface = gr.Interface(
    fn=ai_agent,              # your function
    inputs=gr.Textbox(label="Enter message"),
    outputs=gr.Textbox(label="AI Response"),
    title="Protein Classification Agent",
    description="Input data for a protein to classify."
)

# 3. Launch the interface
iface.launch(share=True)

# 4. Make the interface a chatbot
def chat_agent(history, user_message):
    response = f"Echo: {user_message}"
    history.append((user_message, response))
    return history

iface = gr.Interface(
    fn=chat_agent,
    inputs=[gr.State([]), gr.Textbox()],
    outputs=gr.Chatbot()
)