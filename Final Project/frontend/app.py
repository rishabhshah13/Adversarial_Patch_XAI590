import streamlit as st
import requests
import plotly.graph_objects as go
import json
import pandas as pd
from io import BytesIO
from bertviz import head_view
import numpy as np
import torch
import os

# Backend API URL
API_BASE_URL = "http://127.0.0.1:8000"

# Page Configuration
st.set_page_config(
    page_title="LLM Response Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state variables
if "response" not in st.session_state:
    st.session_state.response = None
if "response1" not in st.session_state:
    st.session_state.response1 = None
if "response2" not in st.session_state:
    st.session_state.response2 = None
if "prob1" not in st.session_state:
    st.session_state.prob1 = None
if "prob2" not in st.session_state:
    st.session_state.prob2 = None

# Sidebar for Model Selection
st.sidebar.title("🔧 Model Configuration")
with st.sidebar.expander("What is Model Configuration?", expanded=False):
    st.markdown("""
    Configure the parameters for your language model here. Adjust options like:
    - **Model Name**: Specify the model you want to use (e.g., GPT-2, GPT-3).
    - **Temperature**: Controls randomness. Lower values make responses more deterministic, while higher values increase randomness.
    - **Top-K Sampling**: Limits the predictions to the top K most likely tokens, ensuring more focused outputs.
    - **Max New Tokens**: Sets the maximum number of tokens to generate in the response.
    """)

model_name = st.sidebar.text_input("Model Name", value="gpt2-xl")
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.01, step=0.01)
top_k = st.sidebar.slider("Top-K Sampling", min_value=1, max_value=100, value=2, step=1)
max_new_tokens = st.sidebar.slider("Max New Tokens", min_value=1, max_value=100, value=10, step=1)

if st.sidebar.button("Load Model"):
    payload = {"model_name": model_name}
    response = requests.post(f"{API_BASE_URL}/load_model", json=payload)
    if response.status_code == 200:
        st.sidebar.success("✅ Model loaded successfully!")
    else:
        st.sidebar.error(f"❌ Error: {response.status_code} - {response.json().get('detail', 'Unknown error')}")

# Main Input Section
st.title("📊 LLM Response Analyzer")
with st.expander("What is LLM Response Analyzer?", expanded=False):
    st.markdown("""
    This tool allows you to:
    - Generate and Analyzer responses from a large language model (LLM) based on your input prompts.
    - Experiment with different sampling parameters to see how they affect the response.
    """)

prompt = st.text_area("Enter your prompt:", placeholder="Type something here...", value="What comes after night? Answer is:")

if st.button("Generate Response"):
    payload = {
        "prompt": prompt,
        "temperature": temperature,
        "top_k": top_k,
        "max_new_tokens": max_new_tokens,
    }
    response = requests.post(f"{API_BASE_URL}/generate_response", json=payload)
    if response.status_code == 200:
        st.session_state.response = response.json()["response"]
        st.success("✅ Response generated successfully!")
    else:
        st.error(f"❌ Error: {response.status_code} - {response.json().get('detail', 'Unknown error')}")

if st.session_state.response:
    st.subheader("📝 Generated Response:")
    # with st.expander("What does Generated Response mean?", expanded=False):
    #     st.markdown("""
    #     This is the text generated by the LLM based on your input prompt. It is influenced by the selected model configuration parameters.
    #     """)
    # st.markdown(f"```{st.session_state.response}```")
    st.write(f"{st.session_state.response}")

st.subheader("⚖️ Compare Responses with Different Parameters")
with st.expander("What is Comparative Analysis?", expanded=False):
    st.markdown("""
    ### Comparative Analysis: Exploring Model Behavior

    Imagine you’re cooking the same dish but using different recipes. Each recipe changes specific ingredients (parameters) to influence the final taste (output). Similarly, **Comparative Analysis** lets you tweak the model's parameters and observe how it alters the generated response.

    #### Why is it Important?
    By comparing responses with different configurations, you can:
    - Understand the impact of randomness (**Temperature**).
    - Observe how tightly the model focuses on likely tokens (**Top-K Sampling**).
    - Experiment with the verbosity of responses (**Max New Tokens**).

    ---
    #### 🔧 Parameters Explained with Examples:
    **1. Temperature:** 
    - Think of temperature as the level of creativity or randomness in the response.  
    - **Example:**
      - A low value (e.g., `0.2`) produces focused and deterministic outputs.  
        _Prompt:_ "What is AI?"  
        _Response:_ "AI is artificial intelligence."  
      - A high value (e.g., `1.0`) introduces more creativity.  
        _Prompt:_ "What is AI?"  
        _Response:_ "AI is like a digital mind, thinking faster than any human."

    **2. Top-K Sampling:** 
    - Picture this as the model picking its next word from a bag of the top K most likely choices.  
    - **Example:**  
      - **Top-K = 5:** The model selects words only from the top 5 most probable tokens.  
        _Prompt:_ "The dog is very..."  
        _Response:_ "...friendly and playful."  
      - **Top-K = 50:** The model has a much larger pool to choose from, leading to more variability.  
        _Prompt:_ "The dog is very..."  
        _Response:_ "...excited about the strange possibilities of life."

    **3. Max New Tokens:**  
    - This controls how long the generated response can be.  
    - **Example:**  
      - **Max New Tokens = 10:**  
        _Prompt:_ "Once upon a time,"  
        _Response:_ "...there was a king who ruled the land."  
      - **Max New Tokens = 50:**  
        _Prompt:_ "Once upon a time,"  
        _Response:_ "...there was a king who ruled the land, known for his wisdom and compassion. His kingdom prospered under his reign, with people living in harmony."

    ---
    #### ✨ Example Analysis:
    - **Configuration 1:**  
      - Temperature = 0.7, Top-K = 50, Max New Tokens = 20  
      - _Prompt:_ "The universe is vast and mysterious."  
      - _Response:_ "It holds secrets that humanity has yet to uncover, with endless possibilities."

    - **Configuration 2:**  
      - Temperature = 1.0, Top-K = 30, Max New Tokens = 30  
      - _Prompt:_ "The universe is vast and mysterious."  
      - _Response:_ "It’s a place of infinite wonder, where galaxies dance in the fabric of time."

    ---
    #### 🧪 What to Do Here:
    1. Set up two configurations using the sliders below.  
    2. Click "Generate for Config 1" and "Generate for Config 2" to see how the model responds differently.  
    3. Compare the outputs side by side and analyze how your parameter tweaks influenced the results.
    """)


col1, col2 = st.columns(2)

with col1:
    st.write("### Configuration 1")
    temp1 = st.slider("Temperature (Config 1)", 0.0, 1.0, 0.7, 0.01, key="temp1")
    top_k1 = st.slider("Top-K (Config 1)", 1, 100, 10, 1, key="top_k1")
    max_new_tokens1 = st.slider("Max New Tokens (Config 1)", 1, 100, 20, 1, key="max_new_tokens1")
    if st.button("Generate for Config 1"):
        payload1 = {"prompt": prompt, "temperature": temp1, "top_k": top_k1, "max_new_tokens": max_new_tokens1}
        response1 = requests.post(f"{API_BASE_URL}/generate_response", json=payload1)
        if response1.status_code == 200:
            st.session_state.response1 = response1.json()["response"]

with col2:
    st.write("### Configuration 2")
    temp2 = st.slider("Temperature (Config 2)", 0.0, 1.0, 0.2, 0.01, key="temp2")
    top_k2 = st.slider("Top-K (Config 2)", 1, 100, 50, 1, key="top_k2")
    max_new_tokens2 = st.slider("Max New Tokens (Config 2)", 1, 100, 20, 1, key="max_new_tokens2")
    if st.button("Generate for Config 2"):
        payload2 = {"prompt": prompt, "temperature": temp2, "top_k": top_k2, "max_new_tokens": max_new_tokens2}
        response2 = requests.post(f"{API_BASE_URL}/generate_response", json=payload2)
        if response2.status_code == 200:
            st.session_state.response2 = response2.json()["response"]

if st.session_state.response1 or st.session_state.response2:
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Response 1:")
        if st.session_state.response1:
            # st.markdown(f"```{st.session_state.response1}```")
            st.write(f"{st.session_state.response1}")
    with col2:
        st.write("#### Response 2:")
        if st.session_state.response2:
            # st.markdown(f"```{st.session_state.response2}```")
            st.write(f"{st.session_state.response2}")

# Attention Heatmap Section
st.subheader("🌐 Attention Heatmap Visualization")
with st.expander("What is Attention Heatmap?", expanded=False):
    st.markdown("""
    ### Understanding Attention Heatmaps: Unveiling the Model's Focus 🎯

    Imagine you're reading a story. As you encounter each word, your mind subconsciously connects it with earlier words to understand the sentence better. Similarly, language models like GPT-2 use **attention mechanisms** to decide which parts of the input are most relevant when generating the next token.

    ---
    #### Why Use Attention Heatmaps? 🧐
    Attention heatmaps help us:
    - Visualize which tokens (words or subwords) the model focuses on.
    - Interpret the reasoning behind a model's generated response.
    - Gain trust and transparency into the "black box" nature of language models.

    ---
    #### 🔍 Example:
    **Input Prompt:** "The quick brown fox jumps over the lazy dog."

    **Attention Heatmap Output:**
    - When predicting "jumps," the model might focus heavily on "fox."
    - For "lazy," it might attend more to "dog."

    These attention patterns highlight how the model connects tokens to make sense of the input.

    ---
    #### How Does This Work?
    - We use **BertViz**, an interactive tool, to visualize attention layers in GPT-2.
    - Each layer consists of multiple "attention heads," which focus on different aspects of the input.

    **Key Views in BertViz:**
    1. **Head View:** Displays attention patterns for each head in a specific layer.
    2. **Model View:** Shows how attention flows across all layers.
    3. **Neuron View:** Provides fine-grained insights into individual neurons.

    ---
    #### What Can You Do Here?
    1. Enter a prompt in the text box above.
    2. Click "Visualize Attention."
    3. Explore how the model distributes its focus using BertViz.

    **Example Prompt:**  
    _"In a galaxy far, far away, a young Jedi was training."_  
    - Observe which tokens influence "Jedi" (e.g., "galaxy" or "young").
    """)


if st.button("Visualize Attention"):
    payload = {"prompt": prompt}
    attention_response = requests.post(f"{API_BASE_URL}/attention_weights", json=payload)
    if attention_response.status_code == 200:
        attention_data = attention_response.json()
        attention = torch.tensor(attention_data["attention"])
        tokens = attention_data["tokens"]

        html_head_view = head_view(attention, tokens, html_action="return")
        output_path = "frontend/outputs/head_view.html"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as file:
            file.write(html_head_view.data)

        with open(output_path, "r") as f:
            html_data = f.read()
        st.components.v1.html(html_data, height=400, scrolling=True)

# LLM Steering Section
st.subheader("🎛️ LLM Steering")
with st.expander("What is LLM Steering?", expanded=False):
    st.markdown("""
    ### Steering the Model’s Output: Influencing the AI’s Thought Process 🧠

    **LLM Steering** allows us to modify the internal activations of a language model to influence its output in a controlled way. Think of it as nudging the model's "thinking" in a desired direction without changing the input prompt.

    ---
    #### How Does It Work? 🛠️
    At a high level:
    1. **Select a Layer:** Choose which part of the model (e.g., a specific transformer layer) to intervene in.
    2. **Define a Steering Vector:** Create a vector that represents the "direction" you want to steer the model.  
       - For example, the vector can encode the difference between "Love" and "Hate."
    3. **Modify Activations:** Add this vector to the model’s internal activations during the forward pass to influence its output.

    ---
    #### 📖 Example: Steering Sentiment
    - Input Prompt: `"I think dogs are"`
    - Steering Vector: The difference between the activations for `"Love"` and `"Hate"`  
    - **Positive Steering:** Add the "Love" direction to the activations.  
      _Output:_ `"I think dogs are wonderful companions."`  
    - **Negative Steering:** Add the "Hate" direction to the activations.  
      _Output:_ `"I think dogs are the worst pets ever."`

    ---
    #### Why is LLM Steering Useful? 🔍
    - **Controllable Output:** Fine-tune the tone or style of the model's response (e.g., formal, humorous).
    - **Exploration:** Study how specific concepts (e.g., "Love") are represented internally by the model.
    - **Applications:** Useful in scenarios like sentiment control, bias mitigation, and creative writing.

    ---
    #### How Do We Achieve This? 🚀
    - **TransformerLens Library:** A popular library that makes it easy to perform activation steering.  
    - **PyTorch Hooks:** Low-level functions to modify the model's behavior directly.  
    - **Wrapper Modules:** Custom code that adds the steering functionality to a specific layer.

    ---
    #### Ready to Try? Here’s What to Do:
    1. Select a steering option (e.g., `"happy-sad"`) from the dropdown menu below.
    2. Enter a prompt like `"The movie was"`.
    3. Click the "Generate Steering Output" button to see:
       - A positive response (e.g., `"The movie was delightful and uplifting."`)
       - A negative response (e.g., `"The movie was disappointing and dull."`)
       - A neutral response (e.g., `"The movie was directed by a talented filmmaker."`)

    ---
    #### Additional Resources:
    - **[Activation Steering Blog Post](https://www.lesswrong.com/posts/ndyngghzFY388Dnew/implementing-activation-steering#fn0j9pmu8ahchs):** A comprehensive guide to activation steering.
    - **[TransformerLens GitHub](https://github.com/neelnanda-io/TransformerLens):** Tools for studying and steering transformer models.
    """)


steering_option = st.selectbox("Choose a steering option:", ["happy-sad", "good-bad", "love-hate"])
steering_prompt = st.text_area("Enter your prompt for steering:", value="I think dogs are ")
layer_num = st.text_area("Layer Number", value=5)

if st.button("Generate Steering Output"):
    payload = {"prompt": steering_prompt, "steering_option": steering_option, "layer_num": int(layer_num)}
    response = requests.post(f"{API_BASE_URL}/llm_steering", json=payload)

    if response.status_code == 200:
        results = response.json()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("#### Positive Steering:")
            st.markdown(f"```{results['positive']}```")
        with col2:
            st.write("#### Negative Steering:")
            st.markdown(f"```{results['negative']}```")
        with col3:
            st.write("#### Neutral Output:")
            st.markdown(f"```{results['neutral']}```")
    else:
        st.error(f"❌ Error: {response.status_code} - {response.json().get('detail', 'Unknown error')}")