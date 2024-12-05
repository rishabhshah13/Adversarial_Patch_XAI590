import streamlit as st
import requests
import plotly.graph_objects as go
import json
import pandas as pd
from io import BytesIO
from bertviz import head_view
import numpy as np
import torch

# Backend API URL
API_BASE_URL = "http://127.0.0.1:8000"

# Page Configuration
st.set_page_config(
    page_title="LLM Response Visualizer",
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
st.sidebar.title("Model Configuration")
model_name = st.sidebar.text_input("Model Name", value="gpt2-xl")
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
top_k = st.sidebar.slider("Top-K Sampling", min_value=1, max_value=100, value=50, step=1)
max_new_tokens = st.sidebar.slider("Max New Tokens", min_value=1, max_value=100, value=20, step=1)

# Load Model Button
if st.sidebar.button("Load Model"):
    payload = {"model_name": model_name}
    response = requests.post(f"{API_BASE_URL}/load_model", json=payload)
    if response.status_code == 200:
        st.sidebar.write(response.json())
    else:
        st.sidebar.error(f"Error: {response.status_code} - {response.json().get('detail', 'Unknown error')}")

# Main Input Section
st.title("LLM Response Visualizer")
st.write("Experiment with LLM responses and visualize token probabilities.")

# Input Text Area
# prompt = st.text_area("Enter your prompt:", placeholder="Type something here...", value="Machine learning is great for humanity. It helps a lot of people.")

prompt = st.text_area("Enter your prompt:", placeholder="Type something here...", value="Q: What comes after night? Answer: ")
# prompt = st.text_area("Enter your prompt:", placeholder="Type something here...", value="What comes after night? The answer is: ")



# Generate Response Button
if st.button("Generate Response"):
    payload = {
        "prompt": prompt,
        "temperature": temperature,
        "top_k": top_k,
        "max_new_tokens": max_new_tokens
    }
    response = requests.post(f"{API_BASE_URL}/generate_response", json=payload)
    if response.status_code == 200:
        st.session_state.response = response.json()["response"]
    else:
        st.error(f"Error: {response.status_code} - {response.json().get('detail', 'Unknown error')}")

# Display Generated Response
if st.session_state.response:
    st.subheader("Generated Response:")
    st.write(st.session_state.response)

# Comparative Analysis Section
st.subheader("Compare Responses with Different Parameters")

col1, col2 = st.columns(2)

# Left Column: First configuration
with col1:
    st.write("**Configuration 1**")
    temp1 = st.slider("Temperature (Config 1)", 0.0, 1.0, 0.7, 0.01, key="temp1")
    top_k1 = st.slider("Top-K (Config 1)", 1, 100, 50, 1, key="top_k1")
    max_new_tokens1 = st.slider("Max New Tokens (Config 1)", 1, 100, 20, 1, key="max_new_tokens1")
    if st.button("Generate for Config 1"):
        payload1 = {"prompt": prompt, "temperature": temp1, "top_k": top_k1, "max_new_tokens": max_new_tokens1}
        response1 = requests.post(f"{API_BASE_URL}/generate_response", json=payload1)
        if response1.status_code == 200:
            st.session_state.response1 = response1.json()["response"]
            # st.write("**Response 1:**", st.session_state.response1)


# Right Column: Second configuration
with col2:
    st.write("**Configuration 2**")
    temp2 = st.slider("Temperature (Config 2)", 0.0, 1.0, 0.7, 0.01, key="temp2")
    top_k2 = st.slider("Top-K (Config 2)", 1, 100, 50, 1, key="top_k2")
    max_new_tokens2 = st.slider("Max New Tokens (Config 2)", 1, 100, 20, 1, key="max_new_tokens2")
    if st.button("Generate for Config 2"):
        payload2 = {"prompt": prompt, "temperature": temp2, "top_k": top_k2, "max_new_tokens": max_new_tokens2}
        response2 = requests.post(f"{API_BASE_URL}/generate_response", json=payload2)
        if response2.status_code == 200:
            st.session_state.response2 = response2.json()["response"]
            # st.write("**Response 2:**", st.session_state.response2)


# Display Responses for Configurations
if st.session_state.response1:
    with col1:
        st.write("**Response 1:**", st.session_state.response1)
if st.session_state.response2:
    with col2:
        st.write("**Response 2:**", st.session_state.response2)



# # Comparative Visualization
# if st.session_state.response1 and st.session_state.response2:
#     payload1 = {"prompt": prompt}
#     payload2 = {"prompt": prompt}

#     prob1_response = requests.post(f"{API_BASE_URL}/token_probabilities", json=payload1)
#     prob2_response = requests.post(f"{API_BASE_URL}/token_probabilities", json=payload2)

#     if prob1_response.status_code == 200:
#         st.session_state.prob1 = prob1_response.json()["probabilities"]
#     if prob2_response.status_code == 200:
#         st.session_state.prob2 = prob2_response.json()["probabilities"]

#     if st.session_state.prob1 and st.session_state.prob2:
#         fig = go.Figure()
#         tokens = [f"Token {i}" for i in range(len(st.session_state.prob1))]
#         fig.add_trace(go.Bar(x=tokens, y=st.session_state.prob1, name="Config 1"))
#         fig.add_trace(go.Bar(x=tokens, y=st.session_state.prob2, name="Config 2"))
#         fig.update_layout(
#             title="Comparative Token Probabilities",
#             xaxis_title="Tokens",
#             yaxis_title="Probability",
#             barmode="group",
#             template="plotly_white"
#         )
#         st.plotly_chart(fig, use_container_width=True)


st.subheader("Attention Heatmap Visualization")

if st.button("Visualize Attention"):
    payload = {"prompt": prompt}
    st.write(payload)
    attention_response = requests.post(f"{API_BASE_URL}/attention_weights", json=payload)

    if attention_response.status_code == 200:
        attention_data = attention_response.json()
        attention = torch.tensor(attention_data["attention"])  # Convert to NumPy array
        tokens = attention_data["tokens"]

        # st.write("Attention tensor shape:", attention.shape)

        # # Expected axes for 4D tensors: (batch_size, seq_len, num_heads, seq_len)
        # # Debug: Check the attention tensor shape
        # print("Attention tensor shape before processing:", attention.shape)

        # if len(attention.shape) == 5:  # (num_layers, batch_size, num_heads, seq_len, seq_len)
        #     formatted_attention = []
        #     for layer_idx, layer_attention in enumerate(attention):
        #         print(f"Processing layer {layer_idx} attention shape: {layer_attention.shape}")
        #         # Correct transposition for 4D tensor
        #         transposed_attention = np.transpose(layer_attention, (0, 2, 1, 3))  # (batch_size, seq_len, num_heads, seq_len)
        #         formatted_attention.append(transposed_attention)
        #         print(f"Layer {layer_idx} transposed shape: {transposed_attention.shape}")
        #     print("All layers processed. Ready for visualization.")
        # else:
        #     raise ValueError(f"Unexpected attention tensor shape: {attention.shape}")

        # Adjust attention shape to match bertviz input requirements
        # attention_tensor = np.transpose(formatted_attention, (0, 2, 1, 3))


        # Display the heatmap using bertviz's head_view
        st.write("**Attention Heatmap:**")
        # head_view(attention_tensor, tokens, tokens)
        st.write(attention.shape)
        # head_view(attention, tokens)

        import streamlit.components.v1 as components
        import os
        # Generate HTML from head_view
        html_head_view = head_view(attention, tokens, html_action="return")
        output_path = "frontend/outputs/head_view.html"

        st.write(tokens)
        # Save HTML file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as file:
            file.write(html_head_view.data)

        # Display HTML file in Streamlit
        st.write("**Attention Heatmap:**")
        with open(output_path,'r') as f:
            html_data = f.read()

        components.html(html_data, height=400, scrolling=True)

    else:
        st.error(f"Error: {attention_response.status_code} - {attention_response.json().get('detail', 'Unknown error')}")


# LLM Steering Section
st.subheader("LLM Steering")

# Dropdown for Steering Options
steering_option = st.selectbox(
    "Choose a steering option:",
    ["happy-sad", "good-bad", "love-hate"],
    index=0  # Default to the first option
)

# Input Text Area
steering_prompt = st.text_area("Enter your prompt for steering:", placeholder="Type something here...",value="The movie was")

# Button to Trigger Steering
if st.button("Generate Steering Output"):
    payload = {
        "prompt": steering_prompt,
        "steering_option": steering_option
    }
    response = requests.post(f"{API_BASE_URL}/llm_steering", json=payload)

    if response.status_code == 200:
        results = response.json()

        # Display Outputs Side by Side
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Positive Steering:**")
            st.write(results["positive"])
        with col2:
            st.write("**Negative Steering:**")
            st.write(results["negative"])
        with col3:
            st.write("**Neutral Output:**")
            st.write(results["neutral"])
    else:
        st.error(f"Error: {response.status_code} - {response.json().get('detail', 'Unknown error')}")



# # Export Results Section
# st.subheader("Export Results")

# # Export responses and probabilities
# export_data = {}

# if st.session_state.response1 is not None and st.session_state.response2 is not None:
#     export_data = {
#         "Prompt": prompt,
#         "Config 1": {
#             "Temperature": temp1,
#             "Top-K": top_k1,
#             "Response": st.session_state.response1,
#             "Probabilities": st.session_state.prob1,
#         },
#         "Config 2": {
#             "Temperature": temp2,
#             "Top-K": top_k2,
#             "Response": st.session_state.response2,
#             "Probabilities": st.session_state.prob2,
#         }
#     }

#     # Export as JSON
#     json_data = json.dumps(export_data, indent=4)
#     st.download_button(
#         label="Download as JSON",
#         data=json_data,
#         file_name="response_analysis.json",
#         mime="application/json"
#     )

#     # Export as CSV
#     flat_data = []
#     for config, details in export_data.items():
#         if config != "Prompt":
#             flat_data.append({
#                 "Prompt": prompt,
#                 "Configuration": config,
#                 "Temperature": details["Temperature"],
#                 "Top-K": details["Top-K"],
#                 "Response": details["Response"]
#             })
#     csv_data = pd.DataFrame(flat_data).to_csv(index=False).encode('utf-8')
#     st.download_button(
#         label="Download as CSV",
#         data=csv_data,
#         file_name="response_analysis.csv",
#         mime="text/csv"
#     )
# else:
#     st.write("Generate responses to enable export functionality.")
