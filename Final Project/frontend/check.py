import streamlit.components.v1 as components
import streamlit as st
# components.iframe(
#     src="/Users/rishabhshah/Desktop/XAI/LLM_Response_Visualizer/frontend/outputs/head_view.html",  # URL of the page to embed
#     width=700,                 # Width of the iframe in pixels
#     height=500,                # Height of the iframe in pixels
#     scrolling=True             # Enable scrolling within the iframe
# )


import streamlit as st
import streamlit.components.v1 as components


with open("/Users/rishabhshah/Desktop/XAI/LLM_Response_Visualizer/frontend/outputs/head_view.html",'r') as f: 
    html_data = f.read()

components.html(html_data, height=600)

