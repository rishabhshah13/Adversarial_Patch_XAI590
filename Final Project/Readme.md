# LLM Response Visualizer

# TO BE ADDED SOON
[Final Project Ideation Video](https://youtu.be/VmPNgspBAeU)

## Project Overview

This tool generates multiple responses from a Large Language Model (LLM) for a single input question using various parameter combinations (temperature, top-K, etc.). It then visualizes the log probabilities for each response, allowing users to explore how different parameters affect the model's output, and token probabilities.

## Features

- Generate multiple LLM responses with varying parameters
- Display log probabilities for generated tokens
- Compare outputs, visualizations, and probabilities across different parameter settings
- Interactive user interface for easy exploration

## Tools and Technologies

- Python 3.8+
- Hugging Face Transformers (for LLM implementation)
- LangChain (for handling log probabilities)
- BertViz (for attention visualization)
- Streamlit (for web interface)
- Matplotlib/Plotly (for additional visualizations)

## Key Components

- LLM Integration: Utilizes LangChain's capabilities to interact with LLMs and retrieve log probabilities. (See: [LangChain Log Probabilities Guide](https://python.langchain.com/docs/how_to/logprobs/))
- Visualization: Inspired by interactive visualization tools like [Perplexity](https://perplexity.vercel.app/), which demonstrates effective ways to visualize language model behavior.
- Interpretability Insights: Draws inspiration from [Neuronpedia's Gemma Scope](https://www.neuronpedia.org/gemma-scope#main), which provides detailed visualizations and explanations of the inner workings of language models like Gemma. While our project focuses on a different model and usecase, the principles and visualization techniques demonstrated in Gemma Scope have greatly influenced the approach to making LLMs more interpretable and transparent.
