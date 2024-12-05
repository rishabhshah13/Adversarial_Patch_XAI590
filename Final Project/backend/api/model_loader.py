import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import pipeline, set_seed

# Cache to store loaded models
model_cache = {}

def load_model(model_name: str):
    """
    Load a specified language model and its tokenizer into memory.

    Args:
        model_name (str): The name of the model to be loaded, e.g., "gpt2", "gpt2-xl".

    Returns:
        dict: A dictionary containing the status of the model loading process or an error message if the loading fails.
            - "status": A message indicating success or if the model is already loaded.
            - "model_name": The name of the loaded model.
            - "error": A description of the error if the loading fails.
    """
    try:
        # Check if the model is already loaded in the cache
        if model_name in model_cache:
            return {"status": "Model already loaded", "model_name": model_name}
        
        # Set random seed for reproducibility
        set_seed(42)

        # Load tokenizer and model with attention outputs enabled
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name, output_attentions=True)

        # Set padding token to the end-of-sequence token
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

        # Cache the loaded model and tokenizer for future use
        model_cache[model_name] = {"model": model, "tokenizer": tokenizer}

        return {"status": "Model loaded successfully", "model_name": model_name}
    except Exception as e:
        # Return error details in case of an exception
        return {"error": str(e)}
