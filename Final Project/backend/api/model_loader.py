import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import pipeline, set_seed
# Cache to store loaded models
model_cache = {}

# def load_model(model_name: str):
#     try:
#         if model_name in model_cache:
#             return {"status": "Model already loaded", "model_name": model_name}
        
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)  # Enable attention outputs
#         model_cache[model_name] = {"model": model, "tokenizer": tokenizer}

#         return {"status": "Model loaded successfully", "model_name": model_name}
#     except Exception as e:
#         return {"error": str(e)}
    


def load_model(model_name: str):
    try:
        if model_name in model_cache:
            return {"status": "Model already loaded", "model_name": model_name}
        
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        # model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)  # Enable attention outputs
        set_seed(42)

        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name, output_attentions=True)

        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

        model_cache[model_name] = {"model": model, "tokenizer": tokenizer}

        return {"status": "Model loaded successfully", "model_name": model_name}
    except Exception as e:
        return {"error": str(e)}