import torch
from backend.api.model_loader import model_cache

def generate_response(prompt: str, temperature: float = 0.7, top_k: int = 50, max_new_tokens: int = 20):
    """
    Generate a response from the language model based on the input prompt.

    Args:
        prompt (str): The input text to provide to the model.
        temperature (float): Controls the randomness of predictions (default: 0.7).
        top_k (int): Limits the next token selection to the top K most probable tokens (default: 50).
        max_new_tokens (int): Maximum number of tokens to generate (default: 20).

    Returns:
        dict: Contains the generated response as a string or an error message if no model is loaded.
    """
    if not model_cache:
        return {"error": "No model is loaded. Please load a model first."}
    
    model_name = list(model_cache.keys())[0]  # Assume one model for simplicity
    model = model_cache[model_name]["model"]
    tokenizer = model_cache[model_name]["tokenizer"]

    # Tokenize input with padding
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    
    # Generate response
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return {"response": response}


def generate_response_with_log_probs(prompt: str, temperature: float = 0.7, top_k: int = 50, max_new_tokens: int = 20):
    """
    Generate a response and compute the log probabilities for each generated token.

    Args:
        prompt (str): The input text to provide to the model.
        temperature (float): Controls the randomness of predictions (default: 0.7).
        top_k (int): Limits the next token selection to the top K most probable tokens (default: 50).
        max_new_tokens (int): Maximum number of tokens to generate (default: 20).

    Returns:
        dict: Contains the generated response, tokens, and log probabilities for each token, or an error message if no model is loaded.
    """
    if not model_cache:
        return {"error": "No model is loaded. Please load a model first."}

    model_name = list(model_cache.keys())[0]
    model = model_cache[model_name]["model"]
    tokenizer = model_cache[model_name]["tokenizer"]

    # Tokenize input with padding
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    # Generate response with output logits
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Decode generated tokens
    generated_tokens = outputs.sequences[0]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Extract logits and compute probabilities for each token
    logits = torch.stack(outputs.scores, dim=0)
    probabilities = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probabilities)

    # Align generated tokens with logits
    generated_token_ids = generated_tokens[1:]  # Exclude the first input token
    log_probs = log_probs[:-1]  # Exclude the final logits since they are unused
    token_log_probs = log_probs.gather(index=generated_token_ids.unsqueeze(-1), dim=-1).squeeze(-1)

    # Convert log probs and tokens to a readable format
    tokens = tokenizer.convert_ids_to_tokens(generated_token_ids)
    token_log_probs = token_log_probs.tolist()

    return {
        "response": response,
        "tokens": tokens,
        "log_probs": token_log_probs,
    }


def get_token_probabilities(prompt: str):
    """
    Compute the token probabilities for the given prompt.

    Args:
        prompt (str): The input text to provide to the model.

    Returns:
        dict: Contains token probabilities as a list or an error message if the prompt is empty.
    """
    if not prompt:
        return {"error": "Prompt cannot be empty"}
    model_name = list(model_cache.keys())[0]
    model = model_cache[model_name]["model"]
    tokenizer = model_cache[model_name]["tokenizer"]
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model(**inputs, output_attentions=False, output_hidden_states=False)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)
    return {"probabilities": probabilities.tolist()}


def get_attention_weights(prompt: str):
    """
    Retrieve attention weights from the model for the given prompt.

    Args:
        prompt (str): The input text to provide to the model.

    Returns:
        dict: Contains the attention weights and corresponding tokens, or an error message if the prompt is empty.
    """
    if not prompt:
        return {"error": "Prompt cannot be empty"}

    model_name = list(model_cache.keys())[0]
    model = model_cache[model_name]["model"]
    tokenizer = model_cache[model_name]["tokenizer"]

    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model(inputs)
    attention = outputs[-1]  # Output includes attention weights when output_attentions=True
    tokens = tokenizer.convert_ids_to_tokens(inputs[0]) 

    # Format attention as a list for further processing
    formatted_attention = [layer.tolist() for layer in attention]
    
    return {"attention": formatted_attention, "tokens": tokens}
