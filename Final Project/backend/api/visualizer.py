import torch
from backend.api.model_loader import model_cache



# def generate_response(prompt: str, temperature: float = 0.7, top_k: int = 50):
#     if not model_cache:
#         return {"error": "No model is loaded. Please load a model first."}
    
#     model_name = list(model_cache.keys())[0]  # Assume one model for simplicity
#     model = model_cache[model_name]["model"]
#     tokenizer = model_cache[model_name]["tokenizer"]

#     # torch.manual_seed(2341)

#     inputs = tokenizer(prompt, return_tensors="pt")
#     outputs = model.generate(
#         inputs["input_ids"], 
#         do_sample=True, 
#         temperature=temperature, 
#         top_k=top_k, 
#         max_new_tokens=20,
#         # return_dict_in_generate=True,  # Include this to enable dictionary-style output
#     output_attentions=True
#     )
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return {"response": response}

def generate_response(prompt: str, temperature: float = 0.7, top_k: int = 50, max_new_tokens: int = 20):
    if not model_cache:
        return {"error": "No model is loaded. Please load a model first."}
    
    model_name = list(model_cache.keys())[0]  # Assume one model for simplicity
    model = model_cache[model_name]["model"]
    tokenizer = model_cache[model_name]["tokenizer"]

    # torch.manual_seed(2341)

    # inputs = tokenizer(prompt, return_tensors="pt")
    # outputs = model.generate(
    #     inputs["input_ids"], 
    #     do_sample=True, 
    #     temperature=temperature, 
    #     top_k=top_k, 
    #     max_new_tokens=20,
    #     # return_dict_in_generate=True,  # Include this to enable dictionary-style output
    # output_attentions=True
    # )
    # response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # return {"response": response}

    # Tokenize input with padding
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    
    # Generate response
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],  # Use attention mask for reliable results
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,  # Set pad_token_id explicitly
        eos_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return {"response": response}


def get_token_probabilities(prompt: str):
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


# def get_attention_weights(prompt: str):
#     if not prompt:
#         return {"error": "Prompt cannot be empty"}
#     model_name = list(model_cache.keys())[0]
#     model = model_cache[model_name]["model"]
#     tokenizer = model_cache[model_name]["tokenizer"]
    
#     # Tokenize input
#     inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
#     outputs = model(**inputs, output_attentions=True)
    
#     # Extract attention weights
#     attention = outputs[-1]  # List of tensors (num_layers, batch_size, num_heads, seq_len, seq_len)
#     formatted_attention = [layer.tolist() for layer in attention]  # Remove batch dimension
    
#     # Extract tokens
#     tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

#     print(torch.tensor(formatted_attention).shape)
    
#     return {"attention": formatted_attention, "tokens": tokens}


def get_attention_weights(prompt: str):
    # if not prompt:
    #     return {"error": "Prompt cannot be empty"}
    model_name = list(model_cache.keys())[0]
    model = model_cache[model_name]["model"]
    tokenizer = model_cache[model_name]["tokenizer"]
    
    # # Tokenize input
    # inputs = tokenizer(prompt, return_tensors="pt")
    # outputs = model(**inputs)
    
    # attention = outputs[-1]  # Output includes attention weights when output_attentions=True
    # tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]) 

    # formatted_attention = [layer.tolist() for layer in attention]

    # print(torch.tensor(formatted_attention).shape)

    # return {"attention": formatted_attention, "tokens": tokens}

    if not prompt:
        return {"error": "Prompt cannot be empty"}
    
    from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModel
    
    # tokenizer = AutoTokenizer.from_pretrained('gpt2')
    # model = AutoModel.from_pretrained("distilbert-base-uncased", output_attentions=True)
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model(inputs)
    attention = outputs[-1]  # Output includes attention weights when output_attentions=True
    tokens = tokenizer.convert_ids_to_tokens(inputs[0]) 
    print(torch.tensor([layer.tolist() for layer in attention]).shape)


    formatted_attention = [layer.tolist() for layer in attention]
        
    print(torch.tensor(formatted_attention).shape)



    return {"attention": formatted_attention, "tokens": tokens}


