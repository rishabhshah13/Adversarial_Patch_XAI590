from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.api.model_loader import load_model
from transformer_lens import HookedTransformer
from backend.api.visualizer import generate_response, get_token_probabilities, get_attention_weights
import torch

router = APIRouter()

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and cache the model globally
model = HookedTransformer.from_pretrained_no_processing("gpt2-xl", default_prepend_bos=False).to(device).eval()

# Schema for /llm_steering
class SteeringRequest(BaseModel):
    prompt: str
    steering_option: str

@router.post("/llm_steering")
def llm_steering_route(request: SteeringRequest):

    try:
        layer_id = 5
        cache_name = f"blocks.{layer_id}.hook_resid_post" # we do activation steering on the activation (the output) of the residual layer

        # Precompute steering vectors based on option
        if request.steering_option == "happy-sad":
            _, cache = model.run_with_cache("Happy")
            act_positive = cache[cache_name]
            _, cache = model.run_with_cache("Sad")
            act_negative = cache[cache_name]
        elif request.steering_option == "good-bad":
            _, cache = model.run_with_cache("Good")
            act_positive = cache[cache_name]
            _, cache = model.run_with_cache("Bad")
            act_negative = cache[cache_name]
        elif request.steering_option == "love-hate":
            _, cache = model.run_with_cache("Love")
            act_positive = cache[cache_name]
            _, cache = model.run_with_cache("Hate")
            act_negative = cache[cache_name]
        else:
            raise ValueError("Invalid steering option.")

        # Compute steering vector
        steering_vec = act_positive[:, -1:, :] - act_negative[:, -1:, :]

        steering_vec /= steering_vec.norm()

        # Generate outputs
        def act_add(steering_vec):
            def hook(activation, hook):
                return activation + steering_vec
            return hook

        test_sentence = request.prompt

        # Positive steering
        coeff = 10
        model.add_hook(name=cache_name, hook=act_add(coeff * steering_vec))
        positive_output = model.generate(test_sentence, max_new_tokens=10, do_sample=False)
        model.reset_hooks()

        # Negative steering
        coeff = -10
        model.add_hook(name=cache_name, hook=act_add(coeff * steering_vec))
        negative_output = model.generate(test_sentence, max_new_tokens=10, do_sample=False)
        model.reset_hooks()

        # Neutral output
        neutral_output = model.generate(test_sentence, max_new_tokens=10, do_sample=False)

        return {
            "positive": positive_output,
            "negative": negative_output,
            "neutral": neutral_output
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Schema for /load_model
class LoadModelRequest(BaseModel):
    model_name: str

@router.post("/load_model")
def load_model_route(request: LoadModelRequest):
    return load_model(request.model_name)

# Schema for /generate_response
class GenerateResponseRequest(BaseModel):
    prompt: str
    temperature: float
    top_k: int
    max_new_tokens: int

@router.post("/generate_response")
def generate_response_route(request: GenerateResponseRequest):
    result = generate_response(request.prompt, request.temperature, request.top_k,request.max_new_tokens)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

# Schema for /token_probabilities
class TokenProbabilitiesRequest(BaseModel):
    prompt: str

@router.post("/token_probabilities")
def token_probabilities_route(request: TokenProbabilitiesRequest):
    return get_token_probabilities(request.prompt)

# Schema for /attention_weights
class AttentionWeightsRequest(BaseModel):
    prompt: str

@router.post("/attention_weights")
def attention_weights_route(request: AttentionWeightsRequest):
    return get_attention_weights(request.prompt)



# from fastapi import APIRouter
# from pydantic import BaseModel
# from backend.api.model_loader import load_model
# from backend.api.visualizer import generate_response, get_token_probabilities, get_attention_weights

# router = APIRouter()

# # Schema for /load_model
# class LoadModelRequest(BaseModel):
#     model_name: str

# @router.post("/load_model")
# def load_model_route(request: LoadModelRequest):
#     return load_model(request.model_name)

# # Schema for /generate_response
# class GenerateResponseRequest(BaseModel):
#     prompt: str
#     temperature: float
#     top_k: int

# @router.post("/generate_response")
# def generate_response_route(request: GenerateResponseRequest):
#     result = generate_response(request.prompt, request.temperature, request.top_k)
#     if "error" in result:
#         raise HTTPException(status_code=400, detail=result["error"])
#     return result

# # Schema for /token_probabilities
# class TokenProbabilitiesRequest(BaseModel):
#     prompt: str

# @router.post("/token_probabilities")
# def token_probabilities_route(request: TokenProbabilitiesRequest):
#     return get_token_probabilities(request.prompt)

# # Schema for /attention_weights
# class AttentionWeightsRequest(BaseModel):
#     prompt: str

# @router.post("/attention_weights")
# def attention_weights_route(request: AttentionWeightsRequest):
#     return get_attention_weights(request.prompt)
