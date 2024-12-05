from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.api.model_loader import load_model
from transformer_lens import HookedTransformer
from backend.api.visualizer import generate_response, get_token_probabilities, get_attention_weights, generate_response_with_log_probs
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
    layer_num: int

@router.post("/llm_steering")
def llm_steering_route(request: SteeringRequest):
    """
    Perform LLM steering by manipulating internal activations based on a steering vector.

    Args:
        request (SteeringRequest): Contains the prompt and the steering option (e.g., happy-sad, good-bad).

    Returns:
        dict: Contains positive, negative, and neutral outputs after applying activation steering.
    """
    try:
        layer_id = request.layer_num
        cache_name = f"blocks.{layer_id}.hook_resid_post"  # Perform activation steering at the residual layer

        # Precompute steering vectors based on the selected option
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
    """
    Load a specific language model into memory.

    Args:
        request (LoadModelRequest): Contains the model name to be loaded.

    Returns:
        dict: Confirmation that the model has been loaded, or an error if unsuccessful.
    """
    return load_model(request.model_name)


# Schema for /generate_response
class GenerateResponseRequest(BaseModel):
    prompt: str
    temperature: float
    top_k: int
    max_new_tokens: int

@router.post("/generate_response")
def generate_response_route(request: GenerateResponseRequest):
    """
    Generate a response from the language model based on the input prompt and parameters.

    Args:
        request (GenerateResponseRequest): Contains the prompt, temperature, top_k, and max_new_tokens.

    Returns:
        dict: The generated response or an error message if the operation fails.
    """
    result = generate_response(request.prompt, request.temperature, request.top_k, request.max_new_tokens)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# Schema for /token_probabilities
class TokenProbabilitiesRequest(BaseModel):
    prompt: str

@router.post("/token_probabilities")
def token_probabilities_route(request: TokenProbabilitiesRequest):
    """
    Compute token probabilities for the given input prompt.

    Args:
        request (TokenProbabilitiesRequest): Contains the input prompt.

    Returns:
        dict: Token probabilities or an error message if the prompt is invalid.
    """
    return get_token_probabilities(request.prompt)


# Schema for /attention_weights
class AttentionWeightsRequest(BaseModel):
    prompt: str

@router.post("/attention_weights")
def attention_weights_route(request: AttentionWeightsRequest):
    """
    Retrieve attention weights for the input prompt.

    Args:
        request (AttentionWeightsRequest): Contains the input prompt.

    Returns:
        dict: Attention weights and corresponding tokens.
    """
    return get_attention_weights(request.prompt)


# Schema for /generate_response_with_log_probs
class GenerateResponseWithLogProbsRequest(BaseModel):
    prompt: str
    temperature: float
    top_k: int
    max_new_tokens: int

@router.post("/generate_response_with_log_probs")
def generate_response_with_log_probs_route(request: GenerateResponseWithLogProbsRequest):
    """
    Generate a response and compute log probabilities for each token.

    Args:
        request (GenerateResponseWithLogProbsRequest): Contains the prompt, temperature, top_k, and max_new_tokens.

    Returns:
        dict: Generated response, tokens, and log probabilities for each token.
    """
    result = generate_response_with_log_probs(
        request.prompt, request.temperature, request.top_k, request.max_new_tokens
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result
