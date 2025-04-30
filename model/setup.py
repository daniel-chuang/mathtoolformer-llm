import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from constants import HUGGINGFACE_TOKEN
from datetime import datetime

def setup_model(model_name="mistralai/Mistral-7B-v0.1", use_4bit=False):
    """
    Load and prepare the base model with quantization if needed
    
    Args:
        model_name: HuggingFace model identifier
        use_4bit: Whether to use 4-bit quantization (recommended for larger models)
    
    Returns:
        tokenizer, model: The loaded tokenizer and model
    """
    # Configure quantization if needed
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    else:
        quantization_config = None
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=HUGGINGFACE_TOKEN
        )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with quantization config
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=HUGGINGFACE_TOKEN,
        quantization_config=quantization_config,
        device_map="auto"
    )
    
    return tokenizer, model

def save_model(model, tokenizer, save_path):
    """
    Save model and tokenizer to the specified path
    
    Args:
        model: The model to save
        tokenizer: The tokenizer to save
        save_path: Directory path to save to
    """
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Save model state
    model.save_pretrained(save_path)
    
    # Save tokenizer
    tokenizer.save_pretrained(save_path)
    
    # Save metadata (like original model name, quantization status, etc.)
    metadata = {
        "save_date": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "quantized": hasattr(model, "quantization_config") and model.quantization_config is not None
    }
    
    with open(os.path.join(save_path, "metadata.json"), "w") as f:
        json.dump(metadata, f)
    
    print(f"Model saved to {save_path}")

def load_model(model_path, device_map="auto", use_4bit=False):
    """
    Load a model and tokenizer from a local path if it exists.
    
    Args:
        model_path: Local directory path where model is saved
        device_map: Device mapping strategy
        use_4bit: Whether to use 4-bit quantization when loading
        
    Returns:
        tuple: (tokenizer, model)
    """
    # Check if this is a local path that exists
    if os.path.exists(model_path) and os.path.isdir(model_path):
        print(f"Loading model from local path: {model_path}")
        
        # Check if metadata exists to determine quantization
        metadata_path = os.path.join(model_path, "metadata.json")
        metadata_exists = os.path.exists(metadata_path)
        
        # Configure quantization based on parameter or metadata
        if use_4bit or (metadata_exists and json.load(open(metadata_path)).get("quantized", False)):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        else:
            quantization_config = None
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map=device_map
        )
        
        return tokenizer, model
    else:
        raise FileNotFoundError(f"Model not found at path: {model_path}")