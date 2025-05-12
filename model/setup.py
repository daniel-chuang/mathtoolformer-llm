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
        tuple: (tokenizer, model, metadata) - The loaded tokenizer, model and initial metadata
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
    # Create initial metadata
    metadata = {
        "base_model": model_name,
        "creation_date": datetime.now().isoformat(),
        "quantization": "4-bit" if use_4bit else "none",
        "total_epochs": 0,
        "training_history": []
    }
    
    return tokenizer, model, metadata

def save_model(model, tokenizer, path, epochs=None, total_epochs=None, include_timestamp=False):
    """
    Save model and tokenizer to the specified path.
    
    Args:
        model: The model to save
        tokenizer: The tokenizer to save
        path: The path to save the model to
        epochs: Number of training epochs (optional)
        include_timestamp: Whether to include a timestamp in the saved path
    
    Returns:
        The actual path where the model was saved
    """
    # Extract the directory and filename components
    directory = os.path.dirname(path)
    filename = os.path.basename(path)
    
    # Start with the base path
    save_path = path
    
    # Add epochs to the path if provided
    if epochs is not None:
        save_path = f"{path}_epochs{epochs}"
    
    if total_epochs is not None:
        save_path = f"{save_path}_total_epochs{total_epochs}"

    # Add timestamp if requested
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"{save_path}_{timestamp}"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the model and tokenizer
    print(f"Saving model to {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    return save_path

def load_tokenizer(model_path, device_map="auto", use_4bit=False):
    # Check if this is a local path that exists
    if os.path.exists(model_path) and os.path.isdir(model_path):
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    else:
        raise FileNotFoundError(f"Tokenizer not found at path: {model_path}")


def load_model(model_path, device_map="auto", use_4bit=False):
    # Check if this is a local path that exists
    if os.path.exists(model_path) and os.path.isdir(model_path):
        print(f"Loading model from local path: {model_path}")
        
        # Load metadata if it exists
        metadata_path = os.path.join(model_path, "training_metadata.json")
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                print(f"Model training history: {metadata.get('total_epochs', 'unknown')} total epochs")
        
        # Configure quantization based on parameter
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        else:
            quantization_config = None
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map=device_map
        )
        
        return model, metadata
    else:
        raise FileNotFoundError(f"Model not found at path: {model_path}")