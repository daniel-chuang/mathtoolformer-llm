# constants.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Global constants
TOOL_PATTERN = r'<tool:(\w+)>(.*?)</tool>'
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
CHECKPOINTS = "./checkpoints"
SEED = 183
SPLIT = 0.9  # Proportion of data to use for training (default: 0.8)

# DATASET
DATASET = "arithmetic"  # Options: "arithmetic", "svamp", "gsm8k"
TOOL_TRAIN_DATASET_PATH = ".preprocessed/preprocessed_train_dataset"
PURE_TRAIN_DATASET_PATH = ".preprocessed/preprocessed_pure_train_dataset"
EVAL_DATASET_PATH = ".preprocessed/preprocessed_eval_dataset"

# SMOLLM
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
INITIAL_SAVE_PATH = "SmolLM2-initial"
TOOL_FINETUNED_SAVE_PATH = "SmolLM2-tool-finetuned"
PURE_FINETUNED_SAVE_PATH = "SmolLM2-pure-finetuned"

# GPT 2 - Does poorly
# MODEL_NAME = "gpt2"
# INITIAL_SAVE_PATH = "/gpt2-initial"
# TOOL_FINETUNED_SAVE_PATH = "/gpt2-math-finetuned"

# Microsoft Phi 3 - Does really well but slow
# MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
# INITIAL_SAVE_PATH = "phi3-mini-initial"
# TOOL_FINETUNED_SAVE_PATH = "phi3-mini-math-finetuned"