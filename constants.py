# constants.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Global constants
TOOL_PATTERN = r'<tool:(\w+)>(.*?)</tool>'
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# DATASET
DATASET = "arithmetic"  # Options: "arithmetic", "svamp", "gsm8k"

# QWEN Qwen2.5-Math-1.5B
MODEL_NAME = "Qwen2.5-Math-1.5B"
INITIAL_SAVE_PATH = "./qwen-initial"
MATH_FINETUNED_SAVE_PATH = "./qwen-math-finetuned"

# GPT 2 - Does poorly
# MODEL_NAME = "gpt2"
# INITIAL_SAVE_PATH = "./gpt2-initial"
# MATH_FINETUNED_SAVE_PATH = "./gpt2-math-finetuned"

# Microsoft Phi 3 - Does really well but slow
# MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
# INITIAL_SAVE_PATH = "./phi3-mini-initial"
# MATH_FINETUNED_SAVE_PATH = "./phi3-mini-math-finetuned"