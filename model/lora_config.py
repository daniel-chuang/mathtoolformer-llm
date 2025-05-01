from peft import LoraConfig

def setup_lora_config():
    """Configure LoRA for parameter-efficient fine-tuning"""
    lora_config = LoraConfig(
        r=16,  # Rank dimension
        lora_alpha=32,  # LoRA scaling factor
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Adjust based on model architecture
        #target_modules=["q_proj", "v_proj"]  # Adjust based on model architecture
    )
    return lora_config