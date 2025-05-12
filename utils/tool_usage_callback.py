from transformers import TrainerCallback
import torch
import wandb
from inference import generate

class ToolUsageMonitor(TrainerCallback):
    def __init__(self, tokenizer, check_steps=50):
        self.tokenizer = tokenizer
        self.check_steps = check_steps
        self.test_inputs = [
            "Question: What is 7 + 8?\nAnswer:",
            "Question: What is 12 * 3?\nAnswer:"
        ]
        
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None or state.global_step % self.check_steps != 0:
            return
            
        print(f"\n--- Checking tool usage at step {state.global_step} ---")
        for test_input in self.test_inputs:
            inputs = self.tokenizer(test_input, return_tensors="pt").to(model.device)
            outputs = generate(
                inputs,
                model,
                self.tokenizer,
                max_new_tokens=30
            ) 
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Input: {test_input}")
            print(f"Output: {generated}")
            print(f"Uses tool: {'<tool:' in generated}")
            print("-" * 50)