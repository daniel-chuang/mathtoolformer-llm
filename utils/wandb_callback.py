from transformers import TrainerCallback
import wandb
import torch
import random
import numpy as np

class WandbPredictionCallback(TrainerCallback):
    """
    Callback to log model predictions to Weights & Biases during training.
    """
    def __init__(self, tokenizer, eval_dataset, num_examples=5, steps=100):
        """
        Initialize the callback.
        
        Args:
            tokenizer: The tokenizer used by the model
            eval_dataset: Dataset to sample examples from
            num_examples: Number of examples to log per evaluation
            steps: How often to log predictions (in steps)
        """
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.num_examples = min(num_examples, len(eval_dataset))
        self.steps = steps
        
        # Sample a fixed set of examples to track
        self.example_indices = random.sample(range(len(eval_dataset)), self.num_examples)
        self.tracked_examples = [eval_dataset[i] for i in self.example_indices]
        
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Log predictions at regular intervals"""
        if state.global_step % self.steps == 0 and model is not None:
            # Put model in eval mode
            model.eval()
            
            # Initialize a table for this step
            predictions_table = wandb.Table(columns=["Step", "Example_ID", "Input", "Target", "Prediction"])
            
            # Process each tracked example
            for i, example in enumerate(self.tracked_examples):
                # Get model input
                input_text = example["question"]
                input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
                input_ids = input_ids.to(model.device)
                
                # Get target (ground truth)
                target_text = example["final_answer"]
                
                # Generate prediction
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids, 
                        max_new_tokens=50, 
                        do_sample=False,
                        num_beams=1
                    )
                    
                # Decode the prediction (skip input tokens)
                input_length = input_ids.shape[1]
                prediction_ids = output_ids[0, input_length:]
                prediction_text = self.tokenizer.decode(prediction_ids, skip_special_tokens=True)
                
                # Add to table
                predictions_table.add_data(
                    state.global_step,
                    f"Example_{i}",
                    input_text,
                    target_text,
                    prediction_text
                )
            
            # Log the table to wandb
            wandb.log({"predictions": predictions_table}, step=state.global_step)
            
            # Put model back in train mode
            model.train()
            
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """Log predictions during evaluation"""
        if model is not None:
            # We'll reuse the same logic as step_end but force it to run
            self.on_step_end(args, state, control, model=model, **kwargs)