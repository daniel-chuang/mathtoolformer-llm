from transformers import TrainerCallback
import wandb
import torch
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from inference.generate import generate

class WandbToolformerCallback(TrainerCallback):
    """Enhanced W&B callback for Toolformer training with LoRA.
    
    This callback handles comprehensive logging of:
    - Training and validation losses
    - LoRA parameter updates and gradients
    - Model predictions on validation data
    - Attention visualizations
    """
    
    def __init__(self, 
                tokenizer, 
                eval_dataset=None, 
                log_freq=50, 
                sample_size=5, 
                log_lora_params=True,
                log_attention=True):
        """Initialize the callback.
        
        Args:
            tokenizer: The tokenizer for decoding predictions
            eval_dataset: Optional evaluation dataset for generating predictions
            log_freq: Frequency (in steps) for logging samples and parameters
            sample_size: Number of examples to log
            log_lora_params: Whether to log LoRA parameter values and gradients
            log_attention: Whether to log attention visualizations
        """
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.log_freq = log_freq
        self.sample_size = min(sample_size, len(eval_dataset) if eval_dataset is not None else 0)
        self.log_lora_params = log_lora_params
        self.log_attention = log_attention
        self.samples = None
        self.examples_seen = 0
        
        # Tracking for loss values
        self.train_losses = []
        self.current_step = 0
        
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Set up initial logging and prepare validation samples."""
        if not wandb.run:
            raise ValueError("WandB run not initialized. Call wandb.init() before training.")
            
        # Log model architecture diagram
        wandb.run.summary["model_parameter_count"] = sum(
            p.numel() for p in model.parameters()
        )
        
        # Log non-frozen parameter count (LoRA parameters)
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        wandb.run.summary["trainable_parameter_count"] = trainable_params
        
        # Prepare samples for tracking predictions if we have eval data
        if self.eval_dataset is not None and self.sample_size > 0:
            # Select random samples for consistent tracking
            indices = random.sample(range(len(self.eval_dataset)), self.sample_size)
            self.samples = [self.eval_dataset[i] for i in indices]
            
            # Log the initial input texts
            sample_texts = []
            for sample in self.samples:
                # Decode input tokens to text
                text = self.tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
                sample_texts.append(text)
                
            # Create a table of input samples
            samples_table = wandb.Table(columns=["Sample ID", "Input Text"])
            for i, text in enumerate(sample_texts):
                samples_table.add_data(i, text)
                
            wandb.log({"validation_samples": samples_table})
            
    def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        """Track training progress, losses, and LoRA parameters."""
        self.current_step = state.global_step
        
        # Get the latest loss from training state (do this EVERY step)
        if state.log_history:
            latest_step_info = state.log_history[-1]
            if "loss" in latest_step_info:
                loss = latest_step_info["loss"]
                # Log loss every step
                wandb.log({"training/step_loss": loss}, step=state.global_step)
                self.train_losses.append((state.global_step, loss))
                
        # For heavier logging, only do at specified frequency
        if state.global_step % self.log_freq != 0:
            return
        
        # Get the current learning rate
        if optimizer is not None:
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                wandb.log({"learning_rate": lr}, step=state.global_step)
                break
        
        # Log LoRA parameters if enabled
        if self.log_lora_params and model is not None:
            self._log_lora_parameters(model, state.global_step)
    
    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        """Log evaluation metrics and generate predictions."""
        if metrics:
            # Prefix eval metrics for clarity in the UI
            eval_metrics = {"eval/" + k: v for k, v in metrics.items()}
            wandb.log(eval_metrics, step=state.global_step)
        
        # Generate and log predictions on sample data
        if self.samples and model:
            self._log_predictions(model, state.global_step)
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Enhanced logging with additional metrics."""
        if not logs:
            return
            
        # Augment logs with additional metrics
        logs_to_record = logs.copy()
        
        # Calculate and add training progress percentage
        if args.max_steps > 0:
            progress = (state.global_step / args.max_steps) * 100
            logs_to_record["training/progress_percentage"] = progress
            
        # Track examples seen so far (for data throughput analysis)
        batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
        if hasattr(args, "n_gpu") and args.n_gpu > 0:
            batch_size *= args.n_gpu
            
        self.examples_seen += batch_size
        logs_to_record["training/examples_seen"] = self.examples_seen
        
        # Calculate training throughput (examples/second)
        if "train_runtime" in logs:
            runtime = logs["train_runtime"]
            if runtime > 0:
                throughput = self.examples_seen / runtime
                logs_to_record["training/throughput"] = throughput
    
    def on_train_end(self, args, state, control, model=None, **kwargs):
        """Finalize logging with summary metrics and visualizations."""
        # Create loss curve
        if self.train_losses:
            steps, losses = zip(*self.train_losses)
            data = [[x, y] for (x, y) in zip(steps, losses)]
            table = wandb.Table(data=data, columns=["step", "loss"])
            wandb.log({"training/loss_curve": wandb.plot.line(
                table, "step", "loss", title="Training Loss Curve")
            })
            
        # Log final model predictions
        if self.samples and model:
            self._log_predictions(model, state.global_step, is_final=True)
            
        # Log final LoRA parameter state
        if self.log_lora_params and model:
            self._log_lora_parameters(model, state.global_step, is_final=True)
            
        # Log overall training summary
        wandb.run.summary.update({
            "final_step": state.global_step,
            "total_examples_seen": self.examples_seen,
        })
    
    def _log_lora_parameters(self, model, step, is_final=False):
        """Log LoRA parameter values and gradients."""
        # Find all LoRA modules in the model
        prefix = "final_" if is_final else ""
        lora_data = {}
        lora_parameter_norms = {}
        lora_gradient_norms = {}
        
        # Iterate through named parameters to find LoRA parameters
        for name, param in model.named_parameters():
            if 'lora_' in name:
                # Store parameter values
                param_data = param.detach().cpu().numpy()
                
                # Calculate norm for summary metrics
                param_norm = np.linalg.norm(param_data)
                lora_parameter_norms[name] = param_norm
                
                # Track gradients if available
                if param.grad is not None:
                    grad_data = param.grad.detach().cpu().numpy()
                    grad_norm = np.linalg.norm(grad_data)
                    lora_gradient_norms[name] = grad_norm
                
                # For selected parameters, create detailed visualizations
                # (avoiding too many plots by focusing on key layers)
                if (is_final or step % (self.log_freq * 10) == 0) and (
                        'q_proj' in name or 'v_proj' in name):
                    # Create heatmap for parameter values
                    param_fig, param_ax = plt.subplots(figsize=(10, 4))
                    sns.heatmap(param_data, cmap='viridis', ax=param_ax)
                    param_ax.set_title(f"LoRA Parameters: {name}")
                    lora_data[f"{prefix}lora_param_{name}"] = wandb.Image(param_fig)
                    plt.close(param_fig)
                    
                    # Create histogram of parameter values
                    hist_fig, hist_ax = plt.subplots(figsize=(8, 3))
                    hist_ax.hist(param_data.flatten(), bins=50)
                    hist_ax.set_title(f"LoRA Parameter Distribution: {name}")
                    lora_data[f"{prefix}lora_hist_{name}"] = wandb.Image(hist_fig)
                    plt.close(hist_fig)
                    
        # Log all parameter norms as metrics
        for name, norm in lora_parameter_norms.items():
            lora_data[f"{prefix}lora_param_norm/{name}"] = norm
            
        # Log all gradient norms as metrics
        for name, norm in lora_gradient_norms.items():
            lora_data[f"{prefix}lora_grad_norm/{name}"] = norm
            
        # Calculate average norms across parameter types
        if lora_parameter_norms:
            lora_data[f"{prefix}lora_param_norm_avg"] = sum(lora_parameter_norms.values()) / len(lora_parameter_norms)
        if lora_gradient_norms:
            lora_data[f"{prefix}lora_grad_norm_avg"] = sum(lora_gradient_norms.values()) / len(lora_gradient_norms)
            
        wandb.log(lora_data, step=step)
    
    def _log_predictions(self, model, step, is_final=False):
        """Generate and log model predictions on sample data."""
        if not self.samples:
            return
            
        # Set model to evaluation mode
        model.eval()
        device = model.device
        prefix = "final_" if is_final else ""
        
        # Create table for predictions
        columns = ["Sample ID", "Input", "Target Completion", "Model Prediction", "Match?"]
        predictions_table = wandb.Table(columns=columns)
        
        # Generate predictions for each sample
        for i, sample in enumerate(self.samples):
            # Move inputs to the device
            inputs = {k: torch.tensor(v).unsqueeze(0).to(device) 
                     for k, v in sample.items() if k != 'labels'}
            
            # Get reference completion (from labels)
            labels = sample.get('labels', [])
            target_text = self.tokenizer.decode(
                [t for t in labels if t != -100], 
                skip_special_tokens=True
            )
            
            # Get input text
            input_text = self.tokenizer.decode(
                sample['input_ids'], 
                skip_special_tokens=True
            )
            
            # Generate prediction
            outputs = generate(
                inputs, 
                model, 
                self.tokenizer, 
                max_new_tokens=150
            )
                
            # Decode prediction
            prediction = self.tokenizer.decode(
                outputs[0][len(inputs['input_ids'][0]):], 
                skip_special_tokens=True
            )
            
            # Check if prediction matches target (simplistic check)
            # For a real implementation, use proper NLP metrics
            match = "✓" if prediction.strip() == target_text.strip() else "✗"
            
            # Add to table
            predictions_table.add_data(i, input_text, target_text, prediction, match)
            
        # Log the predictions table
        wandb.log({f"{prefix}predictions": predictions_table}, step=step)
        
        # Return model to training mode
        model.train()
    
    def _log_attention(self, model, step, is_final=False):
        """Log attention visualizations."""
        if not self.log_attention or not self.samples:
            return
            
        # This is complex and model-specific
        # For a full implementation, you'd need to:
        # 1. Hook into model's attention mechanisms
        # 2. Collect attention maps during forward pass
        # 3. Visualize and log them
        
        prefix = "final_" if is_final else ""
        
        # Example placeholder - implementation depends on model architecture
        attention_data = {}
        
        # Log attention data if available
        if attention_data:
            wandb.log({f"{prefix}attention": attention_data}, step=step)