import torch
import re
from transformers import Trainer, TrainingArguments
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import wandb
from datetime import datetime
from transformers import DataCollatorForLanguageModeling

class ToolformerTrainer(Trainer):
    """
    Custom trainer that rewards the model for proper tool usage
    """
    def __init__(
        self,
        tokenizer=None,
        tool_reward_multiplier=3.0,
        tool_pattern=r'<tool:([^>]+)>([^<]+)</tool>',
        log_tool_usage_freq=100,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.tool_reward_multiplier = tool_reward_multiplier
        self.tool_pattern = tool_pattern
        self.log_tool_usage_freq = log_tool_usage_freq
        self.step_counter = 0
        
        # Compile regex pattern for efficiency
        self.tool_regex = re.compile(tool_pattern)
        
        # Counter for various tool usage statistics
        self.stats = {
            "total_batches": 0,
            "batches_with_tools": 0,
            "total_tool_calls": 0,
            "valid_tool_calls": 0
        }

    def _is_valid_tool_input(self, tool_name: str, tool_input: str) -> bool:
        """
        Checks if the input for a specific tool is valid
        
        Args:
            tool_name: Name of the tool (e.g., 'calculator')
            tool_input: Input string passed to the tool
            
        Returns:
            bool: Whether the input is valid for the tool
        """
        if tool_name == "calculator":
            # For calculator, input should be a mathematical expression
            # This is a simple check - could be made more sophisticated
            return any(op in tool_input for op in ['+', '-', '*', '/', '(', ')']) and \
                   tool_input.strip() != "" and \
                   not tool_input.isalpha()
        
        # Add validation for other tools here
        # elif tool_name == "another_tool":
        #     return validation_logic
            
        # Default to True for unknown tools
        return True
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Custom loss function that rewards proper tool usage
        """
        # Get standard language modeling loss
        outputs = model(**inputs)
        standard_loss = outputs.loss
        
        # Get logits and labels
        logits = outputs.logits
        labels = inputs["labels"]
        
        # For tracking tool usage, we'll use the highest probability tokens (greedy decoding)
        predictions = torch.argmax(logits, dim=-1)
        
        # Decode each sequence in the batch
        batch_size = predictions.shape[0]
        reward_factor = torch.ones(batch_size, device=standard_loss.device)
        
        # Track statistics for this batch
        batch_has_tools = False
        batch_tool_calls = 0
        batch_valid_calls = 0
        
        for i in range(batch_size):
            # Get the predicted sequence for this example
            pred_tokens = predictions[i].detach().cpu().tolist()
            
            # Skip padding tokens (-100 in HF datasets)
            if -100 in labels[i]:
                valid_indices = labels[i] != -100
                pred_tokens = [t for j, t in enumerate(pred_tokens) if j < len(valid_indices) and valid_indices[j]]
            
            # Decode to text
            pred_text = self.tokenizer.decode(pred_tokens)
            
            # Find all tool usage instances
            tool_matches = self.tool_regex.finditer(pred_text)
            
            tool_count = 0
            valid_tool_count = 0
            
            for match in tool_matches:
                tool_count += 1
                tool_name = match.group(1)  # Extract tool name
                tool_input = match.group(2)  # Extract tool input
                
                # Validate the tool input
                if self._is_valid_tool_input(tool_name, tool_input):
                    valid_tool_count += 1
            
            # Update batch statistics
            if tool_count > 0:
                batch_has_tools = True
                batch_tool_calls += tool_count
                batch_valid_calls += valid_tool_count
            
            # Calculate reward factor based on tool usage
            if tool_count > 0:
                # Base reward for using tools at all
                base_reward = 1.0
                
                # Additional reward for valid tool usage
                validity_ratio = valid_tool_count / tool_count if tool_count > 0 else 0
                validity_reward = validity_ratio * 1.0
                
                # Combined reward
                total_reward = base_reward + validity_reward
                
                # Apply reward by reducing the loss for this example
                reward_factor[i] = max(0.1, 1.0 - (total_reward * self.tool_reward_multiplier / 10.0))
        
        # Update global statistics
        self.stats["total_batches"] += 1
        if batch_has_tools:
            self.stats["batches_with_tools"] += 1
        self.stats["total_tool_calls"] += batch_tool_calls
        self.stats["valid_tool_calls"] += batch_valid_calls
        
        # Apply the reward factor to the loss
        # We're reducing loss for examples with good tool usage
        modified_loss = standard_loss * reward_factor.mean()
        
        # Log stats periodically
        self.step_counter += 1
        if self.step_counter % self.log_tool_usage_freq == 0:
            self._log_tool_usage_stats()
        
        return (modified_loss, outputs) if return_outputs else modified_loss
    
    def _log_tool_usage_stats(self):
        """Log tool usage statistics to wandb or console"""
        tool_usage_rate = self.stats["batches_with_tools"] / max(1, self.stats["total_batches"])
        valid_tool_rate = self.stats["valid_tool_calls"] / max(1, self.stats["total_tool_calls"])
        
        stats_dict = {
            "tool_usage/batch_rate": tool_usage_rate,
            "tool_usage/total_calls": self.stats["total_tool_calls"],
            "tool_usage/valid_call_rate": valid_tool_rate,
        }
        
        # Log to wandb if available
        if wandb.run is not None:
            wandb.log(stats_dict)
        
        # Also print to console
        print(f"Tool Usage Stats - Step {self.step_counter}:")
        print(f"  - Tool usage batch rate: {tool_usage_rate:.4f}")
        print(f"  - Total tool calls: {self.stats['total_tool_calls']}")
        print(f"  - Valid tool call rate: {valid_tool_rate:.4f}")


def train_toolformer_model(
    model, 
    tokenizer, 
    train_dataset, 
    eval_dataset=None, 
    output_dir="toolformer_model", 
    num_epochs=3, 
    tool_reward_multiplier=3.0,
    learning_rate=5e-4,
    previous_metadata=None
):
    """
    Train Toolformer model with custom rewards for tool usage
    
    Args:
        model: Model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (optional)
        output_dir: Directory to save model
        num_epochs: Number of training epochs
        tool_reward_multiplier: Factor by which to multiply tool usage rewards
        learning_rate: Learning rate for optimizer
        previous_metadata: Previous model metadata if continuing training
    
    Returns:
        tuple: (model, tokenizer, metadata)
    """
    print(f"Training Toolformer model with tool reward multiplier: {tool_reward_multiplier}")
    
    # Configure and apply LoRA as in your original code
    from peft import LoraConfig, get_peft_model, TaskType
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    # Configure and Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = get_peft_model(model, lora_config)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=learning_rate,
        weight_decay=0.01,
        num_train_epochs=num_epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_dir="./logs",
        report_to="wandb",
        logging_steps=16,
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=True,
        load_best_model_at_end=True if eval_dataset else False,
        resume_from_checkpoint=True,
        label_names=["labels"],
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        optim="adamw_torch"
    )
    
    # Create wandb config as in your original code
    config = {
        "output_dir": output_dir,
        "num_epochs": num_epochs,
        "tool_reward_multiplier": tool_reward_multiplier,
        
        # LoRA configuration
        "lora": {
            "task_type": lora_config.task_type.value if hasattr(lora_config.task_type, 'value') else str(lora_config.task_type),
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "target_modules": lora_config.target_modules
        },
        
        # Training configuration
        "training": {
            "per_device_batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "learning_rate": training_args.learning_rate,
            "weight_decay": training_args.weight_decay,
            "lr_scheduler_type": training_args.lr_scheduler_type,
            "warmup_ratio": training_args.warmup_ratio,
            "fp16": training_args.fp16,
            "dataloader_num_workers": training_args.dataloader_num_workers,
            "gradient_checkpointing": training_args.gradient_checkpointing,
            "optim": training_args.optim
        },
        
        # Logging configuration
        "logging": {
            "logging_dir": training_args.logging_dir,
            "report_to": training_args.report_to,
            "logging_steps": training_args.logging_steps,
            "save_strategy": training_args.save_strategy,
            "eval_strategy": training_args.eval_strategy
        }
    }
    
    # Initialize wandb
    wandb.init(
        project="toolformer",
        config=config,
        name=f"toolformer_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Data collator for language modeling
    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're not using masked language modeling
    )
    
    # Set up callbacks as in your original code
    from transformers import EarlyStoppingCallback
    from utils.tool_usage_callback import ToolUsageMonitor
    from utils.wandb_callback import WandbToolformerCallback
    
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=3),
        ToolUsageMonitor(
            tokenizer=tokenizer,
            check_steps=training_args.logging_steps,
        ),
        WandbToolformerCallback(
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            log_freq=training_args.logging_steps,
            sample_size=5,  # Number of examples to track
            log_lora_params=True,
            log_attention=True
        )
    ]
    
    # Initialize our custom trainer
    trainer = ToolformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
        tokenizer=tokenizer,
        tool_reward_multiplier=tool_reward_multiplier
    )
    
    # Record training start time
    training_start_time = datetime.now()
    
    # Start training
    train_result = trainer.train()
    
    # Record training end time
    training_end_time = datetime.now()
    training_duration = (training_end_time - training_start_time).total_seconds() / 60  # in minutes
    
    # Save the fine-tuned model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Create or update metadata as in your original code
    import os
    import json
    
    if previous_metadata is None:
        previous_metadata = {
            "base_model": getattr(model.config, "model_name_or_path", "unknown"),
            "creation_date": datetime.now().isoformat(),
            "total_epochs": 0,
            "training_history": []
        }
    
    # Calculate total epochs
    total_epochs = previous_metadata.get("total_epochs", 0) + num_epochs
    
    # Create training session info with the extracted configuration
    training_session = {
        "date": datetime.now().isoformat(),
        "epochs": num_epochs,
        "examples": len(train_dataset),
        "training_duration_minutes": training_duration,
        "loss": train_result.training_loss,
        "tool_reward_multiplier": tool_reward_multiplier,
        "tool_usage_stats": {
            "batch_with_tools_rate": trainer.stats["batches_with_tools"] / max(1, trainer.stats["total_batches"]),
            "total_tool_calls": trainer.stats["total_tool_calls"],
            "valid_tool_call_rate": trainer.stats["valid_tool_calls"] / max(1, trainer.stats["total_tool_calls"])
        },
        "dataset_stats": {
            "train_size": len(train_dataset),
            "eval_size": len(eval_dataset) if eval_dataset else 0
        },
        "config": config  # Use the extracted config for metadata
    }
    
    # Add evaluation metrics if available
    if hasattr(train_result, "metrics") and eval_dataset:
        training_session["eval_metrics"] = train_result.metrics
    
    # Update metadata
    training_history = previous_metadata.get("training_history", [])
    training_history.append(training_session)
    
    metadata = {
        **previous_metadata,
        "total_epochs": total_epochs,
        "last_training_date": datetime.now().isoformat(),
        "training_history": training_history
    }
    
    # Save metadata to file
    metadata_path = os.path.join(output_dir, "training_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Final metrics are logged by the WandbToolformerCallback
    wandb.finish()
    
    return model, tokenizer, metadata