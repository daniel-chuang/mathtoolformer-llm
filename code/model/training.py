# Import necessary libraries
from trl import SFTTrainer
from transformers import TrainingArguments
from transformers import TrainerCallback
from datetime import datetime
import os
import json
from utils.wandb_callback import WandbToolformerCallback
from peft import LoraConfig, TaskType
import wandb
from transformers import EarlyStoppingCallback
from utils.tool_usage_callback import ToolUsageMonitor
import numpy as np
import re


def format_messages_for_sft(examples):
    """Format examples for SFTTrainer using chat template"""
    system_prompt = """You are an AI assistant that can use tools to solve problems. When you encounter mathematical calculations, use the <tool:calculator> format to perform the calculation. For example, if asked "What is 2+3?", respond with:

<tool:calculator>2+3</tool>

Then provide the answer. Use tools only when necessary for calculations that require precision."""
    
    # Format each example as a chat conversation
    formatted_texts = []
    print(f"Formatting {len(examples)} examples for SFTTrainer")
    print(type(examples))
    print(f"Available keys: {list(examples.keys())}")
    for question, answer in zip(examples["question"], examples["final_answer"]):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
        formatted_texts.append(messages)
    
    return {"messages": formatted_texts}

def convert_to_sft_format(dataset, tokenizer):
    """Convert dataset to the format expected by SFTTrainer"""
    # Apply the formatting function
    formatted_dataset = dataset.map(format_messages_for_sft, batched=True)
    
    # Apply chat template
    def apply_template(example):
        formatted = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": formatted}
    
    sft_dataset = formatted_dataset.map(apply_template)
    return sft_dataset

def compute_sft_metrics(eval_preds):
    """Simplified metrics computation for SFTTrainer"""
    # Note: SFTTrainer handles most metrics automatically
    # We can add custom metrics here if needed
    return {}

def train_model(model, tokenizer, train_dataset, eval_dataset=None, output_dir="toolformer_model", num_epochs=3, previous_metadata=None):
    """Fine-tune the model on prepared datasets using SFTTrainer
    
    Args:
        model: Prepared model
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (optional)
        output_dir: Directory to save the model
        num_epochs: Number of epochs for this training session
        previous_metadata: Previous model metadata if continuing training
    
    Returns:
        tuple: (model, tokenizer, metadata) - The trained model, tokenizer and updated metadata
    """
    print(f"Model Type: {model.config.model_type}")
    print(f"Tokenizer Type: {type(tokenizer).__name__}")
    
    # Convert datasets to SFT format
    sft_train_dataset = convert_to_sft_format(train_dataset, tokenizer)
    sft_eval_dataset = convert_to_sft_format(eval_dataset, tokenizer) if eval_dataset else None
    
    # Print samples for debugging
    print("=== Sample Training Data ===")
    print(sft_train_dataset[0]["text"])
    if sft_eval_dataset:
        print("=== Sample Eval Data ===")
        print(sft_eval_dataset[0]["text"])

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=5e-4,
        weight_decay=0.01,
        num_train_epochs=num_epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_dir="./logs",
        report_to="wandb",
        logging_steps=2,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset else "no",
        fp16=True,
        load_best_model_at_end=True if eval_dataset else False,
        # remove_unused_columns=False,  # Important for SFTTrainer
        label_names=["labels"],
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        optim="adamw_torch"
    )
    
    # WANDB: Create a consolidated config dictionary
    config = {
        # General configuration
        "output_dir": output_dir,
        "num_epochs": num_epochs,
        
        # Extract LoRA configuration
        "lora": {
            "task_type": lora_config.task_type.value if hasattr(lora_config.task_type, 'value') else str(lora_config.task_type),
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "target_modules": lora_config.target_modules
        },
        
        # Extract training configuration
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
        
        # Extract logging configuration
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
    
    # Custom callbacks for SFTTrainer
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=3) if eval_dataset else None,
        # Note: Tool usage monitoring callback would need to be adapted for SFTTrainer
        # since it handles data differently
    ]
    # Remove None values
    callbacks = [cb for cb in callbacks if cb is not None]
    
    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=sft_train_dataset,
        eval_dataset=sft_eval_dataset,
        tokenizer=tokenizer,
        peft_config=lora_config,  # Pass LoRA config directly
        max_seq_length=1024,  # Adjust as needed
        dataset_text_field="text",  # Column with formatted text
        packing=False,  # Important for maintaining conversation structure
        callbacks=callbacks,
        # formatting_func=None,  # We've already formatted the data
        # compute_metrics=lambda eval_preds: compute_sft_metrics(eval_preds),
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
    
    # Create or update metadata (same as before)
    if previous_metadata is None:
        previous_metadata = {
            "base_model": getattr(model.config, "model_name_or_path", "unknown"),
            "creation_date": datetime.now().isoformat(),
            "total_epochs": 0,
            "training_history": []
        }
    
    # Calculate total epochs
    total_epochs = previous_metadata.get("total_epochs", 0) + num_epochs
    
    # Create training session info
    training_session = {
        "date": datetime.now().isoformat(),
        "epochs": num_epochs,
        "examples": len(train_dataset),
        "training_duration_minutes": training_duration,
        "loss": train_result.training_loss,
        "dataset_stats": {
            "train_size": len(train_dataset),
            "eval_size": len(eval_dataset) if eval_dataset else 0
        },
        "config": config
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
    
    wandb.finish()

    return model, tokenizer, metadata