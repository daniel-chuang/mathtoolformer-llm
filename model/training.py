# Import necessary libraries
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from datetime import datetime
import os
import json
from utils.wandb_callback import WandbToolformerCallback
from peft import LoraConfig, get_peft_model, TaskType
import wandb
from transformers import EarlyStoppingCallback
from utils.tool_usage_callback import ToolUsageMonitor

def preprocess_for_training(examples, tokenizer, max_length=256):
    """Tokenize the examples for training"""
    # Tokenize input context with padding to max_length
    result = tokenizer(
        examples["question"],
        truncation=True,
        max_length=max_length,
        padding='max_length',
    )
    
    # Tokenize completions with the same tokenizer and padding
    completion_encodings = tokenizer(
        examples["final_answer"],
        truncation=True,
        max_length=max_length,
        padding='max_length',
    )
    
    # Set labels to the tokenized completion
    result["labels"] = completion_encodings["input_ids"]
    
    return result

def train_model(model, tokenizer, train_dataset, eval_dataset=None, output_dir="toolformer_model", num_epochs=3, previous_metadata=None):
    """Fine-tune the model on prepared datasets
    
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
    print(tokenizer.decode(train_dataset[0]["input_ids"]))
    print(tokenizer.decode(train_dataset[0]["labels"]))
    print(tokenizer.decode(eval_dataset[0]["input_ids"]))
    print(tokenizer.decode(eval_dataset[0]["labels"]))

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
        learning_rate=5e-4,
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
    
    # WANDB: Create a consolidated config dictionary from the actual objects
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
        
        # Extract training configuration from the training_args object
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
    
    # Initialize wandb with the extracted configuration
    wandb.init(
        project="toolformer",
        config=config,
        name=f"toolformer_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
        
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're not using masked language modeling
    )

    # Callbacks
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
    
    # Initialize trainer with callbacks
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks
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
    
    # Create or update metadata
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