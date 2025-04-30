# Import necessary libraries
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from datetime import datetime

def preprocess_for_training(examples, tokenizer, max_length=2048):
    """Tokenize the examples for training"""
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )
    
    # Create attention masks
    result["attention_mask"] = [
        [1 if token != tokenizer.pad_token_id else 0 for token in tokens]
        for tokens in result["input_ids"]
    ]
    
    # Set labels same as input_ids for causal language modeling
    result["labels"] = result["input_ids"].copy()
    
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
    # Tokenize datasets
    train_dataset = train_dataset.map(
        lambda examples: preprocess_for_training(examples, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    if eval_dataset:
        eval_dataset = eval_dataset.map(
            lambda examples: preprocess_for_training(examples, tokenizer),
            batched=True,
            remove_columns=eval_dataset.column_names
        )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        num_train_epochs=num_epochs,  # Use the parameter value
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="steps",  # Save checkpoints every few steps
        save_steps=500,         # Adjust based on your dataset size
        eval_strategy="epoch" if eval_dataset else "no",
        fp16=True,
        load_best_model_at_end=True if eval_dataset else False,
        resume_from_checkpoint=True,  # Allow resuming from checkpoints
    )
        
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're not using masked language modeling
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
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
        }
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
    
    return model, tokenizer, metadata