# Import necessary libraries
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling

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

def train_model(model, tokenizer, train_dataset, eval_dataset=None, output_dir="toolformer_model"):
    """Fine-tune the model on prepared datasets
    
    Args:
        model: Prepared model
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (optional)
        output_dir: Directory to save the model
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
        num_train_epochs=3,
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
    
    # Start training
    trainer.train()
    
    # Save the fine-tuned model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer