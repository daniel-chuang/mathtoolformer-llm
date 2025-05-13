import os
import json
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from model.setup import setup_model, save_model, load_model, load_tokenizer
from utils.console import isYes, printc, inputc, print_section
from data.gsm8k import prepare_gsm8k_dataset
from data.svamp import prepare_svamp_dataset
from data.arithmetic import prepare_arithmetic_datasets
from evaluation.math_evaluation import evaluate_math_performance
from evaluation.eval_pipeline import eval_model
from constants import MODEL_NAME, INITIAL_SAVE_PATH, TOOL_FINETUNED_SAVE_PATH, DATASET, CHECKPOINTS, TOOL_TRAIN_DATASET_PATH, PURE_TRAIN_DATASET_PATH, EVAL_DATASET_PATH
from data.arithmetic import combine_and_tokenize
import wandb
from datasets import Dataset

def main():
    print_section("Loading Model")
    gc.collect()
    torch.cuda.empty_cache()
    # Try to load from saved path first, if it fails, download from HF
    try:
        model, metadata = load_model(os.path.join(CHECKPOINTS, "pretrained", INITIAL_SAVE_PATH))
        tokenizer = load_tokenizer(os.path.join(CHECKPOINTS, "pretrained", INITIAL_SAVE_PATH))
        print("Loaded model from saved path")
    except FileNotFoundError:
        print(f"Initial model not found. Setting up from {MODEL_NAME}")
        tokenizer, model, metadata = setup_model(MODEL_NAME)
        save_model(model, tokenizer, os.path.join(CHECKPOINTS, "pretrained", INITIAL_SAVE_PATH))

    print_section("Adding Tool Tokens")
    tool_tokens = {
        "additional_special_tokens": [
            "<tool:calculator>",
            "</tool>",
        ]
    }
    num_added = tokenizer.add_special_tokens(tool_tokens)
    print(f"Added {num_added} special tokens to the tokenizer")
    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized model embeddings to {len(tokenizer)} tokens")
    print("Special tokens:", tokenizer.all_special_tokens)

    # Ask if the user wants to load data
    wantToTestPretrained = inputc(f"Do you want to evaluate the pretrained {MODEL_NAME} model? (y/n)").strip().lower()
    wantToTrainTool = inputc("Do you want to train the toolformer model? (y/n)").strip().lower()
    if wantToTrainTool == "y":
        # Ask for the number of epochs
        current_epochs_tool = int(inputc("How many epochs do you want to train the tool FT for? (default: 3)"))
        printc(f"Training for {current_epochs_tool} epochs")
        wantToTestToolFineTuned = inputc("Do you want to evaluate the tool fine-tuned model? (y/n)").strip().lower()
    
    wantToTrainPure = inputc("Do you want to train the pure fine tuning model? (y/n)").strip().lower()
    if wantToTrainPure == "y":
        # Ask for the number of epochs
        current_epochs_pure = int(inputc("How many epochs do you want to train the pure FT for? (default: 3)"))
        printc(f"Training for {current_epochs_pure} epochs")
        wantToTestPureFineTuned = inputc("Do you want to evaluate the pure fine-tuned model? (y/n)").strip().lower()
    
    # Add this after the other evaluation sections
    wantToEvalLatest = inputc("Do you want to evaluate the latest checkpoints? (y/n)").strip().lower()

    wantToStart = inputc("Are you ready to start? (y/n)").strip().lower()
    if wantToStart != "y":
        print("Exiting...")
        return

    # LOAD DATA
    print_section("Loading Data")
    # Prepare datasets
    if DATASET == "arithmetic": # Multiple Datasets
        dataset = prepare_arithmetic_datasets()
        train_data = dataset["train_dict"]
        test_data = dataset["test_dict"]
        if wantToTrainTool == "y":
            train_transformed_data = dataset["train_transformed_dict"]
            test_transformed_data = dataset["test_transformed_dict"]
            print(train_transformed_data['arithmetic_2da'][3])
            print(test_transformed_data['arithmetic_2da'][3])
            print(train_transformed_data['arithmetic_2da'][-1])
            print(train_transformed_data['arithmetic_2da'][-1])
    else:
        if DATASET == "svamp": # Single Datasetse
            train_data, test_data = prepare_svamp_dataset()
        elif DATASET == "gsm8k":
            train_data, test_data = prepare_gsm8k_dataset()
        else:
            raise ValueError(f"Unknown dataset: {DATASET}")

    # PRETAINED MODEL EVALUATION
    if isYes(wantToTestPretrained):
        print_section("Pretrained Model Evaluation")
        eval_model(MODEL_NAME, DATASET, test_data, model, tokenizer, use_tool=False)

    # TOOLFORMER FINE TUNING TRAINING
    if isYes(wantToTrainTool):
        print_section("Toolformer Fine Tuning Training")
        
        # Prepare the training data based on dataset type
        if DATASET == "arithmetic":
            train_dataset = combine_and_tokenize(train_transformed_data, tokenizer, path=TOOL_TRAIN_DATASET_PATH)

            # Create a small evaluation dataset directly instead of using combine_and_tokenize
            eval_examples = []
            for config_name, config_dataset in test_transformed_data.items():
                # Take at most 5 examples from each configuration
                sample_size = 1
                for i in range(sample_size):
                    if isinstance(config_dataset[i], dict):
                        eval_examples.append({
                            "question": config_dataset[i]["question"],
                            "final_answer": config_dataset[i]["final_answer"]
                        })
            
            # Create the evaluation dataset directly
            eval_dataset = Dataset.from_list(eval_examples)
            eval_dataset = eval_dataset.map(
                lambda examples: preprocess_for_training(examples, tokenizer),
                batched=True,
                remove_columns=eval_dataset.column_names
            )
        
        print(f"Created evaluation dataset with {len(eval_dataset)} examples for monitoring")

        # Load previous model if it exists
        try:
            previous_path = os.path.join(CHECKPOINTS, "finetuned", TOOL_FINETUNED_SAVE_PATH)
            tokenizer, model, metadata = load_model(previous_path)
            
            # Get total epochs from metadata
            total_epochs = metadata.get("total_epochs", 0) + current_epochs_tool
            print(f"Continuing training from {metadata.get('total_epochs', 0)} epochs to {total_epochs} epochs")
        except FileNotFoundError:
            # Start fresh training
            total_epochs = current_epochs_tool
            print(f"Starting fresh training for {current_epochs_tool} epochs")
        
        # Train the model
        model, tokenizer, metadata = train_model(model, tokenizer, train_dataset, num_epochs=current_epochs_tool, eval_dataset=eval_dataset)
        
        # Save with updated epoch count
        saved_path = save_model(
            model, 
            tokenizer, 
            os.path.join(CHECKPOINTS, "finetuned", TOOL_FINETUNED_SAVE_PATH),
            epochs=current_epochs_tool,
            total_epochs=total_epochs
        )
        print(f"Saved fine-tuned model to {saved_path} (Total epochs: {total_epochs})")

    # PURE FINE TUNING TRAINING
    if isYes(wantToTrainPure):
        print_section("Pure Fine Tuning Training")
        
        # Prepare the training data based on dataset type
        if DATASET == "arithmetic":
            train_dataset = combine_and_tokenize(train_data, tokenizer, path=PURE_TRAIN_DATASET_PATH)
        else:
            train_dataset = train_dataset.map(preprocess_for_training, batched=True)
        
        # Load previous model if it exists
        try:
            previous_path = os.path.join(CHECKPOINTS, "finetuned", TOOL_FINETUNED_SAVE_PATH)
            tokenizer, model, metadata = load_model(previous_path)
            
            # Get total epochs from metadata
            total_epochs = metadata.get("total_epochs", 0) + current_epochs_pure
            print(f"Continuing training from {metadata.get('total_epochs', 0)} epochs to {total_epochs} epochs")
        except FileNotFoundError:
            # Start fresh training
            total_epochs = current_epochs_pure
            print(f"Starting fresh training for {current_epochs_pure} epochs")
        
        # Train the model
        model, tokenizer, metadata = train_model(model, tokenizer, train_dataset, num_epochs=current_epochs_pure)
        
        # Save with updated epoch count
        saved_path = save_model(
            model, 
            tokenizer, 
            os.path.join(CHECKPOINTS, "finetuned", TOOL_FINETUNED_SAVE_PATH),
            epochs=current_epochs_pure,
            total_epochs=total_epochs
        )
        print(f"Saved fine-tuned model to {saved_path} (Total epochs: {total_epochs})")

    if isYes(wantToEvalLatest):
        print_section("Latest Checkpoint Evaluation")
        model, metadata = load_model(os.path.join(os.curdir, "toolformer_model", "checkpoint-225"))
        print_section("Most recent training Model Evaluation")
        eval_model(MODEL_NAME, DATASET, test_data, model, tokenizer, use_tool=True)

    print_section("Done")
if __name__ == "__main__":
    main()