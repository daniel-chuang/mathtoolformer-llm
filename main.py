import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from model.setup import setup_model, save_model, load_model
from utils.console import isYes, printc, inputc, print_section
from model.training import preprocess_for_training, train_model
from data.gsm8k import prepare_gsm8k_dataset
from data.svamp import prepare_svamp_dataset
from data.arithmetic import prepare_arithmetic_datasets
from evaluation.math_evaluation import evaluate_math_performance
from evaluation.tool_usage_evaluation import evaluate_tool_usage
from constants import MODEL_NAME, INITIAL_SAVE_PATH, TOOL_FINETUNED_SAVE_PATH, DATASET, CHECKPOINTS, TOOL_TRAIN_DATASET_PATH, PURE_TRAIN_DATASET_PATH
from data.arithmetic import combine_and_tokenize

def main():
    print_section("Loading Model")
    # Try to load from saved path first, if it fails, download from HF
    try:
        tokenizer, model, metadata = load_model(os.path.join(CHECKPOINTS, "pretrained", INITIAL_SAVE_PATH))
        print("Loaded model from saved path")
    except FileNotFoundError:
        print(f"Initial model not found. Setting up from {MODEL_NAME}")
        tokenizer, model, metadata = setup_model(MODEL_NAME)
        save_model(model, tokenizer, os.path.join(CHECKPOINTS, "pretrained", INITIAL_SAVE_PATH))

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
    
    inputc("Are you ready to start? (y/n)").strip().lower()

    # LOAD DATA
    print_section("Loading Data")
    # Prepare datasets
    if DATASET == "arithmetic": # Multiple Datasets
        dataset = prepare_arithmetic_datasets()
        train_datasets = dataset["train_dict"]
        test_datasets = dataset["test_dict"]
        if wantToTrainTool == "y":
            train_transformed_datasets = dataset["train_transformed_dict"]
            test_transformed_datasets = dataset["test_transformed_dict"]
            print(train_transformed_datasets['arithmetic_1dc'][3])
            print(test_transformed_datasets['arithmetic_1dc'][3])

    else:
        if DATASET == "svamp": # Single Datasetse
            train_dataset, test_dataset = prepare_svamp_dataset()
        elif DATASET == "gsm8k":
            train_dataset, test_dataset = prepare_gsm8k_dataset()
        else:
            raise ValueError(f"Unknown dataset: {DATASET}")

    # PRETAINED MODEL EVALUATION
    if isYes(wantToTestPretrained):
        print_section("Pretrained Model Evaluation")
        # Update the arithmetic datasets evaluation part:
        if DATASET == "arithmetic":
            # Evaluate math performance
            combined_results = {'metrics': {'correct': 0, 'total': 0}, 'type_statistics': {}}
            for key, test_dataset in test_datasets.items():
                print(f"Evaluating math performance on dataset {key}")
                results = evaluate_math_performance(
                    model, 
                    tokenizer, 
                    test_dataset, 
                    dataset_name=f"arithmetic_{key}", 
                    model_name=MODEL_NAME
                )
                print(f"Math Evaluation Results for {key}:", results['metrics'])
                
                # Aggregate results
                combined_results['metrics']['correct'] += results['metrics']['correct']
                combined_results['metrics']['total'] += results['metrics']['total']
                
                # Merge type statistics
                for q_type, stats in results['type_statistics'].items():
                    if q_type not in combined_results['type_statistics']:
                        combined_results['type_statistics'][q_type] = stats.copy()
                    else:
                        for k in ['total', 'correct', 'incorrect']:
                            combined_results['type_statistics'][q_type][k] += stats[k]
            
            # Calculate combined accuracy
            if combined_results['metrics']['total'] > 0:
                combined_results['metrics']['accuracy'] = combined_results['metrics']['correct'] / combined_results['metrics']['total']
                
            # Update accuracy for each type
            for q_type in combined_results['type_statistics']:
                total = combined_results['type_statistics'][q_type]['total']
                if total > 0:
                    combined_results['type_statistics'][q_type]['accuracy'] = combined_results['type_statistics'][q_type]['correct'] / total
                    
            print("Combined Math Evaluation Results:", combined_results['metrics'])
        else:
            # Evaluate math performance on a single dataset
            results = evaluate_math_performance(
                model, 
                tokenizer, 
                test_dataset, 
                dataset_name=DATASET, 
                model_name=MODEL_NAME
            )
            print("Math Evaluation Results:", results['metrics'])


    # TOOLFORMER FINE TUNING TRAINING
    if isYes(wantToTrainTool):
        print_section("Toolformer Fine Tuning Training")
        
        # Prepare the training data based on dataset type
        if DATASET == "arithmetic":
            train_dataset = combine_and_tokenize(train_transformed_datasets, tokenizer, path=TOOL_TRAIN_DATASET_PATH)
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
        model, tokenizer, metadata = train_model(model, tokenizer, train_dataset, num_epochs=current_epochs_tool)
        
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
            train_dataset = combine_and_tokenize(train_datasets, tokenizer, path=PURE_TRAIN_DATASET_PATH)
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

    print_section("Done")
if __name__ == "__main__":
    main()