import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from model.setup import setup_model, save_model, load_model

from model.training import preprocess_for_training, train_model
from data.gsm8k import prepare_gsm8k_dataset
from data.svamp import prepare_svamp_dataset
from data.arithmetic import prepare_arithmetic_datasets
from evaluation.math_evaluation import evaluate_math_performance
from evaluation.tool_usage_evaluation import evaluate_tool_usage
from inference.pipeline import generate_with_tools
from constants import MODEL_NAME, INITIAL_SAVE_PATH, MATH_FINETUNED_SAVE_PATH, DATASET, CHECKPOINTS

def main():

    # Try to load from saved path first, if it fails, download from HF
    try:
        tokenizer, model = load_model(os.path.join(CHECKPOINTS, INITIAL_SAVE_PATH))
        print("Loaded model from saved path")
    except FileNotFoundError:
        print(f"Initial model not found. Setting up from {MODEL_NAME}")
        tokenizer, model = setup_model(MODEL_NAME)
        save_model(model, tokenizer, os.path.join(CHECKPOINTS, INITIAL_SAVE_PATH))


    # Ask if the user wants to load data
    wantToLoadData = input("Do you want to load the data?").strip().lower()

    # Prepare datasets
    if DATASET == "arithmetic": # Multiple Datasets
        train_datasets, test_datasets = prepare_arithmetic_datasets()
    else:
        if DATASET == "svamp": # Single Datasetse
            train_dataset, test_dataset = prepare_svamp_dataset()
        elif DATASET == "gsm8k":
            train_dataset, test_dataset = prepare_gsm8k_dataset()
        else:
            raise ValueError(f"Unknown dataset: {DATASET}")

    wantToTestPretrained = input(f"Do you want to evaluate the pretrained {MODEL_NAME} model?").strip().lower()
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


    # Ask if the user wants to train
    wantToTrain = input("Do you want to train the model?").strip().lower()
    
    # Train the model
    model, tokenizer = train_model(model, tokenizer, train_dataset)
    save_model(model, tokenizer, MATH_FINETUNED_SAVE_PATH)
    print(f"Saved fine-tuned model to {MATH_FINETUNED_SAVE_PATH}")

    # Ask if the user wants to evaluate the fine-tuned model
    wantToTestFineTuned = input("Do you want to evaluate the fine-tuned model?").strip().lower()

    # Evaluate math performance
    after_math_eval_results = evaluate_math_performance(model, tokenizer, test_dataset)
    print("Math Evaluation Results:", after_math_eval_results)

    # Evaluate tool usage
    tool_eval_results = evaluate_tool_usage(model, tokenizer, test_dataset)
    print("Tool Usage Evaluation Results:", tool_eval_results)

if __name__ == "__main__":
    main()