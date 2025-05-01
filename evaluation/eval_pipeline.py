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
from data.arithmetic import combine_and_tokenize

def eval_model(model_name, dataset_name, test_datasets, model, tokenizer, use_tool):
    # Update the arithmetic datasets evaluation part:
    if type(test_datasets) == dict:
        # Evaluate math performance
        combined_results = {'metrics': {'correct': 0, 'total': 0}, 'type_statistics': {}}
        for key, test_dataset in test_datasets.items():
            print(f"Evaluating math performance on dataset {key}")
            results = evaluate_math_performance(
                model, 
                tokenizer, 
                test_dataset, 
                use_tool=use_tool,
                dataset_name=f"{dataset_name}_{key}", 
                model_name=model_name
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
            dataset_name=dataset_type, 
            model_name=model_name
        )
        print("Math Evaluation Results:", results['metrics'])