from datasets import load_dataset
import numpy as np
import re
from constants import SEED, SPLIT, TOOL_TRAIN_DATASET_PATH
from datasets import Dataset
import os
from model.training import preprocess_for_training

def transform_question(input_text):
    """
    Transform arithmetic questions to use calculator tool format.
    
    Args:
        input_text: The input question text
        
    Returns:
        Transformed question with calculator tool tags
    """
    # First replace word operators
    operator_replacements = [
        (r'\bplus\b', '+'),     # Replace "plus" with "+"
        (r'\bminus\b', '-'),    # Replace "minus" with "-"
        (r'\btimes\b', '*')     # Replace "times" with "*"
    ]
    
    result = input_text
    for pattern, replacement in operator_replacements:
        result = re.sub(pattern, replacement, result)
    
    # Then handle the overall structure
    question_regex = r'Question: What is (.*?)\?\nAnswer:'
    question_replacement = r'<tool:calculator>\1</tool>'
    
    transformed = re.sub(question_regex, question_replacement, result) 
    transformed = transformed.replace('</tool>?', '</tool>')

    return transformed

def prepare_arithmetic_datasets(train_split=SPLIT, random_seed=SEED):
    """
    Prepare arithmetic datasets for all configurations.
    
    Args:
        train_split: Proportion of data to use for training (default: 0.8)
        random_seed: Random seed for reproducibility
    
    Returns:
        tuple: (train_dict, test_dict, train_transformed_dict, test_transformed_dict) 
               where each is a dictionary mapping config names to datasets
    """
    # Setting random seed for reproducibility
    np.random.seed(random_seed)

    # List of available configurations
    configs = [
        # 'arithmetic_1dc',
        'arithmetic_2da',
        # 'arithmetic_2dm',
        # 'arithmetic_2ds',
        # 'arithmetic_3da',
        # 'arithmetic_3ds',
        # 'arithmetic_4da',
        # 'arithmetic_4ds',
        # 'arithmetic_5da',
        # 'arithmetic_5ds'
    ]
    
    train_dict = {}
    test_dict = {}
    train_transformed_dict = {}
    test_transformed_dict = {}
    
    for config in configs:
        print(f"Processing dataset configuration: {config}")
        
        try:
            # Load the dataset for the current configuration
            dataset = load_dataset("EleutherAI/arithmetic", config)
            
            # The dataset only has a validation split with context/completion fields
            validation_data = dataset['validation']
            
            # Process examples to have consistent format (original version)
            def process_example(example):
                return {
                    "question": example["context"],
                    "final_answer": example["completion"].strip()
                }
            
            # Process examples for transformed version
            def process_example_transformed(example):
                return {
                    "question": example["context"],
                    "final_answer": transform_question(example["context"])
                }
            
            # Create both original and transformed datasets
            processed_data = validation_data.map(
                process_example,
                remove_columns=["context", "completion"]  # Remove original columns
            )
            processed_data_transformed = validation_data.map(
                process_example_transformed,
                remove_columns=["context", "completion"]  # Remove original columns
            )

            # Create train/test split manually - use same indices for both datasets
            num_examples = len(processed_data)
            indices = np.random.permutation(num_examples)
            train_size = int(train_split * num_examples)
            
            train_indices = indices[:train_size]
            test_indices = indices[train_size:]
            
            # Original datasets
            train_dataset = processed_data.select(train_indices)
            test_dataset = processed_data.select(test_indices)
            
            # Transformed datasets
            train_dataset_transformed = processed_data_transformed.select(train_indices)
            test_dataset_transformed = processed_data_transformed.select(test_indices)
            
            # Store in dictionaries
            train_dict[config] = train_dataset
            test_dict[config] = test_dataset
            train_transformed_dict[config] = train_dataset_transformed
            test_transformed_dict[config] = test_dataset_transformed
            
        except Exception as e:
            print(f"Error processing {config}: {e}")
    
    return {
        "train_dict": train_dict,
        "test_dict": test_dict,
        "train_transformed_dict": train_transformed_dict,
        "test_transformed_dict": test_transformed_dict
    }

def combine_and_tokenize(datasets, tokenizer, path=TOOL_TRAIN_DATASET_PATH):
    """
    Combine multiple datasets into a single dataset, tokenized.
    
    Args:
        datasets: List of datasets to combine
    
    Returns:
        Combined dataset
    """
    if os.path.exists(path):
        train_dataset = Dataset.load_from_disk(path)
    else:
        # For arithmetic datasets, combine all datasets into one
        print("Combining arithmetic datasets for training...")
        combined_train_dataset = []
        for key, dataset in datasets.items():
            combined_train_dataset.extend(dataset)
        train_dataset = Dataset.from_list(combined_train_dataset)
        print(f"Combined {len(train_dataset)} examples for training")
        train_dataset = train_dataset.map(
            lambda examples: preprocess_for_training(examples, tokenizer),
            batched=True,
            remove_columns=train_dataset.column_names
        )   # YOU NEED TO DELETE THE FOLDER "data/preprocessed_train_dataset" IF YOU SWITCH MODELS
        Dataset.save_to_disk(train_dataset,path)
        print(f"Saved {len(train_dataset)} examples to data")
    return train_dataset