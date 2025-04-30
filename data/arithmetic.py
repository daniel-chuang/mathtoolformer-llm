from datasets import load_dataset, Dataset
import numpy as np

def prepare_arithmetic_datasets(train_split=0.8):
    """
    Prepare arithmetic datasets for all configurations.
    
    Args:
        train_split: Proportion of data to use for training (default: 0.8)
    
    Returns:
        tuple: (train_dict, test_dict) where each is a dictionary mapping config names to datasets
    """
    # List of available configurations
    configs = [
        'arithmetic_1dc', 'arithmetic_2da', 'arithmetic_2dm', 'arithmetic_2ds',
        'arithmetic_3da', 'arithmetic_3ds', 'arithmetic_4da', 'arithmetic_4ds',
        'arithmetic_5da', 'arithmetic_5ds'
    ]
    
    train_dict = {}
    test_dict = {}
    
    for config in configs:
        print(f"Processing dataset configuration: {config}")
        
        try:
            # Load the dataset for the current configuration
            dataset = load_dataset("EleutherAI/arithmetic", config)
            
            # The dataset only has a validation split with context/completion fields
            validation_data = dataset['validation']
            
            # Process examples to have consistent format
            def process_example(example):
                return {
                    "question": example["context"],
                    "final_answer": example["completion"].strip()
                }
            
            processed_data = validation_data.map(process_example)
            
            # Create train/test split manually
            num_examples = len(processed_data)
            indices = np.random.permutation(num_examples)
            train_size = int(train_split * num_examples)
            
            train_indices = indices[:train_size]
            test_indices = indices[train_size:]
            
            train_dataset = processed_data.select(train_indices)
            test_dataset = processed_data.select(test_indices)
            
            # Store in dictionaries
            train_dict[config] = train_dataset
            test_dict[config] = test_dataset
            
        except Exception as e:
            print(f"Error processing {config}: {e}")
    
    return train_dict, test_dict