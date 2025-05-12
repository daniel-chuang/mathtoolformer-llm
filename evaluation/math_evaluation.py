import re
from utils.logging_utils import ResultsLogger
import torch
from inference.tool_execution import inference

def evaluate_math_performance(model, tokenizer, test_dataset, use_tool, dataset_name=None, model_name=None):
    """
    Evaluate model performance on math problems.

    Args:
        model: The model to evaluate
        tokenizer: The tokenizer for the model
        test_dataset: Dataset containing math problems
        dataset_name: Name of the dataset
        model_name: Name of the model

    Returns:
        Dictionary with evaluation metrics.
    """
    print("Evaluating math performance...")
    correct = 0
    total = 0
    
    # Initialize logger
    logger = ResultsLogger()
    logger.log_model_info(model_name or "unknown_model", dataset_name or "unknown_dataset")

    for (i, example) in enumerate(test_dataset):
        # Extract question and expected answer from the example
        question = example["question"]
        expected_answer = example["final_answer"]
        
        # Try to extract question type if available
        question_type = example.get("type", None)
        
        if not expected_answer:
            continue

        # Generate model's response
        if use_tool:
            prompt = f"""
            You can solve math problems using tools. When you need to calculate, use <tool:calculator>expression</tool>.

            Example:
            Question: What is 25 + 17?
            To solve this, I'll use a calculator.
            <tool:calculator>25 + 17</tool>
            Output: 42
            The answer is 42.

            Now solve this:
            {question}\n\n
            Answer:
            """
        else:
            prompt = f"{question}\n\nAnswer:"
    
        print(f"------\n{question}\n------")
        response = inference(model, tokenizer, prompt, max_new_tokens=256, use_tool=use_tool)

        # Extract the last number mentioned in the response
        numbers = re.findall(r'-?\d+', response)  # Find all numbers in the response (with optional minus sign)
        model_answer = numbers[-1] if numbers else None  # Take the last number, or None if no numbers found

        # Check correctness
        is_correct = (model_answer and model_answer == expected_answer)
        if is_correct:
            correct += 1

        total += 1
        print(f"Expected: {expected_answer}\nModel: {model_answer if model_answer else 'No valid number found'}")
        print(f"Correct: {correct}, Total: {total}\n\n")

        # Log individual question result
        logger.log_question_result(
            question=question, 
            expected_answer=expected_answer, 
            model_answer=model_answer,
            question_type=question_type,
            correct=is_correct
        )

    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    
    # Log aggregated metrics
    metrics = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }
    logger.log_aggregated_metrics(metrics)
    
    # Compute statistics by question type
    type_stats = logger.compute_type_statistics()
    
    # Save results to files
    file_paths = logger.save_results()
    print(f"Results saved to: {file_paths['csv_path']}")
    print(f"Type statistics saved to: {file_paths['stats_path']}")

    return {
        "metrics": metrics,
        "type_statistics": type_stats,
        "file_paths": file_paths
    }