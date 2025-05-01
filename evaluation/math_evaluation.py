import re
from utils.logging_utils import ResultsLogger
import torch

def evaluate_math_performance(model, tokenizer, test_dataset, dataset_name=None, model_name=None, device='device'):
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
    model.to(device)
    for (i, example) in enumerate(test_dataset):
        # Extract question and expected answer from the example
        question = example["question"]
        expected_answer = example["final_answer"]
        
        # Try to extract question type if available
        question_type = example.get("type", None)
        
        if not expected_answer:
            continue

        # Generate model's response
        prompt = f'Question: {question}\n\n. Do not use any tools. Answer the question directly with arithmetic.\n\n'
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.0,
                num_beams=2,
                top_p=1.0,
                do_sample=False
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(response + "\n\n")

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