import re
from datasets import load_dataset

def extract_gsm8k_solution(answer_text):
    """
    Extract calculations, steps, and final answer from a GSM8K solution.
    
    Args:
        answer_text: The answer string from the GSM8K dataset
        
    Returns:
        dict: Contains calculations, steps, and final answer
    """
    # Extract the final answer (after the "####" marker)
    final_answer_match = re.search(r'####\s*(\d+)', answer_text)
    final_answer = final_answer_match.group(1) if final_answer_match else None
    
    # Extract all calculation steps
    # The pattern looks for expressions like "X/Y = <<X/Y=Z>>Z" or "X+Y = <<X+Y=Z>>Z"
    calculation_pattern = r'(\S+)\s*([+\-*/])\s*(\S+)\s*=\s*<<(\S+)=(\S+)>>(\S+)'
    calculations = re.findall(calculation_pattern, answer_text)
    
    # Format the calculation steps
    formatted_calcs = []
    for calc in calculations:
        formatted_calcs.append({
            'left_operand': calc[0],
            'operator': calc[1],
            'right_operand': calc[2],
            'expression': calc[3],
            'result_raw': calc[4],
            'result_displayed': calc[5]
        })
    
    # Extract the reasoning steps (split by newlines)
    steps = [s.strip() for s in answer_text.split('\n') if s.strip()]
    
    # Create clean steps without the calculation markup
    clean_steps = []
    for step in steps:
        # Remove the "<<expression=result>>" format but keep the rest
        clean_step = re.sub(r'<<[^>]+>>', '', step)
        clean_steps.append(clean_step)
    
    return {
        'calculations': formatted_calcs,
        'steps': steps,
        'clean_steps': clean_steps,
        'final_answer': final_answer
    }

def format_for_training(example, include_calculator=True):
    """
    Format a GSM8K example for training with the calculator tool.
    
    Args:
        example: A GSM8K dataset example
        include_calculator: Whether to include calculator tool usage
        
    Returns:
        dict: Contains formatted text for training
    """
    question = example["question"]
    
    # Extract solution components
    solution = extract_gsm8k_solution(example["answer"])
    
    # Format with calculator tool use
    formatted_text = f"Question: {question}\n\n"
    formatted_text += f"To solve this problem, I'll use calculation tools.\n"
    
    if include_calculator and solution['calculations']:
        for calc in solution['calculations']:
            expression = f"{calc['left_operand']} {calc['operator']} {calc['right_operand']}"
            result = calc['result_displayed']
            formatted_text += f"<tool:calculator>{expression}</tool> = {result}\n"
    
    formatted_text += f"Therefore, the answer is {solution['final_answer']}."
    
    return {"text": formatted_text}

def prepare_gsm8k_dataset(with_calculator=True):
    """
    Prepare GSM8K dataset with extracted answers and calculations.
    
    Args:
        with_calculator: Whether to format examples with calculator tool
        
    Returns:
        tuple: Processed training and test datasets
    """
    # Load the dataset
    dataset = load_dataset("gsm8k", "main")
    
    # Process the dataset
    def process_example(example):
        # Extract solution components
        solution = extract_gsm8k_solution(example["answer"])
        
        # Add extracted components to the example
        processed = {
            "question": example["question"],
            "answer": example["answer"],
            "final_answer": solution["final_answer"],
            "calculations": solution["calculations"],
            "steps": solution["clean_steps"]
        }
        
        # Add the formatted text for training
        if with_calculator:
            processed["formatted_text"] = format_for_training(example)["text"]
        
        return processed
    
    # Process both training and test sets
    processed_train = dataset["train"].map(process_example)
    processed_test = dataset["test"].map(process_example)
    
    return processed_train, processed_test