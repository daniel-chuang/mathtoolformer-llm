import re
from datasets import load_dataset

def extract_svamp_solution(equation, answer):
    """
    Extract calculations and final answer from an SVAMP solution.
    
    Args:
        equation: The equation string from the SVAMP dataset
        answer: The final answer string from the SVAMP dataset
        
    Returns:
        dict: Contains calculations and final answer
    """
    # Extract the final answer
    final_answer = str(answer).strip()  # Ensure the answer is a string and clean it
    
    # Extract all calculation steps (if any)
    calculation_pattern = r'(\S+)\s*([+\-*/])\s*(\S+)'  # Basic pattern for arithmetic operations
    calculations = re.findall(calculation_pattern, equation)
    
    # Format the calculation steps
    formatted_calcs = []
    for calc in calculations:
        formatted_calcs.append({
            'left_operand': calc[0],
            'operator': calc[1],
            'right_operand': calc[2],
            'expression': f"{calc[0]} {calc[1]} {calc[2]}",
            'result_raw': None,  # Placeholder since SVAMP doesn't provide intermediate results
            'result_displayed': None  # Placeholder
        })
    
    # Use the equation directly as a single step
    clean_steps = [equation.strip()]
    
    return {
        'calculations': formatted_calcs,
        'steps': clean_steps,
        'clean_steps': clean_steps,
        'final_answer': final_answer
    }

def format_for_training_svamp(example, include_calculator=True):
    """
    Format an SVAMP example for training with the calculator tool.
    
    Args:
        example: An SVAMP dataset example
        include_calculator: Whether to include calculator tool usage
        
    Returns:
        dict: Contains formatted text for training
    """
    question = example["Body"] + example["Question"]
    print(question)
    
    # Extract solution components
    solution = extract_svamp_solution(example["Equation"], example["Answer"])
    
    # Format with calculator tool use
    formatted_text = f"Question: {question}\n\n"
    formatted_text += f"To solve this problem, I'll use calculation tools.\n"
    
    if include_calculator and solution['calculations']:
        for calc in solution['calculations']:
            expression = calc['expression']
            formatted_text += f"<tool:calculator>{expression}</tool>\n"
    
    formatted_text += f"Therefore, the answer is {solution['final_answer']}."
    
    return {"text": formatted_text}

def prepare_svamp_dataset(with_calculator=True):
    """
    Prepare SVAMP dataset with extracted answers and calculations.
    
    Args:
        with_calculator: Whether to format examples with calculator tool
        
    Returns:
        tuple: Processed training and test datasets
    """
    # Load the dataset
    dataset = load_dataset("ChilleD/SVAMP")
    
    # Process the dataset
    def process_example(example):
        # Extract solution components
        solution = extract_svamp_solution(example["Equation"], example["Answer"])
        
        # Add extracted components to the example
        processed = {
            "question": example["Body"] + example["Question"],
            "answer": example["Answer"],
            "final_answer": solution["final_answer"],
            "calculations": solution["calculations"],
            "steps": solution["clean_steps"]
        }
        
        # Add the formatted text for training
        if with_calculator:
            processed["formatted_text"] = format_for_training_svamp(example)["text"]
        
        return processed
    
    # Process both training and test sets
    processed_train = dataset["train"].map(process_example)
    processed_test = dataset["test"].map(process_example)
    
    return processed_train, processed_test