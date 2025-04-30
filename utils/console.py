def isYes(input_str):
    return input_str.lower() in ["yes", "y"]

def printc(text, color_code="033"):
    """
    Print text with a specific color code.
    
    Args:
        text (str): The text to print.
        color_code (str): The ANSI color code.
    """
    print(f"\033[{color_code}m{text}\033[0m")

def inputc(prompt, color_code="033"):
    """
    Get user input with a specific color code.
    
    Args:
        prompt (str): The prompt to display.
        color_code (str): The ANSI color code.
        
    Returns:
        str: User input.
    """
    return input(f"\033[{color_code}m{prompt}\033[0m")

def print_section(text, color_code="032"):
    printc("" + "=" * 50, color_code=color_code)
    printc(f"\n{text}\n", color_code=color_code)
    printc("" + "=" * 50, color_code=color_code)