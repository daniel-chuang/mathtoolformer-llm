import re
import torch
from constants import TOOL_PATTERN
from tools import tools

def extract_and_execute_tools(text):
    """
    Extracts tool calls from text, executes them, and replaces the calls with their results.
    
    Args:
        text: Input text with potential tool calls in the format <tool:name>args</tool>
        
    Returns:
        Text with tool calls replaced by their results
    """
    # Define the pattern to match tool calls
    tool_pattern = r'<tool:(\w+)>(.*?)</tool>'
    
    # Find all matches
    matches = list(re.finditer(tool_pattern, text))
    
    # If no tools to execute, return the original text
    if not matches:
        return text
    
    # Process the text in reverse order to avoid offset issues when replacing
    result_text = text
    for match in reversed(matches):
        full_match = match.group(0)  # The entire tool call
        tool_name = match.group(1)   # The tool name
        tool_args = match.group(2)   # The tool arguments
        
        try:
            # Execute the tool and get the result
            tool_result = tools.execute_tool(tool_name, tool_args)
            # Replace the tool call with its result
            print("Tool Result:", tool_result)
            result_text = result_text[:match.start()] + str(tool_result) + result_text[match.end():]
        except Exception as e:
            # If execution fails, replace with an error message
            error_msg = f"[Tool Error: {str(e)}]"
            result_text = result_text[:match.start()] + error_msg + result_text[match.end():]
    
    return result_text

def inference(model, tokenizer, prompt, max_new_tokens=512, use_tool=True):
    """Generate a response with tool use capability.

    Args:
        model: Fine-tuned model.
        tokenizer: Tokenizer.
        prompt: User input prompt.
        max_new_tokens: Maximum number of tokens to generate.

    Returns:
        Generated response with tool outputs.
    """
    # Initial generation
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.05,
            top_p=0.95,
            do_sample=True
        )

    # Get the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if not use_tool:
        return generated_text

    # Process any tool calls in the generated text
    response_with_tool_results = extract_and_execute_tools(generated_text)

    # If there were tool calls, generate a follow-up response
    if response_with_tool_results != generated_text:
        # We need to generate again with the tool results included
        follow_up_inputs = tokenizer(response_with_tool_results, return_tensors="pt").to(model.device)

        with torch.no_grad():
            follow_up_outputs = model.generate(
                **follow_up_inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.95,
                do_sample=True
            )

        final_response = tokenizer.decode(follow_up_outputs[0], skip_special_tokens=True)
        return final_response

    return response_with_tool_results