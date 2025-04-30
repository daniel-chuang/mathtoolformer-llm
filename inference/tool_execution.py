import re
import torch
from constants import TOOL_PATTERN
import tools

def extract_and_execute_tools(text):
    """Parse a text for tool calls and execute them.

    Args:
        text: Text containing tool calls.

    Returns:
        Text with tool call results inserted.
    """
    # Find all tool calls in the text
    tool_calls = re.finditer(TOOL_PATTERN, text)
    result_text = text

    # Process each tool call
    for match in tool_calls:
        tool_name = match.group(1)
        args_str = match.group(2)

        # Clean up args - split by comma if multiple
        args = [arg.strip() for arg in args_str.split(",")]

        # Execute the tool
        tool_result = tools.execute_tool(tool_name, args)

        # Replace the tool call with the result
        start, end = match.span()
        before = result_text[:start]
        after = result_text[end:]

        # Format for readability
        result_text = f"{before}\n[Tool Result: {tool_result}]\n{after}"
    
    return result_text

def generate_with_tools(model, tokenizer, prompt, max_new_tokens=512):
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
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )

    # Get the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

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