from constants import TOOL_PATTERN

def evaluate_tool_usage(model, tokenizer, test_dataset):
    """
    Evaluate the model's ability to use tools correctly.

    Args:
        model: Fine-tuned model.
        tokenizer: Tokenizer.
        test_dataset: Dataset containing math problems.

    Returns:
        Dict with evaluation metrics.
    """
    results = {
        "total": len(test_dataset),
        "tool_invoked": 0,
        "correct_tool": 0,
        "valid_args": 0
    }

    for example in test_dataset:
        # Extract question from the example
        question = example["question"]

        # Generate model's response
        prompt = f"Question: {question}\n\nTo solve this problem, I'll use calculation tools."
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                top_p=0.95,
                do_sample=True
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Check if a tool was invoked
        tool_matches = re.findall(TOOL_PATTERN, response)
        if tool_matches:
            results["tool_invoked"] += 1

            # Check if the correct tool was invoked
            if any(tool for tool, _ in tool_matches if tool in ["calculator", "solve_equation", "wolfram_alpha"]):
                results["correct_tool"] += 1

                # Check for valid arguments
                if all(args.strip() for _, args in tool_matches):
                    results["valid_args"] += 1

    # Calculate percentages
    results["tool_invocation_rate"] = results["tool_invoked"] / results["total"]
    results["correct_tool_rate"] = results["correct_tool"] / results["tool_invoked"] if results["tool_invoked"] > 0 else 0
    results["valid_args_rate"] = results["valid_args"] / results["correct_tool"] if results["correct_tool"] > 0 else 0

    return results