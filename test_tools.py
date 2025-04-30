from inference.tool_execution import extract_and_execute_tools

print("Tool Execution Test")

x = extract_and_execute_tools("What is 2 + 2? <tool:calculator>2 + 2</tool>")
print(x)