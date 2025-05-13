from tools import tools
from inference.tool_execution import extract_and_execute_tools

# Use the calculator
result = tools.execute_tool("calculator", "3 * (2 + 2) * 3")
print(result)  # Output: 8

# Use the equation solver
solution = tools.execute_tool("solve_equation", "x**2 - 4 = 0")
print(solution)  # Output: x = -2, 2

# Test extract_and_execute_tools function
x_0 = "The answer is: <tool:calculator>33 * 54</tool>"
x = extract_and_execute_tools(x_0)
print(x)

# Test with multiple tools in a single string
y_0 = "The answer is: <tool:calculator>33 * 54</tool>, <tool:calculator>33 + 54 * 21</tool>"
y = extract_and_execute_tools(y_0)
print(y)

z_0 = "The answer is: <tool:calculator>33 * 54</tool>, <tool:solve_equation>x**2 - 4 = 0</tool>"
z = extract_and_execute_tools(z_0)
print(z)