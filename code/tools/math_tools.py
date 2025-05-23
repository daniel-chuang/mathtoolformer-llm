import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from .registry import ToolRegistry
from tools import tools  # Import the singleton instance

@tools.register("calculator")
def calculator(expression):
    """
    Basic calculator for arithmetic expressions.

    Args:
        expression: Math expression as string.

    Returns:
        Result of calculation.
    """
    try:
        # Clean and sanitize the input
        expression = expression.strip()
        # Evaluate using sympy for safety and precision
        result = sp.sympify(expression)
        return str(result)
    except Exception as e:
        return f"Error in calculation: {str(e)}"

@tools.register("solve_equation")
def solve_equation(equation):
    """
    Solve algebraic equations using SymPy.

    Args:
        equation: String representation of equation like "x**2 + 2*x - 3 = 0".

    Returns:
        Solutions as string.
    """
    try:
        # Split by equals sign and move everything to LHS
        if "=" in equation:
            left, right = equation.split("=", 1)
            left = left.strip()
            right = right.strip()
            equation = f"({left})-({right})"

        # Parse the equation and find the variables
        expr = parse_expr(equation)
        variables = list(expr.free_symbols)

        if not variables:
            return "No variables found in equation"

        # Sort variables alphabetically for consistent output
        variables.sort(key=lambda x: x.name)

        # Solve for the first variable
        solutions = sp.solve(expr, variables[0])

        # Format the solutions
        if not solutions:
            return f"No solutions found for {variables[0]}"

        solutions_text = ", ".join([str(sol) for sol in solutions])
        return f"{variables[0]} = {solutions_text}"

    except Exception as e:
        return f"Error solving equation: {str(e)}"