from .registry import ToolRegistry

# Create a singleton instance
tools = ToolRegistry()

# Now import the tools after creating the registry
from . import math_tools

# print(f"Available tools: {list(tools.tools.keys())}")

# Make tools available when importing the package
__all__ = ['tools']