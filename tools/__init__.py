def __init__(self):
    self.tools = {}
    
def register(self, name):
    """Decorator to register a tool function"""
    def decorator(func):
        self.tools[name] = func
        return func
    return decorator

def execute_tool(self, tool_name, args):
    """Execute a registered tool with given arguments"""
    if tool_name not in self.tools:
        return f"Error: Tool '{tool_name}' not found"
    
    try:
        return self.tools[tool_name](args)
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"