# Deep Learning Project

## Overview
This project implements a Toolformer-style language model (LLM) that can utilize external tools for various tasks, including mathematical problem solving, course scheduling, and professor rating integration. The model is designed to improve its performance in mathematics and enhance user interaction through tool usage.

## Goals
1. **Quantifiable Improvement at Mathematics**
   - Utilize Sympy and Wolfram Alpha API to solve complex mathematics problems.
   - Benchmark against datasets like GSM8K and DeepMind's Mathematics Dataset.

2. **Integration with External Tools**
   - Teach the LLM to utilize:
     - Google Calendar API
     - Cornell Scheduler API
     - Rate My Professor API

3. **Visualization**
   - Create engaging visualizations using Manim for project presentations.

## Project Structure
```
final_project/
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── setup.py
│   │   ├── lora_config.py
│   │   └── training.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   ├── math_tools.py
│   │   ├── wolfram_alpha.py
│   │   ├── cornell_courses.py
│   │   ├── rate_professor.py
│   │   └── google_calendar.py
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── gsm8k.py
│   │   ├── cornell.py
│   │   └── combined.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── math_evaluation.py
│   │   └── tool_usage_evaluation.py
│   └── inference/
│       ├── __init__.py
│       ├── pipeline.py
│       └── tool_execution.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation
To install the required dependencies, run:
```
pip install -r requirements.txt
```

## Usage
1. **Run the main application**:
   Execute the `main.py` file to initialize the model, tools, datasets, and start the training and evaluation processes.
   
2. **Tool Usage**:
   The model can invoke various tools for specific tasks, such as solving equations, querying course information, and managing calendar events.

## Evaluation
The model's performance will be evaluated based on:
- Accuracy in solving mathematical problems.
- Effectiveness in utilizing external tools.
- User satisfaction through interaction with the model.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.