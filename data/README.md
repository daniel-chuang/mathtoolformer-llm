# Data

For our data, we use a huggingface dataset called [EleutherAI/arithmetic](https://huggingface.co/datasets/EleutherAI/arithmetic), which can be accessed via the link above or by getting the dataset programmatically with Huggingface's "load_dataset" function.

```python
from datasets import load_dataset
configs = [
    'arithmetic_1dc',
    'arithmetic_2da',
    'arithmetic_2dm',
    'arithmetic_2ds',
    'arithmetic_3da',
    'arithmetic_3ds',
    'arithmetic_4da',
    'arithmetic_4ds',
    'arithmetic_5da',
    'arithmetic_5ds'
]
dataset = load_dataset("EleutherAI/arithmetic", configs[0], split="validation")
```

## Dataset Usage

We combine all of the different configs from the dataset, shuffle it, and sample the top n for our fine tuning process.
```
Transformed dataset:
{'completion': '<tool:calculator>6204 + 2521</tool>', 'prompt': 'Question: What is 6204 plus 2521?\nAnswer:'}
{'completion': '<tool:calculator>53441 + 19903</tool>', 'prompt': 'Question: What is 53441 plus 19903?\nAnswer:'}
{'completion': '<tool:calculator>0 - 81</tool>', 'prompt': 'Question: What is 0 minus 81?\nAnswer:'}
{'completion': '<tool:calculator>934 - 935</tool>', 'prompt': 'Question: What is 934 minus 935?\nAnswer:'}
{'completion': '<tool:calculator>42324 + 24298</tool>', 'prompt': 'Question: What is 42324 plus 24298?\nAnswer:'}
{'completion': '<tool:calculator>7116 + 8508</tool>', 'prompt': 'Question: What is 7116 plus 8508?\nAnswer:'}
{'completion': '<tool:calculator>5381 + 7791</tool>', 'prompt': 'Question: What is 5381 plus 7791?\nAnswer:'}
{'completion': '<tool:calculator>99930 - 85074</tool>', 'prompt': 'Question: What is 99930 minus 85074?\nAnswer:'}
{'completion': '<tool:calculator>3 + 4</tool>', 'prompt': 'Question: What is 3 plus 4?\nAnswer:'}
{'completion': '<tool:calculator>7 * 50</tool>', 'prompt': 'Question: What is 7 times 50?\nAnswer:'}
Original dataset:
{'context': 'Question: What is 6204 plus 2521?\nAnswer:', 'completion': ' 8725'}
{'context': 'Question: What is 53441 plus 19903?\nAnswer:', 'completion': ' 73344'}
{'context': 'Question: What is 0 minus 81?\nAnswer:', 'completion': ' -81'}
{'context': 'Question: What is 934 minus 935?\nAnswer:', 'completion': ' -1'}
{'context': 'Question: What is 42324 plus 24298?\nAnswer:', 'completion': ' 66622'}
{'context': 'Question: What is 7116 plus 8508?\nAnswer:', 'completion': ' 15624'}
{'context': 'Question: What is 5381 plus 7791?\nAnswer:', 'completion': ' 13172'}
{'context': 'Question: What is 99930 minus 85074?\nAnswer:', 'completion': ' 14856'}
{'context': 'Question: What is 3 plus 4?\nAnswer:', 'completion': ' 7'}
{'context': 'Question: What is 7 times 50?\nAnswer:', 'completion': ' 350'}

Dataset sizes:
Total combined dataset: 1000
Training set: 900
Evaluation set: 100
Individual dataset sizes:
  arithmetic_1dc: 2000
  arithmetic_2da: 2000
  arithmetic_2dm: 2000
  arithmetic_2ds: 2000
  arithmetic_3da: 2000
  arithmetic_3ds: 2000
  arithmetic_4da: 2000
  arithmetic_4ds: 2000
  arithmetic_5da: 2000
  arithmetic_5ds: 2000
```