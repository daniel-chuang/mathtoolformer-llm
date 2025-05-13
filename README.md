# Toolformers on Minimal Data

#  Introduction
[*Toolformer: Language Models Can Teach Themselves to Use Tools*](https://arxiv.org/abs/2302.04761) is a recent research paper on the now heavily utilized capabilities of LLMs (Large Language Models) to adapt to and use tools with limited fine-tuning and instruction. It analyzes the impact of tool-usage on standard benchmark performance as well as what happens to performance when the fine-tuned model is forced to operate without tool access.

# Chosen Result
We elected to focus on the improvements in LLM benchmark performance after limited fine-tuning on API-call usage examples. This is the most significant contribution of the paper as the improvments in performance are substantial and the ability to leverage APIs is useful accross the board for LLMs.

![Main Result Table](resources/main_table.png)

Notably, we also chose to extend the findings of the original paper to utilize a symbolic python calculator API to try and demonstrate the efficacy of small-model fine-tuning for users who are not billion dollar AI companies. 

# GitHub Contents

<pre> root/
├─ code/
├─ ├─ data/
│  ├─ evaluation/
│  ├─ inference/
│  ├─ model/
│  ├─ results/
│  ├─ tools/
│  ├─ utils/
│  ├─ constants.py
│  ├─ main.py
│  └─ test_tools.py
├─ data/
├─ poster/
├─ report/
├─ results/
├─ resources/
├─ .gitignore
├─ LICENSE
├─ README.md 
└─ requirements.txt </pre>

# Re-Implementation Details

## Approach
We largely mimic the approach taken by the authors of the *Toolformer* paper, electing to fine-tune an LLM on examples of tool usage directly to attempt to manipulate its latent representation to include tool usage.

Specifically, since the paper did not provide information about their fine tuning processes, we decided to fine tune via two methods.

Firstly, we did a pure fine tuning of the model, with causal data generated from combining the math question prompt and the correct answer together.

> For example, given the original data "Question: What is 4+5?\n Answer:", "9", we combine it to "Question: What is 4+5?\n Answer:9"

Secondly, we did a Instructional (prompt / completion) style fine tuning, with the math question as the prompt and the label being the toolformer call to the math question.

> For example, given the original data "Question: What is 4+5\n Answer:", "9", we get "Question: What is 4+5?\n Answer:", "\<tool:calculator\>4+5\</tool\>.

## Models, Datasets, Tools, Evaluation Metrics
We decided to train on Qwen 2.5 Math 1.5B [https://huggingface.co/Qwen/Qwen2.5-Math-1.5B] due to GPU resource constraints. Our evaluation metrics are entirely contained in the evaluation scripts in this project, and involve comparing numerical output to the actual computed answer of the math problems directly. This enables us to deduce accuracy values without having to hand-label or hand-generate data -- a crucial detail given our limited resources and time. 

## Challenges/Modifications
The size of the model was the primary hurdle for our reimplementation as we suspect a larger model would have significantly outperformed ours across nearly all metrics. In the future, we hope to re-attempt this project using a more powerful setup.

We spent upwards of 60 hours trying to fine tune our models, with dozens of attempts with different hyperparameters and code, but every single attempt yielded poor results. Moreover, we were learning how to use Huggingface and Weights and Biases for the first time while doing this project. We feel disappointed that we weren't able to get any good results even after so much effort put in.

We believe that we should've picked an easier paper to implement - this one was too difficult for us.

Here are some specific challenges we encountered:
1. Learning how to use Huggingface and Wandb
2. Using Trainer instead of SFTTrainer, which lead to much slower training because we didn't know
3. Not using LoRA, which led to 160H epoch times
4. Learning how to use LoRA
5. Using SFTTrainer instead of Trainer in a new python notebook to attempt to get results. We ended up getting a significant boost for pure fine tuning from 23% -> 73% on the 1dc config, but after trying to expand our dataset to encompass all types of arithmetic problems, our LLMs just ended up not even outputting any data at inference time anymore
6. Not knowing that we needed to add in special tokens to our model and tokenizer for our tool calls until it was too late
7. Every single time we tried to teach the LLM how to use toolformer, it failed. Here are some of the bizarre results we got:
   1. !!!!!!!!!!! repeated over and over again because we switched models, and our data was still on a previously cached version so the tokenization was wrong
   2. Trying multiple datasets, such as GSM8k, SVAMP, and our final selected arithmetic - nothing worked. Sometimes, we would even just get random python code out of our models
   3. Model asking itself new questions, not knowing when to output EOS token
   4. Most recently, the model just doesn't output anything at all

Even though we weren't able to get great results, we tried really hard on this project.

# Reproduction Steps

1. Clone Repository
2. Install required dependencies by running: <pre>pip install -r requirements.txt</pre>
  - Note: pytorch may need to be manually installed to be compatible with your GPU, or installed via conda
3. Add your HuggingFace access token to .env in the format: <pre>HUGGINGFACE_TOKEN="\<token\>"</pre>
4. Add your WANDB access token to .env in the same way
5. Change to `code/` directory with `cd code` or equivalent
6. Run `python main.py` and follow command line instructions to see our original attempt with different datasets and models
7. Run our python notebook `main.ipynb` to see our ipython notebook with only the arithmetic dataset
8. Run our python notebook `main2.ipynb` to see our ipython notebook with SFTTrainer

# Results/Insights
After rigorous testing, we have determined that the model we used does not reach the minimum representational threshold where tool usage becomes viable. The models (SMOLLM 135M and QWEN 2.5 Math 1.5B) we tried were significantly smaller than the original (GPT-J, 6.6), so this conclusion does not come as a total suprise.

Moreover, the paper was closed source, and there is very limited information online about how to implement toolformer, so we have extreme difficulty with implementation.

We don't know if our toolformer failed due to errors in our code, or because we chose the wrong base model (size / type), or because we chose the wrong hyperparameters.

# Conclusion
- We tried to reimplement the Toolformer paper, which has massive applications in the current LLM space
- We failed to produce the same results even after trying all sorts of code, datasets, models, hyperparameters, and configurations.
- We hope that someone out there can make better progress on this, as toolformer is a very important technology that lacks a good open source implementation.
- We are disappointed in our lack of results. We put in upwards of 60 hours in trying to replicate Toolformer, unfortunately to no avail.

# References
*Toolformer: Language Models Can Teach Themselves to Use Tools* ─ [arXiv:2302.04761](https://arxiv.org/abs/2302.04761)
Qwen 2.5 Math 1.5 B - [link](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B)
SmolLM 135M - [link](https://huggingface.co/HuggingFaceTB/SmolLM-135M)

# Acknowledgements
This project was done as a final project submission for *Computer Science 4782 - Introduction to Deep Learning* at Cornell University taught by Professors Killian Weinberger and Jennifer Sun. 