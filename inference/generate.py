import torch

def generate(inputs, model, tokenizer, max_new_tokens=150):
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=0.95,
            # do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )
        return outputs