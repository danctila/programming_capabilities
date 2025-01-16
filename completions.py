import os
import json
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

COMPLETIONS_DIR = "completions_2"
# exist_ok because I made the directory myself
os.makedirs(COMPLETIONS_DIR, exist_ok=True)

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.2
NUM_SAMPLES_PER_PROMPT = 20
STOP_STRINGS = ["\ndef", "\nclass", "\nif", "\nprint"]

def load_humaneval_dataset():
    ds = load_dataset("nuprl/engineering-llm-systems", "humaneval", split="test")
    return ds

def clip_generated_code(generated_text, stop_strings):
    stop_positions = []
    for s in stop_strings:
        idx = generated_text.find(s)
        if idx != -1:
            stop_positions.append(idx)
    if len(stop_positions) > 0:
        clip_idx = min(stop_positions)
        return generated_text[:clip_idx]
    else:
        return generated_text

def generate_completions_for_prompt(prompt, model, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    completions = []

    for _ in range(NUM_SAMPLES_PER_PROMPT):
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask, # was giving me warning about this
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id # was giving me warning about this
            )
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Remove the original prompt to get only the newly generated text
        completion = generated_text[len(prompt):]
        # Clip
        completion = clip_generated_code(completion, STOP_STRINGS)
        completions.append(completion)
    return completions

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenizer.pad_token = tokenizer.eos_token # was warning me about not having this


    model.eval()

    ds = load_humaneval_dataset()

    print(next(model.parameters()).device)

    for i in tqdm(range(len(ds)), desc="Generating completions"):
        # name should be fine
        problem_id = ds[i]["name"]
        prompt = ds[i]["prompt"]
        solutions = generate_completions_for_prompt(prompt, model, tokenizer, device)

        out_path = os.path.join(COMPLETIONS_DIR, f"{problem_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "problem_id": problem_id,
                    "prompt": prompt,
                    "completions": solutions,
                },
                f,
                indent=2
            )

if __name__ == "__main__":
    main()
