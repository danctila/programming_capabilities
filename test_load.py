import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "HuggingFaceTB/SmolLM2-360M"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model onto the GPU
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Prepare a prompt
prompt = "def hello_world():"

# Tokenize
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# Generate
with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_new_tokens=50,
        temperature=0.2,
        do_sample=True
    )

output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output)