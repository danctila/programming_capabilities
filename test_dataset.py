from datasets import load_dataset

ds = load_dataset("nuprl/engineering-llm-systems", "humaneval", split="test")
print(ds[0])  # should show the first HumanEval example
