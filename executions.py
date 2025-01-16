import os
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset

COMPLETIONS_DIR = "completions_2"
# Could adjust higher to maybe 10ish
TIMEOUT_SECONDS = 5

def load_humaneval_dataset():
    ds = load_dataset("nuprl/engineering-llm-systems", "humaneval", split="test")
    return ds

def run_completion_with_tests(prompt, completion, test_code, timeout=5):
    temp_filename = "temp_solution.py"
    with open(temp_filename, "w", encoding="utf-8") as f:
        f.write(prompt)
        f.write(completion)
        f.write("\n")
        f.write(test_code)

    try:
        start = time.time()
        proc = subprocess.run(
            ["python", temp_filename],
            capture_output=True,
            timeout=timeout
        )
        # For debugging if needed later but we will see...
        #print("stdout:", proc.stdout.decode())
        #print("stderr:", proc.stderr.decode())

        if proc.returncode == 0:
            return True
        else:
            return False
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

def compute_pass_at_1(dataset):
    pass_count = 0
    total_count = 0

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for i in range(len(dataset)):
            problem_id = dataset[i]["name"]
            test_code = dataset[i]["tests"]
            completions_file = os.path.join(COMPLETIONS_DIR, f"{problem_id}.json")
            if not os.path.exists(completions_file):
                continue

            with open(completions_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            prompt = data["prompt"]
            completions = data["completions"]
            if len(completions) == 0:
                continue

            chosen_completion = completions[0]

            future = executor.submit(
                run_completion_with_tests,
                prompt,
                chosen_completion,
                test_code,
                TIMEOUT_SECONDS
            )
            futures.append(future)

        results = []
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception:
                results.append(False)

    pass_count = sum(1 for r in results if r)
    total_count = len(results)
    if total_count == 0:
        return 0.0
    return pass_count / total_count

def main():
    dataset = load_humaneval_dataset()
    pass_rate = compute_pass_at_1(dataset)
    print(f"Mean pass rate (pass@1) = {pass_rate * 100:.2f}%")

if __name__ == "__main__":
    main()

# Output (completions):
# Mean pass rate (pass@1) = 4.97%
# Mean pass rate (pass@1) = 0.62%
# Mean pass rate (pass@1) = 0.00%

# Output (completions_2):
# Mean pass rate (pass@1) = 3.11%
# Mean pass rate (pass@1) = 3.73%
# Mean pass rate (pass@1) = 1.86%
