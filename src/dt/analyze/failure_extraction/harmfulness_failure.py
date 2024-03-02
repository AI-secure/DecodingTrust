import json
import numpy as np
import os

def extract_harmfulness_examples(model, eval_method, result_dir):
    harmful_dir = os.path.join(result_dir, "harmfulness/generations", model)
    all_results = [f for f in os.listdir(harmful_dir) if f.endswith(f"{eval_method}_results.jsonl")]
    failure_cases = []
    for file in all_results:
        with open(os.path.join(harmful_dir, file), "r") as f:
            results = f.readlines()
            for result in results:
                result = json.loads(result)
                if result.get("jailbreak")==1:
                    failure_cases.append({"Query": result.get("prompt"), "Outputs": result.get("output")})
    return failure_cases





if __name__ == "__main__":
    model = "vertexai/gemini-pro"
    eval_method = "key_words" # choose the evaluation method from [key_words, gpt_4_judge]
    failure_examples = extract_harmfulness_examples(model, eval_method, "results")
    print(failure_examples)