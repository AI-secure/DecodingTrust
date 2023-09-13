import os, json
import numpy as np, pandas as pd

base_models = ["alpaca", "vicuna", "stable-vicuna"]
target_models = ["mosaicml/mpt-7b-chat", "togethercomputer/RedPajama-INCITE-7B-Chat", "tiiuae/falcon-7b-instruct",
                 "lmsys/vicuna-7b-v1.3", "chavinlo/alpaca-native", "meta-llama/Llama-2-7b-chat-hf"]
base_dir = "./data/advglue-plus-plus/results/"

df = {"BaseModel": [], "TargetModel": [], "Transferability": [], "Task": [], "RR": []}
for target_model in target_models:
    print(target_model)
    for base_model in base_models:
        if not os.path.exists(os.path.join(base_dir, target_model, f"{base_model}-demo.json")):
            continue
        with open(os.path.join(base_dir, target_model, f"{base_model}-demo.json")) as f:
            j = json.load(f)
            for task in j.keys():
                df["BaseModel"].append(base_model)
                df["TargetModel"].append(target_model)
                df["Task"].append(task)
                df["Transferability"].append(np.mean(np.array(j[task]["predictions"]) != np.array(j[task]["labels"])))
                df["RR"].append(np.mean(np.array(j[task]["predictions"]) == -1))
            refusals = {}
            for task in j.keys():
                preds = j[task]["predictions"]
                responses = j[task]["responses"]
            refusals[task] = [y["choices"][0]["message"]["content"] for x, y in zip(preds, responses) if x == -1]
            with open(os.path.join(base_dir, f"{target_model}/{base_model}-demo-refusal.json"), "w") as f:
                json.dump(refusals, f, indent=4)
                print(target_model, base_model)
                print(json.dumps(refusals, indent=4))
df = pd.DataFrame(df)
