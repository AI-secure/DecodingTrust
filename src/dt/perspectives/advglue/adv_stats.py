import os, json
import numpy as np, pandas as pd

base_models = ["alpaca", "vicuna", "stable-vicuna",]
target_models = ["together/zhangcetogether/ft-15d21ae7-1dfe-48e6-b2f3-12306e3671b6-2023-10-08-13-47-00"]
base_dir = "./results/adv-glue-plus-plus/"
tasks = ["sst2", "qqp", "mnli"]

df = {"BaseModel": [], "TargetModel": [], "Task": [], "RobustAccuracy": [], "Transferability": [], "RR": []}
for target_model in target_models:
    print(target_model)
    for base_model in base_models:
        for task in tasks:
            if not os.path.exists(os.path.join(base_dir, target_model, f"{base_model}-demo-{task}.json")):
                continue
            with open(os.path.join(base_dir, target_model, f"{base_model}-demo-{task}.json")) as f:
                j = json.load(f)
            df["BaseModel"].append(base_model)
            df["TargetModel"].append(target_model)
            df["Task"].append(task)
            df["RobustAccuracy"].append(np.mean(np.array(j[task]["predictions"]) == np.array(j[task]["labels"])))
            df["Transferability"].append(np.mean(np.array(j[task]["predictions"]) != np.array(j[task]["labels"])))
            df["RR"].append(np.mean(np.array(j[task]["predictions"]) == -1))

            refusals = {}
            preds = j[task]["predictions"]
            responses = j[task]["responses"]
            refusals[task] = [y["choices"][0]["message"]["content"] for x, y in zip(preds, responses) if x == -1]
            with open(os.path.join(base_dir, f"{target_model}/{base_model}-demo-{task}-refusal.json"), "w") as dest:
                json.dump(refusals, dest, indent=4)
                print(target_model, base_model)
                print(json.dumps(refusals, indent=4))
df = pd.DataFrame(df)
df.to_csv(os.path.join(base_dir, "summary.csv"), index=False)
del df["BaseModel"]
print(df.groupby(["TargetModel", "Task"]).mean())
