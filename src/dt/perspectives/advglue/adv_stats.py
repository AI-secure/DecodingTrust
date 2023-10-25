import os, json
from glob import glob
import numpy as np, pandas as pd


RESULT_DIR = "./results/adv-glue-plus-plus"
ORGANIZATIONS = ["openai", "hf", "together", "anthropic"]  # TODO: refactor when we have the central database
BASE_MODELS = ["alpaca", "vicuna", "stable-vicuna"]


def main():
    benign_files = glob(os.path.join(RESULT_DIR, "**", "*benign*.json"), recursive=True)
    target_models = [os.path.dirname(x) for x in benign_files]

    df = {
        "BaseModel": [], "TargetModel": [], "Transferability": [], "Accuracy": [], "AccuracyNoRefusal": [],
        "Task": [], "RR+NE": [], "TaskDataCount": []
    }

    base_dir = ""
    for target_model in target_models:
        for base_model in BASE_MODELS:
            print(os.path.join(base_dir, target_model, f"{base_model}-demo.json"))
            if not os.path.exists(os.path.join(base_dir, target_model, f"{base_model}-demo.json")):
                continue
            with open(os.path.join(base_dir, target_model, f"{base_model}-demo.json")) as f:
                j = json.load(f)
                for task in j.keys():
                    df["BaseModel"].append(base_model)
                    df["TargetModel"].append(target_model.removeprefix(RESULT_DIR))
                    df["Task"].append(task)
                    df["TaskDataCount"].append(len(j[task]["labels"]))

                    df["Accuracy"].append(
                        np.mean(np.array(j[task]["predictions"]) == np.array(j[task]["labels"]))
                    )

                    df["Transferability"].append(
                        np.mean(np.array(j[task]["predictions"]) != np.array(j[task]["labels"]))
                    )
                    refusal_mask = np.array(j[task]["predictions"]) == -1
                    df["RR+NE"].append(np.mean(refusal_mask))
                    df["AccuracyNoRefusal"].append(
                        np.mean(
                            np.array(j[task]["predictions"])[~refusal_mask] == np.array(j[task]["labels"])[
                                ~refusal_mask]
                        )
                    )
                refusals = {}
                for task in j.keys():
                    preds = j[task]["predictions"]
                    responses = j[task]["responses"]
                refusals[task] = [y["choices"][0]["message"]["content"] for x, y in zip(preds, responses) if x == -1]
                with open(os.path.join(base_dir, f"{target_model}/{base_model}-demo-refusal.json"), "w") as dest:
                    json.dump(refusals, dest, indent=4)
                    print(json.dumps(refusals, indent=4))
    df = pd.DataFrame(df)
    print(df)
    df.to_csv(os.path.join(RESULT_DIR, "task_breakdown.csv"), index=False)

    # task_weights = df.apply(lambda x: x["TaskDataCount"] / (df[df["Task"] == x["Task"]]["TaskDataCount"].unique(
    # ).sum()), axis=1) df["Accuracy"] *= task_weights df["AccuracyNoRefusal"] *= task_weights
    df.drop(["BaseModel", "Task"], axis=1, inplace=True)
    with open(os.path.join(RESULT_DIR, "summary.json"), "w") as dest:
        json.dump(df.groupby(["TargetModel",]).mean().to_dict(), dest, indent=4)


if __name__ == "__main__":
    main()
