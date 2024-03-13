import os, json
from glob import glob
import numpy as np, pandas as pd


BASE_MODELS = ["alpaca", "vicuna", "stable-vicuna"]


def parse_examples(model, result_dir):

    df = {
        "BaseModel": [], "TargetModel": [], "Transferability": [], "Accuracy": [], "AccuracyNoRefusal": [],
        "Task": [], "RR+NE": [], "TaskDataCount": []
    }

    failures = {model: {}}
    for target_model in [model]:
        model_file = target_model
        for base_model in BASE_MODELS:
            if not os.path.exists(os.path.join(result_dir, model_file, f"{base_model}-demo.json")):
                print(f"{os.path.join(result_dir, model_file, f'{base_model}-demo.json')} does not exist.)")
                continue
            with open(os.path.join(result_dir, model_file, f"{base_model}-demo.json")) as f:
                j = json.load(f)
                for task in j.keys():
                    if task not in failures[target_model]:
                        failures[target_model][task] = []

                    df["BaseModel"].append(base_model)
                    df["TargetModel"].append(target_model.lstrip(result_dir))
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
                    queries = j[task]["requests"]
                    refusals[task] = [

                        y["choices"][0]["message"]["content"] for x, y in zip(preds, responses) if x == -1
                    ]

                    failures[target_model][task].extend(
                        [
                            {
                                "Query": q["messages"][-1]["content"],
                                "Outputs": y["choices"][0]["message"]["content"]
                            } for q, x, y in zip(queries, preds, responses) if x != y
                        ]
                    )

    return failures


def extract_adv_examples(model, sub_perspective, result_dir):
    adv_dir = os.path.join(result_dir, "adv-glue-plus-plus")
    failures = parse_examples(model, adv_dir)
    print(failures[model].keys())
    return failures[model][sub_perspective]


if __name__ == "__main__":
    failure_examples = extract_adv_examples("vertexai/gemini-pro", "mnli", "results")
    print(failure_examples)
