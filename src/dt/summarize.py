import os
import json

import numpy as np
import pandas as pd
from glob import glob

RESULT_DIR = "./results"


def get_adv_demo_scores(breakdown=False):
    fs = glob(os.path.join(RESULT_DIR, "adv_demonstration", "**", "*_score.json"), recursive=True)
    # TODO: This won't work if OpenAI or Anthropic models start to have underscores
    model_names = [os.path.basename(f).removesuffix("_score.json").replace("_", "/", 2) for f in fs]
    model_scores = {}
    model_rejections = {}
    model_breakdowns = {}
    for f, model_name in zip(fs, model_names):
        with open(f) as src:
            scores = json.load(src)
            if not scores:
                continue
            scores["adv_demonstration"] *= 100
            scores["adv_demonstration_rej"] *= 100
            model_scores[model_name] = scores["adv_demonstration"]
            model_rejections[model_name] = scores["adv_demonstration_rej"]
            model_breakdowns[model_name] = scores
    if breakdown:
        return model_breakdowns
    else:
        return {"score": model_scores, "rejection_rate": model_rejections}


def get_advglue_scores(breakdown=False):
    print(os.path.join(RESULT_DIR, "adv-glue-plus-plus", "summary.json"))
    with open(os.path.join(RESULT_DIR, "adv-glue-plus-plus", "summary.json")) as src:
        scores = json.load(src)
    model_scores = {k.removeprefix("/"): v * 100 for k, v in scores["Accuracy"].items()}
    model_rejections = {k.removeprefix("/"): v * 100 for k, v in scores["RR+NE"].items()}
    if breakdown:
        with open(os.path.join(RESULT_DIR, "adv-glue-plus-plus", "breakdown.json")) as src:
            breakdown_scores = json.load(src)
            for k, v in breakdown_scores.items():
                for task_name in v:
                    breakdown_scores[k][task_name]["acc"] *= 100
            return breakdown_scores
    else:
        return {"score": model_scores, "rejection_rate": model_rejections}


def get_fairness_scores(breakdown=False):
    fs = glob(os.path.join(RESULT_DIR, "fairness", "**", "final_scores.json"), recursive=True)
    model_names = [
        os.path.dirname(x).removeprefix(os.path.join(RESULT_DIR, "fairness", "results")).removeprefix("/") for x in fs
    ]
    model_scores = {}
    model_rejections = {}
    model_breakdown = {}
    for f, model_name in zip(fs, model_names):
        with open(f) as src:
            scores = json.load(src)
            model_scores[model_name] = scores["fairness score"]
            model_rejections[model_name] = scores.get("rejection rate", None)
            # model_breakdown[model_name] = {
            #     "zero-shot": {
            #         "Acc": "",
            #         "Demographic Parity Difference": "",
            #         "Equalized Odds Difference": " "
            #     },
            #     "few-shot-1": {},
            #     "few-shot-2": {},
            #     "Averaged Score": {},
            # }
            model_breakdown[model_name] = scores
    if breakdown:
        return model_breakdown
    else:
        return {"score": model_scores, "rejection_rate": model_rejections}


def get_ethics_scores(breakdown=False):
    df = pd.read_json(os.path.join(RESULT_DIR, "machine_ethics", "generations", "scores.jsonl"), lines=True)
    if breakdown:
        keys = ["avg_fpr_ev", "avg_fpr_jb", "acc_few", "acc_zero"]
        df = df[df["dataset"] == "ethics_commonsense_short"].drop_duplicates()
        df = df[["model"] + keys]
        df = df.rename({
            "acc_few": "few-shot benchmark",
            "acc_zero": "zero-shot benchmark",
            "avg_fpr_jb": "jailbreak",
            "avg_fpr_ev": "evasive"
        }, axis=1)

        model_breakdown = {}
        for record in df.to_dict(orient="records"):
            model_breakdown["model"] = {
                "few-shot benchmark": record["few-shot benchmark"],
                "zero-shot benchmark": record["zero-shot benchmark"],
                "jailbreak": record["jailbreak"],
                "evasive": record["evasive"]
            }
        # "jailbreak": {
        #     "brittleness": 1.0
        # },
        # "evasive": {
        #     "brittleness": 1.0
        # },
        # "zero-shot benchmark": {
        #     "performance": 0.533902323376007
        # },
        # "few-shot benchmark": {
        #     "performance": 0.683262209577999
        # }
        return model_breakdown
    else:
        keys = ["agg_score", "ref_rate"]
        df = df[df["dataset"] == "ethics_commonsense_short"].drop_duplicates().set_index("model")[keys]
        return df.to_dict()


def get_ood_scores(breakdown=False):
    path_prefix = os.path.join(RESULT_DIR, "ood", "results/")
    fs = glob(os.path.join(path_prefix, "**", "final_scores.json"), recursive=True)
    model_names = [os.path.dirname(f).removeprefix(path_prefix) for f in fs]
    model_scores = {}
    model_rejections = {}
    model_breakdowns = {}
    for f, model_name in zip(fs, model_names):
        with open(f) as src:
            scores = json.load(src)
            if not scores:
                continue
            model_scores[model_name] = scores.get("score", None)
            model_rejections[model_name] = scores.get("rr", None)
            model_breakdowns[model_name] = scores
    if breakdown:
        return model_breakdowns
    else:
        return {"score": model_scores, "rejection_rate": model_rejections}


def get_privacy_scores(breakdown=False):
    df = pd.read_json(os.path.join(RESULT_DIR, "privacy", "generations", "scores.jsonl"), lines=True)
    # TODO: This won't work if OpenAI or Anthropic models start to have underscores
    df["model"] = df["model"].apply(lambda x: x.replace("_", "/", 2))
    if breakdown:
        keys = ["enron", "pii", "understanding"]
        model_breakdown = {}
        models = df["model"].unique().tolist()
        for model in models:
            model_breakdown[model] = {}
            for key in keys:
                df_key = df[df["dataset"] == key].drop_duplicates().set_index("model")
                model_breakdown[model][key] = {"asr": df_key.loc[model, "leak_rate"]}
        return model_breakdown
    else:
        df = df[df["dataset"] == "all"].drop_duplicates().set_index("model")
        return df[["privacy_score", "reject_rate", "privacy_score_wo_reject"]].to_dict()


def get_stereotype_scores(breakdown=False):
    path_prefix = os.path.join(RESULT_DIR, "stereotype", "generations/")
    fs = glob(os.path.join(path_prefix, "**", "25_compiled.json"), recursive=True)
    model_names = [os.path.dirname(f).removeprefix(path_prefix) for f in fs]
    model_scores = {}
    model_rejections = {}
    model_breakdown = {}
    for f, model_name in zip(fs, model_names):
        with open(f) as src:
            scores = json.load(src)
            if not scores:
                continue
            model_scores[model_name] = scores["overall_score"] * 100
            model_rejections[model_name] = scores["overall_rejection_rate"] * 100
            model_breakdown[model_name] = scores
    if breakdown:
        return model_breakdown
    else:
        return {"score": model_scores, "rejection_rate": model_rejections}


def get_toxicity_scores(breakdown=False):
    path_prefix = os.path.join(RESULT_DIR, "toxicity", "user_prompts", "generations/")
    fs = glob(os.path.join(path_prefix, "**", "report.jsonl"), recursive=True)
    model_names = [os.path.dirname(f).removeprefix(path_prefix) for f in fs]
    model_scores = {}
    model_rejections = {}
    model_breakdown = {}
    for f, model_name in zip(fs, model_names):
        with open(f) as src:
            scores = json.load(src)
            if not scores:
                continue
            score_key = os.path.join(model_name, "aggregated-score")
            if score_key not in scores:
                continue
            if scores[score_key] is not None:
                scores[score_key] *= 100
            model_scores[model_name] = scores[score_key]
            model_rejections[model_name] = np.mean([v for k, v in scores.items() if k.endswith("rej_rates")])
            model_breakdown[model_name] = scores
    if breakdown:
        return model_breakdown
    else:
        return {"score": model_scores, "rejection_rate": model_rejections}


def summarize_results():
    summarized_results = {
        "aggregated_results": {
            "adv_demonstration": get_adv_demo_scores(),
            "adv-glue-plus-plus": get_advglue_scores(),
            "fairness": get_fairness_scores(),
            "machine_ethics": get_ethics_scores(),
            "ood": get_ood_scores(),
            "privacy": get_privacy_scores(),
            "stereotype": get_stereotype_scores(),
            "toxicity": get_toxicity_scores()
        },
        "breakdown_results": {
            "adv_demonstration": get_adv_demo_scores(True),
            "adv-glue-plus-plus": get_advglue_scores(True),
            "fairness": get_fairness_scores(True),
            "machine_ethics": get_ethics_scores(True),
            "ood": get_ood_scores(True),
            "privacy": get_privacy_scores(True),
            "stereotype": get_stereotype_scores(True),
            "toxicity": get_toxicity_scores(True)
        }
    }

    summarized_results = sort_keys(summarized_results)

    with open(os.path.join(RESULT_DIR, "summary.json"), "w") as f:
        json.dump(summarized_results, f, indent=4)
        print(json.dumps(summarized_results["aggregated_results"], indent=4))

    return summarized_results


def sort_keys(obj):
    if isinstance(obj, dict):
        return {k: sort_keys(obj[k]) for k in sorted(obj.keys())}
    elif isinstance(obj, list):
        return [sort_keys(element) for element in obj]
    else:
        return obj


if __name__ == "__main__":
    summarize_results()
