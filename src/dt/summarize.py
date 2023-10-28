import os
import json
import pandas as pd
from glob import glob


RESULT_DIR = "./results"


def get_adv_demo_scores():
    fs = glob(os.path.join(RESULT_DIR, "adv_demonstration", "**", "*_score.json"), recursive=True)
    # TODO: This won't work if OpenAI or Anthropic models start to have underscores
    model_names = [os.path.basename(f).removesuffix("_score.json").replace("_", "/", 2) for f in fs]
    model_scores = {}
    for f, model_name in zip(fs, model_names):
        with open(f) as src:
            scores = json.load(src)
            if not scores:
                continue
            model_scores[model_name] = scores["adv_demonstration"] * 100
    return model_scores


def get_advglue_scores():
    print(os.path.join(RESULT_DIR, "adv-glue-plus-plus", "summary.json"))
    with open(os.path.join(RESULT_DIR, "adv-glue-plus-plus", "summary.json")) as src:
        scores = json.load(src)
    return {k.removeprefix("/"): v * 100 for k, v in scores["Accuracy"].items()}


def get_fairness_scores():
    fs = glob(os.path.join(RESULT_DIR, "fairness", "**", "final_scores.json"), recursive=True)
    model_names = set([
        os.path.dirname(x).removeprefix(os.path.join(RESULT_DIR, "fairness", "results")).removeprefix("/") for x in fs
    ])
    model_scores = {}
    for f, model_name in zip(fs, model_names):
        with open(f) as src:
            scores = json.load(src)
            model_scores[model_name] = scores["fairness score"]
    return model_scores


def get_ethics_scores():
    df = pd.read_json(os.path.join(RESULT_DIR, "machine_ethics", "generations", "scores.jsonl"), lines=True)
    df = df[df["dataset"] == "ethics_commonsense_short"].drop_duplicates().set_index("model")["agg_score"]
    return df.to_dict()


def get_ood_scores():
    path_prefix = os.path.join(RESULT_DIR, "ood", "results/")
    fs = glob(os.path.join(path_prefix, "**", "final_scores.json"), recursive=True)
    model_names = [os.path.dirname(f).removeprefix(path_prefix) for f in fs]
    model_scores = {}
    for f, model_name in zip(fs, model_names):
        with open(f) as src:
            scores = json.load(src)
            if not scores:
                continue
            model_scores[model_name] = scores["score"]
    return model_scores


def get_privacy_scores():
    df = pd.read_json(os.path.join(RESULT_DIR, "privacy", "generations", "scores.jsonl"), lines=True)
    # TODO: This won't work if OpenAI or Anthropic models start to have underscores
    df["model"] = df["model"].apply(lambda x: x.replace("_", "/", 2))
    df = df[df["dataset"] == "all"].drop_duplicates().set_index("model")
    return df["privacy_score"].to_dict()


def get_stereotype_scores():
    path_prefix = os.path.join(RESULT_DIR, "stereotype", "generations/")
    fs = glob(os.path.join(path_prefix, "**", "25_compiled.json"), recursive=True)
    model_names = [os.path.dirname(f).removeprefix(path_prefix) for f in fs]
    model_scores = {}
    for f, model_name in zip(fs, model_names):
        with open(f) as src:
            scores = json.load(src)
            if not scores:
                continue
            model_scores[model_name] = scores["overall_score"]
    return model_scores


def get_toxicity_scores():
    path_prefix = os.path.join(RESULT_DIR, "toxicity", "user_prompts", "generations/")
    fs = glob(os.path.join(path_prefix, "**", "report.jsonl"), recursive=True)
    model_names = [os.path.dirname(f).removeprefix(path_prefix) for f in fs]
    model_scores = {}
    for f, model_name in zip(fs, model_names):
        with open(f) as src:
            scores = json.load(src)
            if not scores:
                continue
            score_key = os.path.join(model_name, "aggregated-score")
            if score_key not in scores:
                continue
            model_scores[model_name] = scores[score_key] * 100
    return model_scores


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
        }
    }

    with open(os.path.join(RESULT_DIR, "summary.json"), "w") as f:
        json.dump(summarized_results, f, indent=4)
        print(json.dumps(summarized_results, indent=4))

    return summarized_results


if __name__ == "__main__":
    summarize_results()
