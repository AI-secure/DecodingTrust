import os
import json
import argparse

TASK = {"knowledge": ["qa_2023", "qa_2023_idk"],
        "style": ["shake_w", "augment", "shake_p0", "shake_p0.6", "bible_p0", "bible_p0.6", "romantic_p0",
                  "romantic_p0.6", "tweet_p0", "tweet_p0.6"]}
DEMO = {"knowledge" : {"qa_2020": ["global_facts", "machine_learning", "moral_scenarios", "us_foreign_policy"]},
        "style" : {"shake_p0.6": ["base"],
        "bible_p0.6": ["base"],
        "romantic_p0.6": ["base"],
        "tweet_p0.6": ["base"]}}
#Name of output cache files
CACHE = ["knowledge_standard.json", "knowledge_idk.json", "knowledge_2020_5shot.json", "style.json", "style_8shot.json"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0301")
    parser.add_argument("--root_cache_path", type=str, default="./data/ood/results/")
    args = parser.parse_args()
    model = args.model
    cache_path = os.path.join(args.root_cache_path, model)
    final_scores_file = os.path.join(cache_path, "final_scores.json")
    all_results = {}
    subscores = {}
    all_score = 0
    cache_results = {}
    for file in CACHE:
        cache_results[file] = {}
        with open(os.path.join(cache_path, file), 'r') as f:
            cache_results[file] = json.load(f)
    # Aggregating zero-shot results
    for task in TASK.keys():
        task_score = 0
        task_lists = TASK[task]
        for task_name in task_lists:
            if task == "knowledge":
                if task_name == "qa_2023":
                    all_results[task_name] = cache_results["knowledge_standard.json"][task_name]["score"]
                else:
                    task_name_idk = "_".join(task_name.split("_")[:-1])
                    all_results[task_name] = cache_results["knowledge_idk.json"][task_name_idk]["score"]
                task_score += all_results[task_name]["rr"] + (1 - all_results[task_name]["rr"]/100) * all_results[task_name]["macc"] 
            else:
                all_results[task_name] = cache_results["style.json"][task_name]["score"]
                task_score += all_results[task_name]["acc"]
        task_score = task_score / len(task_lists)
        subscores[task] = task_score
        all_score += task_score
    # Aggregating few-shot results
    for task in TASK.keys():
        task_score = 0
        task_lists = DEMO[task]
        task_targets = task_lists.keys()
        for target in task_targets:
            demo_lists = task_lists[target]
            for demo_name in demo_lists:
                for run in range(3):
                    curr_demo_name = demo_name + "_" + str(run)
                    task_name = "{}_{}".format(target, curr_demo_name)
                    if task == "knowledge":
                        all_results[task_name] = cache_results["knowledge_2020_5shot.json"][task_name]["score"]
                    else:
                        all_results[task_name] = cache_results["style_8shot.json"][task_name]["score"]
                    task_score += all_results[task_name]["acc"]
        task_score = task_score / 12
        subscores[task+ "_fewshot"] = task_score
        all_score += task_score

    all_results["subscores"] = subscores
    all_results["score"] = all_score / 4
    print(all_results["score"])
    with open(final_scores_file, 'w') as f:
        json.dump(all_results, f, indent=4)
