from tempfile import TemporaryDirectory
import os
import json
from dt.utils import make_sure_path_exists, init_logger
from dt.chat import Chat
from .task_description import task_dic
import numpy as np
from datasets import load_dataset

def test(args, path, seed):
    model = args.model_config.model
    only = args.adv_demonstration.only
    file = f"{seed}.jsonl"

    cache_path = path.replace("data", "results")
    if only:
        cache_path += "_only"
    elif args.adv_demonstration.zero:
        cache_path += "_zero"
    cache_file = model.replace("/", "_") + "_" + file
    if os.path.exists(os.path.join(cache_path, cache_file)):
        print("Skipping", os.path.join(cache_path, cache_file))
        return 0, 0, (0, 0, 0)
    raw_dataset = load_dataset("AI-Secure/DecodingTrust", name="adv_demonstration", data_files=[os.path.join(path, file).split('./data/')[-1]])
    raw_dataset = raw_dataset["train"]
    dataset = []
    for x in raw_dataset:
        dataset.append(x)
        if only:
            assert "cf" in path
            dataset[-1]["examples"] = dataset[-1]["examples"][-1:]
        elif args.adv_demonstration.zero:
            dataset[-1]["examples"] = []  

    with TemporaryDirectory(dir="./.cache") as dirname:
        gpt = Chat.from_helm(args, cache=dirname)
        task_message = task_dic[args.adv_demonstration.task]
        acc, unknown, cost, cache = gpt.do_classification(dataset, task_message, dry_run=args.dry_run)
        print(f"{model}, {path}, {seed}: {acc}, {unknown}, {cost}")
    make_sure_path_exists(cache_path)
    labels = [x["label"] for x in dataset]
    out = {"requests": [x[0] for x in cache], "responses": [x[1] for x in cache], "labels":labels}
    out["scores"] = {"acc": acc}
    print(out["scores"])
    with open(os.path.join(cache_path, cache_file), "w") as f:
        json.dump(out, f)
    return acc, unknown, cost


def main(args):
    seeds = [int(x) for x in str(args.adv_demonstration.seeds).split(",")]
    tot, tp, tc = 0, 0, 0 
    score_lis = []
    for seed in seeds:
        res = test(args, args.adv_demonstration.path, seed)
        score_lis.append(res[0])
        tot += res[-1][0]
        tp += res[-1][1]
        tc += res[-1][2]
        if args.adv_demonstration.zero or args.adv_demonstration.only:
            break
    avg = np.mean(score_lis)
    print(f"average acc: {avg}")
    print(f"total cost: {tot} ({tp}, {tc})")
    return avg
