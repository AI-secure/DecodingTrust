import json
import os
from dt.utils import make_sure_path_exists, init_logger
from dt.chat import Chat
from .task_description import task_dic
from tempfile import TemporaryDirectory
import numpy as np
from datasets import load_dataset

def test(args, path, seed):
    model = args.model_config.model
    file = f"{seed}.jsonl"
    cache_path = path.replace("data", "results")
    cache_file = model.replace("/", "_") + "_" + file
    if os.path.exists(os.path.join(cache_path, cache_file)):
        print("Skipping", os.path.join(cache_path, cache_file))
        return 0, 0, (0, 0, 0)

    raw_dataset = load_dataset("AI-Secure/DecodingTrust", name="adv_demonstration", data_files=[os.path.join(path, file).split('./data/')[-1]])
    raw_dataset = raw_dataset["train"]
    dataset0 = []
    dataset1 = []
    for cur in raw_dataset:
        if args.adv_demonstration.zero:
            cur["examples"] = []
        if cur["label"] == "yes":
            dataset1.append(cur)
        else:
            dataset0.append(cur)
    assert len(dataset0) == len(dataset1)
    with TemporaryDirectory(dir="./.cache") as dirname:
        gpt = Chat.from_helm(args, cache=dirname)
        task_message = task_dic["nli2"]

        acc, unknown, cost, cache = gpt.do_classification(dataset1, task_message, dry_run=args.dry_run)
        print(f"{model}, {path}, {seed} Eacc: {acc}, {unknown}, {cost}")

        acc0, unknown0, cost0, cache0 = gpt.do_classification(dataset0, task_message, dry_run=args.dry_run)
        print(f"{model}, {path}, {seed} Nacc: {acc0}, {unknown0}, {cost0}")

        overall = (acc + acc0) / 2
        gap = abs(acc - acc0)
        print(f"Acc: {overall}; Gap: {gap}")
        cache.extend(cache0)

    make_sure_path_exists(cache_path)
    labels = [x["label"] for x in dataset1] + [x["label"] for x in dataset0]
    out = {"requests": [x[0] for x in cache], "responses": [x[1] for x in cache], "labels": labels}
    out["scores"] = {"Eacc": acc, "Nacc": acc0, "Acc": overall, "Gap": gap}
    print(out["scores"])
    with open(os.path.join(cache_path, cache_file), "w") as f:
        json.dump(out, f)
    return overall, unknown, (cost[0]+cost0[0], cost[1]+cost0[1], cost[2]+cost0[2])


def main(args):
    seeds = [int(x) for x in str(args.adv_demonstration.seeds).split(",")]
    print(args)
    tot, tp, tc = 0, 0, 0 
    score_lis = []
    for seed in seeds:
        res = test(args, args.adv_demonstration.path, seed)
        score_lis.append(res[0])
        tot += res[-1][0]
        tp += res[-1][1]
        tc += res[-1][2]
        if args.adv_demonstration.zero:
            break
    avg = np.mean(score_lis)
    print(f"average acc: {avg}")
    print(f"total cost: {tot} ({tp}, {tc})")
    return avg
