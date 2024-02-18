import json
import os
from tempfile import TemporaryDirectory
import numpy as np
from dt.utils import make_sure_path_exists
from dt.chat import Chat
from .task_description import task_dic
from datasets import load_dataset

def test(args, path, seed):
    model = args.model_config.model
    file = f"{seed}.jsonl"
    cache_path = path.replace("data", "results")
    cache_file = model.replace("/", "_") + "_" + file
    if os.path.exists(os.path.join(cache_path, cache_file)):
        print("Skipping", os.path.join(cache_path, cache_file))
        return 0, 0, (0, 0, 0)

    """
    dataset = []
    with open(os.path.join(path, file)) as f:
        for line in f.readlines():
            cur = json.loads(line)
            dataset.append(cur)
    """
    dataset = load_dataset("AI-Secure/DecodingTrust", name="adv_demonstration", data_files=[os.path.join(path, file).split('./data/')[-1]])
    dataset = dataset["train"]
    with TemporaryDirectory(dir="./.cache") as dirname:
        gpt = Chat.from_helm(args, cache=dirname)
        task_message = task_dic[args.adv_demonstration.task]
        acc, unknown, cost, cache = gpt.do_classification(dataset, task_message, dry_run=args.dry_run)
        print(f"{model}, {path}, {seed} : {acc}, {unknown}, {cost}")

    make_sure_path_exists(cache_path)
    print(cache_path, cache_file)
    labels = [x["label"] for x in dataset]
    out = {"requests": [x[0] for x in cache], "responses": [x[1] for x in cache], "labels": labels}
    out["scores"] = {path.split("_")[-1]: acc}
    print(out["scores"])
    with open(os.path.join(cache_path, cache_file), "w") as f:
        json.dump(out, f)
    return acc, unknown, cost


def main(args):
    seeds = [int(x) for x in str(args.adv_demonstration.seeds).split(",")]
    print(args)
    tot, tp, tc = 0, 0, 0 
    score_lis = {"_asr": [], "_cacc": []}
    for seed in seeds:
        for suffix in ["_asr", "_cacc"]:
            res = test(args, args.adv_demonstration.path + suffix, seed)
            score_lis[suffix].append(res[0])
            tot += res[-1][0]
            tp += res[-1][1]
            tc += res[-1][2]
    asr = np.mean(score_lis["_asr"])
    cacc = np.mean(score_lis["_cacc"])
    print(f"average asr: {asr}")
    print(f"average cacc: {cacc}")
    print(f"total cost: {tot} ({tp}, {tc})")
    return asr