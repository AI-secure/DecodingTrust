import json
import os
from tempfile import TemporaryDirectory

from utils import make_sure_path_exists
from chat import Chat
from .task_description import task_dic


def test(args, path, seed):
    model = args.model
    file = f"{seed}.jsonl"
    cache_path = path.replace("data", "cache")
    cache_file = model.replace("/", "_") + "_" + file
    if os.path.exists(os.path.join(cache_path, cache_file)):
        print("Skipping", os.path.join(cache_path, cache_file))
        return 0, 0, (0, 0, 0)

    dataset = []
    with open(os.path.join(path, file)) as f:
        for line in f.readlines():
            cur = json.loads(line)
            dataset.append(cur)
    with TemporaryDirectory(dir="./.cache") as dirname:
        gpt = Chat.from_helm(model_name=model, conv_template=args.conv_template, cache=dirname, api_key=args.key)
        task_message = task_dic[args.task]
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
    seeds = [int(x) for x in str(args.seeds).split(",")]
    print(args)
    tot, tp, tc = 0, 0, 0 
    for seed in seeds:
        for suffix in ["_asr", "_cacc"]:
            res = test(args, args.path + suffix, seed)
            tot += res[-1][0]
            tp += res[-1][1]
            tc += res[-1][2]
    print(tot, tp, tc)
