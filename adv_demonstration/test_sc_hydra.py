import json
import os
from utils import make_sure_path_exists, init_logger
from chat import Chat
from .task_description import task_dic
from tempfile import TemporaryDirectory


def test(args, path, seed):
    model = args.model
    file = f"{seed}.jsonl"
    cache_path = path.replace("data", "cache")
    cache_file = model.replace("/", "_") + "_" + file
    if os.path.exists(os.path.join(cache_path, cache_file)):
        print("Skipping", os.path.join(cache_path, cache_file))
        return 0, 0, (0, 0, 0)

    dataset0 = []
    dataset1 = []
    with open(os.path.join(path, file)) as f:
        for line in f.readlines():
            cur = json.loads(line)
            if args.zero:
                cur["examples"] = []
            if cur["label"] == "yes":
                dataset1.append(cur)
            else:
                dataset0.append(cur)
    assert len(dataset0) == len(dataset1)
    with TemporaryDirectory(dir="./.cache") as dirname:
        gpt = Chat.from_helm(model_name=model, conv_template=args.conv_template, cache=dirname, api_key=args.key)
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
    return acc, unknown, (cost[0]+cost0[0], cost[1]+cost0[1], cost[2]+cost0[2])


def main(args):
    seeds = [int(x) for x in str(args.seeds).split(",")]
    print(args)
    tot, tp, tc = 0, 0, 0 
    for seed in seeds:
        res = test(args, args.path, seed)
        tot += res[-1][0]
        tp += res[-1][1]
        tc += res[-1][2]
        if args.zero:
            break
    print(tot, tp, tc)
