import json
import os 
import sys 
sys.path.append("..") 
from utils import make_sure_path_exists, init_logger
from chat import Chat
from task_description import task_dic
import openai
import time 
import argparse

def test(model, path, seed, only=False):
    file = f"{seed}.jsonl"
    dataset = []
    with open(os.path.join(path, file)) as f:
        for line in f.readlines():
            dataset.append(json.loads(line))
            if only:
                assert "cf" in path
                dataset[-1]["examples"] = dataset[-1]["examples"][-1:]
            elif args.zero:
                dataset[-1]["examples"] = []
    gpt = Chat(model)
    task_message = task_dic[args.task]
    acc, unknown, cost, cache = gpt.do_classification(dataset, task_message)
    logger.info(f"{model}, {path}, {seed}: {acc}, {unknown}, {cost}")
    cache_path = path.replace("data", "cache")
    if only:
        cache_path += "_only"
    elif args.zero:
        cache_path += "_zero"
    cache_file = model+"_"+file
    make_sure_path_exists(cache_path)
    with open(os.path.join(cache_path, cache_file), "w") as f:
        json.dump(cache, f)
    return (acc, unknown, cost)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', required=True, help='OpenAI API Key', type=str)
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0301")
    parser.add_argument("--task", type=str, default="nli3")
    parser.add_argument("--path", type=str, default="../data/adv_demonstration/counterfactual/snli_hypothesis_cf")
    parser.add_argument("--seeds", type=str, default="42,2333,10007")
    parser.add_argument("--only", default=False, help='only use cf(x)', action="store_true")
    parser.add_argument("--zero", default=False, help='zero-shot', action="store_true")
    args = parser.parse_args()
    openai.api_key = args.key
    make_sure_path_exists("../log/adv_demonstration/cf")
    model = args.model
    path = args.path
    ts = int(time.time())
    logger = init_logger("../log/adv_demonstration/cf", f"cf_{model}_{ts}")
    logger.info(args)
    seeds = [int(x) for x in args.seeds.split(",")]
    tot = 0
    for seed in seeds:
        res = test(model, path, seed, args.only)
        tot += res[-1]
        if args.zero or args.only:
            break
    logger.info(tot)
    