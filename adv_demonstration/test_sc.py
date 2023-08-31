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

def test(model, path, seed):
    file = f"{seed}.jsonl"
    dataset0 = []
    dataset1 = []
    with open(os.path.join(path, file)) as f:
        for line in f.readlines():
            cur = json.loads(line)
            if args.zero:
                cur["examples"] = []
            if cur["label"]=="yes":
                dataset1.append(cur)
            else:
                dataset0.append(cur)
    assert len(dataset0)==len(dataset1)
    gpt = Chat(model)
    task_message = task_dic["nli2"]
    
    acc, unknown, cost, cache = gpt.do_classification(dataset1, task_message)
    logger.info(f"{model}, {path}, {seed} Eacc: {acc}, {unknown}, {cost}")
    
    acc0, unknown0, cost0, cache0 = gpt.do_classification(dataset0, task_message)
    logger.info(f"{model}, {path}, {seed} Nacc: {acc0}, {unknown0}, {cost0}")
    
    overall = (acc+acc0)/2
    logger.info(f"Acc: {overall}; Gap: {abs(acc-acc0)}")
    cache.extend(cache0)
    cost += cost0
    cache_path = path.replace("data", "cache")
    cache_file = model+"_"+file
    make_sure_path_exists(cache_path)
    with open(os.path.join(cache_path, cache_file), "w") as f:
        json.dump(cache, f)
    return (acc, unknown, cost)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', required=True, help='OpenAI API Key', type=str)
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0301")
    parser.add_argument("--path", type=str, default="../data/spurious/l_relative_clause/entail-bias")
    parser.add_argument("--seeds", type=str, default="0,42,2333,10007,12306")
    parser.add_argument("--zero", default=False, help='zero-shot', action="store_true")
    args = parser.parse_args()
    openai.api_key = args.key
    make_sure_path_exists("../log/adv_demonstration/sc")
    model = args.model
    path = args.path
    ts = int(time.time())
    logger = init_logger("../log/adv_demonstration/sc", f"sc_{model}_{ts}")
    logger.info(args)
    seeds = [int(x) for x in args.seeds.split(",")]
    tot = 0
    for seed in seeds:
        res = test(model, path, seed)
        tot += res[-1]
        if args.zero:
            break
    logger.info(tot)
    