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
    dataset = []
    with open(os.path.join(path, file)) as f:
        for line in f.readlines():
            cur = json.loads(line)
            dataset.append(cur)
    gpt = Chat(model)
    task_message = task_dic[args.task]
    acc, unknown, cost, cache = gpt.do_classification(dataset, task_message)
    logger.info(f"{model}, {path}, {seed} : {acc}, {unknown}, {cost}")
    
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
    parser.add_argument("--task", type=str, default="sa")
    parser.add_argument("--path", type=str, default="../data/backdoor/experiment1/sst-2_setup1_badword")
    parser.add_argument("--seeds", type=str, default="42,2333,10007")
    args = parser.parse_args()
    openai.api_key = args.key
    make_sure_path_exists("../log/adv_demonstration/bkd")
    model = args.model
    path = args.path
    ts = int(time.time())
    logger = init_logger("../log/adv_demonstration/bkd", f"bkd_{model}_{ts}")
    logger.info(args)
    seeds = [int(x) for x in args.seeds.split(",")]
    tot = 0
    for seed in seeds:
        res = test(model, args.path+"_asr", seed)
        tot += res[-1]
        res = test(model, args.path+"_cacc", seed)
        tot += res[-1]
    logger.info(tot)
    