import argparse
from tqdm import tqdm
import openai
import sys
sys.path.append("..")
from chat import Chat
import os
import json
from data.privacy.sys_prompts import PROMPTS, PII_INFO
from utils import timeout

from data.privacy.utils import * 

class MessageConstructor(object):
    def __init__(self, sys_demo, task_desc):
        self.sys_demo = sys_demo
        self.task_desc = task_desc

    def get_message(self, input):
      
        if len(self.sys_demo)>0:
            messages = [{"role": "system", "content": f"{self.sys_demo}"}]
        else:
            messages=[]
        
        messages= messages+ input 
        return messages

def parse_args():
    parser = argparse.ArgumentParser('Privacy Email generation')
    parser.add_argument('--model', required=False,
                        choices=["gpt-4-0314", "gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-0301"], help='OpenAI model',  default="gpt-3.5-turbo-0301")
    parser.add_argument('--key', required=False, help='OpenAI API Key', type=str, default="")
    parser.add_argument('--data-file', required=False, type=str, help='privacy data file.', default="five_shot")
    parser.add_argument('--out-file', help='Write evaluation results to file.', default="")
    parser.add_argument('--template', help="generation template", type=int, default=1)
    parser.add_argument('--n', help="number of generations", type=int, default=1)
    parser.add_argument('--t', help="temperature of generations", type=float, default=1)
    parser.add_argument('--max_tokens', help="max tokens of generations", type=int, default=150)
    parser.add_argument('--run', help="the run seed", type=int, default=1)
    parser.add_argument('--dataset-size', help="dataset size", type=int, default=30)
    parser.add_argument('--batch-size', help="batch size", type=int, default=5)
    parser.add_argument('--personal-info', help="personal info", type=str, default="phone number")
    parser.add_argument('--fewshot-type', help="few shot type", type=str, choices=["protect", "attack"],  default="attack")

    parser.add_argument('--k', help="number of demonstrations", type=int, default=3)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    folder_root= "../data/privacy/enron_data/"


    constructor = MessageConstructor(PROMPTS[args.template]['sys_prompt'].format(args.personal_info), PROMPTS[args.template]['task_desc'])

    openai.api_key = args.key

    fname= os.path.join(folder_root, "five_shot.pkl") # load the names from enron data to construct the prompts
    
    print(args.personal_info)
    # print(PII_INFO)
    if args.personal_info not in PII_INFO:
        print("Exiting the program....", args.personal_info)
        exit(0)

    dataset, gt_list  = get_pii_messages_k_shot(filename=fname, k=args.k, pp=args.personal_info, fewshot_type=args.fewshot_type, num_cases= args.dataset_size)
    print("datase tsize", len(dataset) )
    generator = Chat(args.model)

    if len(args.out_file)==0:
        result_dir= "privacy/results/pii"
        os.makedirs(result_dir, exist_ok=True)
        args.out_file= os.path.join(result_dir, "{}_{}shot{}_s{}_{}_{}.jsonl".format(args.personal_info, args.k, args.fewshot_type, args.dataset_size, args.model,args.run))
        
    print(f"Saving to {args.out_file}")


    num_batches = int(args.dataset_size/args.batch_size)

    all_cost=0
    for batch_idx in range(num_batches):
        batch_start= batch_idx* args.batch_size
        batch_end=  (batch_idx+1)* args.batch_size
        batch_data= dataset[batch_start:batch_end]
        batch_gt = gt_list[batch_start:batch_end]
        result = generator.do_generation(batch_data, constructor, n=args.n, t=args.t, max_tokens=args.max_tokens) # only works for n=1 now... 
        print(batch_idx, "cost:", result[0])
        all_cost+=result[0]

        with open(args.out_file, "a") as f:
            for idx, x in enumerate(result[1]):
                res = {"gt": batch_gt[idx], "continuation": x[1],  "isbanned": x[2], "prompt": x[3],"response": x[-1]}
                f.write(json.dumps(res) + "\n")
    print("all_cost:", all_cost)
