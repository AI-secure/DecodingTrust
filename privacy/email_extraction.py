import argparse
import openai
import sys
sys.path.append("..")
from chat import Chat
import json
import os
from data.privacy.sys_prompts import PROMPTS
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
        if len(self.task_desc)>0:
            messages.append({"role": "user", "content": self.task_desc + "\n" + input})
        else:
            messages.append({"role": "user", "content": input})
        return messages


def load_prompts(folder_root, data_file):
    email2name = load_pickle(os.path.join(folder_root, "email2name.pkl"))
    x= data_file
    if x.startswith("context"):
        k = int(x.split('-')[-1])
        fname= os.path.join(folder_root, "context.pkl")  
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        prompts,name_list,email_list = get_prompts_context(fname, email2name, tokenizer,  k=k)
    elif x.startswith("knownzero"):
        pp = x.split('-')[-1]
        fname= os.path.join(folder_root, "one_shot.pkl")  
        prompts,name_list,email_list = get_prompts_known_0_shot(fname, email2name,  pp)
    
    elif x.startswith("zero"):
        pp = x.split('-')[-1]
        fname= os.path.join(folder_root, "one_shot.pkl")  
        prompts,name_list,email_list = get_prompts_0_shot(fname, email2name,  pp)
        
    else:
        pp = x.split('-')[-1]
        prefix=  x[:-2].replace("-", '_') 
        fname= os.path.join(folder_root, f"{prefix}.pkl")  
        prompts,name_list,email_list = get_prompts_k_shot(fname, email2name,pp)
        
    return prompts,name_list,email_list


def parse_args():
    parser = argparse.ArgumentParser('Privacy Email generation')
    parser.add_argument('--model', required=False,
                        choices=["gpt-4-0314", "gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-0301"], help='OpenAI model',  default="gpt-3.5-turbo-0301")
    parser.add_argument('--key', required=False, help='OpenAI API Key', type=str, default="")
    parser.add_argument('--data-file', required=False, type=str, help='privacy data file.', default="context-50")
    parser.add_argument('--out-file', help='Write evaluation results to file.', default="")
    parser.add_argument('--template', help="generation template", type=int, default=0)
    parser.add_argument('--n', help="number of generations", type=int, default=1)
    parser.add_argument('--t', help="temperature of generations", type=float, default=1)
    parser.add_argument('--max_tokens', help="max tokens of generations", type=int, default=150)
    parser.add_argument('--run', help="the run seed", type=int, default=1)
    parser.add_argument('--dataset-size', help="dataset size. 0 means all data; otherwise subsample the dataset", type=int, default=0) 
    parser.add_argument('--batch-size', help="batch size", type=int, default=5)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    folder_root= "../data/privacy/enron_data/"


    constructor = MessageConstructor(PROMPTS[args.template]['sys_prompt'], PROMPTS[args.template]['task_desc'])

    openai.api_key = args.key

    dataset, name_list,email_list  = load_prompts(folder_root, args.data_file)
    generator = Chat(args.model)

    if args.dataset_size <1 or  args.dataset_size>len(dataset):
        args.dataset_size=len(dataset)
    else:
        dataset=dataset[:args.dataset_size]

    
    if len(args.out_file)==0:
        result_dir= "privacy/results/enron"
        os.makedirs(result_dir, exist_ok=True)
        args.out_file= os.path.join(result_dir, "{}_s{}_{}_{}.jsonl".format(args.data_file, args.dataset_size, args.model,args.run))
    print(f"Saving to {args.out_file}")


    total_cost= 0
    num_batches = int(args.dataset_size/args.batch_size)
    for batch_idx in range(num_batches):

        batch_start= batch_idx* args.batch_size
        batch_end=  (batch_idx+1)* args.batch_size
        batch_data= dataset[batch_start:batch_end]
        batch_name = name_list[batch_start:batch_end]
        batch_email = email_list[batch_start:batch_end]

        result = generator.do_generation(batch_data, constructor, n=args.n, t=args.t, max_tokens=args.max_tokens) # only works for n=1 now... 
        print(batch_idx, "cost:", result[0])
        total_cost+=result[0]

        with open(args.out_file, "a") as f:
            for idx, x in enumerate(result[1]):
                res = {"gt-email": batch_email[idx], "continuation": x[1], "gt-name": batch_name[idx],  "isbanned": x[2], "prompt": x[3],"response": x[-1]}
                f.write(json.dumps(res) + "\n")

    print("total cost", total_cost, "dataset size",args.dataset_size)
