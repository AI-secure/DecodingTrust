import json
import os
from utils import make_sure_path_exists
from chat import Chat
from task_description import task_dic
import numpy as np
from fairlearn.metrics import demographic_parity_difference
from fairlearn.metrics import demographic_parity_ratio
from fairlearn.metrics import equalized_odds_ratio
from fairlearn.metrics import equalized_odds_difference
import openai
import argparse

parser = argparse.ArgumentParser(description='GPT fairness evaluation')
parser.add_argument('--data_dir', type=str)
parser.add_argument('--prompt_file', type=str)
parser.add_argument('--gt_file', type=str)
parser.add_argument('--sensitive_attr_file', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--key', type=str)
parser.add_argument('--dataset', type=str, choices=['adult','compas','crime','german','health','lawschool'], default='adult')
parser.add_argument('--num_instance_prompt', type=int, default=0)
parser.add_argument('--num_question', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--protected_att', type=str)
args = parser.parse_args()

openai.api_key = args.key

# sensitive_idx = 8

dataset = []
# path = "./fairness_prompts/"
# file = f"{args.dataset}_{args.num_instance_prompt}_{args.num_question}_{sensitive_idx}.jsonl"

with open(args.data_dir+args.prompt_file) as f:
    for line in f.readlines():
        dataset.append(json.loads(line))

# model = "gpt-3.5-turbo-0301"
# model = "gpt-4-0314"

model = args.model

gpt = Chat(model)
task_message = task_dic[args.dataset]
acc, unknown, cost, cache, prediction = gpt.do_classification(dataset, task_message, example_prefix=True)
print(acc, unknown, cost)

# labels = np.load(f"./fairness_prompts/gt_labels_{args.dataset}_{args.num_instance_prompt}_{args.num_question}_{sensitive_idx}.npy")
# sensitive_attr_str = np.load(f"./fairness_prompts/sensitive_attr_{args.dataset}_{args.num_instance_prompt}_{args.num_question}_{sensitive_idx}.npy")
labels = np.load(args.data_dir+args.gt_file)
sensitive_attr_str = np.load(args.data_dir+args.sensitive_attr_file)

prediction = np.array(prediction)
if args.dataset == 'adult':
    sensitive_attr = []
    for i in range(len(sensitive_attr_str)):
        if sensitive_attr_str[i] == 'Male':
            sensitive_attr.append(1)
        else:
            sensitive_attr.append(0)
        # if sensitive_attr_str[i] == 'White':
        #     sensitive_attr.append(1)
        # else:
        #     sensitive_attr.append(0)
    sensitive_attr = np.array(sensitive_attr)
elif args.dataset == 'compas':
    sensitive_attr = np.array(sensitive_attr_str)
elif args.dataset == 'german':
    sensitive_attr = np.array(sensitive_attr_str)
idx = prediction>=0

# print(prediction)
# print(labels[idx])
# print(prediction[idx])
# print(sensitive_attr[idx])

m = demographic_parity_difference(labels[idx], prediction[idx], sensitive_features=sensitive_attr[idx])
print(f'Demographic parity difference: {m}')
# m = demographic_parity_ratio(labels[idx], prediction[idx], sensitive_features=sensitive_attr[idx])
# print(f'Demographic parity ratio: {m}')
m = equalized_odds_difference(labels[idx], prediction[idx], sensitive_features=sensitive_attr[idx])
print(f'Equalized odds difference: {m}')
# m = equalized_odds_ratio(labels[idx], prediction[idx], sensitive_features=sensitive_attr[idx])
# print(f'Equalized odds ratio: {m}')

# cache_path = "./cache/"
# cache_file = model+"_"+file
# make_sure_path_exists(cache_path)
# #assert not os.path.exists(os.path.join(cache_path, cache_file))
# with open(os.path.join(cache_path, cache_file), "w") as f:
#     json.dump(cache, f)
