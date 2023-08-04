import os
import json
import pandas as pd
import argparse

# setting
parser = argparse.ArgumentParser("Transform cache file into csv format")
parser.add_argument('--dataset', type=str, default='ethics')
parser.add_argument("--is_few_shot", action='store_true')
parser.add_argument("--test_num", type=int, default=None)
parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo-0301')
parser.add_argument('--template', type=str, default='standard')
parser.add_argument('--test_mode', type=str, default='short_test')
args = parser.parse_args()

if args.dataset == 'jiminy':
    # open cache file
    cache_path = os.path.join('./response', 'jiminy', args.model_name, args.template)
    if args.is_few_shot:
        json_file = "few_shot_num_{}.json".format(str(args.test_num))
        csv_file = "few_shot_num_{}.csv".format(str(args.test_num))
    else:
        json_file = "zero_shot_num_{}.json".format(str(args.test_num))
        csv_file = "zero_shot_num_{}.csv".format(str(args.test_num))
    with open(os.path.join(cache_path, json_file), "r") as f:
        cache = json.load(f)
    print("Num of data: ", len(cache))

    # transform cache file into csv format
    input_list, examples_list, label_list, focal_list, degree_list, output_list, num_tokens_list = [], [], [], [], [], [], []
    for i, data in enumerate(cache):
        input = data[0]
        input_list.append(input)

        examples = data[1]
        examples_list.append(examples)

        label = data[2]
        label_list.append(label)

        focal = data[3]
        focal_list.append(focal)

        degree = data[4]
        degree_list.append(degree)

        output = data[5]
        output_list.append(output)

        num_tokens = data[6]
        num_tokens_list.append(num_tokens)

    save = pd.DataFrame({'input':input_list, 'examples':examples_list, 'label':label_list, 'focal':focal_list, 'degree':degree_list, 'output':output_list, 'num_tokens':num_tokens_list})
    save.to_csv(os.path.join(cache_path, csv_file), index=False)


elif args.dataset == 'ethics':
    # open cache file
    cache_path = os.path.join('./response', 'ethics', args.model_name, args.test_mode, args.template)
    if args.is_few_shot:
        json_file = "few_shot_num_{}.json".format(str(args.test_num))
        csv_file = "few_shot_num_{}.csv".format(str(args.test_num))
    else:
        json_file = "zero_shot_num_{}.json".format(str(args.test_num))
        csv_file = "zero_shot_num_{}.csv".format(str(args.test_num))
    with open(os.path.join(cache_path, json_file), "r") as f:
        cache = json.load(f)
    print("Num of data: ", len(cache))

    # transform cache file into csv format
    input_list, examples_list, label_list, output_list, num_tokens_list, is_short_list = [], [], [], [], [], []
    for i, data in enumerate(cache):
        input = data[0]
        input_list.append(input)

        examples = data[1]
        examples_list.append(examples)

        label = data[2]
        label_list.append(label)

        output = data[3]
        output_list.append(output)

        num_tokens = data[4]
        num_tokens_list.append(num_tokens)

        is_short = data[5]
        is_short_list.append(is_short)

    save = pd.DataFrame({'input': input_list, 'examples': examples_list, 'label': label_list, 'output': output_list, 'num_tokens': num_tokens_list, 'is_short': is_short_list})
    save.to_csv(os.path.join(cache_path, csv_file), index=False)
