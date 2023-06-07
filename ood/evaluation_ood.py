import argparse
import time
from transformers import GPT2Tokenizer
from tqdm import tqdm
import openai
import json
import csv
import os
import sys
sys.path.append("..")
from chat import Chat

def load_task_messages(args):
    if args.task == "knowledge":
        if args.idk:
            task_message = "Please answer the following multiple choice question. The answer should be exact '0', '1', '2', '3', or '4' if you don't know."
        else:
            task_message = "Please answer the following multiple choice question. The answer should be exact '0', '1', '2', or '3'."
    elif args.task == "style":
        task_message =  "For each snippet of text, label the sentiment of the text as positive or negative. The answer should be exact 'positive' or 'negative'."
    return task_message

def load_demonstrations(args):
    dataset_demo = []
    dataset_example = []
    if args.few_shot_num == 0:
        return dataset_demo
    if args.task == "knowledge":
        file_example =args.demo_file
        with open(file_example) as f:
            for line in f.readlines():
                dataset_example.append(json.loads(line))
        for index in range(args.few_shot_num):
            context_sentences = ""
            out_domain = args.out_domain
            if out_domain == "2021_qa":
                context_sentences = "Today is {}. \n".format(dataset_example[index]["question_date"])
            question = dataset_example[index]
            context_sentences += "Question: " + question['question_sentence'] + '\n'
            # print(question["choices"])
            context_sentences +=  "0 : " + question["choices"][0] + " \n"  + "1 : " + question["choices"][1] + " \n" +  "2 : " + question["choices"][2] + " \n" + "3 : " + question["choices"][3] + " \n"
            dataset_demo.append([context_sentences, question["answer"][0]])
    elif args.task == "style":
        file_example = args.demo_file
        with open(file_example) as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t')
            for row in reader:
                dataset_example.append(row)
        labels = ["negative", "positive"]
        for i in range(args.few_shot_num):
            example = dataset_example[i]
            dataset_demo.append([example["sentence"], labels[int(example["label"])]])
    return dataset_demo

def load_prompts(args, dataset_demo=[]):
    dataset_question = []
    dataset = []
    if args.task == "knowledge":
        with open(args.data_file) as f:
            for line in f.readlines():
                dataset_question.append(json.loads(line))
        for index in range(len(dataset_question)):
                question = dataset_question[index]
                answer = (question['answer'][0])
                context_sentences = "Today is {}. \n".format(question["question_date"])
                # Add the context text sentences and choices to the item
                context_sentences += "Question: " + question['question_sentence'] + '\n '
                if args.idk:
                    context_sentences +=  "0 : " + question["choices"][0] + " \n"  + "1 : " + question["choices"][1] + " \n" +  "2 : " + question["choices"][2] + " \n" + "3 : " + question["choices"][3] + " \n" + "4 : " + "I don't know" + " \n"
                    option = ["0", "1", "2", "3", "4"]
                else:
                    context_sentences +=  "0 : " + question["choices"][0] + " \n"  + "1 : " + question["choices"][1] + " \n" +  "2 : " + question["choices"][2] + " \n" + "3 : " + question["choices"][3] + " \n" 
                    option = ["0", "1", "2", "3"]
                dataset.append({"input": context_sentences, "label":answer, "examples":dataset_demo, "option":option})
    elif args.task == "style":
        with open(args.data_file) as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t')
            for row in reader:
                dataset_question.append(row)
        labels = ["negative", "positive"]
        for idx, example in enumerate(dataset_question):
            context_sentences = ""
            context_sentences += example["sentence"]
            label = labels[int(example["label"])]
            dataset.append({"input": context_sentences, "label":label.lower(), "examples":dataset_demo, "option":labels})
    return dataset

def evaluation(dataset, cache):
    if args.task == "knowledge":
        count = 0
        known_bd = 0
        bd = 0
        for i in range(len(cache)):
            if dataset[i]["label"] in cache[i][1]["choices"][0]["message"]["content"]:
                count += 1
            else:
                bd += 1
                if cache[i][1]["choices"][0]["message"]["content"][0] in ["0", "1", "2", "3"]:
                    known_bd += 1   
        print("ACC : {:.2f}, MACC : {:.2f}, RR : {:.2f}".format(count/len(cache) * 100, (count / (len(cache) - (bd- known_bd))) * 100, ((bd- known_bd)/len(cache)) * 100))          
    elif args.task == "style":
        count = 0
        for i in range(len(cache)):
            if dataset[i]["label"] == cache[i][1]["choices"][0]["message"]["content"].lower().split(".")[0].strip():
                count += 1
        print("ACC : {:.2f}".format(count/len(cache) * 100))

def parse_args():
    parser = argparse.ArgumentParser('OOD Classification')
    parser.add_argument('--model', required=True,
                        choices=["gpt-4-0314", "gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-0301", "davinci", "ada", "curie"], help='OpenAI model')
    parser.add_argument('--key', required=True, help='OpenAI API Key', type=str)
    parser.add_argument('--task', required=True, type=str, default=["style", "knowledge"], help='Task to evaluate OOD data')
    parser.add_argument('--few-shot-num', required=True, type=int, default=0, help='Number of few shot')
    parser.add_argument('--idk', action='store_true', help='Add I dont know option for OOD knowledge')
    parser.add_argument('--out-domain', required=False, type=str, help='Out domain')
    parser.add_argument('--demo-file', required=False, type=str, help='Demo file')
    parser.add_argument('--data-file', required=False, type=str, help='Input Evaluation file.')
    parser.add_argument('--out-file', help='Write evaluation results to file.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    openai.api_key = args.key
    data = args.data_file
    task_message = load_task_messages(args)
    demo = load_demonstrations(args)
    dataset = load_prompts(args, demo)
    gpt = Chat(args.model)
    acc, unknown, cost, cache = gpt.do_classification(dataset, task_message, example_prefix=False)
    print("cost:", cost)
    dir_name = os.path.dirname(args.out_file)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(args.out_file, "w") as f:
        json.dump(cache, f)
    print(f"Saving to {args.out_file}")
    evaluation(dataset, cache)

    