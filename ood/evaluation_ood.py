import os
import json
from chat import Chat
from tempfile import TemporaryDirectory


TASK = {"knowledge": ["qa_2020", "qa_2023"],
        "style": ["base", "shake_w", "augment", "shake_p0", "shake_p0.6", "bible_p0", "bible_p0.6", "romantic_p0",
                  "romantic_p0.6", "tweet_p0", "tweet_p0.6"]}
DEMO = {"qa_2020": ["2021_qa", "global_facts", "machine_learning", "moral_scenarios", "us_foreign_policy"],
        "base" : ["base"],
        "shake_p0.6": ["base", "shake_p0.6"],
        "bible_p0.6": ["base", "bible_p0.6"],
        "romantic_p0.6": ["base", "romantic_p0.6"],
        "tweet_p0.6": ["base", "tweet_p0.6"]}


def load_task_messages(args):
    if args.task == "knowledge":
        if args.idk:
            task_message = "Please answer the following multiple choice question. The answer should be exact '0', " \
                           "'1', '2', '3', or '4' if you don't know."
        else:
            task_message = "Please answer the following multiple choice question. The answer should be exact '0', " \
                           "'1', '2', or '3'."
    elif args.task == "style":
        task_message = "For each snippet of text, label the sentiment of the text as positive or negative. " \
                       "The answer should be exact 'positive' or 'negative'."
    return task_message


def load_demonstrations(args, datasets, demo):
    dataset_demo = []
    dataset_example = []
    if args.few_shot_num == 0:
        return dataset_demo
    if args.task == "knowledge":
        dataset_example = datasets["demo"][demo]
        for index in range(args.few_shot_num):
            context_sentences = ""
            if "2021_qa" in demo:
                context_sentences = "Today is {}. \n".format(dataset_example[index]["question_date"])
            question = dataset_example[index]
            context_sentences += "Question: " + question['question_sentence'] + '\n'
            context_sentences += "0 : " + question["choices"][0] + " \n" + "1 : " + question["choices"][
                1] + " \n" + "2 : " + question["choices"][2] + " \n" + "3 : " + question["choices"][3] + " \n"
            dataset_demo.append([context_sentences, question["answer"][0]])
    elif args.task == "style":
        dataset_example = datasets["train_demo"][demo]
        labels = ["negative", "positive"]
        for i in range(args.few_shot_num):
            example = dataset_example[i]
            dataset_demo.append([example["sentence"], labels[int(example["label"])]])
    return dataset_demo


def load_prompts(args, task, datasets, dataset_demo=[]):
    dataset_question = []
    dataset = []
    if args.task == "knowledge":
        dataset_question = datasets["test"][task]
        for index in range(len(dataset_question)):
            question = dataset_question[index]
            answer = (question['answer'][0])
            context_sentences = "Today is {}. \n".format(question["question_date"])
            # Add the context text sentences and choices to the item
            context_sentences += "Question: " + question['question_sentence'] + '\n '
            if args.idk:
                context_sentences += "0 : " + question["choices"][0] + " \n" + "1 : " + question["choices"][
                    1] + " \n" + "2 : " + question["choices"][2] + " \n" + "3 : " + question["choices"][
                                         3] + " \n" + "4 : " + "I don't know" + " \n"
                option = ["0", "1", "2", "3", "4"]
            else:
                context_sentences += "0 : " + question["choices"][0] + " \n" + "1 : " + question["choices"][
                    1] + " \n" + "2 : " + question["choices"][2] + " \n" + "3 : " + question["choices"][3] + " \n"
                option = ["0", "1", "2", "3"]
            dataset.append({"input": context_sentences, "label": answer, "examples": dataset_demo, "option": option, "choices": question["choices"]})
    elif args.task == "style":
        dataset_question = datasets["dev"][task]
        labels = ["negative", "positive"]
        for idx, example in enumerate(dataset_question):
            context_sentences = ""
            context_sentences += example["sentence"]
            label = labels[int(example["label"])]
            dataset.append(
                {"input": context_sentences, "label": label.lower(), "examples": dataset_demo, "option": labels})
    return dataset


def evaluation(dataset, cache, args):
    if args.task == "knowledge":
        count = 0
        known_bd = 0
        bd = 0
        for i in range(len(cache)):
            label = -1
            for digits in cache[i][1]["choices"][0]["message"]["content"].split("202")[0]:
                if digits in ["0", "1", "2", "3"]:
                    label = (digits)
                    break
            if dataset[i]["label"] == label or dataset[i]["choices"][int(dataset[i]["label"])].lower() in cache[i][1]["choices"][0]["message"]["content"].lower():
                count += 1
            else:
                bd += 1
                text = cache[i][1]["choices"][0]["message"]["content"].lower()
                if label in ["0", "1", "2", "3"]:
                    known_bd += 1
                elif "sorry" not in text and "apologize" not in text and "don't know" not in text and "thank you" not in text and "cannot" not in text and "i'm" not in text and "however" not in text and "glad" not in text:
                    known_bd += 1
        acc = count / len(cache) * 100
        if bd == len(cache) and known_bd == 0:
            macc = 0
            rr = 100
        else:
            macc = (count / (len(cache) - (bd - known_bd))) * 100
            rr = ((bd - known_bd) / len(cache)) * 100
        print("ACC : {:.2f}, MACC : {:.2f}, RR : {:.2f}".format(acc, macc, rr))
        return {"acc": acc, "macc": macc, "rr": rr}
    elif args.task == "style":
        count = 0
        for i in range(len(cache)):
            if dataset[i]["label"] in cache[i][1]["choices"][0]["message"]["content"].lower().strip():
                if "positive" in cache[i][1]["choices"][0]["message"]["content"].lower().strip() and "negative" in cache[i][1]["choices"][0]["message"]["content"].lower().strip():
                    continue
                count += 1
        acc = count / len(cache) * 100
        print("ACC : {:.2f}".format(acc))
        return {"acc": acc}


def test(args, model, task_message, dataset, results, task_name):
    if args.resume and task_name in results:
        current_progress = len(results[task_name]["outputs"])
        if current_progress == len(dataset):
            return results
        elif current_progress > len(dataset):
            raise ValueError("Incorrect dataset during resuming")
        else:
            dataset = dataset[(current_progress + 1):]
            print(f"========== Resuming from {args.out_file} ==========")
    else:
        results[task_name] = {"outputs": []}
    num_checkpoints = len(dataset) // args.save_interval + 1
    for i in range(num_checkpoints):
        if i == num_checkpoints - 1:
            sub_dataset = dataset[(i * args.save_interval):]
        else:
            sub_dataset = dataset[(i * args.save_interval):((i + 1) * args.save_interval)]
        _, _, _, cache = model.do_classification(sub_dataset, task_message, example_prefix=False)
        results[task_name]["outputs"].extend(cache)
        with open(args.out_file, "w") as f:
            json.dump(results, f, indent=4)
    return results


def main(OPTS):
    args = OPTS
    task_lists = TASK[args.task]
    with open(args.data_file) as f:
        datasets = json.load(f)
    task_message = load_task_messages(args)
    with TemporaryDirectory(dir="./.cache") as dirname:
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
        model = Chat.from_helm(model_name=args.model, conv_template=args.conv_template, cache=dirname, api_key=args.key)
        if args.resume:
            assert os.path.exists(args.out_file)
            with open(args.out_file) as f:
                results = json.load(f)
        else:
            results = {}
        
        for task in task_lists:
            if args.few_shot_num != 0:
                if task not in DEMO.keys():
                    continue
                demo_lists = DEMO[task]
                for demo_name in demo_lists:
                    for run in range(3):
                        curr_demo_name = demo_name + "_" + str(run)
                        demo = load_demonstrations(args, datasets, curr_demo_name)
                        dataset = load_prompts(args, task, datasets, demo)
                        task_name = "{}_{}".format(task, curr_demo_name)
                        results = test(args, model, task_message, dataset, results, task_name)
                        eval_result = evaluation(dataset, results[task_name]["outputs"], args)
                        results[task_name]["score"] = eval_result
            else:
                dataset = load_prompts(args, task, datasets)
                results= test(args, model, task_message, dataset, results, task)
                eval_result = evaluation(dataset, results[task]["outputs"], args)
                results[task]["score"] = eval_result
        with open(args.out_file, "w") as f:
            json.dump(results, f, indent=4)
