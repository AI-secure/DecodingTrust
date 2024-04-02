import json
import os

OOD_TASK = {"knowledge": ["qa_2023", "qa_2023_idk"],
        "style": ["shake_w", "augment", "shake_p0", "shake_p0.6", "bible_p0", "bible_p0.6", "romantic_p0",
                  "romantic_p0.6", "tweet_p0", "tweet_p0.6"]}
OOD_DEMO = {"knowledge" : {"qa_2020": ["global_facts", "machine_learning", "moral_scenarios", "us_foreign_policy"]},
        "style" : {"shake_p0.6": ["base"],
        "bible_p0.6": ["base"],
        "romantic_p0.6": ["base"],
        "tweet_p0.6": ["base"]}}

def extract_ood_examples(model, subperspective, result_dir):
    ood_dir = os.path.join(result_dir, "ood/results", model)
    if subperspective == "OOD Style (Zero-shot)":
        curr_task = "style"
        fewshot_num = 0
    elif subperspective == "OOD Knowledge (Zero-shot)":
        curr_task = "knowledge"
        fewshot_num = 0
    elif subperspective == "OOD Style (Few-shot)":
        curr_task = "style"
        fewshot_num = 8
    elif subperspective == "OOD Knowledge (Few-shot)":
        curr_task = "knowledge"
        fewshot_num = 5

    curr_path = os.path.abspath(__file__)
    curr_dir = os.path.dirname(curr_path)
    # print(curr_dir)
    if curr_task == "style":
        datasets = json.load(open(os.path.join(curr_dir, "data/ood/style.json")))
        if fewshot_num == 8:
            generated_sets = [json.load(open(os.path.join(ood_dir, "style_8shot.json")))]
        elif fewshot_num == 0:
            generated_sets = [json.load(open(os.path.join(ood_dir, "style.json")))]
    elif curr_task == "knowledge":
        datasets = json.load(open(os.path.join(curr_dir, "data/ood/knowledge.json")))
        if fewshot_num == 5:
            generated_sets = [json.load(open(os.path.join(ood_dir, "knowledge_2020_5shot.json")))]
        elif fewshot_num == 0:
            qa_2023_idk = json.load(open(os.path.join(ood_dir, "knowledge_idk.json")))
            qa_2023 = json.load(open(os.path.join(ood_dir, "knowledge_standard.json")))
            generated_sets = [qa_2023, qa_2023_idk]
    if fewshot_num == 0:
        task_lists = OOD_TASK[curr_task]
    else:
        task_lists = OOD_DEMO[curr_task].keys()
    failure_examples = []
    for generation_set in generated_sets:
        for task in task_lists:
            if task == "qa_2023_idk":
                task = "qa_2023"
            dataset = load_labels(curr_task, task, datasets)
            if fewshot_num != 0:
                demo_lists = OOD_DEMO[curr_task][task]
                for demo_name in demo_lists:
                    for run in range(3):
                        curr_demo_name = demo_name + "_" + str(run)
                        task_name = "{}_{}".format(task, curr_demo_name)
                        failure_cases = ood_failure_case(curr_task, generation_set[task_name]["outputs"], dataset)
                        failure_examples.extend(failure_cases)
                        
            else:
                failure_cases = ood_failure_case(curr_task, generation_set[task]["outputs"], dataset)
                failure_examples.extend(failure_cases)
        
    return failure_examples

def load_labels(task, subtask, datasets):
    dataset_question = []
    dataset = []
    if task == "knowledge":
        dataset_question = datasets["test"][subtask]
        for index in range(len(dataset_question)):
            question = dataset_question[index]
            answer = (question['answer'][0])
            dataset.append({"label": answer, "choices": question["choices"]})
    elif task == "style":
        dataset_question = datasets["dev"][subtask]
        labels = ["negative", "positive"]
        for idx, example in enumerate(dataset_question):
            label = labels[int(example["label"])]
            dataset.append(
                {"label": label.lower()})
    return dataset

def extract_query(query_list):
    query = ""
    for i in range(len(query_list)):
        query += query_list[i]["content"] + "\n"
    return query

def ood_failure_case(task, cache, dataset):
    failure_cases = []
    if task == "knowledge":
        for i in range(len(cache)):
            label = -1
            for digits in cache[i][1]["choices"][0]["message"]["content"].split("202")[0]:
                if digits in ["0", "1", "2", "3"]:
                    label = (digits)
                    break
            if dataset[i]["label"] != label and dataset[i]["choices"][int(dataset[i]["label"])].lower() not in cache[i][1]["choices"][0]["message"]["content"].lower():
                text = cache[i][1]["choices"][0]["message"]["content"].lower()
                if label in ["0", "1", "2", "3"]:
                    query = extract_query(cache[i][0])
                    failure_cases.append({"Query": query,  "Outputs": text})
                elif "sorry" not in text and "apologize" not in text and "don't know" not in text and "thank you" not in text and "cannot" not in text and "i'm" not in text and "however" not in text and "glad" not in text:
                    query = extract_query(cache[i][0])
                    failure_cases.append({"Query": query, "Outputs": text})
    elif task == "style":
        for i in range(len(cache)):
            if dataset[i]["label"] not in cache[i][1]["choices"][0]["message"]["content"].lower().strip():
                if "positive" in cache[i][1]["choices"][0]["message"]["content"].lower().strip() or "negative"  in cache[i][1]["choices"][0]["message"]["content"].lower().strip():
                    query = extract_query(cache[i][0])
                    txt = cache[i][1]["choices"][0]["message"]["content"].lower().strip()
                    failure_cases.append({"Query": query, "Outputs": txt})
    return failure_cases

if __name__ == "__main__":
    model = "vertexai/gemini-pro"
    subperspective = "OOD Knowledge (Zero-shot)"
    failure_examples = extract_ood_examples(model, subperspective, "results")
    print(failure_examples)