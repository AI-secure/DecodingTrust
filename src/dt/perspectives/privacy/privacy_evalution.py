import json
from chat import Chat
from tempfile import TemporaryDirectory
from data.privacy.sys_prompts import PROMPT_TEMPLATE, PII_INFO, PII_PROMPT_TYPE, PRIVACY_TOPIC
from data.privacy.utils import *


class MessageConstructor(object):
    def __init__(self, sys_demo, task_desc):
        self.sys_demo = sys_demo
        self.task_desc = task_desc

    def get_message(self, input):
        messages = []
        if len(self.sys_demo) > 0:
            messages.append({"role": "system", "content": self.sys_demo})

        if len(self.task_desc) > 0:
            messages.append({"role": "user", "content": self.task_desc})

        messages = messages + input
        return messages


class EnronMessageConstructor(object):
    def __init__(self, sys_demo, task_desc):
        self.sys_demo = sys_demo
        self.task_desc = task_desc

    def get_message(self, input):
        if len(self.sys_demo) > 0:
            messages = [{"role": "system", "content": f"{self.sys_demo}"}]
        else:
            messages = []
        if len(self.task_desc) > 0:
            messages.append({"role": "user", "content": self.task_desc + "\n" + input})
        else:
            messages.append({"role": "user", "content": input})
        return messages


ScenarioName2ResultDir = {
    "pii": "./data/privacy/generations/pii",
    "privacy_understanding": "./data/privacy/generations/understanding",
    "enron_email_extraction": "./data/privacy/generations/enron",
}


def main(args):
    print(args)
    out_file = args.out_file
    dataset_size = args.dataset_size

    all_cost = {}
    all_prompt_tokens={}
    all_comple_tokens={}
    all_num_prompt={}

    if args.scenario_name == "pii":

        for pii in args.personal_infos:
            if pii not in PII_INFO:
                print(f"Skip {pii} as it is not defined in supported PII types {PII_INFO}")
                continue
            constructor = MessageConstructor(PROMPT_TEMPLATE[args.template]['sys_prompt'].format(pii),
                                             PROMPT_TEMPLATE[args.template]['task_desc'])

            for fewshot_type in args.prompt_types:
                if fewshot_type not in PII_PROMPT_TYPE:
                    print(f"Skip {fewshot_type} as it is not defined in supported prompt types {PII_PROMPT_TYPE}")
                    continue

                dataset, gt_list = get_pii_messages_k_shot(filename=args.data_file, k=args.few_shot_num, pp=pii,
                                                           fewshot_type=fewshot_type, num_cases=dataset_size)

                
            
                result_dir = ScenarioName2ResultDir[args.scenario_name]
                os.makedirs(result_dir, exist_ok=True)
                out_file = os.path.join(
                    result_dir,
                    "{}_{}shot{}_s{}_{}_{}.jsonl".format(
                        pii, args.few_shot_num, fewshot_type, dataset_size, args.model, args.seed
                    ).replace("/", "_").replace(" ", "")
                )
                print(f"Saving to {out_file}")

                num_batches = int(dataset_size / args.batch_size)
                price_cost = 0
                prompt_tokens=0
                comple_tokens=0
                num_prompt=0
                with TemporaryDirectory(dir="./.cache") as dirname:
                    model = Chat.from_helm(model_name=args.model, conv_template=args.conv_template, cache=dirname,
                                           api_key=args.key)

                    for batch_idx in range(num_batches):
                        batch_start = batch_idx * args.batch_size
                        batch_end = (batch_idx + 1) * args.batch_size
                        batch_data = dataset[batch_start:batch_end]
                        batch_gt = gt_list[batch_start:batch_end]

                        cost, cache = model.do_generation(batch_data, constructor, n=args.n, t=args.t,
                                                          max_tokens=args.max_tokens, dry_run=args.dry_run)  # only works for n=1 now...
                        
                        # computing resource
                        price_cost += cost[0]
                        prompt_tokens+=cost[1]
                        comple_tokens+=cost[2]
                        num_prompt+=len(cache)

                        with open(out_file, "a") as f:
                            for idx, x in enumerate(cache):
                                res = {"gt": batch_gt[idx], "continuation": x[1], "isbanned": x[2], "prompt": x[3],
                                       "response": x[-1]}
                                f.write(json.dumps(res) + "\n")
                    print("Cost: ", price_cost)
                    print("Num of prompt tokens: ", prompt_tokens)
                    print("Num of completion tokens: ", comple_tokens)
                    all_cost[pii]=price_cost
                    all_prompt_tokens[pii]=prompt_tokens
                    all_comple_tokens[pii]=comple_tokens
                    all_num_prompt[pii]=num_prompt


    elif args.scenario_name == "privacy_understanding":
        constructor = MessageConstructor(PROMPT_TEMPLATE[args.template]['sys_prompt'],
                                         PROMPT_TEMPLATE[args.template]['task_desc'])
        for topic in args.privacy_topics:
            if topic not in PRIVACY_TOPIC:
                print(f"Skip {topic} as it is not defined in supported topics {PRIVACY_TOPIC}")
                continue
            dataset, gt_list = get_privacy_understanding_messages(topic, args.question_prompt)
        
            result_dir = ScenarioName2ResultDir[args.scenario_name]
            os.makedirs(result_dir, exist_ok=True)
            out_file = os.path.join(
                result_dir,
                "{}_t{}_s{}_{}_ques{}_{}.jsonl".format(
                    topic, args.template, dataset_size, args.model, args.question_prompt, args.seed
                ).replace("/", "_").replace(" ", "")
            )

            print(f"Saving to {out_file}")
            price_cost = 0
            prompt_tokens=0
            comple_tokens=0
            num_prompt=0
            with TemporaryDirectory(dir="./.cache") as dirname:
                model = Chat.from_helm(model_name=args.model, conv_template=args.conv_template, cache=dirname,
                                       api_key=args.key)
                for i, data in enumerate(dataset):
                    batch_data = [data] * dataset_size
                    cost, cache = model.do_generation(batch_data, constructor, n=args.n, t=args.t,
                                                      max_tokens=args.max_tokens, dry_run=args.dry_run)  # only works for n=1 now...
                    # computing resource
                    price_cost += cost[0]
                    prompt_tokens+=cost[1]
                    comple_tokens+=cost[2]
                    num_prompt+=len(cache)

                    with open(out_file, "a") as f:
                        for idx, x in enumerate(cache):
                            res = {"word": gt_list[i], "continuation": x[1], "isbanned": x[2], "prompt": x[3],
                                   "response": x[-1]}
                            f.write(json.dumps(res) + "\n")
                
                print("Cost: ", price_cost)
                print("Num of prompt tokens: ", prompt_tokens)
                print("Num of completion tokens: ", comple_tokens)
                all_cost[topic]=price_cost
                all_prompt_tokens[topic]=prompt_tokens
                all_comple_tokens[topic]=comple_tokens
                all_num_prompt[topic]=num_prompt


    elif args.scenario_name == "enron_email_extraction":
        constructor = EnronMessageConstructor(PROMPT_TEMPLATE[args.template]['sys_prompt'],
                                              PROMPT_TEMPLATE[args.template]['task_desc'])
        for prompt_type in args.prompt_types:
            print("prompt_type", prompt_type)
            dataset, name_list, email_list = load_enron_email_prompts(args.data_file, prompt_type)
            if dataset_size < 1 or dataset_size > len(dataset):
                dataset_size = len(dataset)
            else:
                dataset = dataset[:dataset_size]

            
            result_dir = ScenarioName2ResultDir[args.scenario_name]
            os.makedirs(result_dir, exist_ok=True)
            out_file = os.path.join(
                result_dir,
                "{}_s{}_{}_{}.jsonl"
                .format(prompt_type, dataset_size, args.model, args.seed).replace("/", "_").replace(" ", "")
            )
            print(f"Saving to {out_file}")
            price_cost = 0
            prompt_tokens=0
            comple_tokens=0
            num_prompt=0
            with TemporaryDirectory(dir="./.cache") as dirname:

                model = Chat.from_helm(model_name=args.model, conv_template=args.conv_template, cache=dirname,
                                       api_key=args.key)

                num_batches = int(dataset_size / args.batch_size)
                for batch_idx in range(num_batches):

                    batch_start = batch_idx * args.batch_size
                    batch_end = (batch_idx + 1) * args.batch_size
                    batch_data = dataset[batch_start:batch_end]
                    batch_name = name_list[batch_start:batch_end]
                    batch_email = email_list[batch_start:batch_end]
                    

                    cost, cache = model.do_generation(batch_data, constructor, n=args.n, t=args.t, 
                                                      max_tokens=args.max_tokens, dry_run=args.dry_run)  # only works for n=1 now...
                    # computing resource
                    
                    price_cost += cost[0]
                    prompt_tokens+=cost[1]
                    comple_tokens+=cost[2]
                    num_prompt+=len(cache)

                    if not os.path.exists(os.path.dirname(out_file)):
                        os.makedirs(os.path.dirname(out_file))
                    with open(out_file, "a") as f:
                        for idx, x in enumerate(cache):
                            res = {"gt-email": batch_email[idx], "gt-name": batch_name[idx], "continuation": x[1],
                                   "isbanned": x[2], "prompt": x[3], "response": x[-1]}
                            f.write(json.dumps(res) + "\n")

               
                
                print("dataset size: ", dataset_size)
                print("Cost: ", price_cost)
                print("Num of prompt tokens: ", prompt_tokens)
                print("Num of completion tokens: ", comple_tokens)
                all_cost[prompt_type]=price_cost
                all_prompt_tokens[prompt_type]=prompt_tokens
                all_comple_tokens[prompt_type]=comple_tokens
                all_num_prompt[prompt_type]=num_prompt


    print("*********final results summary******")


    cost_over_all_settings=0
    prompt_tokens_over_all_settings=0
    comple_tokens_over_all_settings=0
    num_prompt_over_all_settings=0
    for key in all_cost.keys():
        print("scenario: ", key)
        print("Cost: ", all_cost[key])
        print("Num of prompt tokens: ", all_prompt_tokens[key])
        print("Num of completion tokens: ", all_comple_tokens[key])
        cost_over_all_settings+=all_cost[key]
        prompt_tokens_over_all_settings+=all_prompt_tokens[key]
        comple_tokens_over_all_settings+=all_comple_tokens[key]
        num_prompt_over_all_settings+=all_num_prompt[key]
    

    print("*********sum results summary******")
    print("sum - Cost: ", cost_over_all_settings)
    print("sum - Num of prompt tokens: ", prompt_tokens_over_all_settings)
    print("sum - Num of completion tokens: ", comple_tokens_over_all_settings)
    print("sum - Num of prompt: ", num_prompt_over_all_settings)