import os
import json
from fairness.fair_chat import FairChat
from fairness.task_description import task_dic
import numpy as np
from tempfile import TemporaryDirectory
from fairlearn.metrics import equalized_odds_difference
from fairlearn.metrics import demographic_parity_difference
from chat import Chat


def main(args):
    dataset = []
    with open(args.data_dir + args.prompt_file) as f:
        for line in f.readlines():
            dataset.append(json.loads(line))

    with TemporaryDirectory(dir="./.cache") as dirname:
        if args.dry_run:
            model = Chat.from_helm(model_name=args.model, conv_template=args.conv_template, cache=dirname,
                                   api_key=args.key)

            acc, unknown, cost, cache = model.do_classification(dataset, task_message=task_dic[args.dataset],
                                                                example_prefix=False,
                                                                dry_run=args.dry_run)

            # computing resource
            price_cost, prompt_tokens, comple_tokens = cost[0], cost[1], cost[2]
            print("Cost: ", price_cost)
            print("Num of prompt tokens: ", prompt_tokens)
            print("Num of completion tokens: ", comple_tokens)
        else:
            gpt = FairChat(
                model_name=args.model.replace("hf/", ""), conv_template=args.conv_template, cache=dirname, api_key=args.key
            )
            task_message = task_dic[args.dataset]
            acc, unknown, cost, cache, prediction = gpt.do_classification(dataset, task_message, example_prefix=True)

            print(acc, unknown, cost)

            out = {
                "requests": [x[0] for x in cache],
                "responses": [x[1] for x in cache],
                "labels": [x["label"] for x in dataset],
                "scores": {"accuracy": acc}
            }
            print(out["scores"])
            with open(args.out_file, "w") as dest:
                json.dump(out, dest)

        labels = np.load(args.data_dir + args.gt_file)
        sensitive_attr_str = np.load(args.data_dir + args.sensitive_attr_file)

        prediction = np.array(prediction)
        if args.dataset == 'adult':
            sensitive_attr = []
            for i in range(len(sensitive_attr_str)):
                if sensitive_attr_str[i] == 'Male':
                    sensitive_attr.append(1)
                else:
                    sensitive_attr.append(0)
            sensitive_attr = np.array(sensitive_attr)
        elif args.dataset == 'compas':
            sensitive_attr = np.array(sensitive_attr_str)
        elif args.dataset == 'german':
            sensitive_attr = np.array(sensitive_attr_str)
        else:
            raise ValueError("Dataset not supported")
        idx = prediction >= 0

        m = demographic_parity_difference(labels[idx], prediction[idx], sensitive_features=sensitive_attr[idx])
        print(f'Demographic parity difference: {m}')
        m = equalized_odds_difference(labels[idx], prediction[idx], sensitive_features=sensitive_attr[idx])
        print(f'Equalized odds difference: {m}')