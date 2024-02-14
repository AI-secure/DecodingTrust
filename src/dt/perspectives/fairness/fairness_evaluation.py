import os
import json
import numpy as np
from tempfile import TemporaryDirectory
from fairlearn.metrics import equalized_odds_difference
from fairlearn.metrics import demographic_parity_difference

from dt.perspectives.fairness.fair_chat import do_classification
from dt.perspectives.fairness.task_description import task_dic
from dt.chat import Chat
from dt.perspectives.fairness.score_calculation_script import aggregate_current_scores


def main(args):
    dataset = []
    with open(args.fairness.data_dir + args.fairness.prompt_file) as f:
        for line in f.readlines():
            dataset.append(json.loads(line))

    with TemporaryDirectory(dir="./.cache") as dirname:
        if args.dry_run:
            model = Chat.from_helm(args, cache=dirname)
            acc, unknown, (cost, prompt_tokens, cont_tokens), cache = model.do_classification(dataset, task_message=task_dic[args.fairness.dataset], example_prefix=True, dry_run=args.dry_run)

            print("cost:", cost)
            print("#/ prompt tokens:", prompt_tokens)
            print("#/ completion tokens:", cont_tokens)
        else:
            Chat.do_classification = do_classification
            gpt = Chat.from_helm(args, cache=dirname)
            # TODO: Very dumb monkey patch - refactor later
            task_message = task_dic[args.fairness.dataset]
            acc, unknown, cost, cache, prediction = gpt.do_classification(dataset, task_message, example_prefix=True, args=args)

            out = {
                "requests": [x[0] for x in cache],
                "responses": [x[1] for x in cache],
                "labels": [x["label"] for x in dataset],
                "scores": {"accuracy": acc}
            }
            os.makedirs(os.path.dirname(args.fairness.out_file), exist_ok=True)
            with open(args.fairness.out_file, "w") as dest:
                json.dump(out, dest)

            labels = np.load(args.fairness.data_dir + args.fairness.gt_file)
            sensitive_attr_str = np.load(args.fairness.data_dir + args.fairness.sensitive_attr_file)

            prediction = np.array(prediction)
            if args.fairness.dataset == 'adult':
                sensitive_attr = []
                for i in range(len(sensitive_attr_str)):
                    if sensitive_attr_str[i] == 'Male':
                        sensitive_attr.append(1)
                    else:
                        sensitive_attr.append(0)
                sensitive_attr = np.array(sensitive_attr)
            elif args.fairness.dataset == 'crime':
                labels_ = []
                for i in range(len(labels)):
                    if labels[i] == 'Low' or labels[i] == 'low':
                        labels_.append(1)
                    else:
                        labels_.append(0)
                labels = np.array(labels_)
                sensitive_attr = np.array(sensitive_attr_str)
            else:
                raise ValueError("Dataset not supported")
            idx = prediction >= 0

            # if not all(idx==False):
            #     m_dpd = demographic_parity_difference(labels[idx], prediction[idx], sensitive_features=sensitive_attr[idx])
            #     m_eod = equalized_odds_difference(labels[idx], prediction[idx], sensitive_features=sensitive_attr[idx])
            # else:
            #     m_dpd = m_eod = 0

            aggregate_current_scores(args)
