from .test_cf_hydra import main as counterfactual_main
from .test_sc_hydra import main as spurious_main
from .test_bkd_hydra import main as backdoor_main
from .aggregate_score import counterfactual_score, spurious_score, backdoor_score, aggregate_all
import json


def main(args):
    if args.adv_demonstration.path.startswith("./data/adv_demonstration/counterfactual/"):
        counterfactual_main(args)
    elif args.adv_demonstration.path.startswith("./data/adv_demonstration/spurious/"):
        spurious_main(args)
    elif args.adv_demonstration.path.startswith("./data/adv_demonstration/backdoor/"):
        backdoor_main(args)
    else:
        raise ValueError
    score = {}
    model = args.model_config.model
    score["counterfactual"] = counterfactual_score(model)
    score["spurious"] = spurious_score(model)
    score["backdoor"] = backdoor_score(model)
    final = aggregate_all(score)
    with open(f"./results/adv_demonstration/{model.replace('/', '_')}_score.json", "w") as f:
        json.dump(final, f, indent=4)
