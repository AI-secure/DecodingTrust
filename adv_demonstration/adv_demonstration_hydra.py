from .test_cf_hydra import main as counterfactual_main
from .test_sc_hydra import main as spurious_main
from .test_bkd_hydra import main as backdoor_main


def main(args):
    if args.path.startswith("./data/adv_demonstration/counterfactual/"):
        return counterfactual_main(args)
    elif args.path.startswith("./data/adv_demonstration/spurious/"):
        return spurious_main(args)
    elif args.path.startswith("./data/adv_demonstration/backdoor/"):
        return backdoor_main(args)
    else:
        raise ValueError
