import hydra
from omegaconf import OmegaConf
from collections import namedtuple
from importlib import import_module
from hydra.core.config_store import ConfigStore
from configs import BaseConfig, AdvGLUEConfig, EthicsConfig, FairnessConfig, PrivacyConfig, StereotypeConfig

PERSPECTIVES = {
    "stereotype": "stereotype.bias_generation",
    "adv-glue-plus-plus": "advglue.gpt_eval",
    "toxicity": "toxicity.text_generation_hydra",
    "fairness": "fairness.fairness_evaluation",
    "privacy": "privacy.privacy_evalution",
    "adv_demonstration": "adv_demonstration.adv_demonstration_hydra",
    "machine_ethics": "machine_ethics.test_machine_ethics",
    "ood": "ood.evaluation_ood"
}


def copy_base(source, target):
    for k in ["model", "conv_template", "key", "dry_run"]:
        target[k] = source[k]


@hydra.main(config_path="./configs", config_name="config.yaml", version_base="1.2")
def main(config) -> None:
    # The 'validator' methods will be called when you run the line below
    config = OmegaConf.to_object(config)
    print(config)

    for name, module_name in PERSPECTIVES.items():
        if name in config:
            perspective_config = config[name].copy()
            copy_base(config, perspective_config)
            perspective_args = namedtuple(f"Config", list(perspective_config.keys()))

            perspective_module = import_module(module_name)
            perspective_module.main(perspective_args(**perspective_config))


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="BaseConfig", node=BaseConfig)
    cs.store(name="AdvGLUE", node=AdvGLUEConfig)
    cs.store(name="Privacy", node=PrivacyConfig)
    cs.store(name="Stereotype", node=StereotypeConfig)
    main()
