import hydra
from omegaconf import OmegaConf, DictConfig
from collections import namedtuple
from importlib import import_module
from dataclasses import asdict

from hydra.core.config_store import ConfigStore
from .configs import BaseConfig, AdvGLUEConfig, EthicsConfig, FairnessConfig, PrivacyConfig, StereotypeConfig, ModelType

PERSPECTIVES = {
    "stereotype": "stereotype.bias_generation",
    "advglue": "advglue.gpt_eval",
    "toxicity": "toxicity.text_generation_hydra",
    "fairness": "fairness.fairness_evaluation",
    "privacy": "privacy.privacy_evalution",
    "adv_demonstration": "adv_demonstration.adv_demonstration_hydra",
    "machine_ethics": "machine_ethics.test_machine_ethics",
    "ood": "ood.evaluation_ood"
}


cs = ConfigStore.instance()
cs.store(name="config", node=BaseConfig)


@hydra.main(config_path="./configs", config_name="config", version_base="1.2")
def main(config: DictConfig) -> None:
    # The 'validator' methods will be called when you run the line below
    config: BaseConfig = OmegaConf.to_object(config)

    for name, module_name in PERSPECTIVES.items():
        if getattr(config, name) is not None:
            perspective_config = asdict(config)
            del perspective_config[name]
            for k, v in asdict(getattr(config, name)).items():
                perspective_config[k] = v
            perspective_args = namedtuple(f"Config", list(perspective_config.keys()))

            perspective_module = import_module(module_name)
            perspective_module.main(perspective_args(**perspective_config))


if __name__ == "__main__":
    main()
