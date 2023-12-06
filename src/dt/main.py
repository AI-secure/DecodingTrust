import hydra
from importlib import import_module
from dt.configs.configs import BaseConfig
from omegaconf import OmegaConf, DictConfig
from hydra.core.config_store import ConfigStore
from dt.summarize import summarize_results

PERSPECTIVES = {
    "stereotype": "dt.perspectives.stereotype.bias_generation",
    "advglue": "dt.perspectives.advglue.gpt_eval",
    "toxicity": "dt.perspectives.toxicity.text_generation_hydra",
    "fairness": "dt.perspectives.fairness.fairness_evaluation",
    "privacy": "dt.perspectives.privacy.privacy_evaluation",
    "adv_demonstration": "dt.perspectives.adv_demonstration.adv_demonstration_hydra",
    "machine_ethics": "dt.perspectives.machine_ethics.test_machine_ethics",
    "ood": "dt.perspectives.ood.evaluation_ood"
}


cs = ConfigStore.instance()
cs.store(name="config", node=BaseConfig)
cs.store(name="slurm_config", node=BaseConfig)


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(config: DictConfig) -> None:
    # The 'validator' methods will be called when you run the line below
    config: BaseConfig = OmegaConf.to_object(config)
    assert isinstance(config, BaseConfig)
    print(config)

    for name, module_name in PERSPECTIVES.items():
        if getattr(config, name) is not None:
            perspective_module = import_module(module_name)
            perspective_module.main(config)

    summarize_results()


if __name__ == "__main__":
    main()
