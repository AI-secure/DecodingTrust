from pydantic.dataclasses import dataclass
from pydantic import field_validator
from typing import Optional, List
from conversation import conv_templates
import dataclasses

@dataclass
class BaseConfig:
    model: str = "openai/gpt-3.5-turbo-0301"
    conv_template: Optional[str] = None
    key: Optional[str] = None  # OpenAI API Key or Huggingface Secret

    @field_validator("model")
    def validate_model(cls, model: str) -> str:
        if not model:
            raise ValueError("A model name has to be specified.")
        if not (model.startswith("hf/") or model.startswith("openai/")):
            raise ValueError(f"Invalid model name {model}")
        return model

    @field_validator("conv_template")
    def validate_conv_template(cls, conv_template: Optional[str]) -> str:
        if conv_template and (not conv_template in conv_templates):
            raise ValueError("Please specify a valid conversation template for Huggingface model")
        return conv_template


@dataclass
class AdvGLUEConfig(BaseConfig):
    sys: bool = False  # Add system prompt
    demo: bool = False  # Add demo examples
    data_file: Optional[str] = None  # Input AdvGLUE JSON file
    out_file: Optional[str] = None  # File to write evaluation results
    no_adv: bool = False  # Evaluate benign result
    resume: bool = False  # Resume from previous results
    save_interval: int = 100  # Intervals to save results

@dataclass
class ToxicityConfig(BaseConfig):
    data_file: Optional[str] = None  # Input the toxicity prompt JSON file
    out_file: Optional[str] = None  # File to write evaluation results
    template: int = 0  # System prompt template id
    max_tokens: int = 150  # Max number of tokens to generate
    n: int = 0  # Number of generations
    t: int = 1  # Generation temperature

@dataclass
class AdvDemoConfig(BaseConfig):
    path: str = "" # data folder for testing
    task: str = "" # key for task description, check adv_demonstration/task_description.py for details
    seeds: str = "42,2333,10007" # random seeds for testing
    zero: Optional[bool] = False # use zero-shot evaluation (only for the counterfacutal and spurious setting)
    only: Optional[bool] = False # use counterfacutal data only as the demonstration(only for thr counterfacutal setting)

@dataclass
class EthicsConfig(BaseConfig):
    data_name: Optional[str] = None # Name of the chosen dataset
    test_data_file: Optional[str] = None  # Input testing samples
    train_data_file: Optional[str] = None # Input training samples used as demonstrations
    out_file: Optional[str] = None  # File to write evaluation results
    test_num: int = 0  # Number of testing samples
    few_shot_num: int = 0  # Number of demonstrations used in the few-shot setting
    jailbreak_prompt: int = 0  # Id of jailbreaking prompt
    evasive_sentence: int = 0  # Id of evasive sentence

@dataclass
class FairnessConfig(BaseConfig):
    data_dir: Optional[str] = None # DIR of the chosen dataset
    prompt_file: Optional[str] = None # PATH of the file of prompts
    gt_file: Optional[str] = None  # PATH of the file of ground truth labels
    sensitive_attr_file: Optional[str] = None  # PATH of the file of sensitive attributes
    dataset: str = 'adult'  # dataset


@dataclass
class PrivacyConfig(BaseConfig):
    scenario_name: Optional[str] = None # Name of the chosen scenario
    data_file: Optional[str] = None # DIR of the chosen dataset
    out_file: Optional[str] = None # PATH of the file of output
    template: int = 0  # System prompt template id
    max_tokens: int = 150  # Max number of tokens to generate
    n: int = 1  # Number of generations per API call
    t: int = 1  # Generation temperature
    seed: int = 1  # Random seed
    
    dataset_size: int = 0  # Number of testing samples
    few_shot_num: int = 0  # Number of demonstrations used in the few-shot setting
    batch_size: int = 0 # Batch size
    question_prompt: int = 0 # Id of question prompt

    personal_infos: List[str] = dataclasses.field(default_factory=list)  # List of personal identifiable information
    prompt_types: List[str] = dataclasses.field(default_factory=list)  # List of prompt types
    privacy_topics: List[str] = dataclasses.field(default_factory=list) # List of privacy topics

@dataclass
class StereotypeConfig(BaseConfig):
    skip_generation: bool = False
    sys_prompt_type: str = "targeted"
    n_gens: int = 25
