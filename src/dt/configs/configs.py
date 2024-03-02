from helm.proxy.clients.huggingface_model_registry import ModelLoader, WeightType
from dataclasses import dataclass, field
from typing import Optional, List, Union
from enum import Enum, auto


class ModelType(Enum):
    CHAT = auto()
    COMPLETION = auto()


@dataclass
class AdvGLUEConfig:
    sys: bool = False  # Add system prompt
    demo: bool = False  # Add demo examples
    data_file: Optional[str] = None  # Input AdvGLUE JSON file
    out_file: Optional[str] = None  # File to write evaluation results
    no_adv: bool = False  # Evaluate benign result
    resume: bool = False  # Resume from previous results
    save_interval: int = 100  # Intervals to save results
    remove_newline: bool = False
    task: List[str] = field(default_factory=list)


@dataclass
class ToxicityConfig:
    split: Optional[str] = None  # data split
    data_file: Optional[str] = None  # Input the toxicity prompt JSON file
    out_file: Optional[str] = None  # File to write evaluation results
    template: int = 0  # System prompt template id
    max_tokens: int = 150  # Max number of tokens to generate
    n: int = 0  # Number of generations
    t: int = 1  # Generation temperature
    save_interval: int = 100  # Intervals to save results
    api: Optional[str] = None


@dataclass
class AdvDemoConfig:
    path: str = ""  # data folder for testing
    task: str = ""  # key for task description, check adv_demonstration/task_description.py for details
    seeds: str = "42,2333,10007"  # random seeds for testing
    zero: Optional[bool] = False  # use zero-shot evaluation (only for the counterfactual and spurious setting)
    only: Optional[
        bool] = False  # use counterfactual data only as the demonstration(only for thr counterfactual setting)


@dataclass
class EthicsConfig:
    data_name: Optional[str] = None  # Name of the chosen dataset
    test_data_file: Optional[str] = None  # Input testing samples
    train_data_file: Optional[str] = None  # Input training samples used as demonstrations
    out_file: Optional[str] = None  # File to write evaluation results
    test_num: int = 0  # Number of testing samples
    few_shot_num: int = 0  # Number of demonstrations used in the few-shot setting
    jailbreak_prompt: int = 0  # ID of jailbreaking prompt
    evasive_sentence: int = 0  # ID of evasive sentence


@dataclass
class FairnessConfig:
    data_dir: Optional[str] = None  # DIR of the chosen dataset
    prompt_file: Optional[str] = None  # PATH of the file of prompts
    gt_file: Optional[str] = None  # PATH of the file of ground truth labels
    sensitive_attr_file: Optional[str] = None  # PATH of the file of sensitive attributes
    dataset: Optional[str] = 'adult'  # dataset
    out_file: Optional[str] = ''
    score_calculation_only: Optional[bool] = False
    max_tokens: Optional[int] = 20


@dataclass
class PrivacyConfig:
    scenario_name: Optional[str] = None  # Name of the chosen scenario
    data_file: Optional[str] = None  # DIR of the chosen dataset
    out_file: Optional[str] = None  # PATH of the file of output
    template: int = 0  # System prompt template id
    max_tokens: int = 150  # Max number of tokens to generate
    n: int = 1  # Number of generations per API call
    t: int = 1  # Generation temperature
    seed: int = 1  # Random seed

    dataset_size: int = 0  # Number of testing samples
    few_shot_num: int = 0  # Number of demonstrations used in the few-shot setting
    batch_size: int = 0  # Batch size
    question_prompt: int = 0  # Id of question prompt

    personal_infos: List[str] = field(default_factory=list)  # List of personal identifiable information
    prompt_types: List[str] = field(default_factory=list)  # List of prompt types
    privacy_topics: List[str] = field(default_factory=list)  # List of privacy topics


@dataclass
class StereotypeConfig:
    skip_generation: bool = False
    sys_prompt_type: str = "targeted"
    n_gens: int = 25
    out_dir: Optional[str] = None


@dataclass
class OODConfig:
    data_file: str
    out_file: str
    result_file: str
    few_shot_num: int
    task: str
    idk: Optional[bool]
    resume: bool
    save_interval: int
    
@dataclass
class HarmfulnessConfig:
    template: int = 0  # System prompt template id
    data_file: Optional[str] = None  # Input the toxicity prompt JSON file
    out_file: Optional[str] = None  # File to write evaluation results
    max_tokens: int = 300  # Max number of tokens to generate
    n: int = 10  # Number of generations
    t: int = 1  # Generation temperature
    save_interval: int = 100  # Intervals to save results
    api: Optional[str] = None
    eval_method: Optional[str] = 'key_words'
    policy: Optional[str] = 'openai_old'
    main_category: int = 11 


@dataclass
class ModelConfig:
    model: str = "openai/gpt-3.5-turbo-0301"
    type: ModelType = ModelType.CHAT
    conv_template: Optional[str] = None
    model_loader: ModelLoader = ModelLoader.HF
    torch_dtype: Optional[WeightType] = WeightType.BFLOAT16
    trust_remote_code: bool = True
    use_auth_token: bool = True
    disable_exllama: bool = False  # Configuration from AutoGPTQForCausalLM
    inject_fused_attention: bool = True  # Configuration from AutoGPTQForCausalLM
    quant_file: Optional[str] = None
    tokenizer_name: Optional[str] = None
    device_map: Optional[str] = "auto"


@dataclass
class BaseConfig:
    # TODO: Handle device map - change HELM's default behavior
    model_config: ModelConfig
    disable_sys_prompt: Optional[bool] = False

    key: Optional[str] = None  # OpenAI API Key or Huggingface Secret
    dry_run: bool = False

    advglue: Optional[AdvGLUEConfig] = None
    adv_demonstration: Optional[AdvDemoConfig] = None
    fairness: Optional[FairnessConfig] = None
    machine_ethics: Optional[EthicsConfig] = None
    ood: Optional[OODConfig] = None
    privacy: Optional[PrivacyConfig] = None
    stereotype: Optional[StereotypeConfig] = None
    toxicity: Optional[ToxicityConfig] = None
    harmfulness: Optional[HarmfulnessConfig] = None
