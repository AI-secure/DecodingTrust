# DecodingTrust: A Comprehensive Assessment of Trustworthiness in GPT Models

## Overview

This repo contains the source code of DecodingTrust. This research endeavor is designed to help researchers better understand the capabilities, limitations, and potential risks associated with deploying these state-of-the-art Large Language Models (LLMs). See our paper for details.

[**DecodingTrust: A Comprehensive Assessment of Trustworthiness in GPT Models**](https://arxiv.org/abs//2306.11698) 

*Boxin Wang, Weixin Chen, Hengzhi Pei, Chulin Xie, Mintong Kang, Chenhui Zhang, Chejian Xu, Zidi Xiong, Ritik Dutta, Rylan Schaeffer, Sang T. Truong, Simran Arora, Mantas Mazeika, Dan Hendrycks, Zinan Lin, Yu Cheng, Sanmi Koyejo, Dawn Song, Bo Li.*


This project is organized around the following **eight** primary areas of trustworthiness, including:
1. Toxicity
2. Stereotype and bias
3. Adversarial robustness
4. Out-of-Distribution Robustness
5. Privacy 
6. Robustness to Adversarial Demonstrations
7. Machine Ethics
8. Fairness

## Project Structure
This project is structured around subdirectories dedicated to each area of trustworthiness. Each subdir includes scripts, data, and a dedicated README for easy comprehension.

```
.
├── data
│   ├── adv_demonstration
│   │   ├── backdoor
│   │   │   ├── experiment1
│   │   │   ├── experiment2
│   │   │   └── experiment3
│   │   ├── counterfactual
│   │   │   ├── control_raising
│   │   │   ├── control_raising_cf
│   │   │   ├── irregular_form
│   │   │   ├── irregular_form_cf
│   │   │   ├── main_verb
│   │   │   ├── main_verb_cf
│   │   │   ├── snli_hypothesis
│   │   │   ├── snli_hypothesis_cf
│   │   │   ├── snli_premise
│   │   │   ├── snli_premise_cf
│   │   │   ├── syntactic_category
│   │   │   └── syntactic_category_cf
│   │   ├── generations
│   │   │   ├── backdoor
│   │   │   ├── counterfactual
│   │   │   └── spurious
│   │   ├── README.md
│   │   └── spurious
│   │       ├── adverb
│   │       ├── embedded_under_verb
│   │       ├── l_relative_clause
│   │       ├── passive
│   │       ├── PP
│   │       └── s_relative_clause
│   ├── adv-glue-plus-plus
│   │   ├── data
│   │   │   ├── advglue_plus_plus.json
│   │   │   ├── alpaca.json
│   │   │   ├── stable-vicuna.json
│   │   │   └── vicuna.json
│   │   └── README.md
│   ├── fairness
│   │   ├── data
│   │   │   └── adult
│   │   ├── data_generation.py
│   │   ├── datasets
│   │   │   ├── abstract_dataset.py
│   │   │   ├── adult.py
│   │   │   └── __init__.py
│   │   ├── fairness_data
│   │   │   ├── adult_0_200_test_base_rate_0.0.jsonl
│   │   │   ├── adult_0_200_test_base_rate_0.5.jsonl
│   │   │   ├── adult_0_200_test_base_rate_1.0.jsonl
│   │   │   ├── adult_0_200_train_br_0.0_test_br_0.0.jsonl
│   │   │   ├── adult_0_200_train_br_0.0_test_br_0.5.jsonl
│   │   │   ├── adult_0_200_train_br_0.0_test_br_1.0.jsonl
│   │   │   ├── adult_0_8.jsonl
│   │   │   ├── adult_16_200_train_br_0.0_test_br_0.5.jsonl
│   │   │   ├── adult_32_200_train_base_rate_0.0.jsonl
│   │   │   ├── adult_32_200_train_base_rate_0.5.jsonl
│   │   │   ├── adult_32_200_train_base_rate_1.0.jsonl
│   │   │   ├── adult_32_200_train_br_0.0_test_br_0.0.jsonl
│   │   │   ├── adult_32_200_train_br_0.0_test_br_0.5.jsonl
│   │   │   ├── adult_32_200_train_br_0.5_test_br_0.0.jsonl
│   │   │   ├── adult_32_200_train_br_1.0_test_br_0.0.jsonl
│   │   │   ├── gt_labels_adult_0_200_test_base_rate_0.0.npy
│   │   │   ├── gt_labels_adult_0_200_test_base_rate_0.5.npy
│   │   │   ├── gt_labels_adult_0_200_test_base_rate_1.0.npy
│   │   │   ├── gt_labels_adult_0_8.npy
│   │   │   ├── gt_labels_adult_16_200_train_br_0.0_test_br_0.5.npy
│   │   │   ├── gt_labels_adult_32_200_train_base_rate_0.0.npy
│   │   │   ├── gt_labels_adult_32_200_train_base_rate_0.5.npy
│   │   │   ├── gt_labels_adult_32_200_train_base_rate_1.0.npy
│   │   │   ├── gt_labels_adult_32_200_train_br_0.0_test_br_0.5.npy
│   │   │   ├── sensitive_attr_adult_0_200_test_base_rate_0.0.npy
│   │   │   ├── sensitive_attr_adult_0_200_test_base_rate_0.5.npy
│   │   │   ├── sensitive_attr_adult_0_200_test_base_rate_1.0.npy
│   │   │   ├── sensitive_attr_adult_0_8.npy
│   │   │   ├── sensitive_attr_adult_16_200_train_br_0.0_test_br_0.5.npy
│   │   │   ├── sensitive_attr_adult_32_200_train_base_rate_0.0.npy
│   │   │   ├── sensitive_attr_adult_32_200_train_base_rate_0.5.npy
│   │   │   ├── sensitive_attr_adult_32_200_train_base_rate_1.0..npy
│   │   │   └── sensitive_attr_adult_32_200_train_br_0.0_test_br_0.5.npy
│   │   ├── generations
│   │   │   ├── gpt-3.5-turbo-0301_adult_0_200_8.jsonl
│   │   │   ├── gpt-3.5-turbo-0301_adult_0_200.jsonl
│   │   │   ├── gpt-3.5-turbo-0301_adult_0.jsonl
│   │   │   ├── gpt-3.5-turbo-0301_adult_15.jsonl
│   │   │   ├── gpt-3.5-turbo-0301_adult_16_200.jsonl
│   │   │   ├── gpt-3.5-turbo-0301_adult_30.jsonl
│   │   │   ├── gpt-3.5-turbo-0301_adult_32_200.jsonl
│   │   │   ├── gpt-3.5-turbo-0301_adult_32_60.jsonl
│   │   │   ├── gpt-3.5-turbo-0301_adult_32.jsonl
│   │   │   ├── gpt-3.5-turbo-0301_adult_34.jsonl
│   │   │   ├── gpt-3.5-turbo-0301_adult_4_200.jsonl
│   │   │   ├── gpt-3.5-turbo-0301_adult_8_200.jsonl
│   │   │   ├── gpt-3.5-turbo-0301_adult.jsonl
│   │   │   ├── gpt-3.5-turbo-0301_compas_0_60.jsonl
│   │   │   ├── gpt-3.5-turbo-0301_compas_4_8.jsonl
│   │   │   ├── gpt-3.5-turbo-0301_german_0_60.jsonl
│   │   │   ├── gpt-3.5-turbo-0301_german_0_80.jsonl
│   │   │   ├── gpt-4-0314_adult_0_200_8.jsonl
│   │   │   ├── gpt-4-0314_adult_0_200.jsonl
│   │   │   ├── gpt-4-0314_adult_0_40_8.jsonl
│   │   │   ├── gpt-4-0314_adult_0.jsonl
│   │   │   ├── gpt-4-0314_adult_15.jsonl
│   │   │   ├── gpt-4-0314_adult_16_200.jsonl
│   │   │   ├── gpt-4-0314_adult_30.jsonl
│   │   │   ├── gpt-4-0314_adult_32_200.jsonl
│   │   │   ├── gpt-4-0314_adult_32_60.jsonl
│   │   │   ├── gpt-4-0314_adult_32.jsonl
│   │   │   ├── gpt-4-0314_adult_34.jsonl
│   │   │   ├── gpt-4-0314_adult_4_200.jsonl
│   │   │   ├── gpt-4-0314_adult_8_200.jsonl
│   │   │   ├── gpt-4-0314_adult.jsonl
│   │   │   ├── gpt-4-0314_compas_0_60.jsonl
│   │   │   ├── gpt-4-0314_german_0_60.jsonl
│   │   │   ├── gpt-4-0314_german_0_80.jsonl
│   │   │   ├── gpt-4-0314_german_16_60.jsonl
│   │   │   └── gpt-4-0314_german_32_60.jsonl
│   │   └── README.md
│   ├── machine_ethics
│   │   ├── ethics
│   │   │   ├── commonsense
│   │   │   ├── deontology
│   │   │   ├── justice
│   │   │   ├── README.txt
│   │   │   ├── utilitarianism
│   │   │   └── virtue
│   │   ├── jiminy_subset.csv
│   │   ├── jiminy_test.json
│   │   ├── jiminy_train.json
│   │   └── README.md
│   ├── ood
│   │   ├── generations
│   │   │   ├── knowledge
│   │   │   └── styles
│   │   ├── knowledge
│   │   │   └── knowledge.json
│   │   ├── knowledge.json
│   │   ├── README.md
│   │   ├── style.json
│   │   └── styles
│   │       └── style.json
│   ├── privacy
│   │   ├── enron_data
│   │   │   ├── context.pkl
│   │   │   ├── email2name.pkl
│   │   │   ├── email_freq.pkl
│   │   │   ├── five_shot_non_domain.pkl
│   │   │   ├── five_shot.pkl
│   │   │   ├── name2email.pkl
│   │   │   ├── one_shot_non_domain.pkl
│   │   │   ├── one_shot.pkl
│   │   │   ├── two_shot_non_domain.pkl
│   │   │   └── two_shot.pkl
│   │   ├── generations
│   │   │   ├── enron
│   │   │   ├── pii
│   │   │   └── understanding
│   │   └── README.md
│   ├── stereotype
│   │   ├── dataset
│   │   │   ├── stereotype_bias_data.jsonl
│   │   │   ├── system_prompts.csv
│   │   │   └── user_prompts.csv
│   │   ├── generations
│   │   │   ├── benign
│   │   │   ├── chavinlo
│   │   │   ├── lmsys
│   │   │   ├── meta-llama
│   │   │   ├── mosaicml
│   │   │   ├── openai
│   │   │   ├── README.md
│   │   │   ├── targeted
│   │   │   ├── tiiuae
│   │   │   ├── together
│   │   │   └── untargeted
│   │   └── README.md
│   └── toxicity
│       ├── generations
│       │   ├── realtoxicityprompts
│       │   ├── system_prompts
│       │   └── user_prompts
│       ├── README.md
│       ├── toxicity_prompts.py
│       └── user_prompts
│           ├── nontoxic.jsonl
│           ├── toxic.jsonl
│           ├── toxic.jsonl.gpt3.n=25.out1.split.annotated.jsonl.all.jsonl
│           ├── toxic.jsonl.gpt3.n=25.out1.split.annotated.jsonl.filtered.jsonl
│           ├── toxic.jsonl.gpt4.n=25.out1.split.annotated.jsonl.all.jsonl
│           └── toxic.jsonl.gpt4.n=25.out1.split.annotated.jsonl.filtered.jsonl
├── Dockerfile
├── dry_run.sh
├── LICENSE
├── pyproject.toml
├── README.md
├── requirements.txt
├── setup.cfg
├── src
│   └── dt
│       ├── chat.py
│       ├── configs
│       │   ├── adv_demonstration
│       │   ├── advglue
│       │   ├── configs.py
│       │   ├── config.yaml
│       │   ├── fairness
│       │   ├── __init__.py
│       │   ├── machine_ethics
│       │   ├── ood
│       │   ├── privacy
│       │   ├── stereotype
│       │   └── toxicity
│       ├── conversation.py
│       ├── __init__.py
│       ├── main.py
│       ├── perspectives
│       │   ├── adv_demonstration
│       │   ├── advglue
│       │   ├── fairness
│       │   ├── __init__.py
│       │   ├── machine_ethics
│       │   ├── ood
│       │   ├── privacy
│       │   ├── stereotype
│       │   └── toxicity
│       ├── response.py
│       └── utils.py
└── Tutorial.md
```

The data subdir includes our proposed/generated datasets essential for the evaluation of trustworthiness.
(⚠️WARNING: This paper contains model outputs that may be considered offensive.)

## Getting Started

### (Conda +) Pip

For now, we suggest installing DecodingTrust by cloning our repository and install it in editable mode. This will keep the data, code, and configurations in the same place. 

```bash
git clone https://github.com/AI-secure/DecodingTrust.git && cd DecodingTrust
pip install -e .
```

Please note that this will install PyTorch with `pip`. If your system does not have a `CUDA` version compatible with the PyTorch `pip` wheel. To install `PyTorch` with `Conda` first, as shown below.

```bash
conda create --name dt-test python=3.9 pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda activate dt-test
pip install "decoding-trust @ git+https://github.com/AI-secure/DecodingTrust.git"
```

It is also possible to install DecodingTrust as a standalone package, but you will need to clone our repository again to run it with our data.
```bash
conda create --name dt-test python=3.9
conda activate dt-test
pip install "decoding-trust @ git+https://github.com/AI-secure/DecodingTrust.git"
```

### Support for the `ppc64le` Architecture 

We also support the `ppc64le` architecture of IBM Power-9 platforms. To install on this platform, please first make sure you have the following `conda` channels so that we can utilize pre-built packages.

```
--add channels 'defaults'   # lowest priority
--add channels 'https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda-early-access/'
--add channels 'https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/'
--add channels 'https://opence.mit.edu'
--add channels 'https://ftp.osuosl.org/pub/open-ce/current/'
--add channels 'conda-forge'   # highest priority
```

Then, install the following pre-built packages.

```bash
mamba create --name dt-test python==3.9 pytorch=2.0.1 torchvision=0.15.2 spacy=3.5.3 scipy=1.10.1 fairlearn~=0.9.0 scikit-learn~=1.1.2 pandas~=2.0.3 pyarrow~=11.0.0 rust -c conda-forge
```

Finally, install DecodingTrust with `pip` as usual.

### Docker / Singularity

To use DecodingTrust with docker, simply pull the following docker image.
```bash
sudo docker pull danielz01/decoding-trust
docker run -it \
    -v /path/on/host:/path/in/container \
    --gpus all \
    decoding-trust/v1.0:latest [arg1 arg2 ...]
```
To use it in through singularity or apptainer container environments on HPC environments, simply run the following.
```bash
module load singularity  # Change it to whatever module name your singularity / apptainer environment was given
singularity pull decoding-trust-v1.0.sif docker://danielz01/decoding-trust
singularity exec --nv --bind /path/on/host:/path/in/container decoding-trust-v1.0.sif [arg1 arg2]
```

We will also have a container build for `ppc64le` platforms soon. Stay tuned!

### Notes
+ Each of the eight areas has its own subdirectory containing the respective code and README.

+ Follow the specific `README`: Every subdirectory has its own README. Refer to these documents for information on how to run the scripts and interpret the results.


## [Important] Candidate models
In our benchmark, to have consistent conclusions and results, currently we mainly focus on evaluating the following two OpenAI models:

- `gpt-3.5-turbo-0301`          
- `gpt-4-0314`

**Note we use `gpt-3.5-turbo-0301` (with time stamp) released in March instead of `gpt-3.5-turbo` for the sake of model evolution to ensure reproducibility.**

Currently, we have supported evaluating all the causal LLMs **hosted in Huggingface** or hosted locally. Specifically, we have tested the following open LLMs:

- `Llama-v2-7B-Chat`
- `Vicuna-7BAlpaca-7B`
- `MPT-7B`
- `Falcon-7B`
- `Alpaca-7B`
- `RedPajama-INCITE-7B-Instruct`

## Tutorial

We have provided a [Tutorial](Tutorial.md) to help you walk through the usage of API to evaluate different trustworthiness perspectives and LLMs.  

## Useful tips

- Please first evaluate your experiments with `++dry_run=True` flags on to check the input / output format, and use `gpt-3.5-turbo-0301` to check the generation since it has lower costs.
- Suggesting saving the responses from OpenAI.
- You can check `https://arxiv.org/pdf/2302.06476.pdf` to know if your performance of ChatGPT is reasonable for standard tasks. Also you may find their task descriptions useful.

## File usage

- `main.py` provides a unified entry point to evaluate all the perspectives and different LLMs with proper configuration
- `chat.py` provides robust APIs for creating requests to OpenAI **Chat Completion** models and Huggingface autoregressive LLMs. Recommend implementing experiments based on this file. If you think `chat.py` is not good enough and want to make modifications, please let @acphile and @boxinw know.
- `utils.py` provide auxiliary functions   

For other files, please refer to each subdirs for more information.

## License
This project is licensed under the [CC BY-SA 4.0 ]("http://creativecommons.org/licenses/by-sa/4.0/legalcode")  - see the LICENSE file for details.

## Citation
Please cite the paper as follows if you use the data or code from DecodingTrust:
```
@article{wang2023decodingtrust,
  title={DecodingTrust: A Comprehensive Assessment of Trustworthiness in GPT Models},
  author={Wang, Boxin and Chen, Weixin and Pei, Hengzhi and Xie, Chulin and Kang, Mintong and Zhang, Chenhui and Xu, Chejian and Xiong, Zidi and Dutta, Ritik and Schaeffer, Rylan and others},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2023}
}
```

## Contact
Please reach out to us if you have any questions or suggestions. You can submit an issue or pull request, or send an email to boxinw2@illinois.edu.

Thank you for your interest in DecodingTrust. We hope our work will contribute to a more trustworthy, fair, and robust AI future.
