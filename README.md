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
├── README.md
├── data/
├── toxicity/
├── stereotype/
├── adv-glue-plus-plus/
├── adv_demonstration/
├── privacy/
├── ood/
├── machine_ethics/
└── fairness/
```

The data subdir includes our proposed/generated datasets essential for the evaluation of trustworthiness.
(⚠️WARNING: This paper contains model outputs that may be considered offensive.)

## Getting Started

1. Clone the repository: Start by cloning this repository to your local machine. You can use the following command:

```bash
git clone https://github.com/boxin-wbx/DecodingTrust.git
cd DecodingTrust
```

2. Create a new conda environment (requires python=3.9) and activate it
```bash
conda create --name decoding_trust_env python=3.9
conda activate decoding_trust_env
```

3. Install our customized HELM repo, from which we will call HELM request to different chat models.
```bash
git clone https://github.com/danielz02/helm
cd helm
./pre-commit.sh
```

4. Navigate to the DecodingTrust repo, and install our dependencies

```bash
pip install -r requirements.txt
```

5. Each of the eight areas has its own subdirectory containing the respective code and README.

6. Follow the specific README: Every subdirectory has its own README. Refer to these documents for information on how to run the scripts and interpret the results.


## [Important] Candidate models
In our benchmark, to have consistent conclusions and results, currently we mianly focus on evaluating the following two OpenAI models:

- `gpt-3.5-turbo-0301`          
- `gpt-4-0314`

**Note we use `gpt-3.5-turbo-0301` (with time stamp) released in March instead of `gpt-3.5-turbo` for sake of model evolution to ensure reproducibility.**

Currently, we have supported evaluating all the causal LLMs **hosted in Huggingface**. Specifically, we have tested the following open LLMs:

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
- `chat.py` provides robust APIs for creating requests to OpenAI **Chat Compleition** models and Huggingface autoregressive LLMs. Recommend implementing experiments based on this file. If you think `chat.py` is not good enough and want to make modifications, please let @acphile and @boxinw know.
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
  journal={arXiv preprint arXiv:2306.11698},
  year={2023}
}
```

## Contact
Please reach out to us if you have any questions or suggestions. You can submit an issue or pull request, or send an email to boxinw2@illinois.edu.

Thank you for your interest in DecodingTrust. We hope our work will contribute to a more trustworthy, fair, and robust AI future.