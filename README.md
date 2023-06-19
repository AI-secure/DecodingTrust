# Decoding Trust: A Comprehensive Assessment of Trustworthiness in GPT Models

## Overview
This project aims at providing a thorough assessment of trustworthiness in GPT models. 

This research endeavor is designed to help researchers better understand the capabilities, limitations, and potential risks associated with deploying these state-of-the-art Large Language Models (LLMs).

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
├── adv_demo/
├── privacy/
├── ood/
├── morality/
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

2. Create a new conda environment and activate it
```bash
conda create --name decoding_trust_env python=3.9
conda activate decoding_trust_env
pip install -r requirements.txt
```

3. Navigate to a subdirectory: Each of the eight areas has its own subdirectory containing the respective code and README.

4. Follow the specific README: Every subdirectory has its own README. Refer to these documents for information on how to run the scripts and interpret the results.


## [Important] Candidate models
In our benchmark, to have consistent conclusions and results, currently we focus on evaluating the following two models:
```
gpt-3.5-turbo-0301          
gpt-4-0314
```

**Note we use `gpt-3.5-turbo-0301` (with time stamp) instead of `gpt-3.5-turbo` for sake of model evolution.**

## Useful tips
Please first evaluate your experiments on `gpt-3.5-turbo-0301` to check the sanity since it has lower costs.

Suggesting saving the responses from OpenAI.

You can check `https://arxiv.org/pdf/2302.06476.pdf` to know if your performance of ChatGPT is reasonable for standard tasks. Also you may find their task descriptions useful.

## File usage

- `chat.py` provides robust APIs for creating requests to OpenAI **Chat Compleition** models. Recommend implementing experiments based on this file. If you think `chat.py` is not good enough and want to make modifications, please let @acphile and @boxinw know.
- `utils.py` provide auxiliary functions   

For other files, please refer to each subdirs for more information.

## License
This project is licensed under the [CC BY-SA 4.0 ]("http://creativecommons.org/licenses/by-sa/4.0/legalcode")  - see the LICENSE file for details.

## Contact
Please reach out to us if you have any questions or suggestions. You can submit an issue or pull request, or send an email to boxinw2@illinois.edu.

Thank you for your interest in Decoding Trust. We hope our work will contribute to a more trustworthy, fair, and robust AI future.