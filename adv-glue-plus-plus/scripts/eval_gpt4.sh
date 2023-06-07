#!/bin/bash

cd ProjectRootDirectory || exit

export OPENAI_API_KEY=YourOpenAIKey
python adv-glue-plus-plus/gpt_eval.py --model gpt-4-0314 --key YourOpenAIKey --data-file data/adv-glue-plus-plus/data/alpaca.json --out-file data/adv-glue-plus-plus/data/advglue-plus-plus-adv-gpt-4-0314-system-user-no-demo-short-alt-label-alpaca.json
python adv-glue-plus-plus/gpt_eval.py --model gpt-4-0314 --key YourOpenAIKey --data-file data/adv-glue-plus-plus/data/vicuna.json --out-file data/adv-glue-plus-plus/data/advglue-plus-plus-adv-gpt-4-0314-system-user-no-demo-short-alt-label-vicuna.json
python adv-glue-plus-plus/gpt_eval.py --model gpt-4-0314 --key YourOpenAIKey --data-file data/adv-glue-plus-plus/data/stable-vicuna.json --out-file data/adv-glue-plus-plus/data/advglue-plus-plus-adv-gpt-4-0314-system-user-no-demo-short-alt-label-stable-vicuna.json