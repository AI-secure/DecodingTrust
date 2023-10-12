#!/bin/bash

declare -a folders=("adv_demonstration" "advglue" "fairness" "machine_ethics" "ood" "privacy" "stereotype" "toxicity")

command="dt-run --config-name slurm_config --multirun +model_config=autogptq,hf"

for folder in "${folders[@]}"; do
    # get all config names without the .yaml extension
    configs=$(ls src/dt/configs/${folder}/*.yaml | xargs -n 1 basename | sed 's/.yaml//' | tr '\n' ',' | sed 's/,$//')
    eval "${command} +${folder}=${configs}"
done

# run the constructed command
#eval $command
