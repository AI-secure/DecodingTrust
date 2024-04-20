#!/bin/bash
i=0
target_model=$1

for setting in "ethics_commonsense_long" "ethics_commonsense_short" "ethics_deontology" "ethics_justice" "ethics_util" "ethics_virtue" "jiminy_conditional_harm_others" "jiminy_conditional_harm_self" "jiminy_conditional_minor_harm" "jiminy_conditional_moderate_harm" "jiminy"
do
    for few_shot_num in 0 32
    do
        for test_num in 0 200
        do
            for jailbreak_prompt in 0 1 2 3 4 5
            do
                for evasive_sentence in 0 1 2 3 4 5
                do
                    if [ $jailbreak_prompt -gt 0 ] && [ $evasive_sentence -gt 0 ]; then
                        continue
                    fi
                    if [ $jailbreak_prompt -eq 0 ] && [ $evasive_sentence -eq 0 ] && [ $few_shot_num -eq 32 ] && [ $test_num -eq 200 ]; then
                        continue
                    fi
                    if [ $jailbreak_prompt -gt 0 ] || [ $evasive_sentence -gt 0 ]; then
                        if [ $few_shot_num -eq 32 ] || [ $test_num -eq 0 ]; then
                            continue
                        fi
                    fi
                    command_to_run="dt-run +model_config='${target_model}' +machine_ethics='${setting}' ++machine_ethics.few_shot_num='${few_shot_num}' ++machine_ethics.test_num='${test_num}' ++machine_ethics.jailbreak_prompt='${jailbreak_prompt}' ++machine_ethics.evasive_sentence='${evasive_sentence}'"
                    eval "${command_to_run}"
                done
            done
        done
    ((i++))
    done
done