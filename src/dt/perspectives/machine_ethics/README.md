# Machine Ethics Evaluation on GPT Models

This directory provides a reproducible guide to run machine ethics evaluation on GPT-3.5 and GPT-4. 

You should currently in the root directory of the repo.  

## Scenario 1: Evaluation on standard machine ethics benchmarks 
### Evaluation on ETHICS dataset
GPT-3.5 Evaluation on short commonsense morality subset under zero-shot setting
```
dt-run +machine_ethics=ethics_commonsense_short ++dry_run=True \
    ++model=openai/gpt-3.5-turbo-0301
```

GPT-4 Evaluation on short commonsense morality subset under zero-shot setting
```
dt-run +machine_ethics=ethics_commonsense_short ++dry_run=True \
    ++model=openai/gpt-4-0314
```

GPT-3.5 Evaluation on short commonsense morality subset under few-shot setting
```
dt-run +machine_ethics=ethics_commonsense_short ++dry_run=True \
    ++model=openai/gpt-3.5-turbo-0301 \
    ++machine_ethics.few_shot_num=32
```

GPT-4 Evaluation on short commonsense morality subset under few-shot setting
```
dt-run +machine_ethics=ethics_commonsense_short ++dry_run=True \
    ++model=openai/gpt-4-0314 \
    ++machine_ethics.few_shot_num=32
```

GPT-3.5 Evaluation on long commonsense morality subset under zero-shot setting
```
dt-run +machine_ethics=ethics_commonsense_long ++dry_run=True \
    ++model=openai/gpt-3.5-turbo-0301
```

GPT-4 Evaluation on long commonsense morality subset under zero-shot setting
```
dt-run +machine_ethics=ethics_commonsense_long ++dry_run=True \
    ++model=openai/gpt-4-0314
```

GPT-3.5 Evaluation on long commonsense morality subset under few-shot setting
```
dt-run +machine_ethics=ethics_commonsense_long ++dry_run=True \
    ++model=openai/gpt-3.5-turbo-0301 \
    ++machine_ethics.few_shot_num=8
```

GPT-4 Evaluation on long commonsense morality subset under few-shot setting
```
dt-run +machine_ethics=ethics_commonsense_long ++dry_run=True \
    ++model=openai/gpt-4-0314 \
    ++machine_ethics.few_shot_num=8
```


### Evaluation on Jiminy Cricket dataset
GPT-3.5 Evaluation under zero-shot setting
```
dt-run +machine_ethics=jiminy ++dry_run=True \
    ++model=openai/gpt-3.5-turbo-0301
```

GPT-4 Evaluation under zero-shot setting
```
dt-run +machine_ethics=jiminy ++dry_run=True \
    ++model=openai/gpt-4-0314
```

GPT-3.5 Evaluation under few-shot setting
```
dt-run +machine_ethics=jiminy ++dry_run=True \
    ++model=openai/gpt-3.5-turbo-0301 \
    ++machine_ethics.few_shot_num=3
```

GPT-4 Evaluation under few-shot setting
```
dt-run +machine_ethics=jiminy ++dry_run=True \
    ++model=openai/gpt-4-0314 \
    ++machine_ethics.few_shot_num=3
```


## Scenario 2: Evaluation on jailbreaking prompts
### Evaluation on ETHICS dataset
GPT-3.5 Evaluation
```
for i in {1..5}; do
dt-run +machine_ethics=ethics_commonsense_short ++dry_run=True \
    ++model=openai/gpt-3.5-turbo-0301 \
    ++machine_ethics.test_num=200 \
    ++machine_ethics.jailbreak_prompt=${i}
done
```

GPT-4 Evaluation
```
for i in {1..5}; do
dt-run +machine_ethics=ethics_commonsense_short ++dry_run=True \
    ++model=openai/gpt-4-0314 \
    ++machine_ethics.test_num=200 \
    ++machine_ethics.jailbreak_prompt=${i}
done
```

### Evaluation on Jiminy Cricket dataset
GPT-3.5 Evaluation
```
for i in {1..5}; do
dt-run +machine_ethics=jiminy ++dry_run=True \
    ++model=openai/gpt-3.5-turbo-0301 \
    ++machine_ethics.test_num=200 \
    ++machine_ethics.jailbreak_prompt=${i}
done
```

GPT-4 Evaluation
```
for i in {1..5}; do
dt-run +machine_ethics=jiminy ++dry_run=True \
    ++model=openai/gpt-4-0314 \
    ++machine_ethics.test_num=200 \
    ++machine_ethics.jailbreak_prompt=${i}
done
```


## Scenario 3: Evaluation on evasive sentences
### Evaluation on ETHICS dataset
GPT-3.5 Evaluation
```
for i in {1..5}; do
dt-run +machine_ethics=ethics_commonsense_short ++dry_run=True \
    ++model=openai/gpt-3.5-turbo-0301 \
    ++machine_ethics.test_num=200 \
    ++machine_ethics.evasive_sentence=${i}
done
```

GPT-4 Evaluation
```
for i in {1..5}; do
dt-run +machine_ethics=ethics_commonsense_short ++dry_run=True \
    ++model=openai/gpt-4-0314 \
    ++machine_ethics.test_num=200 \
    ++machine_ethics.evasive_sentence=${i}
done
```

### Evaluation on Jiminy Cricket dataset
GPT-3.5 Evaluation
```
for i in {1..5}; do
dt-run +machine_ethics=jiminy ++dry_run=True \
    ++model=openai/gpt-3.5-turbo-0301 \
    ++machine_ethics.test_num=200 \
    ++machine_ethics.evasive_sentence=${i}
done
```

GPT-4 Evaluation
```
for i in {1..5}; do
dt-run +machine_ethics=jiminy ++dry_run=True \
    ++model=openai/gpt-4-0314 \
    ++machine_ethics.test_num=200 \
    ++machine_ethics.evasive_sentence=${i}
done
```


## Scenario 4: Evaluation on conditional actions
### Evaluation on Jiminy Cricket subset
GPT-3.5 Evaluation on self-harm
```
dt-run +machine_ethics=jiminy_conditional_harm_self ++dry_run=True \
    ++model=openai/gpt-3.5-turbo-0301
```

GPT-4 Evaluation on self-harm
```
dt-run +machine_ethics=jiminy_conditional_harm_self ++dry_run=True \
    ++model=openai/gpt-4-0314
```

GPT-3.5 Evaluation on harm to others
```
dt-run +machine_ethics=jiminy_conditional_harm_others ++dry_run=True \
    ++model=openai/gpt-3.5-turbo-0301
```

GPT-4 Evaluation on harm to others
```
dt-run +machine_ethics=jiminy_conditional_harm_others ++dry_run=True \
    ++model=openai/gpt-4-0314
```

GPT-3.5 Evaluation on minor harm
```
dt-run +machine_ethics=jiminy_conditional_minor_harm ++dry_run=True \
    ++model=openai/gpt-3.5-turbo-0301
```

GPT-4 Evaluation on minor harm
```
dt-run +machine_ethics=jiminy_conditional_minor_harm ++dry_run=True \
    ++model=openai/gpt-4-0314
```

GPT-3.5 Evaluation on moderate harm
```
dt-run +machine_ethics=jiminy_conditional_moderate_harm ++dry_run=True \
    ++model=openai/gpt-3.5-turbo-0301
```

GPT-4 Evaluation on moderate harm
```
dt-run +machine_ethics=jiminy_conditional_moderate_harm ++dry_run=True \
    ++model=openai/gpt-4-0314
```


## Evaluation in LeaderBoard
If you want to get the results in LeaderBoard, please use the following setup, and then aggregate the results using `calculate_score.py`.
```
for target_model in TARGET_MODEL_NAME
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
					if [ $jailbreak_prompt -eq 0 ] && [ $evasive_sentence -eq 0 ] &&[ $few_shot_num -eq 32 ] && [ $test_num -eq 200 ]; then
						continue
					fi
					if [ $jailbreak_prompt -gt 0 ] || [ $evasive_sentence -gt 0 ]; then
						if [ $few_shot_num -eq 32 ] || [ $test_num eq 0 ]; then
							continue
						fi
					fi
					setting=ethics_commonsense_short.yaml
					PREFIX_TO_REMOVE=configs/machine_ethics/
					SUFFIX_TO_REMOVE=.yaml
					setting="${setting//$PREFIX_TO_REMOVE/}"
					setting="${setting//$SUFFIX TO_REMOVE/}"
					echo "${setting} jailbreak_prompt ${jailbreak_prompt} evasive_sentence ${evasive_sentence} few_shot_num ${few_shot_num} test_num ${test_num}"
					cat << EOF | sbatch
```

