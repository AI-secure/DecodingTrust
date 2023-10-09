# Fairness Evaluation on GPT models

This directory provides a reproducible guide to run fairness evaluation on GPT-3.5 and GPT-4.


## Evaluation in zero-shot settings

Evaluation on Adult test set with base rate parity 0.0 on GPT-3.5.
```commandline
dt-run ++model=openai/gpt-3.5-turbo-0301 ++dry_run=True ++key=''  +fairness=zero_shot_br_0.0.yaml
```

Evaluation on Adult test set with base rate parity 0.0 on GPT-4.
```commandline
dt-run ++model=openai/gpt-4-0314 ++dry_run=True ++key=''  +fairness=zero_shot_br_0.0.yaml
```

Evaluation on Adult test set with base rate parity 0.5 on GPT-3.5.
```commandline
dt-run ++model=openai/gpt-3.5-turbo-0301 ++dry_run=True ++key=''  +fairness=zero_shot_br_0.5.yaml
```

Evaluation on Adult test set with base rate parity 0.5 on GPT-4.
```commandline
dt-run ++model=openai/gpt-4-0314 ++dry_run=True ++key=''  +fairness=zero_shot_br_0.5.yaml
```

Evaluation on Adult test set with base rate parity 1.0 on GPT-3.5.
```commandline
dt-run ++model=openai/gpt-3.5-turbo-0301 ++dry_run=True ++key=''  +fairness=zero_shot_br_1.0.yaml
```

Evaluation on Adult test set with base rate parity 1.0 on GPT-4.
```commandline
dt-run ++model=openai/gpt-4-0314 ++dry_run=True ++key=''  +fairness=zero_shot_br_1.0.yaml
```



## Evaluation under unfair context in few-shot settings

Evaluation using demonstrations with base rate parity 0.0 in 32-shot learning on GPT-3.5.
```commandline
dt-run ++model=openai/gpt-3.5-turbo-0301 ++dry_run=True ++key=''  +fairness=few_shot_tr_br_0.0.yaml
```
Evaluation using demonstrations with base rate parity 0.0 in 32-shot learning on GPT-4.
```commandline
dt-run ++model=openai/gpt-4-0314 ++dry_run=True ++key=''  +fairness=few_shot_tr_br_0.0.yaml
```

Evaluation using demonstrations with base rate parity 0.5 in 32-shot learning on GPT-3.5.
```commandline
dt-run ++model=openai/gpt-3.5-turbo-0301 ++dry_run=True ++key=''  +fairness=few_shot_tr_br_0.5.yaml
```

Evaluation using demonstrations with base rate parity 0.5 in 32-shot learning on GPT-4.
```commandline
dt-run ++model=openai/gpt-4-0314 ++dry_run=True ++key=''  +fairness=few_shot_tr_br_0.5.yaml
```

Evaluation using demonstrations with base rate parity 1.0 in 32-shot learning on GPT-3.5.
```commandline
dt-run ++model=openai/gpt-3.5-turbo-0301 ++dry_run=True ++key=''  +fairness=few_shot_tr_br_1.0.yaml
```

Evaluation using demonstrations with base rate parity 1.0 in 32-shot learning on GPT-4.
```commandline
dt-run ++model=openai/gpt-4-0314 ++dry_run=True ++key=''  +fairness=few_shot_tr_br_1.0.yaml
```

## Evaluation based on different numbers of fair demonstrations

Evaluation using 16 fair demonstrations on GPT-3.5.
```commandline
dt-run ++model=openai/gpt-3.5-turbo-0301 ++dry_run=True ++key=''  +fairness=few_shot_16_fair_demon.yaml
```

Evaluation using 16 fair demonstrations on GPT-4.
```commandline
dt-run ++model=openai/gpt-4-0314 ++dry_run=True ++key=''  +fairness=few_shot_16_fair_demon.yaml
```

Evaluation using 32 fair demonstrations on GPT-3.5.
```commandline
dt-run ++model=openai/gpt-3.5-turbo-0301 ++dry_run=True ++key=''  +fairness=few_shot_32_fair_demon.yaml
```

Evaluation using 32 fair demonstrations on GPT-4.
```commandline
dt-run ++model=openai/gpt-4-0314 ++dry_run=True ++key=''  +fairness=few_shot_32_fair_demon.yaml
```

## Calculate Fairness Scores given Model Output
```commandline
python score_calculation_script.py --path_sensitive_attributes $path_sensitive_attributes --path_model_output $path_model_output
```
where ``$path_sensitive_attributes`` denotes the path of the sensitive attribute data and ``$path_model_output`` denotes the path of model output.