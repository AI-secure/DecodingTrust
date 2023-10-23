# Privacy Evaluation on GPT models

Warning: the model outputs may contain personally identifiable information.


This directory provides a reproducible guide to run privacy evaluation on GPT-3.5 and GPT-4.



## Scenario 1: Evaluation on training data privacy

Evaluation on Enron data under context prompting.

```bash
dt-run +privacy=email_extraction_context50 ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-3.5-turbo-0301 
dt-run +privacy=email_extraction_context100 ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-3.5-turbo-0301 
dt-run +privacy=email_extraction_context200 ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-3.5-turbo-0301 

dt-run +privacy=email_extraction_context50 ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-4-0314
dt-run +privacy=email_extraction_context100 ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-4-0314
dt-run +privacy=email_extraction_context200 ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-4-0314
```


Evaluation on Enron data under zero-shot prompting with known email domain.

```bash
dt-run +privacy=email_extraction_known_zeroshot_a ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-3.5-turbo-0301 
dt-run +privacy=email_extraction_known_zeroshot_b ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-3.5-turbo-0301 
dt-run +privacy=email_extraction_known_zeroshot_c ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-3.5-turbo-0301 
dt-run +privacy=email_extraction_known_zeroshot_d ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-3.5-turbo-0301 

dt-run +privacy=email_extraction_known_zeroshot_a ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-4-0314
dt-run +privacy=email_extraction_known_zeroshot_b ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-4-0314
dt-run +privacy=email_extraction_known_zeroshot_c ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-4-0314
dt-run +privacy=email_extraction_known_zeroshot_d ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-4-0314

```


Evaluation on Enron data under few-shot prompting with known email domain.

```bash

dt-run +privacy=email_extraction_known_oneshot_a ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-3.5-turbo-0301 
dt-run +privacy=email_extraction_known_oneshot_b ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-3.5-turbo-0301 
dt-run +privacy=email_extraction_known_oneshot_c ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-3.5-turbo-0301 
dt-run +privacy=email_extraction_known_oneshot_d ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-3.5-turbo-0301 

dt-run +privacy=email_extraction_known_oneshot_a ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-4-0314
dt-run +privacy=email_extraction_known_oneshot_b ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-4-0314
dt-run +privacy=email_extraction_known_oneshot_c ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-4-0314
dt-run +privacy=email_extraction_known_oneshot_d ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-4-0314

dt-run +privacy=email_extraction_known_fiveshot_a ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-3.5-turbo-0301 
dt-run +privacy=email_extraction_known_fiveshot_b ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-3.5-turbo-0301 
dt-run +privacy=email_extraction_known_fiveshot_c ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-3.5-turbo-0301 
dt-run +privacy=email_extraction_known_fiveshot_d ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-3.5-turbo-0301 

dt-run +privacy=email_extraction_known_fiveshot_a ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-4-0314
dt-run +privacy=email_extraction_known_fiveshot_b ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-4-0314
dt-run +privacy=email_extraction_known_fiveshot_c ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-4-0314
dt-run +privacy=email_extraction_known_fiveshot_d ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-4-0314

```




Evaluation on Enron data under zero-shot prompting with unknown email domain.


```bash
dt-run +privacy=email_extraction_unknown_zeroshot_a ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-3.5-turbo-0301 
dt-run +privacy=email_extraction_unknown_zeroshot_b ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-3.5-turbo-0301 
dt-run +privacy=email_extraction_unknown_zeroshot_c ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-3.5-turbo-0301 
dt-run +privacy=email_extraction_unknown_zeroshot_d ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-3.5-turbo-0301 

dt-run +privacy=email_extraction_unknown_zeroshot_a ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-4-0314
dt-run +privacy=email_extraction_unknown_zeroshot_b ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-4-0314
dt-run +privacy=email_extraction_unknown_zeroshot_c ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-4-0314
dt-run +privacy=email_extraction_unknown_zeroshot_d ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-4-0314
```



Evaluation on Enron data under few-shot prompting with unknown email domain.

```bash

dt-run +privacy=email_extraction_unknown_oneshot_a ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-3.5-turbo-0301 
dt-run +privacy=email_extraction_unknown_oneshot_b ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-3.5-turbo-0301 
dt-run +privacy=email_extraction_unknown_oneshot_c ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-3.5-turbo-0301 
dt-run +privacy=email_extraction_unknown_oneshot_d ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-3.5-turbo-0301 

dt-run +privacy=email_extraction_unknown_oneshot_a ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-4-0314
dt-run +privacy=email_extraction_unknown_oneshot_b ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-4-0314
dt-run +privacy=email_extraction_unknown_oneshot_c ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-4-0314
dt-run +privacy=email_extraction_unknown_oneshot_d ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-4-0314

dt-run +privacy=email_extraction_unknown_fiveshot_a ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-3.5-turbo-0301 
dt-run +privacy=email_extraction_unknown_fiveshot_b ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-3.5-turbo-0301 
dt-run +privacy=email_extraction_unknown_fiveshot_c ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-3.5-turbo-0301 
dt-run +privacy=email_extraction_unknown_fiveshot_d ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-3.5-turbo-0301 


dt-run +privacy=email_extraction_unknown_fiveshot_a ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-4-0314
dt-run +privacy=email_extraction_unknown_fiveshot_b ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-4-0314
dt-run +privacy=email_extraction_unknown_fiveshot_c ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-4-0314
dt-run +privacy=email_extraction_unknown_fiveshot_d ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-4-0314

```




## Evaluation on PII injected during inference

Evaluation using zero-shot prompting.
```bash
dt-run +privacy=pii_zeroshot ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-3.5-turbo-0301 
dt-run +privacy=pii_zeroshot ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-4-0314
```


Evaluation using few-shot privacy-protection demonstrations.
```bash
dt-run +privacy=pii_fewshot_protect ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-3.5-turbo-0301 
dt-run +privacy=pii_fewshot_protect ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-4-0314
dt-run +privacy=pii_fewshot_protect ++dry_run=True +model_config=hf
```


Evaluation using few-shot privacy-leakage demonstrations.
```bash
dt-run +privacy=pii_fewshot_attack ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-3.5-turbo-0301 
dt-run +privacy=pii_fewshot_attack ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-4-0314
```


## Evaluation on different privacy-protecting words and privacy events

Question prompt 1. 
```bash
dt-run +privacy=understanding_q1 ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-3.5-turbo-0301 
dt-run +privacy=understanding_q1 ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-4-0314
dt-run +privacy=understanding_q1 ++dry_run=True +model_config=hf
```

Question prompt 2. 
```bash
dt-run +privacy=understanding_q2 ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-3.5-turbo-0301 
dt-run +privacy=understanding_q2 ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-4-0314
```

Question prompt 3. 
```bash
dt-run +privacy=understanding_q3 ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-3.5-turbo-0301 
dt-run +privacy=understanding_q3 ++dry_run=True ++model_config.key="" ++model_config.model=openai/gpt-4-0314
```



## Calculate final privacy score

```bash
python src/dt/perspectives/privacy/result_agg.py
```