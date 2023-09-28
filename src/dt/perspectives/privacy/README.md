# Privacy Evaluation on GPT models

Warning: the model outputs may contain personally identifiable information.


This directory provides a reproducible guide to run privacy evaluation on GPT-3.5 and GPT-4.



## Scenario 1: Evaluation on training data privacy

Evaluation on Enron data under context prompting.

```commandline
python main.py +privacy=email_extraction_context50 ++dry_run=True ++key="" ++model=openai/gpt-3.5-turbo-0301 
python main.py +privacy=email_extraction_context100 ++dry_run=True ++key="" ++model=openai/gpt-3.5-turbo-0301 
python main.py +privacy=email_extraction_context200 ++dry_run=True ++key="" ++model=openai/gpt-3.5-turbo-0301 

python main.py +privacy=email_extraction_context50 ++dry_run=True ++key="" ++model=openai/gpt-4-0314
python main.py +privacy=email_extraction_context100 ++dry_run=True ++key="" ++model=openai/gpt-4-0314
python main.py +privacy=email_extraction_context200 ++dry_run=True ++key="" ++model=openai/gpt-4-0314
```


Evaluation on Enron data under zero-shot prompting with known email domain.

```commandline
python main.py +privacy=email_extraction_known_zeroshot_a ++dry_run=True ++key="" ++model=openai/gpt-3.5-turbo-0301 
python main.py +privacy=email_extraction_known_zeroshot_b ++dry_run=True ++key="" ++model=openai/gpt-3.5-turbo-0301 
python main.py +privacy=email_extraction_known_zeroshot_c ++dry_run=True ++key="" ++model=openai/gpt-3.5-turbo-0301 
python main.py +privacy=email_extraction_known_zeroshot_d ++dry_run=True ++key="" ++model=openai/gpt-3.5-turbo-0301 

python main.py +privacy=email_extraction_known_zeroshot_a ++dry_run=True ++key="" ++model=openai/gpt-4-0314
python main.py +privacy=email_extraction_known_zeroshot_b ++dry_run=True ++key="" ++model=openai/gpt-4-0314
python main.py +privacy=email_extraction_known_zeroshot_c ++dry_run=True ++key="" ++model=openai/gpt-4-0314
python main.py +privacy=email_extraction_known_zeroshot_d ++dry_run=True ++key="" ++model=openai/gpt-4-0314

```


Evaluation on Enron data under few-shot prompting with known email domain.

```commandline

python main.py +privacy=email_extraction_known_oneshot_a ++dry_run=True ++key="" ++model=openai/gpt-3.5-turbo-0301 
python main.py +privacy=email_extraction_known_oneshot_b ++dry_run=True ++key="" ++model=openai/gpt-3.5-turbo-0301 
python main.py +privacy=email_extraction_known_oneshot_c ++dry_run=True ++key="" ++model=openai/gpt-3.5-turbo-0301 
python main.py +privacy=email_extraction_known_oneshot_d ++dry_run=True ++key="" ++model=openai/gpt-3.5-turbo-0301 

python main.py +privacy=email_extraction_known_oneshot_a ++dry_run=True ++key="" ++model=openai/gpt-4-0314
python main.py +privacy=email_extraction_known_oneshot_b ++dry_run=True ++key="" ++model=openai/gpt-4-0314
python main.py +privacy=email_extraction_known_oneshot_c ++dry_run=True ++key="" ++model=openai/gpt-4-0314
python main.py +privacy=email_extraction_known_oneshot_d ++dry_run=True ++key="" ++model=openai/gpt-4-0314

python main.py +privacy=email_extraction_known_fiveshot_a ++dry_run=True ++key="" ++model=openai/gpt-3.5-turbo-0301 
python main.py +privacy=email_extraction_known_fiveshot_b ++dry_run=True ++key="" ++model=openai/gpt-3.5-turbo-0301 
python main.py +privacy=email_extraction_known_fiveshot_c ++dry_run=True ++key="" ++model=openai/gpt-3.5-turbo-0301 
python main.py +privacy=email_extraction_known_fiveshot_d ++dry_run=True ++key="" ++model=openai/gpt-3.5-turbo-0301 

python main.py +privacy=email_extraction_known_fiveshot_a ++dry_run=True ++key="" ++model=openai/gpt-4-0314
python main.py +privacy=email_extraction_known_fiveshot_b ++dry_run=True ++key="" ++model=openai/gpt-4-0314
python main.py +privacy=email_extraction_known_fiveshot_c ++dry_run=True ++key="" ++model=openai/gpt-4-0314
python main.py +privacy=email_extraction_known_fiveshot_d ++dry_run=True ++key="" ++model=openai/gpt-4-0314

```




Evaluation on Enron data under zero-shot prompting with unknown email domain.


```commandline
python main.py +privacy=email_extraction_unknown_zeroshot_a ++dry_run=True ++key="" ++model=openai/gpt-3.5-turbo-0301 
python main.py +privacy=email_extraction_unknown_zeroshot_b ++dry_run=True ++key="" ++model=openai/gpt-3.5-turbo-0301 
python main.py +privacy=email_extraction_unknown_zeroshot_c ++dry_run=True ++key="" ++model=openai/gpt-3.5-turbo-0301 
python main.py +privacy=email_extraction_unknown_zeroshot_d ++dry_run=True ++key="" ++model=openai/gpt-3.5-turbo-0301 

python main.py +privacy=email_extraction_unknown_zeroshot_a ++dry_run=True ++key="" ++model=openai/gpt-4-0314
python main.py +privacy=email_extraction_unknown_zeroshot_b ++dry_run=True ++key="" ++model=openai/gpt-4-0314
python main.py +privacy=email_extraction_unknown_zeroshot_c ++dry_run=True ++key="" ++model=openai/gpt-4-0314
python main.py +privacy=email_extraction_unknown_zeroshot_d ++dry_run=True ++key="" ++model=openai/gpt-4-0314
```



Evaluation on Enron data under few-shot prompting with unknown email domain.

```commandline

python main.py +privacy=email_extraction_unknown_oneshot_a ++dry_run=True ++key="" ++model=openai/gpt-3.5-turbo-0301 
python main.py +privacy=email_extraction_unknown_oneshot_b ++dry_run=True ++key="" ++model=openai/gpt-3.5-turbo-0301 
python main.py +privacy=email_extraction_unknown_oneshot_c ++dry_run=True ++key="" ++model=openai/gpt-3.5-turbo-0301 
python main.py +privacy=email_extraction_unknown_oneshot_d ++dry_run=True ++key="" ++model=openai/gpt-3.5-turbo-0301 

python main.py +privacy=email_extraction_unknown_oneshot_a ++dry_run=True ++key="" ++model=openai/gpt-4-0314
python main.py +privacy=email_extraction_unknown_oneshot_b ++dry_run=True ++key="" ++model=openai/gpt-4-0314
python main.py +privacy=email_extraction_unknown_oneshot_c ++dry_run=True ++key="" ++model=openai/gpt-4-0314
python main.py +privacy=email_extraction_unknown_oneshot_d ++dry_run=True ++key="" ++model=openai/gpt-4-0314

python main.py +privacy=email_extraction_unknown_fiveshot_a ++dry_run=True ++key="" ++model=openai/gpt-3.5-turbo-0301 
python main.py +privacy=email_extraction_unknown_fiveshot_b ++dry_run=True ++key="" ++model=openai/gpt-3.5-turbo-0301 
python main.py +privacy=email_extraction_unknown_fiveshot_c ++dry_run=True ++key="" ++model=openai/gpt-3.5-turbo-0301 
python main.py +privacy=email_extraction_unknown_fiveshot_d ++dry_run=True ++key="" ++model=openai/gpt-3.5-turbo-0301 


python main.py +privacy=email_extraction_unknown_fiveshot_a ++dry_run=True ++key="" ++model=openai/gpt-4-0314
python main.py +privacy=email_extraction_unknown_fiveshot_b ++dry_run=True ++key="" ++model=openai/gpt-4-0314
python main.py +privacy=email_extraction_unknown_fiveshot_c ++dry_run=True ++key="" ++model=openai/gpt-4-0314
python main.py +privacy=email_extraction_unknown_fiveshot_d ++dry_run=True ++key="" ++model=openai/gpt-4-0314

```




## Evaluation on PII injected during inference

Evaluation using zero-shot prompting.
```commandline
python main.py +privacy=pii_zeroshot ++dry_run=True ++key="" ++model=openai/gpt-3.5-turbo-0301 
python main.py +privacy=pii_zeroshot ++dry_run=True ++key="" ++model=openai/gpt-4-0314
```


Evaluation using few-shot privacy-protection demonstrations.
```commandline
python main.py +privacy=pii_fewshot_protect ++dry_run=True ++key="" ++model=openai/gpt-3.5-turbo-0301 
python main.py +privacy=pii_fewshot_protect ++dry_run=True ++key="" ++model=openai/gpt-4-0314
```


Evaluation using few-shot privacy-leakage demonstrations.
```commandline
python main.py +privacy=pii_fewshot_attack ++dry_run=True ++key="" ++model=openai/gpt-3.5-turbo-0301 
python main.py +privacy=pii_fewshot_attack ++dry_run=True ++key="" ++model=openai/gpt-4-0314
```


## Evaluation on different privacy-protecting words and privacy events

Question prompt 1. 
```commandline
python main.py +privacy=understanding_q1 ++dry_run=True ++key="" ++model=openai/gpt-3.5-turbo-0301 
python main.py +privacy=understanding_q1 ++dry_run=True ++key="" ++model=openai/gpt-4-0314
```

Question prompt 2. 
```commandline
python main.py +privacy=understanding_q2 ++dry_run=True ++key="" ++model=openai/gpt-3.5-turbo-0301 
python main.py +privacy=understanding_q2 ++dry_run=True ++key="" ++model=openai/gpt-4-0314
```

Question prompt 3. 
```commandline
python main.py +privacy=understanding_q3 ++dry_run=True ++key="" ++model=openai/gpt-3.5-turbo-0301 
python main.py +privacy=understanding_q3 ++dry_run=True ++key="" ++model=openai/gpt-4-0314
```
