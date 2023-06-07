# Privacy Evaluation on GPT models

This directory provides a reproducible guide to run privacy evaluation on GPT-3.5 and GPT-4.


## Evaluation on training data privacy


Evaluation on Enron data under context prompting.

```commandline
python email_extraction.py --data-file context-100 --model gpt-3.5-turbo-0301 --key "<api-key>"
```
- `--data-file` can be  `context-50`, `context-100`, `context-200` for context prompting.
- `--model` can be `gpt-3.5-turbo-0301` and `gpt-4-0314`.
- `--key` is the OpenAPI access key.

Evaluation on Enron data under zero-shot prompting with known email domain.

```commandline
python email_extraction.py --data-file knownzero-shot-a --model gpt-3.5-turbo-0301 --key "<api-key>"
```
- `--data-file` can be  `knownzero-shot-a`, `knownzero-shot-b`, `knownzero-shot-c`, `knownzero-shot-d`  for different zero-shot prompt templates.



Evaluation on Enron data under few-shot prompting with known email domain.

```commandline
python email_extraction.py --data-file one-shot-a --model gpt-3.5-turbo-0301 --key "<api-key>"
```
- `--data-file` can be  `one-shot-a`, `one-shot-b`, `one-shot-c`, `one-shot-d`, `five-shot-a`, `five-shot-b`, `five-shot-c`, `five-shot-d`   for different number of few-shot demonstrations and different templates.




Evaluation on Enron data under zero-shot prompting with unknown email domain.

```commandline
python email_extraction.py --data-file zero-shot-a --model gpt-3.5-turbo-0301  --key "<api-key>"
```
- `--data-file` can be  `zero-shot-a`, `zero-shot-b`, `zero-shot-c`, `zero-shot-d`  for different zero-shot prompt templates.




Evaluation on Enron data under few-shot prompting with known email domain.

```commandline
python email_extraction.py --data-file one-shot-non-domain-a
--model gpt-3.5-turbo-0301 
--key "<api-key>"
```
- `--data-file` can be  `one-shot-non-domain-a`, `one-shot-non-domain-b`, `one-shot-non-domain-c`, `one-shot-non-domain-d`, 
`five-shot-non-domain-a`, `five-shot-non-domain-b`, `five-shot-non-domain-c`, `five-shot-non-domain-d`   for different number of few-shot demonstrations and different templates.





## Evaluation on PII injected during inference

Evaluation using zero-shot prompting.
```commandline
 python pii.py --k 0 --fewshot-type "attack" --personal-info "phone number" --dataset-size 100  --model gpt-3.5-turbo-0301 --key "<api-key>" 
```
- `--personal-info` can be  ``"phone number" "email address"  "address" "password"  "credit card number"  "passport number"  "ssh private key"  "secret key" "canary number"  "canary code" "ssn"  "social security number" "SSN"  "Social Security Number" "S.S.N"  "Social.Security.Number"    "social.security.number"  "S-S-N"   "Social-Security-Number"    "social-security-number" "[SSN]"  "[S-S-N]"   "[S.S.N]"``
- `--model` can be `gpt-3.5-turbo-0301` and `gpt-4-0314`.
- `--dataset-size` is the number of test samples. 


Evaluation using few-shot privacy-protection demonstrations.
```commandline 
python pii.py --k 1 --fewshot-type protect --personal-info "phone number" --dataset-size 100 --model gpt-3.5-turbo-0301 --key "<api-key>" 
```
```commandline
python pii.py --k 3 --fewshot-type protect --personal-info "phone number" --dataset-size 100 --model gpt-3.5-turbo-0301 --key "<api-key>" 
```
- `--k` means the number of privacy-protection demonstrations


Evaluation using few-shot privacy-leakage demonstrations.
```commandline 
python pii.py --k 1 --fewshot-type attack --personal-info "phone number"  --dataset-size 100 --model gpt-3.5-turbo-0301 --key "<api-key>" 
```
```commandline
python pii.py --k 3 --fewshot-type attack --personal-info "phone number" --dataset-size 100 --model gpt-3.5-turbo-0301 --key "<api-key>" 
```
- `--k` means the number of privacy-leakage demonstrations




## Evaluation on different privacy-protecting words and privacy events

Question prompt 1. 
```commandline
python understanding.py --topic "divorce" --dataset-size 100 --question 1 --model gpt-3.5-turbo-0301 --key "<api-key>"
```
- `--topic` can be `"divorce" "health" "party" "layoff" "exam" "bankrupt" "promotion" "vote"`
- `--model` can be `gpt-3.5-turbo-0301` and `gpt-4-0314`.
- `--dataset-size` is the number of runs.


Question prompt 2. 
```commandline
python understanding.py --topic "divorce" --dataset-size 100 --question 2 --model gpt-3.5-turbo-0301 --key "<api-key>"
```

Question prompt 3. 
```commandline
python understanding.py --topic "divorce" --dataset-size 100 --question 3 --model gpt-3.5-turbo-0301 --key "<api-key>"
```
