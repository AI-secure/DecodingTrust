# OOD Evaluation on GPT Models

This directory provides a reproducible guide to run OOD evaluation on GPT-3.5 and GPT-4. 

We use the environment variable `$WORK_DIR` to your current work dir of the repo.

## Evaluation on OOD Style

### Evaluation on the Shake style with p=0.6
In this section, we employ zero-shot evaluation on OOD Style.

GPT-3.5 Evaluation
```
python3 evaluation_ood.py --data-file ../data/ood/styles/dev_shake_p0.6.tsv \
--out-file "<output-file>" \
--key "<api-key>" \
--task style --model gpt-3.5-turbo-0301 --few-shot-num 0
```

GPT-4 Evaluation
```
python3 evaluation_ood.py --data-file ../data/ood/styles/dev_shake_p0.6.tsv \
--out-file "<output-file>" \
--key "<api-key>" \
--task style --model gpt-4-0314 --few-shot-num 0
 ```


## Evaluation on OOD knowledge
### QA2023 Standard
GPT-3.5 Evaluation
```
python3 evaluation_ood.py --data-file ../data/ood/knowledge/qa_2023.jsonl \
--out-file "<output-file>" \
--key "<api-key>" \
--task knowledge --model gpt-3.5-turbo-0301  --few-shot-num 0

```

GPT-4 Evaluation
```
python3 evaluation_ood.py --data-file ../data/ood/knowledge/qa_2023.jsonl \
--out-file "<output-file>" \
--key "<api-key>" \
--task knowledge --model gpt-4-0314  --few-shot-num 0
```

### QA2023 w/IDK
GPT-3.5 Evaluation
```
python3 evaluation_ood.py --data-file ../data/ood/knowledge/qa_2023.jsonl \
--out-file "<output-file>" \
--key "<api-key>" \
--task knowledge --model gpt-3.5-turbo-0301 --idk  --few-shot-num 0

```

GPT-4 Evaluation
```
python3 evaluation_ood.py --data-file ../data/ood/knowledge/qa_2023.jsonl \
--out-file "<output-file>" \
--key "<api-key>" \
--task knowledge --model gpt-4-0314  --idk  --few-shot-num 0
```

## Evaluation on OOD demonstrations in in-context learning

### 8-shot evaluation on the Shake style with p=0.6 and demonstrations from same style

GPT-3.5 Evaluation
```
python3 evaluation_ood.py --data-file ../data/ood/styles/dev_shake_w.tsv \
--out-file "<output-file>" \
--demo-file ../data/ood/styles/train_demo_shake_p0.6_0.tsv \
--key "<api-key>" \
--task style --model gpt-3.5-turbo-0301 --few-shot-num 8 --out-domain shake_p0.6
```

GPT-4 Evaluation
```
python3 evaluation_ood.py --data-file ../data/ood/styles/dev_shake_p0.6.tsv \
--out-file "<output-file>" \
--demo-file ../data/ood/styles/train_demo_shake_p0.6_0.tsv \
--key "<api-key>" \
--task style --model gpt-4-0314--few-shot-num 8 --out-domain shake_p0.6
```

### 5-shot evaluation on QA2020 with QA2021 as demonstrations
GPT-3.5 Evaluation
```
python3 evaluation_ood.py --data-file ../data/ood/knowledge/qa_2020.jsonl \
--out-file "<output-file>" \
--demo-file ../data/ood/knowledge/demo_2021_qa0.jsonl \
--key "<api-key>" \
--task knowledge --model gpt-3.5-turbo-0301 --few-shot-num 5 --out-domain 2021_qa 
```

GPT-4 Evaluation
```
python3 evaluation_ood.py --data-file ../data/ood/knowledge/qa_2020.jsonl \
--out-file "<output-file>" \
--demo-file ../data/ood/knowledge/demo_2021_qa0.jsonl \
--key "<api-key>" \
--task knowledge --model gpt-4-0314  --few-shot-num 5 --out-domain 2021_qa 
```
