# Fairness Evaluation on GPT models

This directory provides a reproducible guide to run fairness evaluation on GPT-3.5 and GPT-4.


## Evaluation in zero-shot settings

Evaluation on Adult test set with base rate parity 0.0 on GPT-3.5.
```commandline
 python fairness_evaluation.py --data_dir ../data/fairness/fairness_data/ \
 --prompt_file adult_0_200_test_base_rate_0.0.jsonl \
 --gt_file gt_labels_adult_0_200_test_base_rate_0.0.npy \
 --sensitive_attr_file sensitive_attr_adult_0_200_test_base_rate_0.0.npy \
 --key "<api-key>" \
 --model gpt-3.5-turbo-0301
```

Evaluation on Adult test set with base rate parity 0.0 on GPT-4.
```commandline
 python fairness_evaluation.py --data_dir ../data/fairness/fairness_data/ \
 --prompt_file adult_0_200_test_base_rate_0.0.jsonl \
 --gt_file gt_labels_adult_0_200_test_base_rate_0.0.npy \
 --sensitive_attr_file sensitive_attr_adult_0_200_test_base_rate_0.0.npy \
 --key "<api-key>" \
 --model gpt-4-0314
```

Evaluation on Adult test set with base rate parity 0.5 on GPT-3.5.
```commandline
 python fairness_evaluation.py --data_dir ../data/fairness/fairness_data/ \
 --prompt_file adult_0_200_test_base_rate_0.5.jsonl \
 --gt_file gt_labels_adult_0_200_test_base_rate_0.5.npy \
 --sensitive_attr_file sensitive_attr_adult_0_200_test_base_rate_0.5.npy \
 --key "<api-key>" \
 --model gpt-3.5-turbo-0301
```

Evaluation on Adult test set with base rate parity 0.5 on GPT-4.
```commandline
 python fairness_evaluation.py --data_dir ../data/fairness/fairness_data/ \
 --prompt_file adult_0_200_test_base_rate_0.5.jsonl \
 --gt_file gt_labels_adult_0_200_test_base_rate_0.5.npy \
 --sensitive_attr_file sensitive_attr_adult_0_200_test_base_rate_0.5.npy \
 --key "<api-key>" \
 --model gpt-4-0314
```

Evaluation on Adult test set with base rate parity 1.0 on GPT-3.5.
```commandline
 python fairness_evaluation.py --data_dir ../data/fairness/fairness_data/ \
 --prompt_file adult_0_200_test_base_rate_1.0.jsonl \
 --gt_file gt_labels_adult_0_200_test_base_rate_1.0.npy \
 --sensitive_attr_file sensitive_attr_adult_0_200_test_base_rate_1.0.npy \
 --key "<api-key>" \
 --model gpt-3.5-turbo-0301
```

Evaluation on Adult test set with base rate parity 1.0 on GPT-4.
```commandline
 python fairness_evaluation.py --data_dir ../data/fairness/fairness_data/ \
 --prompt_file adult_0_200_test_base_rate_1.0.jsonl \
 --gt_file gt_labels_adult_0_200_test_base_rate_1.0.npy \
 --sensitive_attr_file sensitive_attr_adult_0_200_test_base_rate_1.0.npy \
 --key "<api-key>" \
 --model gpt-4-0314
```



## Evaluation under unfair context in few-shot settings

Evaluation using demonstrations with base rate parity 0.0 in 32-shot learning on GPT-3.5.
```commandline
 python fairness_evaluation.py --data_dir ../data/fairness/fairness_data/ \
 --prompt_file adult_32_200_train_base_rate_0.0.jsonl \
 --gt_file gt_labels_adult_32_200_train_base_rate_0.0.npy \
 --sensitive_attr_file sensitive_attr_adult_32_200_train_base_rate_0.0.npy \
 --key "<api-key>" \
 --model gpt-3.5-turbo-0301
```
Evaluation using demonstrations with base rate parity 0.0 in 32-shot learning on GPT-4.
```commandline
 python fairness_evaluation.py --data_dir ../data/fairness/fairness_data/ \
 --prompt_file adult_32_200_train_base_rate_0.0.jsonl \
 --gt_file gt_labels_adult_32_200_train_base_rate_0.0.npy \
 --sensitive_attr_file sensitive_attr_adult_32_200_train_base_rate_0.0.npy \
 --key "<api-key>" \
 --model gpt-4-0314
```

Evaluation using demonstrations with base rate parity 0.5 in 32-shot learning on GPT-3.5.
```commandline
 python fairness_evaluation.py --data_dir ../data/fairness/fairness_data/ \
 --prompt_file adult_32_200_train_base_rate_0.5.jsonl \
 --gt_file gt_labels_adult_32_200_train_base_rate_0.5.npy \
 --sensitive_attr_file sensitive_attr_adult_32_200_train_base_rate_0.5.npy \
 --key "<api-key>" \
 --model gpt-3.5-turbo-0301
```

Evaluation using demonstrations with base rate parity 0.5 in 32-shot learning on GPT-4.
```commandline
 python fairness_evaluation.py --data_dir ../data/fairness/fairness_data/ \
 --prompt_file adult_32_200_train_base_rate_0.5.jsonl \
 --gt_file gt_labels_adult_32_200_train_base_rate_0.5.npy \
 --sensitive_attr_file sensitive_attr_adult_32_200_train_base_rate_0.5.npy \
 --key "<api-key>" \
 --model gpt-4-0314
```

Evaluation using demonstrations with base rate parity 1.0 in 32-shot learning on GPT-3.5.
```commandline
 python fairness_evaluation.py --data_dir ../data/fairness/fairness_data/ \
 --prompt_file adult_32_200_train_base_rate_1.0.jsonl \
 --gt_file gt_labels_adult_32_200_train_base_rate_1.0.npy \
 --sensitive_attr_file sensitive_attr_adult_32_200_train_base_rate_1.0.npy \
 --key "<api-key>" \
 --model gpt-3.5-turbo-0301
```

Evaluation using demonstrations with base rate parity 1.0 in 32-shot learning on GPT-4.
```commandline
 python fairness_evaluation.py --data_dir ../data/fairness/fairness_data/ \
 --prompt_file adult_32_200_train_base_rate_1.0.jsonl \
 --gt_file gt_labels_adult_32_200_train_base_rate_1.0.npy \
 --sensitive_attr_file sensitive_attr_adult_32_200_train_base_rate_1.0.npy \
 --key "<api-key>" \
 --model gpt-4-0314
```

## Evaluation based on different numbers of fair demonstrations

Evaluation using 16 fair demonstrations on GPT-3.5.
```commandline
 python fairness_evaluation.py --data_dir ../data/fairness/fairness_data/ \
 --prompt_file adult_16_200_train_br_0.0_test_br_0.5.jsonl \
 --gt_file gt_labels_adult_16_200_train_br_0.0_test_br_0.5.npy \
 --sensitive_attr_file sensitive_attr_adult_16_200_train_br_0.0_test_br_0.5.npy \
 --key "<api-key>" \
 --model gpt-3.5-turbo-0301
```

Evaluation using 16 fair demonstrations on GPT-4.
```commandline
 python fairness_evaluation.py --data_dir ../data/fairness/fairness_data/ \
 --prompt_file adult_16_200_train_br_0.0_test_br_0.5.jsonl \
 --gt_file gt_labels_adult_16_200_train_br_0.0_test_br_0.5.npy \
 --sensitive_attr_file sensitive_attr_adult_16_200_train_br_0.0_test_br_0.5.npy \
 --key "<api-key>" \
 --model gpt-4-0314
```

Evaluation using 32 fair demonstrations on GPT-3.5.
```commandline
 python fairness_evaluation.py --data_dir ../data/fairness/fairness_data/ \
 --prompt_file adult_32_200_train_br_0.0_test_br_0.5.jsonl \
 --gt_file gt_labels_adult_32_200_train_br_0.0_test_br_0.5.npy \
 --sensitive_attr_file sensitive_attr_adult_32_200_train_br_0.0_test_br_0.5.npy \
 --key "<api-key>" \
 --model gpt-3.5-turbo-0301
```

Evaluation using 32 fair demonstrations on GPT-4.
```commandline
 python fairness_evaluation.py --data_dir ../data/fairness/fairness_data/ \
 --prompt_file adult_32_200_train_br_0.0_test_br_0.5.jsonl \
 --gt_file gt_labels_adult_32_200_train_br_0.0_test_br_0.5.npy \
 --sensitive_attr_file sensitive_attr_adult_32_200_train_br_0.0_test_br_0.5.npy \
 --key "<api-key>" \
 --model gpt-4-0314
```
