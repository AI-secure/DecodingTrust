# Evaluation on machine ethics

This directory provides a reproducible guide to run machine ethics evaluation on GPT-3.5 and GPT-4.

## Prepare datasets
Please download the ETHICS dataset according to https://github.com/hendrycks/ethics and put it under data/morality folder.

## Evaluation on standard benchmarks

Evaluation on ETHICS dataset:

```python
python test_ethics.py --key '<api-key>' --test_mode 'short_test' --template 'standard' --model_name 'gpt-3.5-turbo-0301' # on short samples
python test_ethics.py --key '<api-key>' --test_mode 'long_test' --template 'standard_long' --model_name 'gpt-3.5-turbo-0301' # on long samples
```

If you want to specify the number of testing samples, please use '--test_num \$num\$'.

If you want to test in the few-shot setting, please use '--is_few_shot'.

'model_name' can be 'gpt-3.5-turbo-0301' and 'gpt-4-0314'.



Evaluation on Jiminy Cricket dataset:

```python
python test_jiminy.py --key '<api-key>' --template jiminy_zero --model_name 'gpt-3.5-turbo-0301' # in zero-shot setting
python test_jiminy.py --key '<api-key>' --template jiminy_few --model_name 'gpt-3.5-turbo-0301' --is_few_shot # in few-shot setting
```

If you want to specify the number of testing samples, please use '--test_num \$num\$'.

'model_name' can be 'gpt-3.5-turbo-0301' and 'gpt-4-0314'.



## Evaluation on jailbreaking prompts

Taking the jailbreaking prompt jb_prompt1 as an example:

```python
python test_ethics.py --key '<api-key>' --test_mode 'short_test' --model_name 'gpt-3.5-turbo-0301' --test_num 200 --template 'standard_jb1' # on ETHICS dataset
python test_jiminy.py --key '<api-key>' --model_name 'gpt-3.5-turbo-0301' --test_num 200 --template 'jiminy_jb1' # on Jiminy Cricket dataset
```



## Evaluation on evasive sentences

Taking the evasive sentence adv_s1 as an example:

```python
python test_ethics.py --key '<api-key>' --test_mode 'short_test' --model_name 'gpt-3.5-turbo-0301' --test_num 200 --template 'standard_adv1' # on ETHICS dataset
python test_jiminy.py --key '<api-key>' --model_name 'gpt-3.5-turbo-0301' --test_num 200 --template 'jiminy_adv1' # on Jiminy Cricket dataset
```



## Evaluation on conditional actions

```python
python test_jiminy_subset.py --key '<api-key>' --model_name 'gpt-3.5-turbo-0301' --template 'jiminy_sub'
```


## Visualization
```python
python cache2csv.py --dataset $dataset$ --test_num $test_num$ --model_name $model_name$ --template $template$
```
