# Evaluation on robustness against adversarial demonstrations 
## Common arguments
key: `<api-key>`; api-key of OpenAI
path: path of the data, should start with `./data/adv_demo/`
task: `<task>`; key of the task description (check `task_description.py`)
model: `<model>`; `gpt-3.5-turbo-0301` or `gpt-4-0314` in our paper
seeds: random seeds 

## Counterfactual
The folder ending with `_cf` means the demonstration has a counterfactual example. Here we show an example of using `<dataset>=snli_premise`. You can change `<dataset>` to `snli_hypothesis`, `control_raising`, `irregular_form`,`main_verb`, `syntactic_category` to reproduce all the results.  

Available seeds: `42,2333,10007`
### Demo+CF
Template:
```
python test_cf.py --path ../data/adv_demo/counterfactual/<dataset>_cf --task <task> --model <model> --key <api-key>
```

`snli_premise` example:
```
python test_cf.py --path ../data/adv_demo/counterfactual/snli_premise_cf --task nli3 --model <model> --key <api-key>
```
### Demo
Template:
```
python test_cf.py --path ../data/adv_demo/counterfactual/<dataset> --task <task> --model <model> --key <api-key>
```

`snli_premise` example
```
python test_cf.py --path ../data/adv_demo/counterfactual/snli_premise --task nli3 --model <model> --key <api-key>
```
### CF
Template:
```
python test_cf.py --path ../data/adv_demo/counterfactual/<dataset>_cf --task <task> --model <model> --key <api-key> --only
```

`snli_premise` example
```
python test_cf.py --path ../data/adv_demo/counterfactual/snli_premise_cf --task nli3 --model <model> --key <api-key> --only
```
### Zero-shot
Template:
```
python test_cf.py --path ../data/adv_demo/counterfactual/<dataset> --task <task> --model <model> --key <api-key> --zero
```

`snli_premise` example
```
python test_cf.py --path ../data/adv_demo/counterfactual/snli_premise --task nli3 --model <model> --key <api-key> --zero
```

## Spurious Correlation
We have six heuristic types (`<heuristic>`): `PP`, `adverb`,`embedded_under_verb` (Verb),`l_relative_clause` (L\_RC), `passive`, `PP`, `s_relative_clause` (S\_RC).

Available seeds: `0,42,2333,10007,12306`
### Zero-shot evaluation 
```
python test_sc.py --path ../data/adv_demo/spurious/<heuristic>/entail-bias --model <model> --key <api-key> --zero
```
### Evaluation on entailment-correlated demonstration
```
python test_sc.py --path ../data/adv_demo/spurious/<heuristic>/entail-bias --model <model> --key <api-key>
```
### Evaluation on non-entailment-correlated demonstration
```
python test_sc.py --path ../data/adv_demo/spurious/<heuristic>/non-entail-bias --model <model> --key <api-key>
```

## Backdoor
Available seeds: `42,2333,10007`

### Experiment I
We have four backdoor generation methods `<method>`: `badword`, `addsent`, `synbkd`, `stylebkd`.

And three backdoor setups `<setup>`: `setup1`, `setup2`, `setup3`.

Please use the following template to run the evaluation:
```
python test_backdoor.py --path ../data/adv_demo/backdoor/experiment1/sst-2_<setup>_<method> --model <model> --key <api-key>
```

### Experiment II
Here we only use `<setup>=setup2` and `<method>=badword`. Use the following commands to run the evaluation:

For backdoor beginning:
```
python test_backdoor.py --path ../data/adv_demo/backdoor/experiment2/sst-2_setup2_badword_1 --model <model> --key <api-key>
```

For backdoor last:
```
python test_backdoor.py --path ../data/adv_demo/backdoor/experiment2/sst-2_setup2_badword_0 --model <model> --key <api-key>
```

### Experiment III
Here we only use `<setup>=setup2` or `<setup>=setup3`. We consider the location `<location>` to be `first`, `end` or `middle`.

Please use the following template to run the evaluation:
```
python test_backdoor.py --path ../data/adv_demo/backdoor/experiment3/<setup>_cf_<location> --model <model> --key <api-key>
```

### Experiment IV
Use `<task>=badword` to add backdoor instruction in the task description:
```
python test_backdoor.py --path ../data/adv_demo/backdoor/experiment1/sst-2_<setup>_badword --model <model> --key <api-key> --task badword
```

