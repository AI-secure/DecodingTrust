# Evaluation on robustness against adversarial demonstrations 
## Common arguments
key: `<api-key>`; api-key of OpenAI
path: path of the data, should start with `./data/adv_demonstration/`
task: `<task>`; key of the task description (check `task_description.py`)
model: `<model>`; `gpt-3.5-turbo-0301` or `gpt-4-0314` in our paper
seeds: random seeds 

## Counterfactual
The folder ending with `_cf` means the demonstration has a counterfactual example. Here we show an example of using `<dataset>=snli_premise`. You can change `<dataset>` to `snli_hypothesis`, `control_raising`, `irregular_form`,`main_verb`, `syntactic_category` to reproduce all the results.  

Available seeds: `42,2333,10007`
### Demo+CF
Template:
```
dt-run +adv_demonstration=counterfactual_<dataset>_cf ++model=<model> ++key=<key>
```

### Demo
Template:
```
dt-run +adv_demonstration=counterfactual_<dataset> ++model=<model> ++key=<key>
```

### CF
Template:
```
dt-run +adv_demonstration=counterfactual_<dataset>_cf ++model=<model> ++key=<key> adv_demonstration.only=True
```

### Zero-shot
Template:
```
dt-run +adv_demonstration=counterfactual_<dataset> ++model=<model> ++key=<key> adv_demonstration.zero=True
```

## Spurious Correlation
We have six heuristic types (`<heuristic>`): `PP`, `adverb`,`embedded_under_verb` (Verb),`l_relative_clause` (L\_RC), `passive`, `PP`, `s_relative_clause` (S\_RC).

Available seeds: `0,42,2333,10007,12306`
### Zero-shot evaluation 
```
dt-run +adv_demonstration=spurious_<heuristic>_zero ++model=<model> ++key=<key>
```
### Evaluation on entailment-correlated demonstration
```
dt-run +adv_demonstration=spurious_<heuristic>_entail-bias ++model=<model> ++key=<key>
```
### Evaluation on non-entailment-correlated demonstration
```
dt-run +adv_demonstration=spurious_<heuristic>_non-entail-bias ++model=<model> ++key=<key>
```

## Backdoor
Available seeds: `42,2333,10007`

### Experiment I
We have four backdoor generation methods `<method>`: `badword`, `addsent`, `synbkd`, `stylebkd`.

And three backdoor setups `<setup>`: `setup1`, `setup2`, `setup3`.

Please use the following template to run the evaluation:
```
dt-run +adv_demonstration=backdoor_exp1_<method>_<setup> ++model=<model> ++key=<key>
```

### Experiment II
Here we only use `<setup>=setup2` and `<method>=badword`. Use the following commands to run the evaluation:

For backdoor beginning:
```
dt-run +adv_demonstration=backdoor adv_demonstration.path=./data/adv_demonstration/backdoor/experiment2/sst-2_setup2_badword_1 ++model=<model> ++key=<key>
```

For backdoor last:
```
dt-run +adv_demonstration=backdoor adv_demonstration.path=./data/adv_demonstration/backdoor/experiment2/sst-2_setup2_badword_0 ++model=<model> ++key=<key>
```

### Experiment III
Here we only use `<setup>=setup2` or `<setup>=setup3`. We consider the location `<location>` to be `first`, `end` or `middle`.

Please use the following template to run the evaluation:
```
dt-run +adv_demonstration=backdoor adv_demonstration.path=./data/adv_demonstration/backdoor/experiment3/<setup>_cf_<location> ++model=<model> ++key=<key>
```

### Experiment IV
Use `<task>=badword` to add backdoor instruction in the task description:
```
dt-run +adv_demonstration=backdoor_exp1_badword_setup1 ++model=<model> ++key=<key> adv_demonstration.task=badword
```

