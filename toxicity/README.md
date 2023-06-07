# Toxicity Evaluation on GPT Models

This directory provides a reproducible guide to run toxicity evaluation on GPT-3.5 and GPT-4. 

You should currently in the root directory of the repo.  

## Scenario 1: Evaluation on standard benchmarks 

In this section, we employ standard benchmarks to evaluate the toxicity of LLM generations.

### Evaluation on the toxic subset of RealToxicity

GPT-3.5 Evaluation (under an adversarial system prompt, use `--template 0` for the benign system prompt)
```
PYTHONPATH=. python3 toxicity/text_generation.py --data-file data/toxicity/user_prompts/toxic.jsonl \
 --out-file data/toxicity/user_prompts/generations/realtoxicityprompts/toxic.jsonl.gpt3.n=25.out1 \
 --key "<api-key>" \
 --model gpt-3.5-turbo-0301 --template 1 --n 25
```

GPT-4 Evaluation (under an adversarial system prompt, use `--template 0` for the benign system prompt)
```
PYTHONPATH=. python3 toxicity/text_generation.py --data-file data/toxicity/user_prompts/toxic.jsonl \
 --out-file data/toxicity/user_prompts/generations/realtoxicityprompts/toxic.jsonl.gpt4.n=25.out1 \
 --key "<api-key>" \
 --model gpt-4-0314 --template 1 --n 25
 ```

GPT-3 Evaluation (Ada, Curie, Davinci)
```
PYTHONPATH=. python3 toxicity/text_generation.py --data-file data/toxicity/user_prompts/toxic.jsonl \
 --out-file data/toxicity/user_prompts/generations/realtoxicityprompts/toxic.jsonl.davinci.n=25.out \
 --key "<api-key>" \
 --model davinci --template 0 --n 25

PYTHONPATH=. python3 toxicity/text_generation.py --data-file data/toxicity/user_prompts/toxic.jsonl \
 --out-file data/toxicity/user_prompts/generations/realtoxicityprompts/toxic.jsonl.curie.n=25.out \
 --key "<api-key>" \
 --model curie --template 0 --n 25

PYTHONPATH=. python3 toxicity/text_generation.py --data-file data/toxicity/user_prompts/toxic.jsonl \
 --out-file data/toxicity/user_prompts/generations/realtoxicityprompts/toxic.jsonl.ada.n=25.out \
 --key "<api-key>" \
 --model ada --template 0 --n 25
```

### Evaluation on the nontoxic subset of RealToxicity

GPT-3.5 Evaluation (under an adversarial system prompt, use `--template 0` for the benign system prompt)
```
PYTHONPATH=. python3 toxicity/text_generation.py --data-file data/toxicity/user_prompts/nontoxic.jsonl \
 --out-file data/toxicity/user_prompts/generations/realtoxicityprompts/nontoxic.jsonl.gpt3.n=25.out1 \
 --key "<api-key>" \
 --model gpt-3.5-turbo-0301 --template 1 --n 25
```

GPT-4 Evaluation
```
PYTHONPATH=. python3 toxicity/text_generation.py --data-file data/toxicity/user_prompts/nontoxic.jsonl \
 --out-file data/toxicity/user_prompts/generations/realtoxicityprompts/nontoxic.jsonl.gpt4.n=25.out1 \
 --key "<api-key>" \
 --model gpt-4-0314 --template 1 --n 25
 ```

GPT-3 Evaluation (Ada, Curie, Davinci)
```
PYTHONPATH=. python3 toxicity/text_generation.py --data-file data/toxicity/user_prompts/nontoxic.jsonl \
 --out-file data/toxicity/user_prompts/generations/realtoxicityprompts/nontoxic.jsonl.davinci.n=25.out \
 --key "<api-key>" \
 --model davinci --template 0 --n 25

PYTHONPATH=. python3 toxicity/text_generation.py --data-file data/toxicity/user_prompts/nontoxic.jsonl \
 --out-file data/toxicity/user_prompts/generations/realtoxicityprompts/nontoxic.jsonl.curie.n=25.out \
 --key "<api-key>" \
 --model curie --template 0 --n 25

PYTHONPATH=. python3 toxicity/text_generation.py --data-file data/toxicity/user_prompts/nontoxic.jsonl \
 --out-file data/toxicity/user_prompts/generations/realtoxicityprompts/nontoxic.jsonl.ada.n=25.out \
 --key "<api-key>" \
 --model ada --template 0 --n 25
```

### Toxicity Evaluation using Perspective API

```
PYTHONPATH=. python3 toxicity/perspective_api.py --data-path data/toxicity/generations/realtoxicityprompts/toxic.jsonl.gpt3.n=25.out1 \
                --prompt-path data/toxicity/user_prompts/toxic.jsonl  --api "<perspective-api-key>"

PYTHONPATH=. python3 toxicity/perspective_api.py --data-path data/toxicity/generations/realtoxicityprompts/nontoxic.jsonl.gpt3.n=25.out1 \
                --prompt-path data/toxicity/user_prompts/nontoxic.jsonl --api "<perspective-api-key>"
```


## Scenario 2: Evaluation on diverse system prompts
GPT-3.5 Evaluation
```
for i in {0..32}; do
  (sleep $((i* 600)) && PYTHONPATH=. python3 toxicity/text_generation.py --data-file data/toxicity/user_prompts/toxic.jsonl \
 --out-file data/toxicity/generations/system_prompts/toxic.jsonl.gpt3.t=0.out${i} \
 --key "<api-key>" \
 --model gpt-3.5-turbo-0301 --template ${i} --t 0) &
done
```

GPT-4 Evaluation
```
 for i in {0..32}; do
  (sleep $((i* 600)) && PYTHONPATH=. python3 toxicity/text_generation.py --data-file data/toxicity/user_prompts/toxic.jsonl \
 --out-file data/toxicity/generations/system_prompts/toxic.jsonl.gpt4.t=0.out${i} \
 --key "<api-key>" \
 --model gpt-4-0314 --template ${i} --t 0) &
done
```

## Scenario 3: Evaluation on challenging user prompts

### Construction of challenging user prompts 

Take GPT-4's generation as an example.

We use nltk to tokenize the generations. 

```
python3 toxicity/augment_prompts.py --data-path data/toxicity/generations/realtoxicityprompts/toxic.jsonl.gpt4.n=25.out1
```

An example output would be `data/toxicity/generations/realtoxicityprompts/toxic.jsonl.gpt4.n=25.out1.split`

Then annotate the generations using Perspective API
```bash
python3 toxicity/perspective_api_annotate.py --data-path data/toxicity/generations/realtoxicityprompts/toxic.jsonl.gpt4.n=25.out1.split \
--out-path data/toxicity/generations/realtoxicityprompts/toxic.jsonl.gpt4.n=25.out1.split.annotated.jsonl --workers 30
```
Then filter the generations to select the most 1.2K toxic generations as new challenging prompts.

```bash
python3 filter.py --data-path data/toxicity/generations/realtoxicityprompts/toxic.jsonl.gpt4.n=25.out1.split.annotated.jsonl
````

An example output would be `data/toxicity/generations/realtoxicityprompts/toxic.jsonl.gpt4.n=25.out1.split.annotated.jsonl.filtered.jsonl`.

Move the generated prompts to `data/toxicity/userprompts/toxic.jsonl.gpt4.n=25.out1.split.annotated.jsonl.filtered.jsonl` as new challenging task prompts.

### Evaluation on LLM-generated toxic prompts 

GPT-3.5 Evaluation

```
PYTHONPATH=. python3 toxicity/text_generation.py --data-file data/toxicity/user_prompts/toxic.jsonl.gpt4.n=25.out1.split.annotated.jsonl.filtered.jsonl \
 --out-file data/toxicity/generations/user_prompts/toxic.jsonl.gpt4.n=25.out1.split.annotated.jsonl.filtered.jsonl.gpt3.n=25.out0 \
 --key "<api-key>" \
 --model gpt-3.5-turbo-0301 --template 0 --n 25
```

GPT-4 Evaluation
```
PYTHONPATH=. python3 toxicity/text_generation.py --data-file data/toxicity/user_prompts/toxic.jsonl.gpt4.n=25.out1.split.annotated.jsonl.filtered.jsonl \
 --out-file data/toxicity/generations/user_prompts/toxic.jsonl.gpt4.n=25.out1.split.annotated.jsonl.filtered.jsonl.gpt4.n=25.out0 \
 --key "<api-key>" \
 --model gpt-4-0314 --template 0 --n 25
```

GPT-3 Evaluation (Ada, Curie, Davinci)
```
PYTHONPATH=. python3 toxicity/text_generation.py --data-file data/toxicity/user_prompts/toxic.jsonl.gpt3.n=25.out1.split.annotated.jsonl.filtered.jsonl \
 --out-file data/toxicity/generations/user_prompts/toxic.jsonl.gpt3.n=25.out1.split.annotated.jsonl.filtered.jsonl.davinci.n=25.out \
 --key "<api-key>" \
 --model avinci --template 0 --n 25

PYTHONPATH=. python3 toxicity/text_generation.py --data-file data/toxicity/user_prompts/toxic.jsonl.gpt4.n=25.out1.split.annotated.jsonl.filtered.jsonl \
 --out-file data/toxicity/generations/user_prompts/toxic.jsonl.gpt4.n=25.out1.split.annotated.jsonl.filtered.jsonl.davinci.n=25.out \
 --key "<api-key>" \
 --model davinci --template 0 --n 25

PYTHONPATH=. python3 toxicity/text_generation.py --data-file data/toxicity/user_prompts/toxic.jsonl.gpt3.n=25.out1.split.annotated.jsonl.filtered.jsonl \
 --out-file data/toxicity/generations/user_prompts/toxic.jsonl.gpt3.n=25.out1.split.annotated.jsonl.filtered.jsonl.ada.n=25.out \
 --key "<api-key>" \
 --model ada --template 0 --n 25

PYTHONPATH=. python3 toxicity/text_generation.py --data-file data/toxicity/user_prompts/toxic.jsonl.gpt4.n=25.out1.split.annotated.jsonl.filtered.jsonl \
 --out-file data/toxicity/generations/user_prompts/toxic.jsonl.gpt4.n=25.out1.split.annotated.jsonl.filtered.jsonl.ada.n=25.out \
 --key "<api-key>" \
 --model ada --template 0 --n 25

PYTHONPATH=. python3 toxicity/text_generation.py --data-file data/toxicity/user_prompts/toxic.jsonl.gpt3.n=25.out1.split.annotated.jsonl.filtered.jsonl \
 --out-file data/toxicity/generations/user_prompts/toxic.jsonl.gpt3.n=25.out1.split.annotated.jsonl.filtered.jsonl.curie.n=25.out \
 --key "<api-key>" \
 --model curie --template 0 --n 25

PYTHONPATH=. python3 toxicity/text_generation.py --data-file data/toxicity/user_prompts/toxic.jsonl.gpt4.n=25.out1.split.annotated.jsonl.filtered.jsonl \
 --out-file data/toxicity/generations/user_prompts/toxic.jsonl.gpt4.n=25.out1.split.annotated.jsonl.filtered.jsonl.curie.n=25.out \
 --key "<api-key>" \
 --model curie --template 0 --n 25
```

### Toxicity Evaluation using Perspective API

```
PYTHONPATH=. python3 toxicity/perspective_api.py --data-path data/toxicity/generations/user_prompts/toxic.jsonl.gpt4.n=25.out1.split.annotated.jsonl.filtered.jsonl.gpt4.n=25.out0 \
                --prompt-path data/toxicity/user_prompts/toxic.jsonl.gpt4.n=25.out1.split.annotated.jsonl.filtered.jsonl  --api "<perspective-api-key>"
```
