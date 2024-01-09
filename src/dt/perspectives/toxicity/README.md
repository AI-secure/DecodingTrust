# Toxicity Evaluation on GPT Models

This directory provides a reproducible guide to run toxicity evaluation on GPT-3.5 and GPT-4. 

You should currently in the root directory of the repo.  

## Scenario 1: Evaluation on standard benchmarks 

In this section, we employ standard benchmarks to evaluate the toxicity of LLM generations.

### Evaluation on the toxic subset of RealToxicity

GPT-3.5 Evaluation (under an adversarial system prompt, use `++template=0` for the benign system prompt)
```
dt-run +toxicity=realtoxicityprompts-toxic  \
    ++model=openai/gpt-3.5-turbo-0301 \
    ++toxicity.n=25 \
    ++toxicity.template=1
```

GPT-4 Evaluation (under an adversarial system prompt, use `++template=0` for the benign system prompt)
```
dt-run +toxicity=realtoxicityprompts-toxic  \
    ++model=openai/gpt-4-0314 \
    ++toxicity.n=25 \
    ++toxicity.template=1
 ```

### Evaluation on the nontoxic subset of RealToxicity

GPT-3.5 Evaluation (under an adversarial system prompt, use `++template=0` for the benign system prompt)
```
dt-run +toxicity=realtoxicityprompts-toxic  \
    ++model=openai/gpt-3-turbo-0301 \
    ++toxicity.n=25 \
    ++toxicity.template=1
```

GPT-4 Evaluation
```
dt-run +toxicity=realtoxicityprompts-nontoxic  \
    ++model=openai/gpt-4-0314 \
    ++toxicity.n=25 \
    ++toxicity.template=1
 ```


### Toxicity Evaluation using Perspective API

```
PYTHONPATH=. python3 src/dt/perspectives/toxicity/perspective_api.py --data-path data/toxicity/generations/realtoxicityprompts/toxic.jsonl.gpt3.n=25.out1 \
                --prompt-path data/toxicity/user_prompts/toxic.jsonl  --api "<perspective-api-key>"

PYTHONPATH=. python3 src/dt/perspectives/toxicity/perspective_api.py --data-path data/toxicity/generations/realtoxicityprompts/nontoxic.jsonl.gpt3.n=25.out1 \
                --prompt-path data/toxicity/user_prompts/nontoxic.jsonl --api "<perspective-api-key>"
```


## Scenario 2: Evaluation on diverse system prompts
GPT-3.5 Evaluation
```
for i in {0..32}; do
dt-run +toxicity=realtoxicityprompts-toxic  \
    ++model=openai/gpt-3-turbo-0301 \
    ++toxicity.n=1 \
    ++toxicity.template=${i} \
    ++toxicity.t=0
done
```

GPT-4 Evaluation
```
for i in {0..32}; do
dt-run +toxicity=realtoxicityprompts-toxic  \
    ++model=openai/gpt-4-0314 \
    ++toxicity.n=1 \
    ++toxicity.template=${i} \
    ++toxicity.t=0
done
```

## Scenario 3: Evaluation on challenging user prompts

### Construction of challenging user prompts 

Take GPT-4's generation as an example.

We use nltk to tokenize the generations. 

```
python3 src/dt/perspectives/toxicity/augment_prompts.py --data-path data/toxicity/generations/realtoxicityprompts/toxic.jsonl.gpt4.n=25.out1
```

An example output would be `data/toxicity/generations/realtoxicityprompts/toxic.jsonl.gpt4.n=25.out1.split`

Then annotate the generations using Perspective API
```bash
python3 src/dt/perspectives/toxicity/perspective_api_annotate.py --data-path data/toxicity/generations/realtoxicityprompts/toxic.jsonl.gpt4.n=25.out1.split \
--out-path data/toxicity/generations/realtoxicityprompts/toxic.jsonl.gpt4.n=25.out1.split.annotated.jsonl --workers 30
```
Then filter the generations to select the most 1.2K toxic generations as new challenging prompts.

```bash
python3 src/dt/perspectives/toxicity/filter.py --data-path data/toxicity/generations/realtoxicityprompts/toxic.jsonl.gpt4.n=25.out1.split.annotated.jsonl
````

An example output would be `data/toxicity/generations/realtoxicityprompts/toxic.jsonl.gpt4.n=25.out1.split.annotated.jsonl.filtered.jsonl`.

Move the generated prompts to `data/toxicity/userprompts/toxic.jsonl.gpt4.n=25.out1.split.annotated.jsonl.filtered.jsonl` as new challenging task prompts.

### Evaluation on LLM-generated toxic prompts 

GPT-3.5 Evaluation (using gpt-4-generated toxic prompts `++toxicity=toxic-gpt4`, to use gpt-3.5-generated toxic prompts `toxicity=toxic-gpt3.5`.)

```
dt-run +toxicity=toxic-gpt4  \
    ++model=openai/gpt-3.5-turbo-0301 \
    ++toxicity.n=25 \
    ++toxicity.template=1
```

GPT-4 Evaluation
```
dt-run +toxicity=toxic-gpt4  \
    ++model=openai/gpt-4-0314 \
    ++toxicity.n=25 \
    ++toxicity.template=1
```


### Toxicity Evaluation using Perspective API

```
python src/dt/perspectives/toxicity/perspective_api.py --data-path data/toxicity/generations/user_prompts/toxic.jsonl.gpt4.n=25.out1.split.annotated.jsonl.filtered.jsonl.gpt4.n=25.out0 \
                --prompt-path data/toxicity/user_prompts/toxic.jsonl.gpt4.n=25.out1.split.annotated.jsonl.filtered.jsonl  --api "<perspective-api-key>"
```