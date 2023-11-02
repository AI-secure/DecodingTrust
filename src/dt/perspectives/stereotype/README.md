# Stereotype study
This folder contains code to run the stereotype study.

This subdirectory contains the following files:
- `bias_generation.py`: contains code for generating model outputs based on stereotype stereotype topics and system prompt type
- `agreement_func.py`: contain helper functions for classify model output agreement

In order to generate GPT outputs for a specific stereotype topic, run the following command:

## Scenario I: Evaluation on benign system prompts

To generate the data and run the stereotype study for the `benign` system prompt, run the following command:

GPT-4 evaluation
```
dt-run key=OPENAI_API_KEY ++model_config.model=openai/gpt-4-0314 ++dry_run=False +stereotype=benign ++stereotype.n_gens=25 ++stereotype.skip_generation=False
```

GPT-3.5 evaluation
```
dt-run key=OPENAI_API_KEY ++model_config.model=openai/gpt-3.5-turbo-0301 ++dry_run=False +stereotype=benign ++stereotype.n_gens=25 ++stereotype.skip_generation=False
```

## Scenario II: Evaluation on untargeted system prompts

To generate the data and run the stereotype study for the `untargeted` system prompt, run the following command:

GPT-4 evaluation
```
dt-run key=OPENAI_API_KEY ++model_config.model=openai/gpt-4-0314 ++dry_run=False +stereotype=untargeted ++stereotype.n_gens=25 ++stereotype.skip_generation=False
```

GPT-3.5 evaluation
```
dt-run key=OPENAI_API_KEY ++model_config.model=openai/gpt-3.5-turbo-0301 ++dry_run=False +stereotype=untargeted ++stereotype.n_gens=25 ++stereotype.skip_generation=False
```

## Scenario III: Evaluation on targeted system prompts

To generate the data and run the stereotype study for the `targeted` system prompt, run the following command:

GPT-4 evaluation
```
dt-run key=OPENAI_API_KEY ++model_config.model=openai/gpt-4-0314 ++dry_run=False +stereotype=targeted ++stereotype.n_gens=25 ++stereotype.skip_generation=False
```

GPT-3.5 evaluation
```
dt-run key=OPENAI_API_KEY ++model_config.model=openai/gpt-3.5-turbo-0301 ++dry_run=False +stereotype=targeted ++stereotype.n_gens=25 ++stereotype.skip_generation=False
```

`bias_generation.py` will by default classify the agreement of the model output to one of the following three categories: 1 (the model agrees), -1 (the model disagrees), and 0 (the model neither agrees nor disagrees).
