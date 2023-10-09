## Scenario I: Evaluation On Standard Benchmark

dt-run +toxicity=realtoxicityprompts-toxic ++dry_run=True \
    ++model=openai/gpt-3.5-turbo-0301 \
    ++toxicity.n=25 \
    ++toxicity.template=0

#cost: 9.07616199999999
#/ prompt tokens: 53081
#/ completion tokens: 4485000
#Saving to ./data/toxicity/user_prompts/generations/openai/gpt-3.5-turbo-0301/toxic-template-0-generations.jsonl

dt-run +toxicity=realtoxicityprompts-toxic ++dry_run=True \
    ++model=openai/gpt-4-0314 \
    ++toxicity.n=25 \
    ++toxicity.template=0

#cost: 270.6924299999986
#/ prompt tokens: 53081
#/ completion tokens: 4485000
#Directory ./data/toxicity/user_prompts/generations/openai/gpt-4-0314 created.
#Saving to ./data/toxicity/user_prompts/generations/openai/gpt-4-0314/toxic-template-0-generations.jsonl

dt-run +toxicity=realtoxicityprompts-toxic ++dry_run=True \
    ++model=openai/gpt-3.5-turbo-0301 \
    ++toxicity.n=25 \
    ++toxicity.template=1

#cost: 9.308186000000013
#/ prompt tokens: 169093
#/ completion tokens: 4485000
# Saving to ./data/toxicity/user_prompts/generations/openai/gpt-3.5-turbo-0301/toxic-template-1-generations.jsonl

dt-run +toxicity=realtoxicityprompts-toxic ++dry_run=True \
    ++model=openai/gpt-4-0314 \
    ++toxicity.n=25 \
    ++toxicity.template=1

#cost: 274.1727900000005
#/ prompt tokens: 169093
#/ completion tokens: 4485000
#Saving to ./data/toxicity/user_prompts/generations/openai/gpt-4-0314/toxic-template-1-generations.jsonl


dt-run +toxicity=realtoxicityprompts-nontoxic ++dry_run=True \
    ++model=openai/gpt-3.5-turbo-0301 \
    ++toxicity.n=25 \
    ++toxicity.template=0

#cost: 9.103788
#/ prompt tokens: 51894
#/ completion tokens: 4500000
#Saving to ./data/toxicity/user_prompts/generations/openai/gpt-3.5-turbo-0301/nontoxic-template-0-generations.jsonl

dt-run +toxicity=realtoxicityprompts-nontoxic ++dry_run=True \
    ++model=openai/gpt-4-0314 \
    ++toxicity.n=25 \
    ++toxicity.template=0

#cost: 271.5568199999989
#/ prompt tokens: 51894
#/ completion tokens: 4500000
#Saving to ./data/toxicity/user_prompts/generations/openai/gpt-4-0314/nontoxic-template-0-generations.jsonl

dt-run +toxicity=realtoxicityprompts-nontoxic ++dry_run=True \
    ++model=openai/gpt-3.5-turbo-0301 \
    ++toxicity.n=25 \
    ++toxicity.template=1

#1200it [00:01, 794.26it/s]
#cost: 9.336588000000026
#/ prompt tokens: 168294
#/ completion tokens: 4500000
#Saving to ./data/toxicity/user_prompts/generations/openai/gpt-3.5-turbo-0301/nontoxic-template-1-generations.jsonl

dt-run +toxicity=realtoxicityprompts-nontoxic ++dry_run=True \
    ++model=openai/gpt-4-0314 \
    ++toxicity.n=25 \
    ++toxicity.template=1

#1200it [00:01, 834.85it/s]
#cost: 275.0488200000002
#/ prompt tokens: 168294
#/ completion tokens: 4500000
#Saving to ./data/toxicity/user_prompts/generations/openai/gpt-4-0314/nontoxic-template-1-generations.jsonl

## Scenario II: Diverse System Prompts

for i in {0..32}; do
dt-run +toxicity=realtoxicityprompts-toxic ++dry_run=True \
    ++model=openai/gpt-4-0314 \
    ++toxicity.n=1 \
    ++toxicity.template=${i} \
    ++toxicity.t=0
done


for i in {0..32}; do
dt-run +toxicity=realtoxicityprompts-toxic ++dry_run=True \
    ++model=openai/gpt-3-turbo-0301 \
    ++toxicity.n=1 \
    ++toxicity.template=${i} \
    ++toxicity.t=0
done

## Scenario III: Challening User Prompts

dt-run +toxicity=toxic-gpt4 ++dry_run=True \
    ++model=openai/gpt-3.5-turbo-0301 \
    ++toxicity.n=25 \
    ++toxicity.template=0

#1200it [00:01, 889.37it/s]
#cost: 9.20475
#/ prompt tokens: 102375
#/ completion tokens: 4500000
#Saving to ./data/toxicity/user_prompts/generations/openai/gpt-3.5-turbo-0301/toxic-gpt4-template-0-generations.jsonl

dt-run +toxicity=toxic-gpt4 ++dry_run=True \
    ++model=openai/gpt-3.5-turbo-0301 \
    ++toxicity.n=25 \
    ++toxicity.template=1

#1200it [00:01, 789.69it/s]
#cost: 9.43755
#/ prompt tokens: 218775
#/ completion tokens: 4500000
#Saving to ./data/toxicity/user_prompts/generations/openai/gpt-3.5-turbo-0301/toxic-gpt4-template-1-generations.jsonl

dt-run +toxicity=toxic-gpt4 ++dry_run=True \
    ++model=openai/gpt-4-0314 \
    ++toxicity.n=25 \
    ++toxicity.template=0

#1200it [00:01, 855.51it/s]
#cost: 273.0712499999991
#/ prompt tokens: 102375
#/ completion tokens: 4500000
#Saving to ./data/toxicity/user_prompts/generations/openai/gpt-4-0314/toxic-gpt4-template-0-generations.jsonl

dt-run +toxicity=toxic-gpt4 ++dry_run=True \
    ++model=openai/gpt-4-0314 \
    ++toxicity.n=25 \
    ++toxicity.template=1

#1200it [00:01, 823.17it/s]
#cost: 276.56325000000004
#/ prompt tokens: 218775
#/ completion tokens: 4500000
#Saving to ./data/toxicity/user_prompts/generations/openai/gpt-4-0314/toxic-gpt4-template-1-generations.jsonl