# OOD Evaluation with OpenAI API

The usage of the script is as below.

```shell
dt-run +ood=OODScenarioConfig ++dry_run=True \
    ++model="OpenAIModel" \
    ++key=YourOpenAIKey \
    ++ood.out_file="Out File"
```
Specifically, using `gpt-3.5-turbo-0301` as an example,  we have five sets of evaluation configurations:

Evaluation of OOD styles.

```shell
dt-run +ood=style \
    ++model=openai/gpt-3.5-turbo-0301 \
    ++key=YourOpenAIKey \
    ++oods.out_file=data/ood/results/gpt-3.5-turbo-0301/style.json
```

Evaluation of OOD knowledge with standard setting.

```shell
dt-run +ood=knowledge_standard \
    ++model=openai/gpt-3.5-turbo-0301 \
    ++key=YourOpenAIKey \
    ++oods.out_file=data/ood/results/gpt-3.5-turbo-0301/knowledge_standard.json
```


Evaluation of OOD knowledge with additional "I don't know" option.

```shell
dt-run +ood=knowledge_standard \
    ++model=openai/gpt-3.5-turbo-0301 \
    ++key=YourOpenAIKey \
    ++oods.out_file=data/ood/results/gpt-3.5-turbo-0301/knowledge_idk.json
```


Evaluation of OOD in-context demonstrations with different styles.

```shell
dt-run +ood=style_8shot \
    ++model=openai/gpt-3.5-turbo-0301 \
    ++key=YourOpenAIKey \
    ++oods.out_file=data/ood/results/gpt-3.5-turbo-0301/style_8shot.json
```

Evaluation of OOD in-context demonstrations with different domains.

```shell
dt-run +ood=knowledge_2020_5shot \
    ++model=openai/gpt-3.5-turbo-0301 \
    ++key=YourOpenAIKey \
    ++oods.out_file=data/ood/results/gpt-3.5-turbo-0301/knowledge_2020_5shot.json
```
