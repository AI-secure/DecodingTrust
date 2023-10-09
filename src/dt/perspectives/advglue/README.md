# AdvGLUE++ Evaluation with OpenAI API

Please check out `data/adv-glue-plus-plus/README.md` for the details about our AdvGLUE++ dataset.

The usage of the script is as below

```shell
dt-run +advglue=AdvGLUEPlusPlusConfig ++dry_run=True \
    ++model="OpenAIModel" \
    ++key=YourOpenAIKey \
    ++advglue.data_file="AdvGLUE(++) Data"
    ++advglue.out_file="Out File"
```

Adding `++advglue.no_adv` flag will evaluate the benign accuracy of the model by reading from `original_xxx`. Adding `++advglue.resume` will resume the evaluation from a previous result should there be any interruption to a previous run.

For example, the script below evaluates `gpt-3.5-turbo-0301` on the adversarial texts generated against Alpaca-7B model.

```shell
dt-run +advglue=alpaca ++dry_run=True \
    ++model=openai/gpt-3.5-turbo-0301 \
    ++key=YourOpenAIKey \
    ++advglue.data_file=data/adv-glue-plus-plus/data/alpaca.json
    ++advglue.out_file=data/adv-glue-plus-plus/results/advglue-plus-plus-adv-gpt-3.5-turbo-0301-alpaca.json
```
