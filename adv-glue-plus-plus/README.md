# AdvGLUE++ Evaluation with OpenAI API

Please check out `data/adv-glue-plus-plus/README.md` for the details about our AdvGLUE++ dataset.

The usage of the script is as below

```shell
python main.py +adv-glue-plus-plus=AdvGLUEPlusPlusConfig ++dry_run=True \
    ++model="OpenAIModel" \
    ++key=YourOpenAIKey \
    ++adv-glue-plus-plus.data_file="AdvGLUE(++) Data"
    ++adv-glue-plus-plus.out_file="Out File"
```

Adding `++adv-glue-plus-plus.no_adv` flag will evaluate the benign accuracy of the model by reading from `original_xxx`. Adding `++adv-glue-plus-plus.resume` will resume the evaluation from a previous result should there be any interruption to a previous run.

For example, the script below evaluates `gpt-3.5-turbo-0301` on the adversarial texts generated against Alpaca-7B model.

```shell
python main.py +adv-glue-plus-plus=alpaca ++dry_run=True \
    ++model=openai/gpt-3.5-turbo-0301 \
    ++key=YourOpenAIKey \
    ++adv-glue-plus-plus.data_file=data/advglue/data/alpaca.json
    ++adv-glue-plus-plus.out_file=data/advglue/data/advglue-plus-plus-adv-gpt-3.5-turbo-0301-alpaca.json
```
