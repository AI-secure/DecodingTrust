# AdvGLUE++ Evaluation with OpenAI API

Please check out `data/adv-glue-plus-plus/README.md` for the details about our AdvGLUE++ dataset.

The usage of the script is as below

```shell
python adv-glue-plus-plus/gpt_eval.py --model "OpenAIModel" --key "YourOpenAIKey" --data-file "AdvGLUE(++) Data" --out-file "Out File"
```
Adding `--no-adv` flag will evaluate the benign accuracy of the model by reading from `original_xxx`. Adding `--resume` will resume the evaluation from a previous result should there be any interruption to a previous run.

For example, the script below evaluates `gpt-3.5-turbo-0301` on the adversarial texts generated against Alpaca-7B model.

```shell
python adv-glue-plus-plus/gpt_eval.py --model gpt-3.5-turbo-0301 --key YourOpenAIKey --data-file data/adv-glue-plus-plus/data/alpaca.json --out-file data/adv-glue-plus-plus/data/advglue-plus-plus-adv-gpt-3.5-turbo-0301-alpaca.json
```
