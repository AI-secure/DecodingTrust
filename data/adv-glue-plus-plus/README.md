# Dataset Description
We follow the same setting in AdvGLUE and consider the following five most representative and challenging tasks: Sentiment Analysis (SST-2), Duplicate Question Detection (QQP), and
Natural Language Inference (NLI, including MNLI, RTE, QNLI). Specifically, we use the dev sets of
these tasks as our source samples, upon which we perform word-level adversarial attacks based on
attack strategies in AdvGLUE.

To attack GPT-3 and GPT-4 with our adversarial texts, please check out the scripts under `data/adv-glue-plus-plus/scripts`. 

## Data Format

File `alpaca.json`, `vicuna.json`, and `stable-vicuna.json` contain the generated adversarial texts of AdvGLUE++ dataset. Each task forms a key-value pair inside the json object. The structure of the file should look like:
```json
{
    "sst2": [
        {
            "idx": "index",
            "label": "label",
            "sentence": "text"
        }
    ],
    "qqp": [
        {
            "index": "index",
            "label": "label",
            "question1": "text",
            "question2": "text"
        }
    ],
    "mnli": [
        {
            "index": "index",
            "label": "label",
            "premise": "text",
            "hypothesis": "text"
        }
    ],
    "mnli-mm": [
        {
            "index": "index",
            "label": "label",
            "premise": "text",
            "hypothesis": "text"
        }
    ],
    "qnli": [
        {
            "index": "index",
            "label": "label",
            "question": "text",
            "sentence": "text"
        }
    ],
    "rte": [
        {
            "index": "index",
            "label": "label",
            "sentence1": "text",
            "sentence2": "text"
        }
    ]
}
```
Please check out our evaluation script `gpt_eval.py` for more details about the JSON schema.

## Additional AdvGLUE Data

Because AdvGLUE is an active benchmark, we won't be able to release the test set of the standard AdvGLUE benchmark.
However, the readers can download the AdvGLUE dev set [here](https://adversarialglue.github.io/dataset/dev_ann.json) and evaluate them through OpenAI APIs with the same script we provided.

# Attack Methods
We leverage the word-level attacks in AdvGLUE to generate adversarial sentences
against the three base models: Alpaca-7B, Vicuna-13B, and StableVicuna-13B. These adversarial attacks perturb the words through different strategies such that the model’s predictions on the
perturbed sentences are dramatically changed while the semantic meaning of these sentences is
preserved. Specifically, we consider the following five kinds of word-level perturbations: typo-based
perturbation (TextBugger), embedding-similarity-based perturbation (TextFooler), context-
aware perturbation (BERT-ATTACK), knowledge-guided perturbation (SememePSO),
and semantic-optimization-based perturbation (SemAttack).

# Target Models
We target Alpaca-7B, Vicuna-13B v0, and StableVicuna-13B to generate transferable adversarial texts against GPT-3.5 and GPT-4.
