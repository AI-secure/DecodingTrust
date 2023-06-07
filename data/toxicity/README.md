# Toxicity Evaluation Data Structures

## File Structures

```
.
├── generations
│   ├── realtoxicityprompts
│   │   ├── nontoxic.jsonl.ada.n=25.out
│   │   ├── nontoxic.jsonl.curie.n=25.out
│   │   ├── nontoxic.jsonl.davinci.n=25.out
│   │   ├── nontoxic.jsonl.gpt3.n=25.out0
│   │   ├── nontoxic.jsonl.gpt3.n=25.out2
│   │   ├── nontoxic.jsonl.gpt4.n=25.out0
│   │   ├── nontoxic.jsonl.gpt4.n=25.out2
│   │   ├── toxic.jsonl.ada.n=25.out
│   │   ├── toxic.jsonl.curie.n=25.out
│   │   ├── toxic.jsonl.davinci.n=25.out
│   │   ├── toxic.jsonl.gpt3.n=25.out0
│   │   ├── toxic.jsonl.gpt3.n=25.out2
│   │   └── toxic.jsonl.gpt4.n=25.out0
│   ├── system_prompts
│   │   ├── toxic.jsonl.gpt3.t=0.out0
│   │   ├── toxic.jsonl.gpt3.t=0.out1
│   │   ├── toxic.jsonl.gpt3.t=0.out10
│   │   ├── toxic.jsonl.gpt3.t=0.out11
│   │   ├── toxic.jsonl.gpt3.t=0.out12
│   │   ├── toxic.jsonl.gpt3.t=0.out13
│   │   ├── toxic.jsonl.gpt3.t=0.out14
│   │   ├── toxic.jsonl.gpt3.t=0.out15
│   │   ├── toxic.jsonl.gpt3.t=0.out16
│   │   ├── toxic.jsonl.gpt3.t=0.out17
│   │   ├── toxic.jsonl.gpt3.t=0.out18
│   │   ├── toxic.jsonl.gpt3.t=0.out19
│   │   ├── toxic.jsonl.gpt3.t=0.out2
│   │   ├── toxic.jsonl.gpt3.t=0.out20
│   │   ├── toxic.jsonl.gpt3.t=0.out21
│   │   ├── toxic.jsonl.gpt3.t=0.out22
│   │   ├── toxic.jsonl.gpt3.t=0.out23
│   │   ├── toxic.jsonl.gpt3.t=0.out24
│   │   ├── toxic.jsonl.gpt3.t=0.out25
│   │   ├── toxic.jsonl.gpt3.t=0.out26
│   │   ├── toxic.jsonl.gpt3.t=0.out27
│   │   ├── toxic.jsonl.gpt3.t=0.out28
│   │   ├── toxic.jsonl.gpt3.t=0.out29
│   │   ├── toxic.jsonl.gpt3.t=0.out3
│   │   ├── toxic.jsonl.gpt3.t=0.out30
│   │   ├── toxic.jsonl.gpt3.t=0.out31
│   │   ├── toxic.jsonl.gpt3.t=0.out32
│   │   ├── toxic.jsonl.gpt3.t=0.out4
│   │   ├── toxic.jsonl.gpt3.t=0.out5
│   │   ├── toxic.jsonl.gpt3.t=0.out6
│   │   ├── toxic.jsonl.gpt3.t=0.out7
│   │   ├── toxic.jsonl.gpt3.t=0.out8
│   │   ├── toxic.jsonl.gpt3.t=0.out9
│   │   ├── toxic.jsonl.gpt4.t=0.out0
│   │   ├── toxic.jsonl.gpt4.t=0.out1
│   │   ├── toxic.jsonl.gpt4.t=0.out10
│   │   ├── toxic.jsonl.gpt4.t=0.out11
│   │   ├── toxic.jsonl.gpt4.t=0.out12
│   │   ├── toxic.jsonl.gpt4.t=0.out13
│   │   ├── toxic.jsonl.gpt4.t=0.out14
│   │   ├── toxic.jsonl.gpt4.t=0.out15
│   │   ├── toxic.jsonl.gpt4.t=0.out16
│   │   ├── toxic.jsonl.gpt4.t=0.out17
│   │   ├── toxic.jsonl.gpt4.t=0.out18
│   │   ├── toxic.jsonl.gpt4.t=0.out19
│   │   ├── toxic.jsonl.gpt4.t=0.out2
│   │   ├── toxic.jsonl.gpt4.t=0.out20
│   │   ├── toxic.jsonl.gpt4.t=0.out21
│   │   ├── toxic.jsonl.gpt4.t=0.out22
│   │   ├── toxic.jsonl.gpt4.t=0.out23
│   │   ├── toxic.jsonl.gpt4.t=0.out24
│   │   ├── toxic.jsonl.gpt4.t=0.out25
│   │   ├── toxic.jsonl.gpt4.t=0.out26
│   │   ├── toxic.jsonl.gpt4.t=0.out27
│   │   ├── toxic.jsonl.gpt4.t=0.out28
│   │   ├── toxic.jsonl.gpt4.t=0.out29
│   │   ├── toxic.jsonl.gpt4.t=0.out3
│   │   ├── toxic.jsonl.gpt4.t=0.out30
│   │   ├── toxic.jsonl.gpt4.t=0.out31
│   │   ├── toxic.jsonl.gpt4.t=0.out32
│   │   ├── toxic.jsonl.gpt4.t=0.out4
│   │   ├── toxic.jsonl.gpt4.t=0.out5
│   │   ├── toxic.jsonl.gpt4.t=0.out6
│   │   ├── toxic.jsonl.gpt4.t=0.out7
│   │   ├── toxic.jsonl.gpt4.t=0.out8
│   │   └── toxic.jsonl.gpt4.t=0.out9
│   └── user_prompts
│       ├── toxic.jsonl.gpt3.n=25.out2.split.annotated.jsonl.filtered.jsonl.ada.n=25.out
│       ├── toxic.jsonl.gpt3.n=25.out2.split.annotated.jsonl.filtered.jsonl.curie.n=25.out
│       ├── toxic.jsonl.gpt3.n=25.out2.split.annotated.jsonl.filtered.jsonl.davinci.n=25.out
│       ├── toxic.jsonl.gpt3.n=25.out2.split.annotated.jsonl.filtered.jsonl.gpt3.n=25.out0
│       ├── toxic.jsonl.gpt3.n=25.out2.split.annotated.jsonl.filtered.jsonl.gpt3.n=25.out1
│       ├── toxic.jsonl.gpt3.n=25.out2.split.annotated.jsonl.filtered.jsonl.gpt4.n=25.out0
│       ├── toxic.jsonl.gpt3.n=25.out2.split.annotated.jsonl.filtered.jsonl.gpt4.n=25.out1
│       ├── toxic.jsonl.gpt4.n=25.out2.split.annotated.jsonl.filtered.jsonl.ada.n=25.out
│       ├── toxic.jsonl.gpt4.n=25.out2.split.annotated.jsonl.filtered.jsonl.curie.n=25.out
│       ├── toxic.jsonl.gpt4.n=25.out2.split.annotated.jsonl.filtered.jsonl.davinci.n=25.out
│       ├── toxic.jsonl.gpt4.n=25.out2.split.annotated.jsonl.filtered.jsonl.gpt3.n=25.out0
│       ├── toxic.jsonl.gpt4.n=25.out2.split.annotated.jsonl.filtered.jsonl.gpt3.n=25.out1
│       ├── toxic.jsonl.gpt4.n=25.out2.split.annotated.jsonl.filtered.jsonl.gpt4.n=25.out0
│       └── toxic.jsonl.gpt4.n=25.out2.split.annotated.jsonl.filtered.jsonl.gpt4.n=25.out1
├── toxicity_prompts.py
└── user_prompts
    ├── nontoxic.jsonl
    ├── toxic.jsonl
    ├── toxic.jsonl.gpt3.n=25.out2.split.annotated.jsonl.all.jsonl
    ├── toxic.jsonl.gpt3.n=25.out2.split.annotated.jsonl.filtered.jsonl
    ├── toxic.jsonl.gpt4.n=25.out2.split.annotated.jsonl.all.jsonl
    └── toxic.jsonl.gpt4.n=25.out2.split.annotated.jsonl.filtered.jsonl
```

## Details of each subdirectories

- The `generation` folder consists of all the generations of LLMs for our toxicity evaluation experiments. 
- The `user_prompts` folder consists of our sampled prompts from the standard RealToxicityPrompts benchmark.
- The `toxicity_prompts.py` file provides 33 diverse system prompts. 

