# OOD Robustness Data Structures

# Dataset Description
- For OOD Style, we adopt SST-2 as the base in-distribution dataset and apply different style transformations. For word-level subsititution, we have "Augment" and "Shake-W". For sentence-level style transformations, we have "Tweet", "Shake", "Bible, and "Romantic Poetry". For its demonstrations in in-context learning, we sampled three sets of demonstrations from origin training examples, and perform style-transformation accrodingly.
- For OOD knowledge, we organize the 2020 questions and 2023 questions until 3/10/2023 from RealtimeQA. For its demonstrations in in-context learning, we sampled three sets of demonstrations from in-distribution 2021 question and 5 different domains, including US
foreign policy (Policy), global facts (Facts), moral scenarios (Moral), and machine learning (ML), from MMLU.
## Data Format
- Folder `styles` contains the style-transformed SS2-T dev set and three sets of demonstrations with different styles. The formats of base `styles/dev.tsv` and other style-transformed data contains the following columns:
   1. `sentence`: The sentence from the base dev set with corresponding style-transformations.
   2. `label`: Label from the base dev set.
- Folder `knowledge` contains the 2020 questions, 2023 questions, and three sets of demonstrations from different domains of MMLU and in-distribution 2021 questions from RealtimeQA. The formats of these data contains the following key-value pairs:
   1. `question_id`: Question id for questions from RealtimeQA .
   2. `question_date`: Question date for questions from RealtimeQA.
   3. `question_source`: Question source for questions from RealtimeQA.
   4. `question_ur`: Question URL for questions from RealtimeQA.
   5. `question_sentence`: Question from both RealtimeQA or MMLU.
   6. `choices`: Answer choices from both RealtimeQA or MMLU.
   7. `answer`: Answer from both RealtimeQA or MMLU.
   8. `evidence`: Evidence of answer from RealtimeQA.

## File Structures

```
.
└── styles
    ├── dev.tsv
    ├── dev_augment.tsv
    ├── dev_bible_p0.6.tsv
    ├── dev_bible_p0.tsv
    ├── dev_romantic_p0.6.tsv
    ├── dev_romantic_p0.tsv
    ├── dev_shake_p0.6.tsv
    ├── dev_shake_p0.tsv
    ├── dev_shake_w.tsv
    ├── dev_tweet_p0.6.tsv
    ├── dev_tweet_p0.tsv
    ├── train_demo_0.tsv
    ├── train_demo_1.tsv
    ├── train_demo_2.tsv
    ├── train_demo_bible_p0.6_0.tsv
    ├── train_demo_bible_p0.6_1.tsv
    ├── train_demo_bible_p0.6_2.tsv
    ├── train_demo_romantic_p0.6_0.tsv
    ├── train_demo_romantic_p0.6_1.tsv
    ├── train_demo_romantic_p0.6_2.tsv
    ├── train_demo_shake_p0.6_0.tsv
    ├── train_demo_shake_p0.6_1.tsv
    ├── train_demo_shake_p0.6_2.tsv
    ├── train_demo_tweet_p0.6_0.tsv
    ├── train_demo_tweet_p0.6_1.tsv
    └── train_demo_tweet_p0.6_2.tsv
└── knowledge
    ├── demo_2021_qa0.jsonl
    ├── demo_2021_qa1.jsonl
    ├── demo_2021_qa2.jsonl
    ├── demo_global_facts0.jsonl
    ├── demo_global_facts1.jsonl
    ├── demo_global_facts2.jsonl
    ├── demo_machine_learning0.jsonl
    ├── demo_machine_learning1.jsonl
    ├── demo_machine_learning2.jsonl
    ├── demo_moral_scenarios0.jsonl
    ├── demo_moral_scenarios1.jsonl
    ├── demo_moral_scenarios2.jsonl
    ├── demo_us_foreign_policy0.jsonl
    ├── demo_us_foreign_policy1.jsonl
    ├── demo_us_foreign_policy2.jsonl
    ├── qa_2020.jsonl
    └── qa_2023.jsonl
```

## Details of each subdirectories

- The `styles` folder consists of all OOD styles and demonstrations from these styles. 
- The `knolwedge` folder consists of RealtimeQA questions in 2020 and 2023 and sampled demonstrations from MMLU and QA questions from 2021.

