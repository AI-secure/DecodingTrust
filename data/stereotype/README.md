# Stereotype Study Data

⚠️ Warning: the stereotype dataset contains models outputs that are offensive in nature.

The following subdirectory contains data files related to the stereotype study.

- The `generations` folder consists of all the generations of LLMs for our stereotype evaluation study. It consists of 3 subdirectories, each one corresponding to 3 chosen system prompt types: `benign`, `untargeted`, `targeted`. Each subdirectory contains 16 CSV files for the 16 stereotype topics selected in the study.
- The `dataset` folder consists of the user prompts and the system prompts used to generate the LLM outputs.

## Data Format
- `dataset/user_prompts.csv`: This file contains the 1,152 user prompts utilized in the study. These prompts were generated using 16 stereotype topics x 3 variants x 24 demographic groups. The CSV file contains the following columns:
  1. `stereotype_topic`: Topic of the stereotype presented in the user prompt
  2. `target_group`: The target demographic group which the user prompt focuses on
  3. `stereotype_template`: The underlying template which is used to get the final user prompt. Each template has a `[target_group]` which is replaced by the `target_group` in the 2nd column to achieve the final user prompt
  4. `user_prompt`: The final user prompt. Each user prompt also has a task description that instructs the model about the task - to append agree/disagree based on whether the model agrees with the stereotype statement.

- `dataset/system_prompts.csv`: This file contains the 3 system prompts used in the stereotype study. The CSV file contains the following columns:
  1. `type`: One of the three system prompt types considered in the study: `benign`, `untargeted`, and `targeted`.
  2. `system_prompt`: The system prompt used to instruct the models. Note that in case of the `targeted` system prompt, `[target_group]` is replaced by the one used in the user prompt.

- LLM output files: Each CSV file in the generations folder has the following columns:
  1. `model`: One of the two GPT-models considered for the study: `gpt-3.5-turbo-0301` or `gpt-4-0314`
  2. `system_prompt`: System prompt used in generating the LLM output
  3. `user_prompt`: User prompt supplied to generate the LLM output
  4. `gen_0, gen_1, ..., gen_24`: 25 generations received by querying the LLM

  Each generation has the following format:
  ```
  {
    'agreeability_num': "Represents model agreeability. 1 represents agree, -1 represents disagree, 0 represents neither agree nor disagree."
    'output': "Output of the model based on the query."
  }
  ```


## File Structure

```
.
├── generations
│   ├── benign
│   │   ├── benign_hiv.csv
│   │   ├── benign_terrorists.csv
│   │   ├── benign_drug_addicts.csv
│   │   ├── benign_intelligence.csv
│   │   ├── benign_greed.csv
│   │   ├── benign_parents.csv
│   │   ├── benign_country.csv
│   │   ├── benign_technology.csv
│   │   ├── benign_weak.csv
│   │   ├── benign_driving.csv
│   │   ├── benign_criminals.csv
│   │   ├── benign_drug_dealers.csv
│   │   ├── benign_jobs.csv
│   │   ├── benign_leadership.csv
│   │   ├── benign_stem.csv
│   │   └── benign_hygiene.csv 
│   ├── untargeted
│   │   ├── untargeted_hiv.csv
│   │   ├── untargeted_terrorists.csv
│   │   ├── untargeted_drug_addicts.csv
│   │   ├── untargeted_intelligence.csv
│   │   ├── untargeted_greed.csv
│   │   ├── untargeted_parents.csv
│   │   ├── untargeted_country.csv
│   │   ├── untargeted_technology.csv
│   │   ├── untargeted_weak.csv
│   │   ├── untargeted_driving.csv
│   │   ├── untargeted_criminals.csv
│   │   ├── untargeted_drug_dealers.csv
│   │   ├── untargeted_jobs.csv
│   │   ├── untargeted_leadership.csv
│   │   ├── untargeted_stem.csv
│   │   └── untargeted_hygiene.csv 
│   └── targeted
│        ├── targeted_hiv.csv
│        ├── targeted_terrorists.csv
│        ├── targeted_drug_addicts.csv
│        ├── targeted_intelligence.csv
│        ├── targeted_greed.csv
│        ├── targeted_parents.csv
│        ├── targeted_country.csv
│        ├── targeted_technology.csv
│        ├── targeted_weak.csv
│        ├── targeted_driving.csv
│        ├── targeted_criminals.csv
│        ├── targeted_drug_dealers.csv
│        ├── targeted_jobs.csv
│        ├── targeted_leadership.csv
│        ├── targeted_stem.csv
│        └── targeted_hygiene.csv 
└── dataset
    ├── user_prompts.csv
    └── system_prompts.csv
```
