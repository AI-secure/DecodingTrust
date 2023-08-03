# Privacy Evaluation Data Structures

- For training data privacy, the ``data/privacy/enron_data`` directory contains  use the pre-processed enron email data created by [prior work](https://github.com/jeffhj/LM_PersonalInfoLeak/tree/main/data). 
  - ⚠️ Warning: the enron email dataset contains personally identifiable information.
- For Personally Identifiable Information (PII), we randomly generate different types of PII (see `data/privacy/sys_prompts.py` for a full list of PII). 
- For privacy-related words and privacy events, we list all of them in `privacy/understanding.py`.
- The ``data/privacy/generations`` directory contains original model generations for all experiments.