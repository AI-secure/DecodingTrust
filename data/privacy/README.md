# Privacy Evaluation Data Structures

- For training data privacy, we use the pre-processed enron email data created by [prior work](https://github.com/jeffhj/LM_PersonalInfoLeak/tree/main/data).
- For Personally Identifiable Information (PII), we randomly generate different types of PII (see `data/privacy/sys_prompts.py` for a full list of PII). 
- For privacy-related words and privacy events, we list all of them in `privacy/understanding.py`.