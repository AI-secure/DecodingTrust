# Fairness evaluation data generation

The ``datasets`` directory contains Python class of used datasets.

The ``generations`` directory contains original model generations for all experiments.

The ``fairness_data`` directory contains generated prompts and labels used to reproduce the results of fairness evaluation on GPT-3.5 and GPT-4.

The ``data_generation.py`` file is used to generate the prompts and labels.

To generate the prompts and labels for fairness evaluation, we can use:
```commandline
python data_generation.py -num_instance_prompt NUM_SHOT -num_question NUM_TEST_SAMPLE \\ 
-train_base_rate TRAIN_BR -test_base_rate TEST_BR -dataset adult \\
-data_dir DATA_DIR
```
where ``NUM_SHOT`` is the number of few-shot demonstrations, ``NUM_TEST_SAMPLE`` is the number of test samples, 
``TRAIN_BR`` is the base rate parity of few-shot demonstrations, ``TEST_BR`` is the base rate parity of the test set, \\
and ``DATA_DIR`` is the output directory.