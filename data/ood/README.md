# OOD Robustness Data Structures

# Dataset Description
- For OOD Style, we adopt SST-2 as the base in-distribution dataset and apply different style transformations. For word-level substitution, we have "Augment" and "Shake-W." For sentence-level style transformations, we have "Tweet," "Shake," "Bible, and "Romantic Poetry." For its demonstrations in in-context learning, we sampled three sets of demonstrations from origin training examples and performed style transformation accordingly.
- For OOD knowledge, we organize the 2020 questions and 2023 questions until 3/10/2023 from RealtimeQA. For its demonstrations in in-context learning, we sampled three sets of demonstrations from the in-distribution 2021 question and 5 different domains, including US
foreign policy (Policy), global facts (Facts), moral scenarios (Moral), and machine learning (ML), from MMLU.
## Data Format
- Folder `styles` contains the style-transformed SS2-T dev set and three sets of demonstrations with different styles. The format of base `styles/dev.tsv` and other style-transformed data files contains the following columns:
   1. `sentence`: The sentence from the base dev set with corresponding style transformations.
   2. `label`: Label from the base dev set.
- Folder `knowledge` contains the 2020 questions, 2023 questions, and three sets of demonstrations from different domains of MMLU and in-distribution 2021 questions from RealtimeQA. Each data file contains the following key-value pairs:
   1. `question_id`: Question id for questions from RealtimeQA.
   2. `question_date`: Question date for questions from RealtimeQA.
   3. `question_source`: Question source for questions from RealtimeQA.
   4. `question_ur`: Question URL for questions from RealtimeQA.
   5. `question_sentence`: Question from both RealtimeQA or MMLU.
   6. `choices`: Answer choices from both RealtimeQA or MMLU.
   7. `answer`: Answer from both RealtimeQA or MMLU.
   8. `evidence`: Evidence of answer from RealtimeQA.

