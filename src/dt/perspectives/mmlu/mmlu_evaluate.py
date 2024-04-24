import re
from typing import List, Dict, Union

import numpy as np
from datasets import load_dataset, Dataset
from tqdm import tqdm

from dt.perspectives.perspective import DatasetManager, ModelEvaluationMetricsInterface, \
    QueryResult, MetricResult, ModelQuery


class MMLUDatasetManager(DatasetManager):
    """
    This class provides an example for implementing a DecodingTrust perspective. Note that this is just an example.
    Please use the standard way to evaluate MMLU.
    """
    hf_dataset_name: str = "cais/mmlu"
    hf_split = "test"

    query_template = """Answer the following multiple choice question. The last line of your response should be of 
    the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before 
    answering.

    {question}

    A) {a}
    B) {b}
    C) {c}
    D) {d}
    """.strip()

    def load_dataset(self, task_name: str = "all") -> Union[List[Dict], Dataset]:
        self.datasets = load_dataset(self.hf_dataset_name, split=self.hf_split)[task_name]

        return self.datasets

    def prepare_task_message(self, task_name: str) -> List[ModelQuery]:
        dataset = self.load_dataset(task_name)

        model_queries: List[ModelQuery] = []
        pbar = tqdm(enumerate(dataset), total=len(dataset))
        for i, data in pbar:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": self.query_template.format(
                    question=data["question"], a=data["choices"][0], b=data["choices"][1],
                    c=data["choices"][2], d=data["choices"][3]
                )},
            ]

            query = ModelQuery(task_name=task_name, label=data["answer"], messages=messages, temperature=0)

            model_queries.append(query)

        return model_queries


class MMLUMetrics(ModelEvaluationMetricsInterface):
    answer_pattern = r"(?i)ANSWER\s*:\s*([A-D])"

    def prediction_processor(self, response: Dict, task_name: str):
        response_text = response["choices"][0]["message"]["content"].lower()
        match = re.search(self.answer_pattern, response_text)
        extracted_answer = match.group(1) if match else None

        return extracted_answer

    def calculate_metrics(self, results: List[QueryResult]) -> MetricResult:
        query_scores = []
        for query_result in results:
            extracted_answer = self.prediction_processor(query_result.response, self.task_name)
            score = 1.0 if extracted_answer == query_result.query.label else 0.0
            query_scores.append(score)
        return MetricResult(name="accuracy", scenario=self.task_name, value=np.mean(query_scores))
