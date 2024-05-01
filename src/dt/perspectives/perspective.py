import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import asdict
from tempfile import TemporaryDirectory

from datasets import Dataset
from pydantic.dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional, Union, Set

from tqdm import tqdm

from dt.chat import Chat
from dt.configs.configs import BaseConfig


def write_jsonl(dest: str, json_list: List[Dict]) -> None:
    with open(dest, "w") as f:
        f.writelines([json.dumps(line) + "\n" for line in json_list])


@dataclass
class ModelQuery:
    task_name: str

    label: Union[str, int]
    messages: List[Dict]

    temperature: Optional[float] = 1
    top_p: Optional[float] = None
    top_k: Optional[int] = None   # Not available to OpenAI models

    max_tokens: Optional[int]
    n: Optional[int]

    presence_penalty: Optional[float]

    stop: Optional[List[str]]


@dataclass
class QueryUsage:
    prompt_token_count: int
    completion_token_count: int
    cost_in_dollars: Optional[float]


@dataclass
class QueryResult:
    task_name: str
    model_name: str

    query: ModelQuery
    response: Dict

    query_usage: QueryUsage


@dataclass
class MetricResult:
    name: str
    scenario: str
    value: float


@dataclass
class AggregatedMetricResult(ABC):
    name: str
    scenario_results: List[MetricResult]

    @property
    @abstractmethod
    def value(self):
        pass


class DatasetManager(ABC):
    hf_dataset_name: str = "AI-Secure/DecodingTrust"
    hf_subset_name: Optional[str] = None

    def __init__(self, options: BaseConfig):
        self.options: BaseConfig = options
        self.datasets: Optional[Union[List[Dict], Dataset]] = None  # Top-level HF datasets with different tasks

    @abstractmethod
    def load_dataset(self, task_name: str) -> Union[List[Dict], Dataset]:
        """Load and return the dataset for a given task."""
        pass

    @abstractmethod
    def prepare_task_message(self, task_name: str) -> List[ModelQuery]:
        """Prepare and return the task-specific message for model interaction."""
        pass


class BenchmarkExecutor:
    cache_dir = ".cache"
    response_cache_file = "response.jsonl"

    def __init__(self, options: BaseConfig, perspective: str):
        self.options = options
        self.perspective = perspective
        self.dataset_manager = DatasetManager(options)
        self.results: Dict[str, List[QueryResult]] = dict()

        self._init_files()
        with TemporaryDirectory(dir=self.cache_dir) as dirname:
            self.chat_model = Chat.from_helm(options, cache=dirname)

    def _init_files(self):
        os.makedirs(self.cache_dir, exist_ok=True)

    @abstractmethod
    def prepare_task_query(self, messages: List[Dict]) -> Dict:
        pass

    def execute_classification(self, tasks: List[str]):
        for task_name in tasks:
            dataset = self.dataset_manager.load_dataset(task_name)
            task_message = self.dataset_manager.prepare_task_message(task_name)
            accuracy, unknown, cost, cache = self.chat_model.do_classification(dataset, task_message)
            # Process and display the results as needed

    def execute_queries(self, queries: List[ModelQuery]) -> List[QueryResult]:
        distinct_task_names: Set[str] = {q.task_name for q in queries}
        if len(distinct_task_names) > 1:
            raise ValueError(f"Multiple tasks ({distinct_task_names}) in the same list of input is not supported!")

        query_results: List[QueryResult] = []
        for i, query in tqdm(enumerate(queries)):
            response = self.chat_model.call(
                messages=query.messages, t=query.temperature, max_tokens=query.max_tokens,
                dry_run=self.options.dry_run
            )
            query_result = QueryResult(
                task_name=query.task_name, model_name=self.options.model_config.model, query=query, response=response,
                query_usage=QueryUsage(
                    prompt_token_count=response["usage"]["prompt_tokens"],
                    completion_token_count=response["usage"]["completions_tokens"]
                )
            )

            query_results.append(query_result)

            if (i + 1) % self.options.save_interval == 0 or (i + 1) == len(queries):
                dest_path = os.path.join(
                    self.options.result_dir, self.perspective, query.task_name, self.response_cache_file
                )
                logging.info(f"Saving to {dest_path}")
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                write_jsonl(dest_path, [asdict(x) for x in query_results])

        return query_results

    def execute_task(self, task_name: str):
        self.dataset_manager.load_dataset(task_name)
        task_queries = self.dataset_manager.prepare_task_message(task_name)
        query_results = self.execute_queries(task_queries)


class ModelEvaluationMetricsInterface(ABC):

    def __init__(self, task_name: str):
        self.task_name = task_name

    @abstractmethod
    def prediction_processor(self, response: Dict, task_name: str) -> int:
        pass

    @abstractmethod
    def calculate_metrics(self, results: List[QueryResult]) -> MetricResult:
        """
        Calculate evaluation metrics based on predictions and ground truth labels.

        Args:
            results: A list of classification results.

        Returns:
            A dictionary where keys are metric names and values are metric scores.
        """
        pass
