import os
from abc import ABC, abstractmethod
from tempfile import TemporaryDirectory

from datasets import Dataset
from pydantic.dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional, Union
from dt.chat import Chat
from dt.configs.configs import BaseConfig


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
    cost_in_dollars: float


@dataclass
class QueryResult:
    task_name: str

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
    def prepare_task_message(self, task_name: str) -> List[List[Dict]]:
        """Prepare and return the task-specific message for model interaction."""
        pass


class BenchmarkExecutor:
    cache_dir = ".cache"

    def __init__(self, options: BaseConfig):
        self.options = options
        self.dataset_manager = DatasetManager(options)
        self.results: Dict[str, List[QueryResult]] = dict()

        self._init_files()
        with TemporaryDirectory(dir=self.cache_dir) as dirname:
            self.model_interaction = Chat.from_helm(options, cache=dirname)

    def _init_files(self):
        os.makedirs(self.cache_dir, exist_ok=True)

    @abstractmethod
    def prepare_task_query(self, messages: List[Dict]) -> Dict:
        pass

    def execute_classification(self, tasks: List[str]):
        for task_name in tasks:
            dataset = self.dataset_manager.load_dataset(task_name)
            task_message = self.dataset_manager.prepare_task_message(task_name)
            accuracy, unknown, cost, cache = self.model_interaction.do_classification(dataset, task_message)
            # Process and display the results as needed

    def execute_calls(self, tasks: List[str]):
        for task_name in tasks:
            task_messages = self.dataset_manager.prepare_task_message(task_name)

            for task_message in task_messages:
                response = self.model_interaction.call(
                    messages=task_message, t=0, max_tokens=16, dry_run=self.options.dry_run
                )

                self.results[task_name].append(
                    QueryResult(
                        prediction=self.prediction_processor(response, task_name),
                        task_name=task_name, response=response,
                        request=task_message,
                        query_usage=QueryUsage(
                            prompt_token_count=response["usage"]["prompt_tokens"],
                            completion_tokens=response["usage"]["completion_tokens"],
                            price=self.model_interaction.calc_price(response),
                        )
                    )
                )


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
