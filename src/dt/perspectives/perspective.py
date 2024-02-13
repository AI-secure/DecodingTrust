from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any
from dt.chat import Chat


class DatasetManager(ABC):
    @abstractmethod
    def load_dataset(self, task_name: str) -> List[Dict]:
        """Load and return the dataset for a given task."""
        pass

    @abstractmethod
    def prepare_task_message(self, task_name: str) -> str:
        """Prepare and return the task-specific message for model interaction."""
        pass


class BenchmarkExecutor:
    def __init__(self, dataset_manager: DatasetManager, model_interaction: Chat):
        self.dataset_manager = dataset_manager
        self.model_interaction = model_interaction

    def execute_benchmark(self, tasks: List[str]):
        for task_name in tasks:
            dataset = self.dataset_manager.load_dataset(task_name)
            task_message = self.dataset_manager.prepare_task_message(task_name)
            accuracy, unknown, cost, cache = self.model_interaction.do_classification(dataset, task_message)
            # Process and display the results as needed


class ModelEvaluationMetricsInterface(ABC):
    @abstractmethod
    def calculate_metrics(self, predictions: List, labels: List) -> Dict[str, float]:
        """
        Calculate evaluation metrics based on predictions and ground truth labels.

        Args:
            predictions: A list of model predictions.
            labels: A list of ground truth labels.

        Returns:
            A dictionary where keys are metric names and values are metric scores.
        """
        pass
