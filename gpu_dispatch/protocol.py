from dataclasses import dataclass
from typing import Any


@dataclass
class TaskSuccess:
    """Successful task completion.

    Attributes:
        task_id: Unique identifier for the task.
        data: The return value from worker.process().
        worker_id: The GPU/worker identifier that processed the task.
    """
    task_id: int
    data: Any
    worker_id: int


@dataclass
class TaskError:
    """Task execution failure.

    Attributes:
        task_id: Unique identifier for the task.
        error: Traceback string describing the runtime error.
        worker_id: The GPU/worker identifier that processed the task.
    """
    task_id: int
    error: str
    worker_id: int


@dataclass
class TaskTimeout:
    """Task execution timeout.

    Attributes:
        task_id: Unique identifier for the task.
        timeout: The timeout duration (in seconds) that was exceeded.
        worker_id: The GPU/worker identifier that was executing the task.
    """
    task_id: int
    timeout: float
    worker_id: int


@dataclass
class TaskStarted:
    """Notification that a worker has started processing a task."""

    task_id: int
    worker_id: int


@dataclass
class SetupFailed:
    """Worker initialization failure.

    Attributes:
        gpu_id: The GPU ID assigned to the failed worker.
        error: Traceback string describing the initialization error.
    """
    gpu_id: int
    error: str


@dataclass
class CleanupFailed:
    """Worker cleanup failure.

    Attributes:
        gpu_id: The GPU ID assigned to the worker.
        error: Traceback string describing the cleanup error.
    """
    gpu_id: int
    error: str
