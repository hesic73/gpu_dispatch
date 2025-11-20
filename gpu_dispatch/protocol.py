from dataclasses import dataclass
from typing import Any


@dataclass
class TaskSuccess:
    """Successful task completion.

    Attributes:
        task_id: Unique identifier for the task.
        data: The return value from worker.process().
    """
    task_id: int
    data: Any


@dataclass
class TaskError:
    """Task execution failure.

    Attributes:
        task_id: Unique identifier for the task.
        error: Traceback string describing the runtime error.
    """
    task_id: int
    error: str


@dataclass
class TaskTimeout:
    """Task execution timeout.

    Attributes:
        task_id: Unique identifier for the task.
        timeout: The timeout duration (in seconds) that was exceeded.
    """
    task_id: int
    timeout: float


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
