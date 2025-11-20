import abc
import signal
import traceback
from queue import Empty
from typing import Any
from multiprocessing import Queue

from gpu_dispatch.protocol import TaskSuccess, TaskError, TaskTimeout, TaskStarted, SetupFailed, CleanupFailed


class BaseWorker(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def setup(self, gpu_id: int, seed: int, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def process(self, data: Any) -> Any:
        pass

    def cleanup(self) -> None:
        pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Task execution exceeded timeout")


def _worker_main(
    worker_instance: BaseWorker,
    gpu_id: int,
    seed: int,
    task_queue: Queue,
    result_queue: Queue,
    task_timeout: float | None,
    setup_kwargs: dict,
    shutdown_event=None,
) -> None:
    # Setup
    try:
        worker_instance.setup(gpu_id=gpu_id, seed=seed, **setup_kwargs)
    except Exception:
        error_msg = traceback.format_exc()
        result_queue.put(SetupFailed(gpu_id=gpu_id, error=error_msg))
        return

    if task_timeout is not None:
        signal.signal(signal.SIGALRM, _timeout_handler)

    # Main loop
    while True:
        # Check for shutdown signal
        if shutdown_event is not None and shutdown_event.is_set():
            break

        # Use timeout to allow checking shutdown_event periodically
        try:
            item = task_queue.get(timeout=0.5)
        except Empty:
            continue

        if item is None:
            break

        task_id, data = item

        try:
            if task_timeout is not None:
                alarm_seconds = max(1, int(task_timeout + 0.5))
                signal.alarm(alarm_seconds)

            result_queue.put(TaskStarted(task_id=task_id, worker_id=gpu_id))
            result = worker_instance.process(data)

            if task_timeout is not None:
                signal.alarm(0)

            result_queue.put(TaskSuccess(task_id=task_id, data=result, worker_id=gpu_id))

        except TimeoutError:
            if task_timeout is not None:
                signal.alarm(0)
            result_queue.put(TaskTimeout(task_id=task_id, timeout=task_timeout, worker_id=gpu_id))

        except Exception:
            if task_timeout is not None:
                signal.alarm(0)
            error_msg = traceback.format_exc()
            result_queue.put(TaskError(task_id=task_id, error=error_msg, worker_id=gpu_id))

    # Cleanup
    try:
        worker_instance.cleanup()
    except Exception:
        error_msg = traceback.format_exc()
        result_queue.put(CleanupFailed(gpu_id=gpu_id, error=error_msg))
