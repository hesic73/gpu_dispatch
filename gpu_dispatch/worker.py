import abc
import signal
import traceback
from typing import Any
from multiprocessing import Queue

from gpu_dispatch.protocol import TaskSuccess, TaskError, TaskTimeout, SetupFailed, CleanupFailed


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
        item = task_queue.get()

        if item is None:
            break

        task_id, data = item

        try:
            if task_timeout is not None:
                alarm_seconds = max(1, int(task_timeout + 0.5))
                signal.alarm(alarm_seconds)

            result = worker_instance.process(data)

            if task_timeout is not None:
                signal.alarm(0)

            result_queue.put(TaskSuccess(task_id=task_id, data=result))

        except TimeoutError:
            if task_timeout is not None:
                signal.alarm(0)
            result_queue.put(TaskTimeout(task_id=task_id, timeout=task_timeout))

        except Exception:
            if task_timeout is not None:
                signal.alarm(0)
            error_msg = traceback.format_exc()
            result_queue.put(TaskError(task_id=task_id, error=error_msg))

    # Cleanup
    try:
        worker_instance.cleanup()
    except Exception:
        error_msg = traceback.format_exc()
        result_queue.put(CleanupFailed(gpu_id=gpu_id, error=error_msg))
