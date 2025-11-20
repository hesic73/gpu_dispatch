import multiprocessing as mp
import threading
from typing import Any, Callable, Iterable

from gpu_dispatch.worker import BaseWorker, _worker_main
from gpu_dispatch.protocol import TaskSuccess, TaskError, TaskTimeout, TaskStarted, SetupFailed, CleanupFailed

SuccessCallback = Callable[[int, Any, int], None]
ErrorCallback = Callable[[int, str, int], None]
TimeoutCallback = Callable[[int, float, int], None]
SetupFailCallback = Callable[[int, str], None]
StartCallback = Callable[[int, int], None]
ExitCallback = Callable[[], None]


class Dispatcher:
    def __init__(
        self,
        worker_cls: type[BaseWorker],
        gpu_ids: list[int],
        queue_size: int = 1024,
    ):
        if not issubclass(worker_cls, BaseWorker):
            raise TypeError(f"worker_cls must inherit from BaseWorker, got {worker_cls}")

        if not gpu_ids:
            raise ValueError("gpu_ids cannot be empty")

        self.worker_cls = worker_cls
        self.gpu_ids = gpu_ids
        self.queue_size = queue_size

        self.ctx = mp.get_context('spawn')

    def run(
        self,
        generator: Iterable[Any],
        on_success: SuccessCallback,
        on_error: ErrorCallback | None = None,
        on_timeout: TimeoutCallback | None = None,
        on_setup_fail: SetupFailCallback | None = None,
        on_task_start: StartCallback | None = None,
        on_exit: ExitCallback | None = None,
        base_seed: int = 42,
        task_timeout: float | None = None,
        **setup_kwargs,
    ) -> None:
        task_queue = self.ctx.Queue(maxsize=self.queue_size)
        result_queue = self.ctx.Queue()

        processes = []
        for gpu_id in self.gpu_ids:
            worker_instance = self.worker_cls()
            seed = base_seed + gpu_id

            p = self.ctx.Process(
                target=_worker_main,
                args=(
                    worker_instance,
                    gpu_id,
                    seed,
                    task_queue,
                    result_queue,
                    task_timeout,
                    setup_kwargs,
                ),
            )
            p.start()
            processes.append(p)

        feeder_stop = threading.Event()
        task_count = [0]
        feeder_thread = threading.Thread(
            target=self._feeder,
            args=(generator, task_queue, feeder_stop, task_count),
            daemon=True,
        )
        feeder_thread.start()

        try:
            self._monitor(
                result_queue=result_queue,
                feeder_stop=feeder_stop,
                feeder_thread=feeder_thread,
                task_count=task_count,
                on_success=on_success,
                on_error=on_error,
                on_timeout=on_timeout,
                on_setup_fail=on_setup_fail,
                on_task_start=on_task_start,
            )
        finally:
            if on_exit:
                on_exit()

            for _ in self.gpu_ids:
                task_queue.put(None)

            for p in processes:
                p.join()

    def _feeder(
        self,
        generator: Iterable[Any],
        task_queue: mp.Queue,
        stop_event: threading.Event,
        task_count: list,
    ) -> None:
        task_id = 0
        try:
            for data in generator:
                task_queue.put((task_id, data), block=True)
                task_id += 1
        except Exception as e:
            # If generator fails, log and stop
            print(f"Feeder thread error: {e}")
        finally:
            task_count[0] = task_id
            stop_event.set()

    def _monitor(
        self,
        result_queue: mp.Queue,
        feeder_stop: threading.Event,
        feeder_thread: threading.Thread,
        task_count: list,
        on_success: SuccessCallback,
        on_error: ErrorCallback | None,
        on_timeout: TimeoutCallback | None,
        on_setup_fail: SetupFailCallback | None,
        on_task_start: StartCallback | None,
    ) -> None:
        active_workers = len(self.gpu_ids)
        results_received = 0

        while True:
            if feeder_stop.is_set() and results_received >= task_count[0]:
                break

            try:
                result = result_queue.get(timeout=0.1)
            except:
                continue
            if isinstance(result, TaskStarted):
                if on_task_start:
                    on_task_start(result.task_id, result.worker_id)
                continue

            if isinstance(result, TaskSuccess):
                on_success(result.task_id, result.data, result.worker_id)
                results_received += 1

            elif isinstance(result, TaskError):
                if on_error:
                    on_error(result.task_id, result.error, result.worker_id)
                results_received += 1

            elif isinstance(result, TaskTimeout):
                if on_timeout:
                    on_timeout(result.task_id, result.timeout, result.worker_id)
                results_received += 1

            elif isinstance(result, SetupFailed):
                if on_setup_fail:
                    on_setup_fail(result.gpu_id, result.error)
                active_workers -= 1
                if active_workers == 0:
                    raise RuntimeError("All workers failed during setup")

            elif isinstance(result, CleanupFailed):
                print(f"Warning: Cleanup failed for GPU {result.gpu_id}: {result.error}")
