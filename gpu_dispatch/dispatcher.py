import multiprocessing as mp
import os
import signal
import sys
import threading
from queue import Empty, Full
from typing import Any, Callable, Iterable

from gpu_dispatch.worker import BaseWorker, _worker_main
from gpu_dispatch.protocol import TaskSuccess, TaskError, TaskTimeout, TaskStarted, SetupFailed, CleanupFailed

SuccessCallback = Callable[[int, Any, int], None]
ErrorCallback = Callable[[int, str, int], None]
TimeoutCallback = Callable[[int, float, int], None]
SetupFailCallback = Callable[[int, str], None]
StartCallback = Callable[[int, int], None]
ExitCallback = Callable[[], None]


def _worker_main_silenced(*args, **kwargs):
    """Wrapper that suppresses worker stdout/stderr to keep UI clean."""
    # Redirect stdout and stderr at the file descriptor level
    # This catches everything including loguru and direct writes
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)

    try:
        # Redirect file descriptors
        os.dup2(devnull_fd, 1)  # stdout
        os.dup2(devnull_fd, 2)  # stderr

        # Also redirect Python-level streams
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

        _worker_main(*args, **kwargs)
    finally:
        # Restore original file descriptors
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)
        os.close(devnull_fd)


class Dispatcher:
    def __init__(
        self,
        worker_cls: type[BaseWorker],
        gpu_ids: list[int],
        queue_size: int = 1024,
        suppress_worker_output: bool = False,
    ):
        if not issubclass(worker_cls, BaseWorker):
            raise TypeError(f"worker_cls must inherit from BaseWorker, got {worker_cls}")

        if not gpu_ids:
            raise ValueError("gpu_ids cannot be empty")

        self.worker_cls = worker_cls
        self.gpu_ids = gpu_ids
        self.queue_size = queue_size
        self.suppress_worker_output = suppress_worker_output

        self.ctx = mp.get_context('spawn')
        self._shutdown_event = None

    def shutdown(self) -> None:
        """Signal the dispatcher to shut down gracefully."""
        if self._shutdown_event is not None:
            self._shutdown_event.set()

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
        self._shutdown_event = self.ctx.Event()
        shutdown_event = self._shutdown_event

        # Signal handler to set shutdown event
        def signal_handler(signum, frame):
            shutdown_event.set()

        # Register signal handlers only in main thread
        is_main_thread = threading.current_thread() is threading.main_thread()
        original_sigint = None
        original_sigterm = None
        if is_main_thread:
            original_sigint = signal.signal(signal.SIGINT, signal_handler)
            original_sigterm = signal.signal(signal.SIGTERM, signal_handler)

        processes = []
        worker_target = _worker_main_silenced if self.suppress_worker_output else _worker_main
        for gpu_id in self.gpu_ids:
            worker_instance = self.worker_cls()
            seed = base_seed + gpu_id

            p = self.ctx.Process(
                target=worker_target,
                args=(
                    worker_instance,
                    gpu_id,
                    seed,
                    task_queue,
                    result_queue,
                    task_timeout,
                    setup_kwargs,
                    shutdown_event,
                ),
            )
            p.start()
            processes.append(p)

        feeder_stop = threading.Event()
        task_count = [0]
        feeder_thread = threading.Thread(
            target=self._feeder,
            args=(generator, task_queue, feeder_stop, task_count, shutdown_event),
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
                shutdown_event=shutdown_event,
            )
        except KeyboardInterrupt:
            shutdown_event.set()
        finally:
            # Restore signal handlers if we registered them
            if is_main_thread:
                signal.signal(signal.SIGINT, original_sigint)
                signal.signal(signal.SIGTERM, original_sigterm)

            if on_exit:
                on_exit()

            # Shutdown sequence
            shutdown_event.set()
            feeder_stop.set()
            self._shutdown_workers(processes, task_queue)
            self._cleanup_queues(task_queue, result_queue)

    def _feeder(
        self,
        generator: Iterable[Any],
        task_queue: mp.Queue,
        stop_event: threading.Event,
        task_count: list,
        shutdown_event,
    ) -> None:
        task_id = 0
        try:
            for data in generator:
                if shutdown_event.is_set():
                    break
                # Use timeout to allow checking shutdown_event
                while not shutdown_event.is_set():
                    try:
                        task_queue.put((task_id, data), timeout=0.5)
                        break
                    except Full:
                        continue
                if shutdown_event.is_set():
                    break
                task_id += 1
        except Exception as e:
            # If generator fails, log and stop
            print(f"Feeder thread error: {e}")
        finally:
            task_count[0] = task_id
            stop_event.set()

    def _drain_queue(self, queue: mp.Queue) -> None:
        """Drain remaining items from queue to allow clean shutdown."""
        try:
            while True:
                queue.get_nowait()
        except Empty:
            pass

    def _shutdown_workers(self, processes: list, task_queue: mp.Queue) -> None:
        """Shutdown worker processes with escalating force (graceful → terminate → kill)."""
        # Send stop sentinels
        for _ in self.gpu_ids:
            try:
                task_queue.put(None, timeout=0.5)
            except Full:
                pass

        # Wait for graceful exit
        for p in processes:
            p.join(timeout=3.0)

        # Terminate stragglers
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1.0)

        # Force kill if needed
        for p in processes:
            if p.is_alive():
                p.kill()
                p.join(timeout=0.5)

    def _cleanup_queues(self, task_queue: mp.Queue, result_queue: mp.Queue) -> None:
        """Drain and close queues to prevent resource warnings."""
        self._drain_queue(task_queue)
        self._drain_queue(result_queue)
        task_queue.close()
        result_queue.close()
        task_queue.join_thread()
        result_queue.join_thread()

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
        shutdown_event,
    ) -> None:
        active_workers = len(self.gpu_ids)
        results_received = 0

        while True:
            # Check for shutdown signal
            if shutdown_event.is_set():
                break

            if feeder_stop.is_set() and results_received >= task_count[0]:
                break

            try:
                result = result_queue.get(timeout=0.1)
            except Empty:
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
