from __future__ import annotations

import threading
import time
from copy import deepcopy
from typing import Any, Callable, Iterable

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from gpu_dispatch.dispatcher import Dispatcher
from gpu_dispatch.worker import BaseWorker

SuccessCallback = Callable[[int, Any, int], None]
ErrorCallback = Callable[[int, str, int], None]
TimeoutCallback = Callable[[int, float, int], None]
SetupFailCallback = Callable[[int, str], None]
StartCallback = Callable[[int, int], None]
ExitCallback = Callable[[], None]


class RichDispatcher:
    """High-level dispatcher wrapper that renders live status with Rich."""

    def __init__(
        self,
        worker_cls: type[BaseWorker],
        gpu_ids: list[int],
        queue_size: int = 1024,
        show_ui: bool = True,
        refresh_rate: float = 2.0,
        console: Console | None = None,
    ) -> None:
        if refresh_rate <= 0:
            raise ValueError("refresh_rate must be positive")

        self._dispatcher = Dispatcher(worker_cls=worker_cls, gpu_ids=gpu_ids, queue_size=queue_size)
        self._gpu_ids = gpu_ids
        self._show_ui = show_ui
        self._refresh_rate = refresh_rate
        self._console = console

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._exception: BaseException | None = None
        self._stats: dict[str, Any] = {}
        self._reset_stats()

    def run(
        self,
        generator: Iterable[Any],
        on_success: SuccessCallback | None = None,
        on_error: ErrorCallback | None = None,
        on_timeout: TimeoutCallback | None = None,
        on_setup_fail: SetupFailCallback | None = None,
        on_task_start: StartCallback | None = None,
        on_exit: ExitCallback | None = None,
        base_seed: int = 42,
        task_timeout: float | None = None,
        **setup_kwargs,
    ) -> dict[str, Any]:
        """Execute the dispatcher and optionally display a live UI."""
        self._reset_stats()
        self._stop_event = threading.Event()
        self._exception = None
        self._stats["start_time"] = time.perf_counter()

        dispatch_thread = threading.Thread(
            target=self._run_dispatcher,
            args=(
                generator,
                self._wrap_success_callback(on_success),
                self._wrap_error_callback(on_error),
                self._wrap_timeout_callback(on_timeout),
                self._wrap_setup_fail_callback(on_setup_fail),
                self._wrap_task_start_callback(on_task_start),
                self._wrap_exit_callback(on_exit),
                base_seed,
                task_timeout,
                setup_kwargs,
            ),
            daemon=True,
        )
        dispatch_thread.start()

        if self._show_ui:
            self._render_loop(dispatch_thread)
        else:
            dispatch_thread.join()

        if self._exception:
            raise self._exception

        return deepcopy(self._stats)

    def _run_dispatcher(
        self,
        generator: Iterable[Any],
        on_success: SuccessCallback,
        on_error: ErrorCallback | None,
        on_timeout: TimeoutCallback | None,
        on_setup_fail: SetupFailCallback | None,
        on_task_start: StartCallback | None,
        on_exit: ExitCallback | None,
        base_seed: int,
        task_timeout: float | None,
        setup_kwargs: dict[str, Any],
    ) -> None:
        try:
            self._dispatcher.run(
                generator=generator,
                on_success=on_success,
                on_error=on_error,
                on_timeout=on_timeout,
                on_setup_fail=on_setup_fail,
                on_task_start=on_task_start,
                on_exit=on_exit,
                base_seed=base_seed,
                task_timeout=task_timeout,
                **setup_kwargs,
            )
        except BaseException as exc:  # Surface worker failures back to caller
            self._exception = exc
        finally:
            self._stop_event.set()

    def _wrap_task_start_callback(self, user_callback: StartCallback | None) -> StartCallback:
        def wrapper(task_id: int, worker_id: int) -> None:
            with self._lock:
                worker_stats = self._stats["gpu_status"].get(worker_id)
                if worker_stats is None:
                    return
                worker_stats["status"] = "processing"
                worker_stats["current_task"] = task_id
                worker_stats["task_start_time"] = time.perf_counter()
            if user_callback:
                user_callback(task_id, worker_id)

        return wrapper

    def _wrap_success_callback(self, user_callback: SuccessCallback | None) -> SuccessCallback:
        def wrapper(task_id: int, result: Any, worker_id: int) -> None:
            with self._lock:
                self._stats["completed"] += 1
                self._stats["total"] += 1
                worker_stats = self._stats["gpu_status"][worker_id]
                worker_stats["completed"] += 1
                self._finalize_task(worker_stats)
            if user_callback:
                user_callback(task_id, result, worker_id)

        return wrapper

    def _wrap_error_callback(self, user_callback: ErrorCallback | None) -> ErrorCallback:
        def wrapper(task_id: int, error: str, worker_id: int) -> None:
            with self._lock:
                self._stats["failed"] += 1
                self._stats["total"] += 1
                worker_stats = self._stats["gpu_status"][worker_id]
                worker_stats["failed"] += 1
                self._finalize_task(worker_stats)
            if user_callback:
                user_callback(task_id, error, worker_id)

        return wrapper

    def _wrap_timeout_callback(self, user_callback: TimeoutCallback | None) -> TimeoutCallback:
        def wrapper(task_id: int, timeout: float, worker_id: int) -> None:
            with self._lock:
                self._stats["timeouts"] += 1
                self._stats["total"] += 1
                worker_stats = self._stats["gpu_status"][worker_id]
                worker_stats["timeouts"] += 1
                self._finalize_task(worker_stats)
            if user_callback:
                user_callback(task_id, timeout, worker_id)

        return wrapper

    def _wrap_setup_fail_callback(self, user_callback: SetupFailCallback | None) -> SetupFailCallback:
        def wrapper(gpu_id: int, error: str) -> None:
            with self._lock:
                self._stats["setup_failures"] += 1
                worker_stats = self._stats["gpu_status"].get(gpu_id)
                if worker_stats is not None:
                    worker_stats["status"] = "error"
                    worker_stats["current_task"] = None
                    worker_stats["task_start_time"] = None
            if user_callback:
                user_callback(gpu_id, error)

        return wrapper

    def _wrap_exit_callback(self, user_callback: ExitCallback | None) -> ExitCallback:
        def wrapper() -> None:
            with self._lock:
                self._stats["end_time"] = time.perf_counter()
                for worker_stats in self._stats["gpu_status"].values():
                    if worker_stats["status"] != "error":
                        worker_stats["status"] = "finished"
                        worker_stats["current_task"] = None
                        worker_stats["task_start_time"] = None
            if user_callback:
                user_callback()

        return wrapper

    def _finalize_task(self, worker_stats: dict[str, Any]) -> None:
        now = time.perf_counter()
        start_time = worker_stats.get("task_start_time")
        worker_stats["last_duration"] = None
        if start_time is not None:
            worker_stats["last_duration"] = max(0.0, now - start_time)
        worker_stats["task_start_time"] = None
        worker_stats["current_task"] = None
        if worker_stats["status"] != "error":
            worker_stats["status"] = "idle"

    def _render_loop(self, dispatch_thread: threading.Thread) -> None:
        console = self._console or Console()
        refresh_delay = 1.0 / self._refresh_rate
        with Live(self._renderable(), console=console, refresh_per_second=self._refresh_rate, transient=True) as live:
            while dispatch_thread.is_alive() or not self._stop_event.is_set():
                live.update(self._renderable(), refresh=True)
                time.sleep(refresh_delay)
        dispatch_thread.join()

    def _renderable(self):
        stats = self._snapshot()
        overall = self._build_overall_panel(stats)
        table = self._build_gpu_table(stats)
        return Group(overall, table)

    def _build_overall_panel(self, stats: dict[str, Any]) -> Panel:
        completed = stats["completed"]
        failed = stats["failed"]
        timeouts = stats["timeouts"]
        total = stats["total"]

        now = time.perf_counter()
        start_time = stats["start_time"]
        end_time = stats.get("end_time")
        elapsed = None
        if start_time is not None:
            elapsed_source = end_time if end_time is not None else now
            elapsed = max(0.0, elapsed_source - start_time)

        throughput = "-"
        if elapsed and elapsed > 0 and total > 0:
            throughput = f"{total / elapsed:.1f} tasks/s"

        lines = [
            f"Processed: {total} (✓ {completed} ✗ {failed} ⏱ {timeouts})",
            f"Elapsed: {self._format_elapsed(elapsed)} | Throughput: {throughput}",
        ]
        return Panel("\n".join(lines), title="Overall Progress", border_style="cyan")

    def _build_gpu_table(self, stats: dict[str, Any]) -> Table:
        table = Table(title="GPU Status", expand=True)
        table.add_column("GPU", justify="center")
        table.add_column("Status", justify="center")
        table.add_column("Current", justify="center")
        table.add_column("Completed", justify="center")
        table.add_column("Failed", justify="center")
        table.add_column("Timeout", justify="center")
        table.add_column("Last Time", justify="center")

        status_styles = {
            "processing": "yellow",
            "idle": "green",
            "initializing": "cyan",
            "finished": "bold green",
            "error": "bold red",
        }

        for gpu_id in self._gpu_ids:
            worker_stats = stats["gpu_status"][gpu_id]
            status = worker_stats["status"]
            status_text = Text(status.title(), style=status_styles.get(status, "white"))

            current_task = worker_stats["current_task"]
            current_display = f"#{current_task}" if current_task is not None else "-"

            last_time = worker_stats.get("last_duration")
            table.add_row(
                str(gpu_id),
                status_text,
                current_display,
                str(worker_stats["completed"]),
                str(worker_stats["failed"]),
                str(worker_stats["timeouts"]),
                self._format_last_time(last_time),
            )

        return table

    def _format_elapsed(self, elapsed: float | None) -> str:
        if elapsed is None:
            return "--"
        minutes, seconds = divmod(int(elapsed), 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _format_last_time(self, duration: float | None) -> str:
        if duration is None:
            return "-"
        if duration < 1:
            return f"{duration * 1000:.0f} ms"
        return f"{duration:.1f} s"

    def _snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._stats)

    def _reset_stats(self) -> None:
        self._stats = {
            "total": 0,
            "completed": 0,
            "failed": 0,
            "timeouts": 0,
            "setup_failures": 0,
            "start_time": None,
            "end_time": None,
            "gpu_status": {
                gpu_id: {
                    "status": "initializing",
                    "current_task": None,
                    "task_start_time": None,
                    "last_duration": None,
                    "completed": 0,
                    "failed": 0,
                    "timeouts": 0,
                }
                for gpu_id in self._gpu_ids
            },
        }
