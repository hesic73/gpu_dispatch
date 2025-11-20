"""Example: Using RichDispatcher with live UI"""
from gpu_dispatch import BaseWorker
from gpu_dispatch.ui import RichDispatcher
import time


class SimpleInferenceWorker(BaseWorker):
    def setup(self, gpu_id: int, seed: int, **kwargs):
        self.gpu_id = gpu_id
        self.model_name = kwargs.get("model_name", "default")
        # In normal mode, these prints are suppressed by the UI
        print(f"[GPU {gpu_id}] Loading model: {self.model_name}")
        time.sleep(0.5)
        print(f"[GPU {gpu_id}] Ready!")

    def process(self, data: str) -> dict:
        time.sleep(0.1)
        return {
            "input": data,
            "gpu": self.gpu_id,
            "result": f"processed_{data}",
        }

    def cleanup(self):
        print(f"[GPU {self.gpu_id}] Cleaning up...")


def data_generator():
    for i in range(50):
        yield f"data_{i:03d}"


if __name__ == "__main__":
    def on_success(task_id, result, worker_id):
        # Optional: can still log some events
        if task_id % 15 == 0:
            print(f"Milestone: Completed {task_id} on GPU {worker_id}")

    rich_dispatcher = RichDispatcher(
        worker_cls=SimpleInferenceWorker,
        gpu_ids=[0, 1],
        queue_size=32,
        refresh_rate=4.0,
    )

    stats = rich_dispatcher.run(
        generator=data_generator(),
        on_success=on_success,
        model_name="example_model",
    )

    print("\nFinal statistics:")
    print(f"  Completed: {stats['completed']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Timeouts: {stats['timeouts']}")
    for gpu_id, gpu_stats in stats["gpu_status"].items():
        print(f"  GPU {gpu_id}: {gpu_stats['completed']} tasks, status={gpu_stats['status']}")
