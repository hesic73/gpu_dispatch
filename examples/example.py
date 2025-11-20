from gpu_dispatch import BaseWorker, Dispatcher
from gpu_dispatch.ui import RichDispatcher
import time


class SimpleInferenceWorker(BaseWorker):
    def setup(self, gpu_id: int, seed: int, **kwargs):
        self.gpu_id = gpu_id
        self.model_name = kwargs.get("model_name", "default")
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

def run_core_dispatcher():
    results = []

    def on_success(task_id, result, worker_id):
        results.append(result)
        if task_id % 10 == 0:
            print(f"[core] Completed task {task_id} on GPU {worker_id}")

    dispatcher = Dispatcher(
        worker_cls=SimpleInferenceWorker,
        gpu_ids=[0, 1],
        queue_size=32,
    )

    start_time = time.time()
    dispatcher.run(
        generator=data_generator(),
        on_success=on_success,
        model_name="example_model",
    )
    elapsed = time.time() - start_time

    print(f"\nCompleted {len(results)} tasks in {elapsed:.2f}s")
    print(f"Throughput: {len(results)/elapsed:.1f} tasks/sec")
    print(f"\nFirst 5 results:")
    for i, result in enumerate(results[:5]):
        print(f"  {i}: {result}")


def run_rich_dispatcher():
    def on_success(task_id, result, worker_id):
        if task_id % 15 == 0:
            print(f"[rich] Completed {task_id} on GPU {worker_id}")

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

    print("\nRich dispatcher stats:")
    print(f"  Completed: {stats['completed']}")
    print(f"  Failed: {stats['failed']}")
    for gpu_id, gpu_stats in stats["gpu_status"].items():
        print(f"  GPU {gpu_id}: {gpu_stats['completed']} tasks, status={gpu_stats['status']}")


if __name__ == "__main__":
    run_core_dispatcher()
    run_rich_dispatcher()
