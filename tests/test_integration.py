import time
import random
from gpu_dispatch import BaseWorker, Dispatcher


class SimulatedInferenceWorker(BaseWorker):
    """Simulates a realistic inference worker."""

    def setup(self, gpu_id: int, seed: int, **kwargs):
        self.gpu_id = gpu_id
        self.model_name = kwargs.get("model_name", "default_model")
        # Simulate loading a heavy model
        time.sleep(0.1)
        print(f"[GPU {gpu_id}] Loaded model: {self.model_name}")

    def process(self, image_path: str) -> dict:
        """Simulate image inference."""
        # Simulate variable processing time
        time.sleep(random.uniform(0.001, 0.01))
        return {
            "path": image_path,
            "gpu": self.gpu_id,
            "prediction": f"class_{random.randint(0, 999)}",
            "confidence": random.uniform(0.5, 1.0),
        }

    def cleanup(self):
        print(f"[GPU {self.gpu_id}] Cleaning up...")


def test_large_dataset_single_gpu():
    """Test processing a large dataset with a single GPU."""
    def image_generator():
        for i in range(1000):
            yield f"/data/images/img_{i:05d}.jpg"

    results = []
    errors = []

    def on_success(task_id, data):
        results.append(data)

    def on_error(task_id, error):
        errors.append((task_id, error))

    dispatcher = Dispatcher(
        worker_cls=SimulatedInferenceWorker,
        gpu_ids=[0],
        queue_size=128,
    )

    start_time = time.time()
    dispatcher.run(
        generator=image_generator(),
        on_success=on_success,
        on_error=on_error,
        model_name="yolo_v8",
    )
    elapsed = time.time() - start_time

    print(f"\nProcessed 1000 images in {elapsed:.2f}s ({1000/elapsed:.1f} imgs/sec)")
    assert len(results) == 1000
    assert len(errors) == 0

    # Verify all results have expected structure
    for result in results[:10]:  # Check first 10
        assert "path" in result
        assert "gpu" in result
        assert "prediction" in result
        assert result["gpu"] == 0


def test_large_dataset_multi_gpu():
    """Test processing a large dataset with multiple GPUs."""
    def image_generator():
        for i in range(1000):
            yield f"/data/images/img_{i:05d}.jpg"

    results = []
    gpu_usage = {0: 0, 1: 0, 2: 0, 3: 0}

    def on_success(task_id, data):
        results.append(data)
        gpu_usage[data["gpu"]] += 1

    dispatcher = Dispatcher(
        worker_cls=SimulatedInferenceWorker,
        gpu_ids=[0, 1, 2, 3],
        queue_size=256,
    )

    start_time = time.time()
    dispatcher.run(
        generator=image_generator(),
        on_success=on_success,
        model_name="resnet50",
    )
    elapsed = time.time() - start_time

    print(f"\nProcessed 1000 images in {elapsed:.2f}s ({1000/elapsed:.1f} imgs/sec)")
    print(f"GPU usage distribution: {gpu_usage}")

    assert len(results) == 1000

    # Verify work was distributed across GPUs
    for gpu_id, count in gpu_usage.items():
        assert count > 0, f"GPU {gpu_id} was not used"

    # Check distribution is relatively balanced (within 30% of average)
    avg = 1000 / 4
    for gpu_id, count in gpu_usage.items():
        assert abs(count - avg) < avg * 0.3, f"GPU {gpu_id} load imbalance: {count} vs {avg}"


class UnreliableWorker(BaseWorker):
    """Worker that fails on certain inputs."""

    def setup(self, gpu_id: int, seed: int, **kwargs):
        self.gpu_id = gpu_id

    def process(self, data: int) -> int:
        # Fail on multiples of 10
        if data % 10 == 0:
            raise RuntimeError(f"Failed on {data}")
        return data * 2

    def cleanup(self):
        pass


class VariableTimeWorker(BaseWorker):
    """Worker with variable processing time."""

    def setup(self, gpu_id: int, seed: int, **kwargs):
        pass

    def process(self, sleep_time: float) -> str:
        time.sleep(sleep_time)
        return f"completed_{sleep_time}"

    def cleanup(self):
        pass


class MetadataWorker(BaseWorker):
    """Worker that processes metadata."""

    def setup(self, gpu_id: int, seed: int, **kwargs):
        pass

    def process(self, data: dict) -> int:
        return data["index"] * 10

    def cleanup(self):
        pass


class OrderedWorker(BaseWorker):
    """Worker with random delays."""

    def setup(self, gpu_id: int, seed: int, **kwargs):
        pass

    def process(self, data: int) -> int:
        # Small random delay to ensure non-sequential completion
        time.sleep(random.uniform(0.001, 0.005))
        return data

    def cleanup(self):
        pass


def test_error_recovery():
    """Test that workers continue after errors."""

    def generator():
        for i in range(100):
            yield i

    successes = []
    errors = []

    def on_success(task_id, data):
        successes.append(task_id)

    def on_error(task_id, error):
        errors.append(task_id)

    dispatcher = Dispatcher(
        worker_cls=UnreliableWorker,
        gpu_ids=[0, 1],
        queue_size=32,
    )

    dispatcher.run(
        generator=generator(),
        on_success=on_success,
        on_error=on_error,
    )

    # 10 tasks should fail (0, 10, 20, ..., 90)
    # 90 tasks should succeed
    assert len(errors) == 10
    assert len(successes) == 90

    # Verify failed tasks are correct
    assert set(errors) == {0, 10, 20, 30, 40, 50, 60, 70, 80, 90}


def test_mixed_timeout_and_success():
    """Test that timeouts and successes can coexist."""
    import platform
    if platform.system() == "Windows":
        import pytest
        pytest.skip("Timeout not supported on Windows")

    def generator():
        # Mix of fast and slow tasks
        tasks = [0.1, 0.1, 2.0, 0.1, 3.0, 0.1, 0.1]
        for t in tasks:
            yield t

    successes = []
    timeouts = []

    def on_success(task_id, data):
        successes.append(task_id)

    def on_timeout(task_id, timeout_val):
        timeouts.append(task_id)

    dispatcher = Dispatcher(
        worker_cls=VariableTimeWorker,
        gpu_ids=[0],
        queue_size=16,
    )

    dispatcher.run(
        generator=generator(),
        on_success=on_success,
        on_timeout=on_timeout,
        task_timeout=1.0,  # 1 second timeout
    )

    # Tasks with 0.1s should succeed (indices 0, 1, 3, 5, 6)
    # Tasks with 2.0s and 3.0s should timeout (indices 2, 4)
    assert len(successes) == 5
    assert len(timeouts) == 2
    assert set(timeouts) == {2, 4}


class StatefulGenerator:
    """A generator that maintains state."""

    def __init__(self):
        self.count = 0

    def __iter__(self):
        for i in range(100):
            self.count += 1
            yield {"index": i, "timestamp": time.time()}


def test_generator_with_state():
    """Test that generator state is preserved during execution."""

    gen = StatefulGenerator()
    results = []

    def on_success(task_id, data):
        results.append(data)

    dispatcher = Dispatcher(
        worker_cls=MetadataWorker,
        gpu_ids=[0, 1],
        queue_size=32,
    )

    dispatcher.run(
        generator=gen,
        on_success=on_success,
    )

    assert len(results) == 100
    assert gen.count == 100  # Verify generator was fully consumed


def test_callback_ordering():
    """Test that callbacks are called for each result."""

    def generator():
        for i in range(50):
            yield i

    callback_order = []

    def on_success(task_id, data):
        callback_order.append(task_id)

    dispatcher = Dispatcher(
        worker_cls=OrderedWorker,
        gpu_ids=[0, 1, 2],
        queue_size=32,
    )

    dispatcher.run(
        generator=generator(),
        on_success=on_success,
    )

    # All tasks should have callbacks
    assert len(callback_order) == 50

    # Callbacks might not be in order (due to parallel execution)
    # but all task IDs should be present
    assert set(callback_order) == set(range(50))
