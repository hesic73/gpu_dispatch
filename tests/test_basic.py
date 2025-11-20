import time
import pytest
from gpu_dispatch import BaseWorker, Dispatcher


class SimpleWorker(BaseWorker):
    """A simple worker for testing."""

    def setup(self, gpu_id: int, seed: int, **kwargs):
        self.gpu_id = gpu_id
        self.multiplier = kwargs.get("multiplier", 1)

    def process(self, data: int) -> int:
        return data * self.multiplier

    def cleanup(self):
        pass


class SlowWorker(BaseWorker):
    """A worker that sleeps to test timeout."""

    def setup(self, gpu_id: int, seed: int, **kwargs):
        self.sleep_time = kwargs.get("sleep_time", 1.0)

    def process(self, data: int) -> int:
        time.sleep(self.sleep_time)
        return data

    def cleanup(self):
        pass


class FailingSetupWorker(BaseWorker):
    """A worker that fails during setup."""

    def setup(self, gpu_id: int, seed: int, **kwargs):
        raise RuntimeError("Intentional setup failure")

    def process(self, data):
        return data

    def cleanup(self):
        pass


class FailingProcessWorker(BaseWorker):
    """A worker that fails during processing."""

    def setup(self, gpu_id: int, seed: int, **kwargs):
        pass

    def process(self, data: int) -> int:
        if data == 5:
            raise ValueError("Intentional process failure")
        return data

    def cleanup(self):
        pass


def test_basic_pipeline():
    """Test basic pipeline execution."""
    def generator():
        for i in range(10):
            yield i

    results = []

    def on_success(task_id, data):
        results.append((task_id, data))

    dispatcher = Dispatcher(
        worker_cls=SimpleWorker,
        gpu_ids=[0],
        queue_size=16,
    )

    dispatcher.run(
        generator=generator(),
        on_success=on_success,
        multiplier=2,
    )

    # Verify all tasks completed
    assert len(results) == 10
    # Verify results are correct (doubled)
    for task_id, data in results:
        assert data == task_id * 2


def test_multiple_workers():
    """Test with multiple workers (simulating multiple GPUs)."""
    def generator():
        for i in range(100):
            yield i

    results = []

    def on_success(task_id, data):
        results.append((task_id, data))

    dispatcher = Dispatcher(
        worker_cls=SimpleWorker,
        gpu_ids=[0, 1, 2, 3],
        queue_size=32,
    )

    dispatcher.run(
        generator=generator(),
        on_success=on_success,
        multiplier=3,
    )

    # Verify all tasks completed
    assert len(results) == 100
    # Verify results are correct
    for task_id, data in results:
        assert data == task_id * 3


def test_task_error_handling():
    """Test that task errors are properly caught and reported."""
    def generator():
        for i in range(10):
            yield i

    successes = []
    errors = []

    def on_success(task_id, data):
        successes.append(task_id)

    def on_error(task_id, error):
        errors.append(task_id)

    dispatcher = Dispatcher(
        worker_cls=FailingProcessWorker,
        gpu_ids=[0],
        queue_size=16,
    )

    dispatcher.run(
        generator=generator(),
        on_success=on_success,
        on_error=on_error,
    )

    # Task 5 should fail, others should succeed
    assert len(successes) == 9
    assert len(errors) == 1
    assert errors[0] == 5


def test_setup_failure():
    """Test that setup failures are properly caught."""
    def generator():
        for i in range(5):
            yield i

    setup_failures = []

    def on_success(task_id, data):
        pytest.fail("Should not succeed if setup fails")

    def on_setup_fail(gpu_id, error):
        setup_failures.append(gpu_id)

    dispatcher = Dispatcher(
        worker_cls=FailingSetupWorker,
        gpu_ids=[0],
        queue_size=16,
    )

    # Should raise RuntimeError because all workers failed
    with pytest.raises(RuntimeError, match="All workers failed during setup"):
        dispatcher.run(
            generator=generator(),
            on_success=on_success,
            on_setup_fail=on_setup_fail,
        )

    assert len(setup_failures) == 1


@pytest.mark.timeout(10)
def test_timeout():
    """Test task timeout mechanism (Unix/Linux only)."""
    import platform
    if platform.system() == "Windows":
        pytest.skip("Timeout not supported on Windows")

    def generator():
        for i in range(5):
            yield i

    successes = []
    timeouts = []

    def on_success(task_id, data):
        successes.append(task_id)

    def on_timeout(task_id, timeout_val):
        timeouts.append(task_id)

    dispatcher = Dispatcher(
        worker_cls=SlowWorker,
        gpu_ids=[0],
        queue_size=16,
    )

    dispatcher.run(
        generator=generator(),
        on_success=on_success,
        on_timeout=on_timeout,
        task_timeout=0.5,  # 0.5 second timeout
        sleep_time=1.0,    # Worker sleeps for 1 second
    )

    # All tasks should timeout
    assert len(timeouts) == 5
    assert len(successes) == 0


def test_backpressure():
    """Test that the feeder respects queue size (backpressure)."""
    def generator():
        for i in range(1000):
            yield i

    results = []

    def on_success(task_id, data):
        results.append(task_id)

    dispatcher = Dispatcher(
        worker_cls=SimpleWorker,
        gpu_ids=[0],
        queue_size=8,  # Very small queue
    )

    # This should complete without memory issues
    dispatcher.run(
        generator=generator(),
        on_success=on_success,
        multiplier=1,
    )

    assert len(results) == 1000


def test_empty_generator():
    """Test with an empty generator."""
    def generator():
        return
        yield  # Never reached

    results = []

    def on_success(task_id, data):
        results.append(task_id)

    dispatcher = Dispatcher(
        worker_cls=SimpleWorker,
        gpu_ids=[0],
        queue_size=16,
    )

    dispatcher.run(
        generator=generator(),
        on_success=on_success,
    )

    assert len(results) == 0
