# gpu_dispatch

Lightweight task orchestration library for multi-GPU inference workloads.

## Installation

```bash
pip install git+https://github.com/hesic73/gpu_dispatch.git
```

For local development:

```bash
git clone https://github.com/hesic73/gpu_dispatch.git
cd gpu_dispatch
pip install -e ".[dev]"
```

## Quick Start

```python
from gpu_dispatch import BaseWorker, Dispatcher

class InferenceWorker(BaseWorker):
    def setup(self, gpu_id, seed, model_path):
        self.gpu_id = gpu_id
        # Load your model here

    def process(self, data):
        # Run inference
        return result

    def cleanup(self):
        # Release resources
        pass

def data_generator():
    for item in dataset:
        yield item

def on_success(task_id, result, worker_id):
    # Handle result
    pass

def on_exit():
    print("All workers drained!")

dispatcher = Dispatcher(
    worker_cls=InferenceWorker,
    gpu_ids=[0, 1, 2, 3],
    queue_size=512,
)

dispatcher.run(
    generator=data_generator(),
    on_success=on_success,
    on_exit=on_exit,
    model_path="./model.pth",
)
```

## Rich UI Dispatcher

Prefer a live dashboard? Use the optional Rich-native wrapper:

```python
from gpu_dispatch.ui import RichDispatcher

dispatcher = RichDispatcher(
    worker_cls=InferenceWorker,
    gpu_ids=[0, 1, 2, 3],
    refresh_rate=4.0,
)

stats = dispatcher.run(
    generator=data_generator(),
    on_success=on_success,
    model_path="./model.pth",
)

print(f"Completed {stats['completed']} tasks")
```

`RichDispatcher` mirrors the core API, renders an updating summary/table in the terminal, and still returns aggregated stats when `show_ui=False`.

## API

### BaseWorker

Implement these methods:

- `setup(gpu_id, seed, **kwargs)` - Initialize worker (load models)
- `process(data)` - Process single item
- `cleanup()` - Release resources

### Dispatcher

```python
Dispatcher(worker_cls, gpu_ids, queue_size=1024)
```

```python
dispatcher.run(
    generator,              # Iterator yielding tasks/records
    on_success,             # Callback(task_id, result, worker_id)
    on_error=None,          # Callback(task_id, error, worker_id)
    on_timeout=None,        # Callback(task_id, timeout, worker_id)
    on_setup_fail=None,     # Callback(gpu_id, error)
    on_task_start=None,     # Callback(task_id, worker_id)
    on_exit=None,           # Callback()
    base_seed=42,           # Seed offset for each worker
    task_timeout=None,      # Seconds per task (Unix/Linux only)
    **setup_kwargs,         # Passed to worker.setup()
)
```

- `on_success` is required; all other callbacks are optional.
- `on_task_start` fires before `worker.process()` runs, which enables per-GPU tracking for the Rich UI.
- `on_exit` is guaranteed to run (success or failure) so you can flush buffers, save metrics, etc.
- `dispatcher.run()` blocks until the generator is exhausted and workers clean up.

## Testing

```bash
pytest tests/
```

## License

MIT
