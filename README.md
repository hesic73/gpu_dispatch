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

def on_success(task_id, result):
    # Handle result
    pass

dispatcher = Dispatcher(
    worker_cls=InferenceWorker,
    gpu_ids=[0, 1, 2, 3],
    queue_size=512,
)

dispatcher.run(
    generator=data_generator(),
    on_success=on_success,
    model_path="./model.pth",
)
```

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
    generator,              # Iterator yielding tasks
    on_success,             # Callback(task_id, result)
    on_error=None,          # Callback(task_id, error)
    on_timeout=None,        # Callback(task_id, timeout)
    on_setup_fail=None,     # Callback(gpu_id, error)
    task_timeout=None,      # Seconds per task (Unix/Linux only)
    **setup_kwargs,         # Passed to worker.setup()
)
```

## Testing

```bash
pytest tests/
```

## License

MIT
