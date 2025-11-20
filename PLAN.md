# Implementation Plan

## Completed

Core dispatcher with multiprocessing task distribution:
- BaseWorker abstraction with setup/process/cleanup lifecycle
- Protocol messages (TaskSuccess, TaskError, TaskTimeout, SetupFailed, CleanupFailed)
- Automatic backpressure control via bounded queues
- Per-task timeout support (Unix/Linux, signal-based)
- Graceful error handling and worker lifecycle management
- Comprehensive test suite (13 tests)
- Packaging setup (pyproject.toml, installable via pip)

## Next: UI Extension

Add a high-level UI wrapper with rich progress display while maintaining the core dispatcher's simplicity.

## Naming

Use `ui` module instead of `extras` (clearer intent).

```
gpu_dispatch/
├── ui/
│   ├── __init__.py
│   └── rich_dispatcher.py
```

## Dependencies

Add `rich` as a default dependency (not optional):

```toml
dependencies = ["rich>=13.0"]
```

Users get it automatically with `pip install git+https://...`

## Core Changes (Breaking)

### 1. Protocol Messages - Add worker_id

All task-related messages include worker_id:

```python
@dataclass
class TaskSuccess:
    task_id: int
    data: Any
    worker_id: int

@dataclass
class TaskError:
    task_id: int
    error: str
    worker_id: int

@dataclass
class TaskTimeout:
    task_id: int
    timeout: float
    worker_id: int
```

### 2. Worker - Pass gpu_id to messages

Modify `_worker_main` to include `gpu_id` in all result messages.

### 3. Callbacks - Add worker_id parameter

New callback signatures:

```python
on_success(task_id: int, result: Any, worker_id: int)
on_error(task_id: int, error: str, worker_id: int)
on_timeout(task_id: int, timeout: float, worker_id: int)
```

### 4. New Callback - on_exit

Add optional `on_exit` callback to `Dispatcher.run()`:

```python
def run(
    self,
    generator: Iterator[Any],
    on_success: Callable[[int, Any, int], None],
    on_error: Callable[[int, str, int], None] | None = None,
    on_timeout: Callable[[int, float, int], None] | None = None,
    on_setup_fail: Callable[[int, str], None] | None = None,
    on_exit: Callable[[], None] | None = None,  # NEW - no parameters
    base_seed: int = 42,
    task_timeout: float | None = None,
    **setup_kwargs,
) -> None:
```

Called in finally block:

```python
finally:
    if on_exit:
        on_exit()  # User can do cleanup, save results, etc.

    # Cleanup workers
    ...
```

User maintains their own state via closures/class methods, no need to pass stats.

## RichDispatcher Implementation

### API

```python
from gpu_dispatch.ui import RichDispatcher

dispatcher = RichDispatcher(
    worker_cls=MyWorker,
    gpu_ids=[0, 1, 2, 3],
    queue_size=512,
    show_ui=True,          # Can disable for logging/non-interactive
    refresh_rate=2.0,      # UI updates per second
)

stats = dispatcher.run(
    generator=my_generator(),
    on_success=custom_success_handler,  # Optional, receives (task_id, result, worker_id)
    on_error=custom_error_handler,      # Optional
    on_timeout=custom_timeout_handler,  # Optional
    model_path="./model.pth",
)

# Returns final statistics
print(f"Completed: {stats['completed']}")
```

### Internal Architecture

**Composition over inheritance**:
- Wraps core `Dispatcher`
- Maintains internal statistics
- Provides callback wrappers that:
  1. Update UI state
  2. Call user's custom callback (if provided)

**Threading**:
- Main thread: Rich Live display
- Background thread: Dispatcher.run()
- Thread-safe state updates with Lock

**State tracking**:
```python
self._stats = {
    'total': 0,
    'completed': 0,
    'failed': 0,
    'timeout': 0,
    'start_time': None,
    'gpu_status': {
        gpu_id: {
            'status': 'initializing',  # idle, processing, finished, error
            'current_task': None,
            'task_start_time': None,
            'completed': 0,
            'failed': 0,
            'timeout': 0,
        }
        for gpu_id in gpu_ids
    }
}
```

### UI Components

**Overall Progress Panel**:
```
┌─ Overall Progress ───────────────────────────┐
│ Progress: 847/1000 (✓ 820 ✗ 15 ⏱ 12)       │
│ Elapsed: 0:12:34 │ ETA: 0:02:15            │
└──────────────────────────────────────────────┘
```

**GPU Status Table**:
```
┌─ GPU Status ─────────────────────────────────────────────┐
│ GPU │ Status      │ Current │ Completed │ Failed │ Time │
├─────┼─────────────┼─────────┼───────────┼────────┼──────┤
│  0  │ Processing  │  #1523  │    210    │   3    │ 2.1s │
│  1  │ Processing  │  #1524  │    208    │   4    │ 1.8s │
│  2  │ Idle        │    -    │    205    │   2    │  -   │
│  3  │ Processing  │  #1525  │    207    │   6    │ 0.9s │
└─────┴─────────────┴─────────┴───────────┴────────┴──────┘
```

**Status colors**:
- Processing: Yellow
- Idle: Green
- Initializing: Blue
- Finished: Bold Green
- Error: Bold Red

### GPU Status Detection

Track GPU status transitions:
- `initializing` → first message from worker
- `idle` → after completing a task (before getting next)
- `processing` → when task starts
- `finished` → when worker exits
- `error` → when setup fails

Use worker_id from messages to update correct GPU status.

### User Callback Integration

```python
def _wrap_success_callback(self, user_callback):
    def wrapper(task_id, result, worker_id):
        with self._lock:
            # Update UI state
            self._stats['completed'] += 1
            self._stats['gpu_status'][worker_id]['completed'] += 1
            self._stats['gpu_status'][worker_id]['status'] = 'idle'
            self._stats['gpu_status'][worker_id]['current_task'] = None

        # Call user callback
        if user_callback:
            user_callback(task_id, result, worker_id)

    return wrapper
```

## Implementation Steps

1. Add type annotations to Dispatcher.run()
2. Update core protocol (add worker_id)
3. Update worker._worker_main (pass gpu_id to messages)
4. Update dispatcher callbacks (add worker_id parameter, add on_exit)
5. Add rich to dependencies
6. Implement RichDispatcher
7. Write examples
8. Update tests (new callback signatures)
9. Update design.md
10. Update README.md

## Examples

### Basic Rich UI
```python
from gpu_dispatch.ui import RichDispatcher

dispatcher = RichDispatcher(
    worker_cls=MyWorker,
    gpu_ids=[0, 1, 2, 3],
)

stats = dispatcher.run(
    generator=data_generator(),
    model_path="./model.pth",
)
```

### With Custom Callbacks
```python
results = []

def save_result(task_id, result, worker_id):
    results.append(result)

def handle_error(task_id, error, worker_id):
    logging.error(f"Task {task_id} on GPU {worker_id}: {error}")

def on_exit():
    print(f"Finished! Processed {len(results)} items")
    save_to_disk(results)

dispatcher = RichDispatcher(
    worker_cls=MyWorker,
    gpu_ids=[0, 1, 2, 3],
)

stats = dispatcher.run(
    generator=data_generator(),
    on_success=save_result,
    on_error=handle_error,
    on_exit=on_exit,
    model_path="./model.pth",
)
```

### No UI (just statistics)
```python
dispatcher = RichDispatcher(
    worker_cls=MyWorker,
    gpu_ids=[0, 1, 2, 3],
    show_ui=False,  # Disable UI, still track stats
)

stats = dispatcher.run(...)
print(f"Success rate: {stats['completed']/stats['total']*100:.1f}%")
```

## Design Decisions

### On-exit callback

Add `on_exit` callback to `Dispatcher.run()`:
- Called in finally block (guaranteed execution)
- No parameters - user maintains state themselves via closures
- Executes on both normal completion and interruption
- Useful for cleanup, final logging, saving state
- Reliable because it's in the finally block of dispatcher.run()
