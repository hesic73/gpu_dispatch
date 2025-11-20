# Implementation Plan: `gpu_dispatch`

## Project Overview

A lightweight Python 3.10+ library for multi-GPU task orchestration using multiprocessing.

## Design Considerations

### Core Philosophy
- **Minimal & Focused**: Core library does ONE thing well - dispatching tasks to GPU workers
- **Extensible**: Designed to support high-level wrappers (e.g., rich progress UI) without coupling
- **Modern Python**: Python 3.10+ only, use modern packaging (pyproject.toml)

### Future Extension: Rich Progress UI
The core library will be designed to support a future high-level wrapper that provides:
- Rich text progress bars (using `rich` library)
- Real-time task statistics
- Worker status visualization

**Design Implications**:
- Callbacks should support progress tracking metadata (task_id, timing, etc.)
- Dispatcher should expose internal state (queue sizes, worker status) via properties/methods
- Event-driven architecture to allow wrapping without modification

---

## Implementation Tasks

### Phase 1: Project Structure
```
gpu_dispatch/
├── pyproject.toml           # Modern packaging config (PEP 621)
├── README.md                # Usage documentation
├── .gitignore               # Python ignores
├── gpu_dispatch/
│   ├── __init__.py          # Public API exports
│   ├── protocol.py          # IPC message types
│   ├── worker.py            # BaseWorker + worker process logic
│   └── dispatcher.py        # Dispatcher engine
└── tests/
    ├── __init__.py
    ├── test_basic.py        # Unit tests (no GPU required)
    └── test_integration.py  # Integration tests
```

### Phase 2: Core Implementation

#### 2.1 Protocol (`protocol.py`)
Define dataclasses for IPC messages:
- `TaskSuccess(task_id, data)`
- `TaskError(task_id, error)`
- `TaskTimeout(task_id, timeout)` - when task exceeds timeout
- `SetupFailed(gpu_id, error)`
- `CleanupFailed(gpu_id, error)`

#### 2.2 Worker (`worker.py`)
- `BaseWorker` abstract base class with lifecycle methods:
  - `setup(gpu_id, seed, **kwargs)` - initialization
  - `process(data) -> Any` - per-task execution
  - `cleanup()` - resource release
- `_worker_main()` function - the target for `multiprocessing.Process`
  - Handles setup/cleanup lifecycle
  - Main loop: consume from task queue, execute process(), send results
  - **Timeout handling**: Uses `signal.alarm()` (Unix/Linux only) to interrupt long-running tasks
    - Set alarm before `process()` call
    - Clear alarm on success
    - Catch `TimeoutError` and send `TaskTimeout` message

#### 2.3 Dispatcher (`dispatcher.py`)
- `Dispatcher` class with:
  - `__init__(worker_cls, gpu_ids, queue_size)`
  - `run(generator, on_success, on_error, on_timeout, on_setup_fail, task_timeout, **setup_kwargs)`

**Internal Components**:
- **Feeder Thread**: Pulls from generator → task queue (blocks on full queue)
- **Worker Processes**: Persistent processes holding GPU context
  - Each worker receives `task_timeout` parameter
  - Uses signal-based timeout for task interruption
- **Monitor Loop**: Polls result queue → invokes callbacks
  - Routes `TaskTimeout` to `on_timeout` callback
- **Shutdown Logic**: Graceful cleanup with poison pills

**Extension Points** (for future rich UI):
- Add optional `on_progress` callback for task lifecycle events
- Expose `get_stats()` method returning queue sizes, completion count
- Emit timing metadata (task start/end times) in callbacks

### Phase 3: Testing

#### 3.1 Unit Tests (`test_basic.py`)
- Worker lifecycle (setup → process → cleanup)
- Queue communication correctness
- Error propagation (setup failures, process exceptions)
- **Timeout mechanism** (task that sleeps longer than timeout)
- Backpressure mechanism

#### 3.2 Integration Tests (`test_integration.py`)
- Multi-worker scenario (simulate multiple GPUs with cpu)
- Large dataset processing (1000+ items)
- Verify task ordering and completion
- Resource cleanup verification

### Phase 4: Packaging & Documentation

#### 4.1 `pyproject.toml`
```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "gpu_dispatch"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = []
```

#### 4.2 `README.md`
- Installation instructions (`pip install git+https://...`)
- Quick start example
- API reference
- Link to design.md

#### 4.3 `.gitignore`
- Python artifacts (`__pycache__`, `*.pyc`, `.pytest_cache`)
- Build artifacts (`dist/`, `*.egg-info`)

---

## Implementation Order

1. **Project skeleton** - pyproject.toml, .gitignore, directory structure
2. **Protocol** - Simple dataclasses, no dependencies
3. **BaseWorker** - Abstract class definition
4. **Worker process logic** - The actual multiprocessing target function
5. **Dispatcher** - Main orchestration engine
6. **Tests** - Validate correctness
7. **Documentation** - README with examples

---

## Key Technical Decisions

### Multiprocessing Strategy
- Use `multiprocessing.Queue` (not Pipe) for thread-safe, multi-consumer support
- Use `spawn` context explicitly (compatible across platforms, especially macOS)
- Workers are long-lived processes (not spawned per task)

### Error Handling
- Capture full tracebacks using `traceback.format_exc()`
- Distinguish setup errors (fatal for worker) from task errors (non-fatal)
- Use poison pills (`None`) for graceful shutdown

### Backpressure
- Feeder thread blocks on `queue.put(block=True)` when task queue is full
- This naturally throttles generator consumption

### Timeout Mechanism
- **Platform**: Unix/Linux only (uses `signal.SIGALRM`)
- **Implementation**:
  - Worker sets `signal.alarm(timeout_seconds)` before calling `process()`
  - Signal handler raises `TimeoutError` to interrupt execution
  - Worker catches exception and sends `TaskTimeout` message
  - Worker continues to next task (not terminated)
- **Graceful degradation**: If timeout is `None`, no alarm is set
- **Limitation**: Not thread-safe within worker process (acceptable since worker is single-threaded)

### Extensibility for Rich UI
- All callbacks receive `task_id` for tracking
- Optional metadata can be added to protocol messages (timestamps, worker_id)
- Dispatcher can expose `get_stats()` → `{"pending": N, "completed": M, "failed": K}`

---

## Non-Goals (Out of Scope)

- GPU detection/allocation (user provides `gpu_ids`)
- Automatic retry logic (user handles in callbacks)
- Rich UI implementation (future separate module)
- Task persistence or checkpointing
- Distributed workers (single-node only)
- Windows timeout support (signal.alarm is Unix-only; adding Windows support would require threading.Timer which adds complexity)
