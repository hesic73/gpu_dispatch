# Implementation Plan

## Completed (v1.0)

### Core Dispatcher
- BaseWorker abstraction with setup/process/cleanup lifecycle
- Protocol messages (TaskSuccess, TaskError, TaskTimeout, SetupFailed, CleanupFailed)
- Automatic backpressure control via bounded queues
- Per-task timeout support (Unix/Linux, signal-based)
- Graceful error handling and worker lifecycle management
- Comprehensive test suite (13 tests)
- Packaging setup (pyproject.toml, installable via pip)

### UI Extension
- RichDispatcher wrapper with live progress display
- Protocol messages extended with worker_id
- Callback signatures updated (task_id, result/error, worker_id)
- on_exit callback for cleanup
- Rich dependency integration
- Overall progress panel and per-GPU status table
- Threading architecture (main thread for UI, background for dispatcher)
- Thread-safe statistics tracking

### Graceful Shutdown
- Signal handling (SIGINT/SIGTERM) for clean Ctrl+C exit
- shutdown() method on Dispatcher
- Graceful worker termination with fallback to terminate/kill
- Queue draining and resource cleanup
- Timeout-based blocking operations to allow interruption

## TODO: Next Steps

### 1. Debug Mode (High Priority)

**Problem**: When developing workers, errors may occur and output needs to be visible. Current UI hides worker stdout/stderr which makes debugging difficult.

**Solution**: Replace `show_ui` flag with `debug` mode.

**API Design**:
```python
dispatcher = RichDispatcher(
    worker_cls=MyWorker,
    gpu_ids=[0, 1, 2, 3],
    debug=False,  # Default: production mode with UI
)
```

**Debug Mode Behavior** (when `debug=True`):
1. **No Rich UI** - to avoid interference with worker output
2. **Redirect worker stdout/stderr** - capture and display all worker output in main process
3. **Fail fast** - stop immediately on first error (instead of continuing)
4. **Detailed error reporting** - print full tracebacks with context

**Normal Mode Behavior** (when `debug=False`):
1. **Show Rich UI** - live progress display
2. **Suppress worker output** - keep UI clean
3. **Continue on error** - process remaining tasks
4. **Summary error reporting** - errors in callback, not stdout

**Implementation Details**:
- Use `sys.stdout`/`sys.stderr` redirection in worker processes
- In debug mode, use a shared Queue to collect worker outputs
- Main process reads from output queue and prints to terminal
- Add `fail_fast` flag to stop dispatcher on first error

### 2. Code Complexity Review (High Priority)

**Current Issues**:
- Shutdown logic is complex (signal handlers, events, timeouts, terminate/kill)
- Multiple layers: Dispatcher → RichDispatcher → threading → multiprocessing
- Queue management and draining is intricate
- Error handling paths are hard to follow

**Goals**:
- Reduce complexity while maintaining functionality
- Make shutdown logic more understandable
- Simplify the interaction between threading and multiprocessing
- Consider if some features can be simplified or removed

**Questions to Address**:
1. Can we simplify the shutdown event propagation?
2. Is the feeder thread necessary or can it be eliminated?
3. Can we reduce the number of queues/events/locks?
4. Is there a simpler architecture that achieves the same goals?
5. Should RichDispatcher directly use multiprocessing instead of wrapping Dispatcher?

**Approach**:
- Document current architecture and control flow
- Identify pain points and unnecessary complexity
- Propose alternative designs
- Discuss trade-offs (complexity vs features)
- Implement chosen design

### 3. Testing and Validation (Medium Priority)

**Items**:
- Test graceful shutdown (Ctrl+C handling)
- Test debug mode output redirection
- Test fail-fast behavior in debug mode
- Stress test with many workers and tasks
- Test edge cases (all workers fail, empty generator, etc.)

### 4. Documentation (Medium Priority)

**Items**:
- Update README with debug mode usage
- Document shutdown behavior
- Add troubleshooting guide
- API reference documentation
- Architecture diagram

## Design Decisions Log

### Completed Decisions

**On-exit callback**:
- Called in finally block (guaranteed execution)
- No parameters - user maintains state themselves via closures
- Executes on both normal completion and interruption

**Shutdown method over parameter**:
- Added `shutdown()` method to Dispatcher
- Cleaner API than passing _shutdown_event parameter
- Allows external control if needed

### Pending Decisions

**Debug vs show_ui**:
- Leaning towards `debug` mode that bundles multiple behaviors
- More intuitive for users than separate flags
- Needs implementation and user feedback
