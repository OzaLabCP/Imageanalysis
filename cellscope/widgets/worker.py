"""Run work off the UI thread so the interface never freezes.

A ``Task`` wraps any callable as a ``QRunnable`` and reports back via queued
signals (safe across threads). Pass ``with_progress=True`` to inject a
``progress_cb`` keyword that emits integer percentages.
"""

from __future__ import annotations

import traceback
from typing import Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot


class TaskSignals(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(int)


class Task(QRunnable):
    def __init__(self, fn: Callable, *args, with_progress: bool = False, **kwargs) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = TaskSignals()
        if with_progress:
            self.kwargs["progress_cb"] = self.signals.progress.emit

    @Slot()
    def run(self) -> None:
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as exc:  # noqa: BLE001 - report any failure to the UI
            traceback.print_exc()
            self.signals.failed.emit(str(exc))
        else:
            self.signals.finished.emit(result)


def run_async(
    fn: Callable,
    *args,
    on_done: Callable | None = None,
    on_failed: Callable | None = None,
    on_progress: Callable | None = None,
    with_progress: bool = False,
    registry: list | None = None,
    **kwargs,
) -> Task:
    """Schedule ``fn`` on the global thread pool. Returns the Task.

    Pass ``registry`` (a list) to keep the Task alive until it finishes; it is
    appended on start and removed automatically on completion, so the list never
    grows without bound.
    """
    task = Task(fn, *args, with_progress=with_progress or on_progress is not None, **kwargs)
    if on_done is not None:
        task.signals.finished.connect(on_done)
    if on_failed is not None:
        task.signals.failed.connect(on_failed)
    if on_progress is not None:
        task.signals.progress.connect(on_progress)

    if registry is not None:
        registry.append(task)

        def _cleanup(*_args) -> None:
            try:
                registry.remove(task)
            except ValueError:
                pass

        task.signals.finished.connect(_cleanup)
        task.signals.failed.connect(_cleanup)

    QThreadPool.globalInstance().start(task)
    return task
