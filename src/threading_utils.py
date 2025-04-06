import threading
import asyncio
from functools import wraps
from PyQt5.QtCore import QObject, pyqtSignal


class ThreadingSignals(QObject):
    """Signals to communicate between threads and main UI thread"""
    result = pyqtSignal(object)
    error = pyqtSignal(object)
    finished = pyqtSignal()
    progress = pyqtSignal(int)


def run_in_thread(func):
    """
    Decorator to run a function in a separate thread

    Usage:
        @run_in_thread
        def my_function():
            # Long running code here
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        thread = threading.Thread(
            target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread
    return wrapper


def run_async_in_thread(thread_pool=None):
    """
    Decorator to run an async function in a separate thread with its own event loop

    Usage:
        @run_async_in_thread()
        async def my_async_function():
            # Async code here
    """
    def decorator(async_func):
        @wraps(async_func)
        def wrapper(*args, **kwargs):
            def run():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(async_func(*args, **kwargs))
                finally:
                    loop.close()

            if thread_pool:
                return thread_pool.submit(run)
            else:
                thread = threading.Thread(target=run, daemon=True)
                thread.start()
                return thread
        return wrapper
    return decorator


def submit_to_thread_pool(thread_pool, func, *args, **kwargs):
    """
    Submit a function to thread pool and return the future

    Args:
        thread_pool: ThreadPoolExecutor instance
        func: Function to run
        *args, **kwargs: Arguments to pass to func

    Returns:
        concurrent.futures.Future object
    """
    return thread_pool.submit(func, *args, **kwargs)


def setup_qt_connect_async(async_func, callback=None, error_handler=None, finished_callback=None):
    """
    Connect async function to Qt callbacks

    Args:
        async_func: Async function to run
        callback: Function to call with result
        error_handler: Function to call on exception
        finished_callback: Function to call when complete

    Returns:
        ThreadingSignals instance for further connections
    """
    signals = ThreadingSignals()

    if callback:
        signals.result.connect(callback)
    if error_handler:
        signals.error.connect(error_handler)
    if finished_callback:
        signals.finished.connect(finished_callback)

    @run_in_thread
    def executor():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(async_func())
            signals.result.emit(result)
        except Exception as e:
            if error_handler:
                signals.error.emit(e)
        finally:
            loop.close()
            signals.finished.emit()

    return signals
