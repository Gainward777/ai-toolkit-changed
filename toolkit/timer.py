import time
from collections import OrderedDict, deque
import sys
import os

# check if is ui process will have IS_AI_TOOLKIT_UI in env
is_ui = os.environ.get("IS_AI_TOOLKIT_UI", "0") == "1"

class Timer:
    def __init__(self, name='Timer', max_buffer=10):
        self.name = name
        self.max_buffer = max_buffer
        self.timers = OrderedDict()
        self.active_timers = {}
        self.current_timer = None  # Used for the context manager functionality
        # Support nested `with timer('x'):` by tracking a stack of active context timers.
        # Without this, nested timers overwrite `current_timer` and break on __exit__.
        self._timer_stack = []
        self._after_print_hooks = []

    def start(self, timer_name):
        if timer_name not in self.timers:
            self.timers[timer_name] = deque(maxlen=self.max_buffer)
        self.active_timers[timer_name] = time.time()

    def cancel(self, timer_name):
        """Cancel an active timer."""
        if timer_name in self.active_timers:
            del self.active_timers[timer_name]

    def stop(self, timer_name):
        if timer_name not in self.active_timers:
            raise ValueError(f"Timer '{timer_name}' was not started!")

        elapsed_time = time.time() - self.active_timers[timer_name]
        self.timers[timer_name].append(elapsed_time)

        # Clean up active timers
        del self.active_timers[timer_name]

        # Check if this timer's buffer exceeds max_buffer and remove the oldest if it does
        if len(self.timers[timer_name]) > self.max_buffer:
            self.timers[timer_name].popleft()

    def add_after_print_hook(self, hook):
        self._after_print_hooks.append(hook)

    def print(self):
        # Print timings even in UI mode so users can see breakdown in logs.
        print(f"\nTimer '{self.name}':")
        timing_dict = {}
        # sort by longest at top
        for timer_name, timings in sorted(self.timers.items(), key=lambda x: sum(x[1]), reverse=True):
            avg_time = sum(timings) / len(timings)

            print(f" - {avg_time:.4f}s avg - {timer_name}, num = {len(timings)}")
            timing_dict[timer_name] = avg_time

        for hook in self._after_print_hooks:
            hook(timing_dict)
        print('')

    def reset(self):
        self.timers.clear()
        self.active_timers.clear()
        self._timer_stack.clear()

    def __call__(self, timer_name):
        """Enable the use of the Timer class as a context manager."""
        self._timer_stack.append(timer_name)
        self.current_timer = timer_name
        self.start(timer_name)
        return self

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore previous timer context (nested timers are common in training loop).
        timer_name = self._timer_stack.pop() if self._timer_stack else self.current_timer
        if exc_type is None:
            # No exceptions, stop the timer normally
            self.stop(timer_name)
        else:
            # There was an exception, cancel the timer
            self.cancel(timer_name)
        self.current_timer = self._timer_stack[-1] if self._timer_stack else None
