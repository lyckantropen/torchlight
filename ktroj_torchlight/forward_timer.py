import time
from typing import Any, Callable, List

import torch


class ForwardTimer:
    """
    A class to time the forward pass of a module.

    Designed to replace the forward or __call__ method of a module with a timing
    wrapper.
    """

    def __init__(self, module: torch.nn.Module, times: List[float]) -> None:
        """
        Initialize the ForwardTimer.

        Parameters
        ----------
        module : torch.nn.Module
            The module to time.
        times : List[float]
            A list to store the times of each forward pass.
        """
        self.module = module
        self._true_forward: Callable
        if hasattr(module, 'forward'):
            self._true_forward = module.forward
        elif hasattr(module, '__call__'):
            self._true_forward = module.__call__
        self._times: List[float] = times

    def __call__(self, *args, **kwargs) -> Any:
        start = time.perf_counter()
        result = self._true_forward(*args, **kwargs)
        end = time.perf_counter()

        self._times.append(end - start)

        return result

    def enable(self) -> None:
        """Enable the timing wrapper."""
        if hasattr(self.module, 'forward'):
            self.module.forward = self
        elif hasattr(self.module, '__call__'):
            self.module.__call__ = self

    def disable(self) -> None:
        """Disable the timing wrapper."""
        if hasattr(self.module, 'forward'):
            self.module.forward = self._true_forward
        elif hasattr(self.module, '__call__'):
            self.module.__call__ = self._true_forward
