import time
from typing import Any, Callable, List

import torch


class ForwardTimer:
    def __init__(self, module: torch.nn.Module, times: List[float]) -> None:
        self.module = module
        if hasattr(module, 'forward'):
            self._true_forward = module.forward
        elif hasattr(module, '__call__'):
            self._true_forward: Callable = module.__call__
        self._times: List[float] = times

    def __call__(self, *args, **kwargs) -> Any:
        start = time.perf_counter()
        result = self._true_forward(*args, **kwargs)
        end = time.perf_counter()

        self._times.append(end - start)

        return result

    def enable(self) -> None:
        if hasattr(self.module, 'forward'):
            self.module.forward = self
        elif hasattr(self.module, '__call__'):
            self.module.__call__ = self

    def disable(self) -> None:
        if hasattr(self.module, 'forward'):
            self.module.forward = self._true_forward
        elif hasattr(self.module, '__call__'):
            self.module.__call__ = self._true_forward
