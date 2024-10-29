import time
from typing import Any, Callable, List

import torch


class ForwardTimer:
    def __init__(self, module: torch.nn.Module, times: List[float]) -> None:
        self.module = module
        self._true_forward: Callable = module.forward
        self._times: List[float] = times

    def __call__(self, *args, **kwargs) -> Any:
        start = time.perf_counter()
        result = self._true_forward(*args, **kwargs)
        end = time.perf_counter()

        self._times.append(end - start)

        return result

    def enable(self) -> None:
        self.module.forward = self

    def disable(self) -> None:
        self.module.forward = self._true_forward
