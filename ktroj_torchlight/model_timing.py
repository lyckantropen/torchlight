from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from tabulate import tabulate

from .forward_timer import ForwardTimer


@dataclass
class ModuleTimingAtom:
    name: str
    class_: str
    level: int
    times: List[float] = field(default_factory=list)
    children: List[ModuleTimingAtom] = field(default_factory=list)
    cumulative_time: np.float32 = np.float32(0.0)
    self_time: np.float32 = np.float32(0.0)
    total_time: np.float32 = np.float32(0.0)


class ModelTiming:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.forward_timers: List[ForwardTimer] = []
        self.timing_data = ModuleTimingAtom('Model', self.model.__class__.__name__, 0, [], [])

        self._add_instrumentation(self.model, self.timing_data, self.forward_timers)

    @classmethod
    def _add_instrumentation(cls,
                             module: torch.nn.Module,
                             timing_data: ModuleTimingAtom,
                             forward_timers: List[ForwardTimer],
                             name: str = "",
                             level: int = 0
                             ) -> None:
        for name, child in module.named_children():
            child_timing_data = ModuleTimingAtom(name, child.__class__.__name__, level+1, [], [])
            cls._add_instrumentation(child, child_timing_data, forward_timers, name, level+1)
            child.forward = ForwardTimer(child, child_timing_data.times)
            forward_timers.append(child.forward)
            timing_data.children.append(child_timing_data)

    @classmethod
    def _summarize_times(cls, module_timing: ModuleTimingAtom) -> np.float32:
        child_times = np.array([cls._summarize_times(child) for child in module_timing.children], np.float32)

        module_timing.cumulative_time = np.sum(child_times) if len(child_times) > 0 else np.float32(0.0)
        if len(module_timing.times) > 0:
            module_timing.self_time = np.mean(module_timing.times) - module_timing.cumulative_time
            module_timing.total_time = np.mean(module_timing.times)
        else:
            module_timing.self_time = np.float32(0.0)
            module_timing.total_time = module_timing.cumulative_time
        return module_timing.total_time

    def __enter__(self) -> ModelTiming:
        for forward_timer in self.forward_timers:
            forward_timer.enable()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for forward_timer in self.forward_timers:
            forward_timer.disable()
        self.timing_data.total_time = self._summarize_times(self.timing_data)

    def _flat_timing_data(self) -> List[Dict[str, Any]]:
        flat_timing_data = []
        stack = [self.timing_data]
        while stack:
            current = stack.pop()
            flat_timing_data.append({
                'name': current.name,
                'class': current.class_,
                'total_time[ms]': current.total_time * 1000,
                'cumulative_time[ms]': current.cumulative_time * 1000,
                'self_time[ms]': current.self_time * 1000
            })
            for child in current.children:
                stack.append(child)
        return flat_timing_data

    def summarize_table(self, sort_by='self_time[ms]') -> str:
        return tabulate(sorted(self._flat_timing_data(), key=lambda x: x[sort_by], reverse=True), headers='keys')

    def summarize_tree(self) -> str:
        def print_structure(s: ModuleTimingAtom, level: int = 0) -> str:
            repr = ''.join([". "] * level) + s.name + f' ({s.class_})' + f" took {s.total_time * 1000:.2f} ms (self: {s.self_time * 1000:.2f} ms)\n"
            for child in s.children:
                repr += print_structure(child, level + 1)
            return repr

        return print_structure(self.timing_data)


class ModelTimingInner:
    def __init__(self, timing: ModelTiming) -> None:
        self.timing = timing
        self._self_begin = 0.0

    def __enter__(self) -> ModelTimingInner:
        self._self_begin = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        end = time.perf_counter()
        self.timing.timing_data.times.append(end - self._self_begin)
