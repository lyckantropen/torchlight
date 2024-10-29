from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from tabulate import tabulate

from .forward_timer import ForwardTimer


@dataclass
class ModuleTimingAtom:
    """Stores data about a module's timing and all its children."""

    name: str
    class_: str
    level: int
    times: List[float] = field(default_factory=list)
    children: List[ModuleTimingAtom] = field(default_factory=list)
    child_time: np.float32 = np.float32(0.0)  # time spent in children
    self_time: np.float32 = np.float32(0.0)   # time spent in the module itself (total - children)
    total_time: np.float32 = np.float32(0.0)  # total time spent in the module


class ModelTiming:
    """
    Context manager for timing a model and its transforms.

    This class replaces the forward or __call__ methods of a model and
    preprocess and postprocess transforms with wrappers. After the context
    manager exits, the timing data can be accessed from the `timing_data`
    attribute and the model will be restored to its original state.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 pre_transforms: Sequence[torch.nn.Module],
                 post_transforms: Sequence[torch.nn.Module]
                 ) -> None:
        """
        Initialize the ModelTiming context manager.

        Parameters
        ----------
        model : torch.nn.Module
            The model to time.
        pre_transforms : Sequence[torch.nn.Module]
            The transforms to time before the model.
        post_transforms : Sequence[torch.nn.Module]
            The transforms to time after the model.
        """
        self.model = model
        self.pre_transforms = pre_transforms
        self.post_transforms = post_transforms
        self.forward_timers: List[ForwardTimer] = []
        self.timing_data = ModuleTimingAtom('Pipeline', 'PIPELINE', 0, [], [
            ModuleTimingAtom('Preprocessing', 'PREPROCESSING', 0, [], []),
            ModuleTimingAtom('Model', self.model.__class__.__name__, 0, [], []),
            ModuleTimingAtom('Postprocessing', 'POSTPROCESSING', 0, [], []),
        ])

        self._add_instrumentation(self.model, self.timing_data.children[1], self.forward_timers)
        self._add_instrumentation_transform(self.pre_transforms, self.timing_data.children[0], self.forward_timers)
        self._add_instrumentation_transform(self.post_transforms, self.timing_data.children[2], self.forward_timers)

    @classmethod
    def _add_instrumentation_transform(cls,
                                       transforms: Sequence[torch.nn.Module],
                                       root_container: ModuleTimingAtom,
                                       forward_timers: List[ForwardTimer],
                                       level: int = 0) -> None:
        """Add instrumentation (timing wrappers) to a sequence of transforms."""
        for transform in transforms:
            transform_timing_data = ModuleTimingAtom(transform.__class__.__name__, transform.__class__.__name__, level + 1, [], [])
            root_container.children.append(transform_timing_data)

            forward_timer = ForwardTimer(transform, transform_timing_data.times)

            # replace the forward method of the transform
            if hasattr(transform, 'forward'):
                transform.forward = forward_timer
            elif hasattr(transform, '__call__'):
                transform.__call__ = forward_timer
            else:
                raise ValueError(f"Transform {transform.__class__.__name__} does not have a forward method")

            forward_timers.append(forward_timer)

    @classmethod
    def _add_instrumentation(cls,
                             module: torch.nn.Module,
                             timing_data: ModuleTimingAtom,
                             forward_timers: List[ForwardTimer],
                             name: str = "",
                             level: int = 0
                             ) -> None:
        """Add instrumentation (timing wrappers) to a module and its children."""
        if hasattr(module, 'named_children'):
            for name, child in module.named_children():
                child_timing_data = ModuleTimingAtom(name, child.__class__.__name__, level+1, [], [])
                cls._add_instrumentation(child, child_timing_data, forward_timers, name, level+1)

                forward_timer = ForwardTimer(child, child_timing_data.times)

                # replace the forward method of the transform
                if hasattr(child, 'forward'):
                    child.forward = forward_timer
                elif hasattr(child, '__call__'):
                    child.__call__ = forward_timer
                else:
                    raise ValueError(f"Transform {child.__class__.__name__} does not have a forward method")

                forward_timers.append(forward_timer)
                timing_data.children.append(child_timing_data)

        if hasattr(module, 'forward'):
            module.forward = ForwardTimer(module, timing_data.times)
        elif hasattr(module, '__call__'):
            module.__call__ = ForwardTimer(module, timing_data.times)

    @classmethod
    def _summarize_times(cls, module_timing: ModuleTimingAtom) -> np.float32:
        """Summarize the times of a module and its children."""
        child_times = np.array([cls._summarize_times(child) for child in module_timing.children], np.float32)

        module_timing.child_time = np.sum(child_times) if len(child_times) > 0 else np.float32(0.0)
        if len(module_timing.times) > 0:
            module_timing.self_time = np.mean(module_timing.times) - module_timing.child_time
            module_timing.total_time = np.mean(module_timing.times)
        else:
            module_timing.self_time = np.float32(0.0)
            module_timing.total_time = module_timing.child_time
        return module_timing.total_time

    def __enter__(self) -> ModelTiming:
        """Context manager entry point - enable all timing wrappers."""
        for forward_timer in self.forward_timers:
            forward_timer.enable()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Context manager exit point - disable all timing wrappers and summarize the times."""
        for forward_timer in self.forward_timers:
            forward_timer.disable()
        self.timing_data.total_time = self._summarize_times(self.timing_data)

    def _flat_timing_data(self) -> List[Dict[str, Any]]:
        """Flatten the timing data into a list of dictionaries."""
        flat_timing_data = []
        stack = [self.timing_data]
        while stack:
            current = stack.pop()
            flat_timing_data.append({
                'name': current.name,
                'class': current.class_,
                'total_mean_time[ms]': current.total_time * 1000,
                'child_mean_time[ms]': current.child_time * 1000,
                'self_mean_time[ms]': current.self_time * 1000,
                'times_run': len(current.times)
            })
            for child in current.children:
                stack.append(child)
        return flat_timing_data

    def summarize_table(self, sort_by='self_mean_time[ms]', limit: Optional[int] = None) -> str:
        """Summarize the timing data in a tabular format."""
        data = sorted(self._flat_timing_data(), key=lambda x: x[sort_by], reverse=True)
        if limit is not None:
            data = data[:limit]
        return tabulate(data, headers='keys')

    def summarize_tree(self) -> str:
        """Summarize the timing data in a tree format."""
        def print_structure(s: ModuleTimingAtom, level: int = 0) -> str:
            repr = ''.join([". "] * level) + s.name + f' ({s.class_})' + f" took {s.total_time * 1000:.2f}ms (self: {s.self_time * 1000:.2f}ms)\n"
            for child in s.children:
                repr += print_structure(child, level + 1)
            return repr

        return print_structure(self.timing_data)
