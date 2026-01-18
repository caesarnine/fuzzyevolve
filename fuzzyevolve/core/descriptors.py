from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol


class Axis(Protocol):
    def key_for(self, value: Any) -> Any: ...

    @property
    def size(self) -> int: ...


@dataclass(frozen=True, slots=True)
class CategoricalAxis:
    values: tuple[Any, ...]

    def key_for(self, value: Any) -> Any:
        if value not in self.values:
            raise ValueError(f"Descriptor value '{value}' not in {self.values}.")
        return value

    @property
    def size(self) -> int:
        return len(self.values)


@dataclass(frozen=True, slots=True)
class BinnedAxis:
    bins: tuple[float, ...]

    def key_for(self, value: float) -> int:
        if len(self.bins) < 2:
            raise ValueError("Binned axis must define at least two bins.")
        if value >= self.bins[-1]:
            return len(self.bins) - 2
        for idx in range(len(self.bins) - 1):
            if self.bins[idx] <= value < self.bins[idx + 1]:
                return idx
        return 0

    @property
    def size(self) -> int:
        return len(self.bins) - 1


@dataclass(frozen=True, slots=True)
class DescriptorSpace:
    axes: Mapping[str, Axis]

    def cell_key(self, descriptor: Mapping[str, Any]) -> tuple[Any, ...]:
        key: list[Any] = []
        for name, axis in self.axes.items():
            if name not in descriptor:
                raise KeyError(f"Missing descriptor value for axis '{name}'.")
            key.append(axis.key_for(descriptor[name]))
        return tuple(key)

    @property
    def total_cells(self) -> int:
        total = 1
        for axis in self.axes.values():
            total *= axis.size
        return total


def build_descriptor_space(spec: Mapping[str, Any]) -> DescriptorSpace:
    axes: dict[str, Axis] = {}
    for name, axis_spec in spec.items():
        if isinstance(axis_spec, list):
            axes[name] = CategoricalAxis(tuple(axis_spec))
        elif isinstance(axis_spec, dict) and "bins" in axis_spec:
            bins = tuple(float(v) for v in axis_spec["bins"])
            axes[name] = BinnedAxis(bins)
        else:
            raise ValueError(
                "Axis spec must be a list of categories or a dict with 'bins'."
            )
    return DescriptorSpace(axes=axes)
