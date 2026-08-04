#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
from jaxtyping import Array, Key

from .._strict import StrictModule
from ._bsde import BSDEPathBatch


def _shape(value: Sequence[int], /, *, owner: str) -> tuple[int, ...]:
    shape = tuple(int(size) for size in value)
    if not shape or any(size <= 0 for size in shape):
        raise ValueError(f"{owner} must contain positive dimensions.")
    return shape


def _name(value: str, /, *, owner: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


class ReflectedPathDependentBSDEProblem(StrictModule):
    """Brownian BSDE with nonanticipative path features and hard obstacles.

    Every callback receives a path prefix ending at the evaluation time. ``path_features``
    maps that prefix to the finite regression state consumed by a least-squares basis.
    At least one of ``lower_obstacle`` and ``upper_obstacle`` is required.
    """

    forward_sampler: Callable[[Array], BSDEPathBatch]
    path_features: Callable[[Array, Array, Any], Array]
    generator: Callable[[Array, Array, Array, Array, Array, Any], Array]
    terminal: Callable[[Array, Array, Any], Array]
    lower_obstacle: Callable[[Array, Array, Array, Any], Array] | None
    upper_obstacle: Callable[[Array, Array, Array, Any], Array] | None
    args: Any
    state_shape: tuple[int, ...] = eqx.field(static=True)
    noise_shape: tuple[int, ...] = eqx.field(static=True)
    regression_state_shape: tuple[int, ...] = eqx.field(static=True)
    output_shape: tuple[int, ...] = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    process_id: str = eqx.field(static=True)

    def __init__(
        self,
        forward_sampler: Callable[[Array], BSDEPathBatch],
        path_features: Callable[[Array, Array, Any], Array],
        generator: Callable[[Array, Array, Array, Array, Array, Any], Array],
        terminal: Callable[[Array, Array, Any], Array],
        /,
        *,
        state_shape: Sequence[int],
        noise_shape: Sequence[int],
        regression_state_shape: Sequence[int],
        output_shape: Sequence[int],
        problem_id: str,
        process_id: str,
        lower_obstacle: Callable[[Array, Array, Array, Any], Array] | None = None,
        upper_obstacle: Callable[[Array, Array, Array, Any], Array] | None = None,
        args: Any = None,
    ):
        for owner, value in (
            ("forward_sampler", forward_sampler),
            ("path_features", path_features),
            ("generator", generator),
            ("terminal", terminal),
        ):
            if not callable(value):
                raise TypeError(f"{owner} must be callable.")
        if lower_obstacle is None and upper_obstacle is None:
            raise ValueError("At least one reflected obstacle is required.")
        if lower_obstacle is not None and not callable(lower_obstacle):
            raise TypeError("lower_obstacle must be callable or None.")
        if upper_obstacle is not None and not callable(upper_obstacle):
            raise TypeError("upper_obstacle must be callable or None.")
        self.forward_sampler = forward_sampler
        self.path_features = path_features
        self.generator = generator
        self.terminal = terminal
        self.lower_obstacle = lower_obstacle
        self.upper_obstacle = upper_obstacle
        self.args = args
        self.state_shape = _shape(state_shape, owner="state_shape")
        self.noise_shape = _shape(noise_shape, owner="noise_shape")
        self.regression_state_shape = _shape(
            regression_state_shape, owner="regression_state_shape"
        )
        self.output_shape = _shape(output_shape, owner="output_shape")
        self.problem_id = _name(problem_id, owner="problem_id")
        self.process_id = _name(process_id, owner="process_id")

    @property
    def has_lower_obstacle(self) -> bool:
        return self.lower_obstacle is not None

    @property
    def has_upper_obstacle(self) -> bool:
        return self.upper_obstacle is not None

    def sample(self, key: Key[Array, ""], /) -> BSDEPathBatch:
        paths = self.forward_sampler(key)
        if not isinstance(paths, BSDEPathBatch):
            raise TypeError("forward_sampler must return a BSDEPathBatch.")
        if paths.state_shape != self.state_shape or paths.noise_shape != self.noise_shape:
            raise ValueError("Forward path state/noise shapes do not match the problem.")
        if paths.process_id != self.process_id:
            raise ValueError("Forward path and reflected BSDE process IDs do not match.")
        return paths


__all__ = ["ReflectedPathDependentBSDEProblem"]
