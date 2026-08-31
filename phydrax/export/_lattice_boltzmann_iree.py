#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

import equinox as eqx
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from ..backends._types import BackendAvailability
from ..backends.iree import iree_availability
from ..discretization.lattice_boltzmann._execution import (
    ReferenceLatticeBoltzmannExecutionPlan,
)
from ..discretization.lattice_boltzmann._fused import (
    FusedLatticeBoltzmannExecutionPlan,
)
from ._iree import IREEExportPolicy, IREEExportResult, save_iree


LatticeBoltzmannIREEExportMode: TypeAlias = Literal["forward-only"]
_LBMExportPlan = (
    ReferenceLatticeBoltzmannExecutionPlan | FusedLatticeBoltzmannExecutionPlan
)


@dataclass(frozen=True, slots=True)
class LatticeBoltzmannIREEForwardContract:
    """Static forward realization ABI; reverse mode is intentionally unavailable."""

    execution_mode: LatticeBoltzmannIREEExportMode
    execution_plan_id: str
    lattice_id: str
    input_shape: tuple[int, ...]
    input_dtype: str
    step_count: int
    step_size: float
    initial_time: float
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    supports_reverse_mode: bool
    contract_id: str

    def pack_inputs(self, populations: Array, /) -> tuple[Array]:
        if tuple(populations.shape) != self.input_shape:
            raise ValueError("LBM IREE population input shape does not match contract.")
        if np.dtype(populations.dtype).str != self.input_dtype:
            raise TypeError("LBM IREE population input dtype does not match contract.")
        return (populations,)

    def unpack_outputs(self, populations: Array, /) -> tuple[Array]:
        if tuple(populations.shape) != self.input_shape:
            raise ValueError("LBM IREE population output shape does not match contract.")
        if np.dtype(populations.dtype).str != self.input_dtype:
            raise TypeError("LBM IREE population output dtype does not match contract.")
        return (populations,)


@dataclass(frozen=True, slots=True)
class LatticeBoltzmannIREEExportResult:
    """Existing IREE artifact plus the LBM-specific forward-only contract."""

    artifact: IREEExportResult
    contract: LatticeBoltzmannIREEForwardContract

    @property
    def path(self) -> Path:
        return self.artifact.path


def lattice_boltzmann_iree_availability() -> BackendAvailability:
    """Return the existing matched IREE compiler/runtime availability evidence."""

    return iree_availability()


def prepare_lattice_boltzmann_iree_contract(
    plan: _LBMExportPlan,
    initial_populations: Array,
    /,
    *,
    step_count: int,
    step_size: float,
    t0: float = 0.0,
    mode: LatticeBoltzmannIREEExportMode = "forward-only",
) -> LatticeBoltzmannIREEForwardContract:
    """Prepare a static export ABI without importing or invoking IREE."""

    if not isinstance(
        plan,
        (ReferenceLatticeBoltzmannExecutionPlan, FusedLatticeBoltzmannExecutionPlan),
    ):
        raise TypeError("IREE LBM export requires a reference or fused execution plan.")
    if mode != "forward-only":
        raise ValueError("IREE LBM export is explicitly forward-only.")
    if not eqx.is_array(initial_populations):
        raise TypeError("IREE LBM export accepts JAX input arrays only.")
    velocity_set = plan.velocity_set
    if (
        initial_populations.ndim != velocity_set.dimension + 1
        or initial_populations.shape[-1] != velocity_set.population_count
    ):
        raise ValueError("IREE LBM input must use the velocity set's trailing-Q shape.")
    count = int(step_count)
    dt = float(step_size)
    initial_time = float(t0)
    if count <= 0 or not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("IREE LBM export requires positive step_count and step_size.")
    if not math.isfinite(initial_time):
        raise ValueError("IREE LBM initial time must be finite.")
    metadata = {
        "kind": "lattice-boltzmann-iree-forward-contract",
        "execution_mode": "forward-only",
        "execution_plan": plan.plan_id,
        "lattice": velocity_set.lattice_id,
        "input_shape": list(initial_populations.shape),
        "input_dtype": np.dtype(initial_populations.dtype).str,
        "step_count": count,
        "step_size": dt,
        "initial_time": initial_time,
        "supports_reverse_mode": False,
        "input_names": ("populations",),
        "output_names": ("final_populations",),
    }
    return LatticeBoltzmannIREEForwardContract(
        "forward-only",
        plan.plan_id,
        velocity_set.lattice_id,
        tuple(int(size) for size in initial_populations.shape),
        np.dtype(initial_populations.dtype).str,
        count,
        dt,
        initial_time,
        ("populations",),
        ("final_populations",),
        False,
        canonical_fingerprint(metadata),
    )


def save_lattice_boltzmann_iree(
    plan: _LBMExportPlan,
    path: str | Path,
    /,
    *,
    initial_populations: Array,
    step_count: int,
    step_size: float,
    args: Any = None,
    t0: float = 0.0,
    mode: LatticeBoltzmannIREEExportMode = "forward-only",
    policy: IREEExportPolicy | None = None,
    validate: bool = True,
    rtol: float = 1.0e-4,
    atol: float = 1.0e-6,
) -> LatticeBoltzmannIREEExportResult:
    """Export one forward realization through the existing checksummed IREE path."""

    contract = prepare_lattice_boltzmann_iree_contract(
        plan,
        initial_populations,
        step_count=step_count,
        step_size=step_size,
        t0=t0,
        mode=mode,
    )
    lattice_boltzmann_iree_availability().require("compiled-inference")

    def forward(populations, *, key=None):
        if key is not None:
            raise ValueError("IREE LBM forward export requires key=None.")
        return plan.realize(
            populations,
            step_count=contract.step_count,
            step_size=contract.step_size,
            args=args,
            t0=contract.initial_time,
        ).final_populations

    artifact = save_iree(
        forward,
        path,
        inputs=contract.pack_inputs(initial_populations),
        input_names=contract.input_names,
        policy=policy,
        key=None,
        validate=validate,
        rtol=rtol,
        atol=atol,
    )
    return LatticeBoltzmannIREEExportResult(artifact, contract)


__all__ = [
    "LatticeBoltzmannIREEExportMode",
    "LatticeBoltzmannIREEExportResult",
    "LatticeBoltzmannIREEForwardContract",
    "lattice_boltzmann_iree_availability",
    "prepare_lattice_boltzmann_iree_contract",
    "save_lattice_boltzmann_iree",
]
