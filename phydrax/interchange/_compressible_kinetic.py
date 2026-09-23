#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..discretization.discrete_velocity import (
    CompressibleKineticRuntimePlan,
    d3q33_filtered_rule,
    entropic_d3q343_plan,
    FilteredD3Q33Plan,
    FullRangeQuasiEquilibriumPlan,
    guided_d3q39_plan,
    IntegerLatticeTransportPlan,
)


class CompressibleKineticCaseIR(StrictModule):
    grid_shape: tuple[int, int, int] = eqx.field(static=True)
    model: str = eqx.field(static=True)
    collision: str = eqx.field(static=True)
    gamma: float = eqx.field(static=True)
    gas_constant: float = eqx.field(static=True)
    prandtl_number: float = eqx.field(static=True)
    relaxation_rate: float = eqx.field(static=True)
    time_step: float = eqx.field(static=True)
    dtype: str = eqx.field(static=True)
    case_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        grid_shape: tuple[int, int, int],
        model: str,
        collision: str,
        gamma: float,
        gas_constant: float,
        prandtl_number: float,
        relaxation_rate: float,
        time_step: float,
        dtype: str = "float64",
    ):
        shape = tuple(int(value) for value in grid_shape)
        if len(shape) != 3 or any(value < 1 for value in shape):
            raise ValueError("grid_shape must contain three positive extents.")
        if model not in ("guided-d3q39", "entropic-d3q343", "filtered-d3q33"):
            raise ValueError(f"Unknown compressible kinetic model {model!r}.")
        if collision not in ("bgk", "entropic", "quasi-equilibrium", "filtered-mrt"):
            raise ValueError(f"Unknown compressible kinetic collision {collision!r}.")
        values = tuple(
            float(value)
            for value in (
                gamma,
                gas_constant,
                prandtl_number,
                relaxation_rate,
                time_step,
            )
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Kinetic case physical/numerical values must be positive.")
        if dtype not in ("float32", "float64"):
            raise ValueError("Kinetic case dtype must be float32 or float64.")
        self.grid_shape = shape
        self.model = model
        self.collision = collision
        self.dtype = dtype
        (
            self.gamma,
            self.gas_constant,
            self.prandtl_number,
            self.relaxation_rate,
            self.time_step,
        ) = values
        self.case_id = canonical_fingerprint(
            {
                "kind": "compressible-kinetic-case",
                "grid_shape": list(shape),
                "model": model,
                "collision": collision,
                "gamma": values[0],
                "gas_constant": values[1],
                "prandtl_number": values[2],
                "relaxation_rate": values[3],
                "time_step": values[4],
                "dtype": dtype,
            }
        )

    def compile(self) -> tuple[Any, CompressibleKineticRuntimePlan | None]:
        if self.model == "guided-d3q39":
            collision = "entropic" if self.collision == "entropic" else "bgk"
            model = guided_d3q39_plan(
                gamma=self.gamma,
                gas_constant=self.gas_constant,
                collision_kind=collision,
                dtype=np.dtype(self.dtype),
            )
        elif self.model == "entropic-d3q343":
            model = entropic_d3q343_plan(
                gamma=self.gamma,
                gas_constant=self.gas_constant,
                dtype=np.dtype(self.dtype),
            )
        else:
            filtered = FilteredD3Q33Plan(
                d3q33_filtered_rule(dtype=np.dtype(self.dtype)),
                gas_constant=self.gas_constant,
            )
            return filtered, None
        transport = IntegerLatticeTransportPlan(model.rule, self.grid_shape)
        quasi = (
            FullRangeQuasiEquilibriumPlan(model, prandtl_number=self.prandtl_number)
            if self.collision == "quasi-equilibrium"
            else None
        )
        return model, CompressibleKineticRuntimePlan(
            model,
            transport,
            time_step=self.time_step,
            quasi_equilibrium=quasi,
        )


_ALLOWED_CASE_KEYS = frozenset(
    {
        "grid_shape",
        "model",
        "collision",
        "gamma",
        "gas_constant",
        "prandtl_number",
        "relaxation_rate",
        "time_step",
        "dtype",
    }
)


def read_compressible_kinetic_yaml(path: str | Path, /) -> CompressibleKineticCaseIR:
    source = Path(path).expanduser().absolute()
    if importlib.util.find_spec("yaml") is None:
        raise RuntimeError("YAML import requires the optional PyYAML dependency.")
    yaml = importlib.import_module("yaml")
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError("Compressible kinetic YAML must contain one mapping.")
    unknown = set(document) - _ALLOWED_CASE_KEYS
    missing = _ALLOWED_CASE_KEYS - set(document)
    if unknown or missing:
        raise ValueError(
            f"Compressible kinetic YAML fields mismatch: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}."
        )
    return CompressibleKineticCaseIR(
        grid_shape=tuple(document["grid_shape"]),
        model=str(document["model"]),
        collision=str(document["collision"]),
        gamma=float(document["gamma"]),
        gas_constant=float(document["gas_constant"]),
        prandtl_number=float(document["prandtl_number"]),
        relaxation_rate=float(document["relaxation_rate"]),
        time_step=float(document["time_step"]),
        dtype=str(document["dtype"]),
    )


class LegacyKineticBoundaryImport(StrictModule):
    wall_mask: Array
    normal_indices: Array
    extra_tags: Array
    unsupported_tags: Array
    source_id: str = eqx.field(static=True)


def import_packed_kinetic_boundary(
    values: ArrayLike,
    /,
    *,
    reject_extra_tags: bool = True,
) -> LegacyKineticBoundaryImport:
    """Decode a documented packed wall/normal raster into native arrays."""
    raw = np.asarray(values)
    if raw.ndim not in (2, 3) or raw.dtype.kind not in "ui":
        raise ValueError("Packed boundary input must be a 2D/3D unsigned integer array.")
    packed = raw.astype(np.uint32, copy=False)
    kind = (packed >> np.uint32(29)) & np.uint32(0x7)
    normals = (packed >> np.uint32(24)) & np.uint32(0x1F)
    tags = packed & np.uint32(0x00FFFFFF)
    unsupported = tags != 0
    if reject_extra_tags and np.any(unsupported):
        raise ValueError("Packed boundary contains unsupported extra tags.")
    return LegacyKineticBoundaryImport(
        wall_mask=jnp.asarray(kind != 0),
        normal_indices=jnp.asarray(normals, dtype=jnp.int32),
        extra_tags=jnp.asarray(tags, dtype=jnp.uint32),
        unsupported_tags=jnp.asarray(unsupported),
        source_id=canonical_fingerprint(
            {
                "kind": "packed-kinetic-boundary-import",
                "shape": list(packed.shape),
                "dtype": packed.dtype.str,
                "values": packed.tolist(),
            }
        ),
    )


__all__ = [
    "CompressibleKineticCaseIR",
    "LegacyKineticBoundaryImport",
    "import_packed_kinetic_boundary",
    "read_compressible_kinetic_yaml",
]
