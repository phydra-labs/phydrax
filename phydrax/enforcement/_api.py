#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence

import equinox as eqx
from jaxtyping import Array, Key

from phydrax.domain import DomainFunction

from .._doc import DOC_KEY0
from .._strict import StrictModule
from ..domain._base import EnforcementGateMethod
from ._compile import EnforcementProgram, InteriorAnchors
from ._spec import EnforcementSpec


class EnforcementOptions(StrictModule):
    """Static policy for compiling staged hard transforms."""

    evolution_var: str = eqx.field(static=True)
    include_identity_remainder: bool = eqx.field(static=True)
    gate_method: EnforcementGateMethod = eqx.field(static=True)
    gate_saturation_fraction: float = eqx.field(static=True)
    gate_linear_fraction: float = eqx.field(static=True)
    num_reference: int = eqx.field(static=True)
    sampler: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        evolution_var: str = "t",
        include_identity_remainder: bool = True,
        gate_method: EnforcementGateMethod = "auto",
        gate_saturation_fraction: float = 0.5,
        gate_linear_fraction: float = 0.5,
        num_reference: int = 3_000_000,
        sampler: str = "latin_hypercube",
    ):
        if gate_method not in ("auto", "global_r_equivalence", "compact"):
            raise ValueError("Unsupported enforcement gate method.")
        saturation = float(gate_saturation_fraction)
        linear = float(gate_linear_fraction)
        if not 0.0 < saturation <= 1.0:
            raise ValueError("gate_saturation_fraction must lie in (0, 1].")
        if not 0.0 < linear <= 1.0:
            raise ValueError("gate_linear_fraction must lie in (0, 1].")
        references = int(num_reference)
        if references <= 0:
            raise ValueError("num_reference must be positive.")
        self.evolution_var = str(evolution_var)
        self.include_identity_remainder = bool(include_identity_remainder)
        self.gate_method = gate_method
        self.gate_saturation_fraction = saturation
        self.gate_linear_fraction = linear
        self.num_reference = references
        self.sampler = str(sampler)


def compile(
    functions: Mapping[str, DomainFunction],
    specs: Sequence[EnforcementSpec],
    /,
    *,
    interior: Sequence[InteriorAnchors] = (),
    options: EnforcementOptions | None = None,
    key: Key[Array, ""] = DOC_KEY0,
) -> EnforcementProgram:
    """Validate and compile hard specifications into one staged program."""
    resolved_functions = dict(functions)
    if not resolved_functions:
        raise ValueError("Hard enforcement requires at least one field.")
    if any(not isinstance(value, DomainFunction) for value in resolved_functions.values()):
        raise TypeError("Every enforced field must be a DomainFunction.")
    resolved_specs = tuple(specs)
    if any(not isinstance(spec, EnforcementSpec) for spec in resolved_specs):
        raise TypeError("specs must contain only EnforcementSpec values.")
    missing_targets = tuple(
        spec.field for spec in resolved_specs if spec.field not in resolved_functions
    )
    if missing_targets:
        raise KeyError(f"Unknown enforcement target fields {missing_targets!r}.")
    missing_dependencies = tuple(
        dependency
        for spec in resolved_specs
        for dependency in spec.dependencies
        if dependency not in resolved_functions
    )
    if missing_dependencies:
        raise KeyError(f"Unknown enforcement dependencies {missing_dependencies!r}.")
    resolved_options = EnforcementOptions() if options is None else options
    if not isinstance(resolved_options, EnforcementOptions):
        raise TypeError("options must be an EnforcementOptions value.")
    return EnforcementProgram.build(
        functions=resolved_functions,
        specs=resolved_specs,
        interior=tuple(interior),
        evolution_var=resolved_options.evolution_var,
        include_identity_remainder=resolved_options.include_identity_remainder,
        gate_method=resolved_options.gate_method,
        gate_saturation_fraction=resolved_options.gate_saturation_fraction,
        gate_linear_fraction=resolved_options.gate_linear_fraction,
        num_reference=resolved_options.num_reference,
        sampler=resolved_options.sampler,
        key=key,
    )


__all__ = ["EnforcementOptions", "compile"]
