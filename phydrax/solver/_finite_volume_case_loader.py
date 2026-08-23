#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from .._strict import StrictModule
from ..discretization import (
    ExtrapolationBoundary,
    FiniteVolumeBoundaryPair,
    FiniteVolumeBoundarySet,
    FiniteVolumeDiscretization,
    FiniteVolumeMethodPlan,
    FiniteVolumePlan,
    FluxPositivityPlan,
    HLLCFluxPlan,
    HLLFluxPlan,
    MUSCLReconstruction,
    PiecewiseConstantReconstruction,
    RusanovFluxPlan,
    TensorGridPlan,
    UniformCellAxisSpec,
    ViscousFluxPlan,
)
from ..equations import (
    compile_conservation_problem,
    CompressibleNavierStokesSystem,
    ConservationProblemIR,
    ConstantTransport,
    EulerSystem,
    IdealGasMaterial,
    SutherlandTransport,
)
from ._finite_volume_case import (
    FiniteVolumeCaseSpec,
    FiniteVolumeExecutionSpec,
    FiniteVolumePrecisionPolicy,
)
from ._finite_volume_runtime import PreparedFiniteVolumeRuntime


def _require_fields(
    payload: dict[str, Any],
    required: set[str],
    path: str,
    /,
) -> None:
    unknown = set(payload).difference(required)
    missing = required.difference(payload)
    if unknown or missing:
        raise ValueError(
            f"{path} has unknown={sorted(unknown)!r}, missing={sorted(missing)!r}."
        )


class PreparedFiniteVolumeCase(StrictModule):
    case: FiniteVolumeCaseSpec
    runtime: PreparedFiniteVolumeRuntime
    discretization: FiniteVolumeDiscretization


def load_finite_volume_case(
    payload: dict[str, Any],
    /,
) -> PreparedFiniteVolumeCase:
    """Build the allowlisted one-dimensional structured FV case schema."""
    _require_fields(
        payload,
        {
            "schema_version",
            "name",
            "grid",
            "equation",
            "method",
            "boundary",
            "execution",
            "precision",
        },
        "case",
    )
    if payload["schema_version"] != 1:
        raise ValueError("case.schema_version must be 1.")
    grid_payload = payload["grid"]
    _require_fields(grid_payload, {"cells", "lower", "upper", "periodic"}, "case.grid")
    cells = int(grid_payload["cells"])
    lower = float(grid_payload["lower"])
    upper = float(grid_payload["upper"])
    periodic = bool(grid_payload["periodic"])
    grid = TensorGridPlan(
        (UniformCellAxisSpec(cells, periodic=periodic),), axis_names=("x",)
    ).prepare(jnp.asarray([[lower], [upper]]))

    equation_payload = payload["equation"]
    equation_type = equation_payload.get("type")
    if equation_type == "ideal_gas_euler":
        _require_fields(
            equation_payload,
            {"type", "gamma", "gas_constant"},
            "case.equation",
        )
        material = IdealGasMaterial(
            float(equation_payload["gamma"]),
            float(equation_payload["gas_constant"]),
        )
        system = EulerSystem(material=material)
        viscous = None
    elif equation_type == "ideal_gas_navier_stokes":
        _require_fields(
            equation_payload,
            {"type", "gamma", "gas_constant", "transport"},
            "case.equation",
        )
        material = IdealGasMaterial(
            float(equation_payload["gamma"]),
            float(equation_payload["gas_constant"]),
        )
        transport_payload = equation_payload["transport"]
        transport_type = transport_payload.get("type")
        if transport_type == "constant":
            _require_fields(
                transport_payload,
                {"type", "viscosity", "conductivity"},
                "case.equation.transport",
            )
            transport = ConstantTransport(
                float(transport_payload["viscosity"]),
                float(transport_payload["conductivity"]),
            )
        elif transport_type == "sutherland":
            _require_fields(
                transport_payload,
                {
                    "type",
                    "reference_viscosity",
                    "reference_temperature",
                    "sutherland_temperature",
                    "specific_heat_cp",
                    "prandtl_number",
                },
                "case.equation.transport",
            )
            transport = SutherlandTransport(
                float(transport_payload["reference_viscosity"]),
                float(transport_payload["reference_temperature"]),
                float(transport_payload["sutherland_temperature"]),
                float(transport_payload["specific_heat_cp"]),
                float(transport_payload["prandtl_number"]),
            )
        else:
            raise ValueError(f"Unsupported transport type {transport_type!r}.")
        system = CompressibleNavierStokesSystem(transport, material=material)
        viscous = ViscousFluxPlan()
    else:
        raise ValueError(f"Unsupported equation type {equation_type!r}.")

    discretization = FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    method_payload = payload["method"]
    _require_fields(method_payload, {"reconstruction", "flux"}, "case.method")
    reconstruction_name = method_payload["reconstruction"]
    if reconstruction_name == "piecewise_constant":
        reconstruction = PiecewiseConstantReconstruction()
    elif reconstruction_name == "muscl":
        reconstruction = MUSCLReconstruction()
    else:
        raise ValueError(f"Unsupported reconstruction {reconstruction_name!r}.")
    flux_name = method_payload["flux"]
    fluxes = {
        "rusanov": RusanovFluxPlan,
        "hll": HLLFluxPlan,
        "hllc": HLLCFluxPlan,
    }
    if flux_name not in fluxes:
        raise ValueError(f"Unsupported numerical flux {flux_name!r}.")
    method = FiniteVolumeMethodPlan(reconstruction, fluxes[flux_name](), viscous=viscous)

    boundary_payload = payload["boundary"]
    _require_fields(boundary_payload, {"type"}, "case.boundary")
    boundary_type = boundary_payload["type"]
    if periodic:
        if boundary_type != "periodic":
            raise ValueError("Periodic grid requires periodic boundary type.")
        boundaries = FiniteVolumeBoundarySet.periodic(("x",))
    else:
        if boundary_type != "extrapolation":
            raise ValueError("The initial loader supports extrapolation on bounded axes.")
        pair = FiniteVolumeBoundaryPair(ExtrapolationBoundary(), ExtrapolationBoundary())
        boundaries = FiniteVolumeBoundarySet(("x",), (pair,))
    precision_payload = payload["precision"]
    _require_fields(precision_payload, {"dtype"}, "case.precision")
    precision = FiniteVolumePrecisionPolicy(precision_payload["dtype"])
    problem = ConservationProblemIR(str(payload["name"]), "state", system, boundaries)
    compiled = compile_conservation_problem(
        problem,
        discretization,
        method,
        precision=precision,
    )

    execution_payload = payload["execution"]
    _require_fields(
        execution_payload,
        {"end_time", "maximum_steps"},
        "case.execution",
    )
    execution = FiniteVolumeExecutionSpec(
        float(execution_payload["end_time"]),
        int(execution_payload["maximum_steps"]),
    )
    runtime = PreparedFiniteVolumeRuntime(
        compiled.dynamics, FluxPositivityPlan(), execution.step_policy
    )
    case = FiniteVolumeCaseSpec(
        str(payload["name"]),
        runtime,
        execution,
        precision=precision,
    )
    return PreparedFiniteVolumeCase(case, runtime, discretization)


__all__ = ["PreparedFiniteVolumeCase", "load_finite_volume_case"]
