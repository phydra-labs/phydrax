#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._strict import StrictModule
from ..discretization import (
    CellPolynomialReconstructionPlan,
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
    read_unstructured_fv_archive,
    RusanovFluxPlan,
    TensorGridPlan,
    UniformCellAxisSpec,
    UnstructuredFiniteVolumeBoundarySet,
    UnstructuredFiniteVolumeDiscretization,
    UnstructuredFiniteVolumeMethodPlan,
    UnstructuredWENOZReconstructionPlan,
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


def _require_fields(payload: dict[str, Any], required: set[str], path: str, /) -> None:
    unknown = set(payload).difference(required)
    missing = required.difference(payload)
    if unknown or missing:
        raise ValueError(
            f"{path} has unknown={sorted(unknown)!r}, missing={sorted(missing)!r}."
        )


def _sha256(path: Path, /) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PreparedFiniteVolumeCase(StrictModule):
    case: FiniteVolumeCaseSpec
    runtime: PreparedFiniteVolumeRuntime
    discretization: FiniteVolumeDiscretization | UnstructuredFiniteVolumeDiscretization
    initial_state: Array | None
    mesh_archive_path: str = eqx.field(static=True)


def _execution(payload: dict[str, Any], /) -> FiniteVolumeExecutionSpec:
    _require_fields(payload, {"end_time", "maximum_steps"}, "case.execution")
    return FiniteVolumeExecutionSpec(
        float(payload["end_time"]), int(payload["maximum_steps"])
    )


def _precision(payload: dict[str, Any], /) -> FiniteVolumePrecisionPolicy:
    _require_fields(payload, {"dtype"}, "case.precision")
    return FiniteVolumePrecisionPolicy(payload["dtype"])


def _load_structured_case_v1(payload: dict[str, Any], /) -> PreparedFiniteVolumeCase:
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
    fluxes = {
        "rusanov": RusanovFluxPlan,
        "hll": HLLFluxPlan,
        "hllc": HLLCFluxPlan,
    }
    flux_name = method_payload["flux"]
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
            raise ValueError("Bounded structured axes require extrapolation here.")
        boundaries = FiniteVolumeBoundarySet(
            ("x",),
            (FiniteVolumeBoundaryPair(ExtrapolationBoundary(), ExtrapolationBoundary()),),
        )
    precision = _precision(payload["precision"])
    problem = ConservationProblemIR(str(payload["name"]), "state", system, boundaries)
    compiled = compile_conservation_problem(
        problem, discretization, method, precision=precision
    )
    execution = _execution(payload["execution"])
    runtime = PreparedFiniteVolumeRuntime(
        compiled.dynamics, FluxPositivityPlan(), execution.step_policy
    )
    case = FiniteVolumeCaseSpec(
        str(payload["name"]), runtime, execution, precision=precision
    )
    return PreparedFiniteVolumeCase(case, runtime, discretization, None, "")


def _load_unstructured_case_v2(
    payload: dict[str, Any], source_path: str | Path | None, /
) -> PreparedFiniteVolumeCase:
    _require_fields(
        payload,
        {
            "schema_version",
            "name",
            "mesh",
            "equation",
            "method",
            "boundary",
            "execution",
            "precision",
            "initial_state",
        },
        "case",
    )
    mesh_payload = payload["mesh"]
    _require_fields(mesh_payload, {"path", "sha256"}, "case.mesh")
    base = Path.cwd() if source_path is None else Path(source_path).resolve().parent
    mesh_path = Path(mesh_payload["path"])
    if not mesh_path.is_absolute():
        mesh_path = base / mesh_path
    mesh_path = mesh_path.resolve()
    if _sha256(mesh_path) != str(mesh_payload["sha256"]):
        raise ValueError("case.mesh.sha256 does not match the mesh archive.")
    mesh_plan = read_unstructured_fv_archive(mesh_path)

    equation_payload = payload["equation"]
    _require_fields(
        equation_payload,
        {"type", "gamma", "gas_constant"},
        "case.equation",
    )
    if equation_payload["type"] != "ideal_gas_euler":
        raise ValueError("Unstructured case schema currently supports ideal_gas_euler.")
    material = IdealGasMaterial(
        float(equation_payload["gamma"]),
        float(equation_payload["gas_constant"]),
    )
    system = EulerSystem(mesh_plan.cell_dimension, material=material)
    if mesh_plan.component_names != system.component_names:
        raise ValueError("Mesh archive component names do not match the equation.")
    discretization = mesh_plan.prepare()

    method_payload = payload["method"]
    _require_fields(method_payload, {"reconstruction", "flux"}, "case.method")
    reconstruction_name = method_payload["reconstruction"]
    if reconstruction_name == "piecewise_constant":
        reconstruction = PiecewiseConstantReconstruction()
    elif reconstruction_name == "polynomial_degree_1":
        reconstruction = CellPolynomialReconstructionPlan(1).prepare(discretization)
    elif reconstruction_name == "polynomial_degree_2":
        reconstruction = CellPolynomialReconstructionPlan(2).prepare(discretization)
    elif reconstruction_name == "weno_z_degree_2":
        reconstruction = UnstructuredWENOZReconstructionPlan().prepare(discretization)
    else:
        raise ValueError(f"Unsupported reconstruction {reconstruction_name!r}.")
    fluxes = {
        "rusanov": RusanovFluxPlan,
        "hll": HLLFluxPlan,
        "hllc": HLLCFluxPlan,
    }
    flux_name = method_payload["flux"]
    if flux_name not in fluxes:
        raise ValueError(f"Unsupported numerical flux {flux_name!r}.")
    method = UnstructuredFiniteVolumeMethodPlan(reconstruction, fluxes[flux_name]())

    boundary_payload = payload["boundary"]
    if not isinstance(boundary_payload, dict) or set(boundary_payload) != set(
        discretization.boundary_patch_names
    ):
        raise ValueError("case.boundary must cover every mesh patch exactly.")
    boundary_values = {}
    for name in discretization.boundary_patch_names:
        declaration = boundary_payload[name]
        _require_fields(declaration, {"type"}, f"case.boundary.{name}")
        if declaration["type"] != "extrapolation":
            raise ValueError(
                "Unstructured case boundaries currently allow extrapolation."
            )
        boundary_values[name] = ExtrapolationBoundary()
    boundaries = UnstructuredFiniteVolumeBoundarySet(
        discretization.boundary_patch_names, boundary_values
    )
    precision = _precision(payload["precision"])
    problem = ConservationProblemIR(str(payload["name"]), "state", system, boundaries)
    compiled = compile_conservation_problem(
        problem, discretization, method, precision=precision
    )
    execution = _execution(payload["execution"])
    runtime = PreparedFiniteVolumeRuntime(
        compiled.dynamics, FluxPositivityPlan(), execution.step_policy
    )

    initial_payload = payload["initial_state"]
    _require_fields(initial_payload, {"type", "values"}, "case.initial_state")
    values = jnp.asarray(initial_payload["values"], dtype=precision.storage_dtype)
    if values.shape != (system.component_count,):
        raise ValueError("case.initial_state.values must match equation components.")
    if initial_payload["type"] == "constant_conservative":
        initial_state = jnp.broadcast_to(values, discretization.state_shape)
    elif initial_payload["type"] == "constant_primitive":
        primitive = jnp.broadcast_to(values, discretization.state_shape)
        initial_state = system.primitive_to_conserved(primitive)
    else:
        raise ValueError("Unsupported initial_state type.")
    initial_state = precision.storage(initial_state)
    if not jnp.all(system.admissible(initial_state)):
        raise ValueError("case.initial_state is not physically admissible.")
    case = FiniteVolumeCaseSpec(
        str(payload["name"]), runtime, execution, precision=precision
    )
    return PreparedFiniteVolumeCase(
        case, runtime, discretization, initial_state, str(mesh_path)
    )


def load_finite_volume_case(
    payload: dict[str, Any],
    /,
    *,
    source_path: str | Path | None = None,
) -> PreparedFiniteVolumeCase:
    """Build an allowlisted structured-v1 or native-unstructured-v2 FV case."""

    if not isinstance(payload, dict):
        raise TypeError("payload must be a case-document mapping.")
    version = payload.get("schema_version")
    if version == 1:
        return _load_structured_case_v1(payload)
    if version == 2:
        return _load_unstructured_case_v2(payload, source_path)
    raise ValueError("Unsupported finite-volume case document schema version.")


__all__ = ["PreparedFiniteVolumeCase", "load_finite_volume_case"]
