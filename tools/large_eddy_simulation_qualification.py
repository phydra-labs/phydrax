#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Execute preregistered, unsigned qualification campaigns for LES routes."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import jax


jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.applications.incompressible_flow._boundary_turbulence import (
    StochasticTurbulentInflowMACBoundaryState,
    StochasticTurbulentInflowPlan,
    VectorEquilibriumWallStressPlan,
)
from phydrax.applications.incompressible_flow._immersed_les import (
    compile_fixed_immersed_mac_les_flow,
    FixedImmersedMACLESPlan,
)
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._conservation_boundary import ExtrapolationBoundary
from phydrax.discretization.fem._boundary import FiniteElementBoundarySet
from phydrax.discretization.fem._generic import (
    FiniteElementFieldSpec,
    FiniteElementPlan,
)
from phydrax.discretization.fem._reference import discontinuous_element
from phydrax.discretization.finite_volume._riemann import RusanovFluxPlan
from phydrax.discretization.spectral._distributed import SpectralMeshTopology
from phydrax.discretization.spectral._distributed_les import DistributedPeriodicLESPlan
from phydrax.discretization.spectral._transfer import prepare_spectral_modal_transfer
from phydrax.equations._chemical_species import (
    ChemicalPhaseKind,
    ChemicalSpeciesSchema,
)
from phydrax.equations._chemical_thermodynamics import (
    PolynomialSpeciesThermodynamicsPlan,
)
from phydrax.equations._conservation import (
    compile_conservation_problem,
    ConservationProblemIR,
)
from phydrax.equations._dynamic_les import (
    AllowSignedBackscatter,
    DynamicLESInputs,
    DynamicLESProvenance,
    DynamicSmagorinskyPlan,
    ExactDenominatorRegularization,
    GlobalDynamicLESAveraging,
)
from phydrax.equations._gas_dynamics import (
    HomogeneousMixtureCompressibleNavierStokesSystem,
)
from phydrax.equations._homogeneous_thermodynamics import (
    HomogeneousHelmholtzPlan,
    IdealGasReferenceHelmholtzTerm,
    UNIVERSAL_GAS_CONSTANT,
    ZeroResidualHelmholtzTerm,
)
from phydrax.equations._learned_stress import (
    LEARNED_STRESS_FEATURE_NAME,
    LEARNED_STRESS_VELOCITY_GRADIENT_COMPONENTS,
    LEARNED_STRESS_VELOCITY_GRADIENT_UNITS,
    MACLearnedStressPlan,
    PeriodicLearnedStressPlan,
)
from phydrax.equations._periodic_dynamic_les import (
    PeriodicDynamicLESPlan,
    PeriodicFourierTestFilterPlan,
)
from phydrax.equations._periodic_les import (
    PeriodicAlgebraicLESPlan,
    PeriodicFourierGridFilterPlan,
)
from phydrax.equations._transport_closures import ConstantTransport
from phydrax.equations.fem._nodal_conservation import (
    NodalDGConservationMethodPlan,
)
from phydrax.equations.fem._viscous_conservation import (
    ViscousBoundaryClosure,
    ViscousDGPlan,
)
from phydrax.lifecycle._repository import (
    HPCFilesystemProfile,
    POSIXArtifactRepository,
    POSIXRepositoryPolicy,
)
from phydrax.linalg import DenseLU, LinearSolvePolicy, TolerancePolicy
from phydrax.solver._production_runtime import ArtifactCheckpointStore
from phydrax.solver._unstructured_les import (
    _step_status,
    UNSTRUCTURED_LES_ENERGY_FAILURE,
    UnstructuredLowMachLESStepInputs,
)
from tools.lattice_boltzmann_smagorinsky_qualification import (
    qualification as lbm_smagorinsky_qualification,
)


CAPABILITY = "large-eddy-simulation"
_CAMPAIGN_KIND = "les-qualification-campaign"
_MEASUREMENT_KIND = "les-qualification-measurements"
_CANDIDATE_KIND = "les-qualification-candidate"
_COMPARISONS = frozenset(("less-than-or-equal", "greater-than-or-equal"))
_CASE_FIELDS = frozenset(
    (
        "case_id",
        "name",
        "producer",
        "support",
        "dependencies",
        "coefficients",
        "grids",
        "timesteps",
        "parameters",
        "metrics",
        "references",
        "predicates",
    )
)
_METRIC_FIELDS = frozenset(
    (
        "name",
        "scope",
        "evidence_kind",
        "comparison",
        "threshold",
        "units",
        "criterion_id",
        "predicate_id",
    )
)
_CAMPAIGN_FIELDS = frozenset(
    (
        "kind",
        "capability",
        "provider",
        "build_id",
        "environment_id",
        "backend",
        "precision",
        "reduction",
        "reviewer_id",
        "issued_at",
        "expires_at",
        "release_index_id",
        "trust_policy_id",
        "base_profiles",
        "cases",
        "matrix_id",
        "campaign_id",
    )
)
_MODEL_PLANS = {
    "smagorinsky": phx.equations.SmagorinskyLESPlan,
    "wale": phx.equations.WALELESPlan,
    "vreman": phx.equations.VremanLESPlan,
    "amd": phx.equations.AMDLESPlan,
}


def canonical_json(value: object, /) -> str:
    """Return the canonical JSON encoding used by campaign identities."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_address(value: object, /) -> str:
    """Content-address one JSON-ready value with SHA-256."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _mapping(value: object, name: str, /) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return value


def _sequence(value: object, name: str, /) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence.")
    return value


def _identifier(value: object, name: str, /) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return value


def _exact_fields(
    record: Mapping[str, object], expected: frozenset[str], label: str, /
) -> None:
    fields = set(record)
    missing = sorted(expected - fields)
    unknown = sorted(fields - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing fields: " + ", ".join(missing))
        if unknown:
            details.append("unknown fields: " + ", ".join(unknown))
        raise ValueError(f"{label} has {'; '.join(details)}.")


def _read_json_object(path: Path, /) -> Mapping[str, object]:
    def reject_nonfinite(value: str, /) -> None:
        raise ValueError(f"Non-finite JSON value {value!r} is not permitted.")

    result = json.loads(path.read_text(), parse_constant=reject_nonfinite)
    return _mapping(result, f"JSON object {path}")


def _without(record: Mapping[str, object], key: str, /) -> dict[str, object]:
    return {name: value for name, value in record.items() if name != key}


def _criterion_record(metric: Mapping[str, object], /) -> dict[str, object]:
    return {
        "kind": "les-qualification-criterion",
        "name": metric["name"],
        "scope": metric["scope"],
        "evidence_kind": metric["evidence_kind"],
        "comparison": metric["comparison"],
        "threshold": metric["threshold"],
        "units": metric["units"],
    }


def load_matrix(path: Path, /) -> phx.qualification.QualificationMatrix:
    """Load and content-verify the preregistered generic evidence matrix."""
    return phx.qualification.QualificationMatrix.from_record(_read_json_object(path))


def validate_campaign(
    record: Mapping[str, object],
    matrix: phx.qualification.QualificationMatrix,
    /,
) -> Mapping[str, object]:
    """Validate every campaign identity, threshold, tuple, and matrix predicate."""
    campaign = _mapping(record, "campaign")
    _exact_fields(campaign, _CAMPAIGN_FIELDS, "Campaign")
    if campaign["kind"] != _CAMPAIGN_KIND:
        raise ValueError("Campaign kind is unsupported.")
    if campaign["capability"] != CAPABILITY:
        raise ValueError(f"Campaign capability must be {CAPABILITY!r}.")
    if "schema" in campaign or "schema_version" in campaign or "version" in campaign:
        raise ValueError("LES campaign inputs are unversioned and contain no schema tag.")
    if campaign["matrix_id"] != matrix.matrix_id:
        raise ValueError("Campaign matrix_id does not match the supplied matrix.")
    if campaign["campaign_id"] != content_address(_without(campaign, "campaign_id")):
        raise ValueError("Campaign has an invalid content address.")
    issued_at = int(campaign["issued_at"])
    expires_at = int(campaign["expires_at"])
    if issued_at < 0 or expires_at <= issued_at:
        raise ValueError("Campaign evidence window must have positive duration.")

    base_profiles = _sequence(campaign["base_profiles"], "base_profiles")
    base_keys: set[str] = set()
    for value in base_profiles:
        base = _mapping(value, "base profile")
        _exact_fields(
            base,
            frozenset(("key", "name", "provider", "scope", "support")),
            "Base profile",
        )
        key = _identifier(base["key"], "base profile key")
        if key in base_keys:
            raise ValueError("Base profile keys must be unique.")
        base_keys.add(key)
        if base["scope"] not in ("scientific", "deployment"):
            raise ValueError("Base profile scope must be scientific or deployment.")
        phx.qualification.SupportTuple.from_record(
            _mapping(base["support"], "base support tuple")
        )

    matrix_predicates = {name: dict(predicate) for name, predicate in matrix.predicates}
    seen_predicates: set[str] = set()
    seen_cases: set[str] = set()
    cases = _sequence(campaign["cases"], "cases")
    if not cases:
        raise ValueError("Campaign cases must not be empty.")
    for value in cases:
        case = _mapping(value, "campaign case")
        _exact_fields(case, _CASE_FIELDS, "Campaign case")
        if case["case_id"] != content_address(_without(case, "case_id")):
            raise ValueError("Campaign case has an invalid content address.")
        case_id = _identifier(case["case_id"], "case ID")
        if case_id in seen_cases:
            raise ValueError("Campaign case IDs must be unique.")
        seen_cases.add(case_id)
        support = phx.qualification.SupportTuple.from_record(
            _mapping(case["support"], "case support tuple")
        )
        if support.capability != CAPABILITY:
            raise ValueError("Every case support tuple must use the LES capability.")
        dependencies = tuple(
            _identifier(item, "base dependency key")
            for item in _sequence(case["dependencies"], "case dependencies")
        )
        if not dependencies or len(set(dependencies)) != len(dependencies):
            raise ValueError("Case dependencies must be non-empty and unique.")
        unknown_dependencies = set(dependencies) - base_keys
        if unknown_dependencies:
            raise ValueError(
                "Case references unknown base dependencies: "
                + ", ".join(sorted(unknown_dependencies))
            )
        for field in ("coefficients", "grids", "timesteps", "parameters"):
            _mapping(case[field], f"case {field}")
        references = _sequence(case["references"], "case references")
        if not references:
            raise ValueError("Every case must preregister at least one reference.")
        admitted_manifest_count = 0
        for reference_value in references:
            reference = _mapping(reference_value, "case reference")
            if reference.get("kind") == "native-reference":
                _exact_fields(
                    reference,
                    frozenset(("kind", "name", "model", "reference_id")),
                    "Native reference",
                )
                native_core = _without(reference, "reference_id")
                if reference["reference_id"] != content_address(native_core):
                    raise ValueError("Native reference has an invalid content address.")
            elif reference.get("kind") == "reference-artifact":
                admit_reference(reference, required=True)
                admitted_manifest_count += 1
            else:
                raise ValueError("Campaign reference kind is unsupported.")
        if case["producer"] == "periodic-exact-filter" and admitted_manifest_count != 1:
            raise ValueError(
                "Exact a-priori filtering requires exactly one admitted reference."
            )
        metrics = _sequence(case["metrics"], "case metrics")
        predicates = tuple(
            _identifier(item, "case predicate ID")
            for item in _sequence(case["predicates"], "case predicates")
        )
        if not metrics or len(predicates) != len(metrics):
            raise ValueError("Each case must preregister one predicate per metric.")
        metric_predicates: list[str] = []
        metric_names: set[str] = set()
        for metric_value in metrics:
            metric = _mapping(metric_value, "case metric")
            _exact_fields(metric, _METRIC_FIELDS, "Campaign metric")
            metric_name = _identifier(metric["name"], "metric name")
            if metric_name in metric_names:
                raise ValueError("Metric names must be unique within a case.")
            metric_names.add(metric_name)
            if metric["scope"] != case["producer"]:
                raise ValueError("Metric scope must equal its case producer.")
            if metric["comparison"] not in _COMPARISONS:
                raise ValueError("Metric comparison is unsupported.")
            threshold = metric["threshold"]
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
                raise TypeError("Metric thresholds must be real numbers.")
            if not math.isfinite(float(threshold)):
                raise ValueError("Metric thresholds must be finite.")
            expected_criterion = content_address(_criterion_record(metric))
            if metric["criterion_id"] != expected_criterion:
                raise ValueError("Metric criterion_id does not bind its threshold.")
            predicate_id = _identifier(metric["predicate_id"], "metric predicate ID")
            metric_predicates.append(predicate_id)
            if predicate_id not in matrix_predicates:
                raise ValueError("Campaign metric is absent from the supplied matrix.")
            expected_predicate = {
                "evidence_kind": str(metric["evidence_kind"]),
                "subject_id": case_id,
                "criterion_id": expected_criterion,
                "build_id": str(campaign["build_id"]),
                "environment_id": str(campaign["environment_id"]),
                "backend": str(campaign["backend"]),
                "topology": str(dict(support.attributes)["topology"]),
                "precision": str(campaign["precision"]),
                "reduction": str(campaign["reduction"]),
                "reviewer_id": str(campaign["reviewer_id"]),
            }
            if matrix_predicates[predicate_id] != expected_predicate:
                raise ValueError(
                    "Matrix predicate differs from its preregistered case criterion."
                )
            if predicate_id in seen_predicates:
                raise ValueError("Matrix predicates may belong to only one case.")
            seen_predicates.add(predicate_id)
        if tuple(metric_predicates) != predicates:
            raise ValueError("Case predicates must exactly follow metric declarations.")
    if seen_predicates != set(matrix_predicates):
        raise ValueError("Matrix contains post-hoc or unassigned predicates.")
    return campaign


def load_campaign(
    path: Path,
    matrix: phx.qualification.QualificationMatrix,
    /,
) -> Mapping[str, object]:
    """Load and strictly verify one content-addressed, unversioned campaign."""
    return validate_campaign(_read_json_object(path), matrix)


def admit_reference(
    value: Mapping[str, object] | None,
    /,
    *,
    required: bool,
) -> phx.qualification.ReferenceArtifactManifest | None:
    """Verify payload identity and commercial rights before reference use."""
    if value is None:
        if required:
            raise ValueError("This qualification case requires a reference manifest.")
        return None
    reference = _mapping(value, "reference")
    if reference.get("kind") != "reference-artifact":
        if required:
            raise ValueError("This qualification case requires a reference manifest.")
        return None
    _exact_fields(
        reference,
        frozenset(("kind", "payload", "manifest", "requested_rights")),
        "Reference binding",
    )
    manifest = phx.qualification.ReferenceArtifactManifest.from_record(
        _mapping(reference["manifest"], "reference manifest")
    )
    payload = canonical_json(reference["payload"]).encode("utf-8")
    checksum = hashlib.new(manifest.checksum_algorithm, payload).hexdigest()
    if len(payload) != manifest.size_bytes or checksum != manifest.checksum:
        raise ValueError("Reference payload does not match its governed manifest.")
    rights = _mapping(reference["requested_rights"], "requested reference rights")
    _exact_fields(
        rights,
        frozenset(("commercial_use", "redistribution", "training_use", "export")),
        "Requested reference rights",
    )
    manifest.require_rights(
        commercial_use=rights["commercial_use"],
        redistribution=rights["redistribution"],
        training_use=rights["training_use"],
        export=rights["export"],
    )
    return manifest


def _resolved_filter(
    name: str,
    /,
    *,
    family: str,
    topology: str,
    boundary_class: str,
    commutation_status: str,
    repeated_filter_semantics: str,
) -> phx.equations.ResolvedLESFilter:
    scale_rule = {
        "sharp-fourier-projection": "cutoff-equivalent",
        "implicit-grid-volume": "volume-equivalent",
        "explicit-filter": "kernel-equivalent",
    }[family]
    return phx.equations.ResolvedLESFilter(
        name,
        family=family,
        axis_names=("x", "y", "z"),
        topology=topology,
        boundary_class=boundary_class,
        scale_rule=scale_rule,
        commutation_status=commutation_status,
        repeated_filter_semantics=repeated_filter_semantics,
    )


def _periodic_space(count: int, /):
    return phx.discretization.TensorSpectralPlan(
        tuple(phx.discretization.FourierBasisPlan(count) for _ in range(3)),
        axis_names=("x", "y", "z"),
        field_name="velocity",
    ).prepare(
        tuple(
            phx.discretization.AxisDomain.periodic(0.0, 2.0 * math.pi) for _ in range(3)
        )
    )


def _periodic_velocity(space):
    x, y, z = jnp.meshgrid(*(axis.nodes for axis in space.axes), indexing="ij")
    return jnp.stack(
        (
            jnp.sin(x) * jnp.cos(y) * jnp.cos(z) + 0.2 * jnp.sin(y),
            -jnp.cos(x) * jnp.sin(y) * jnp.cos(z) + 0.2 * jnp.sin(z),
            0.2 * jnp.sin(x),
        ),
        axis=-1,
    )


def _periodic_les_plan(space, model: str, coefficient: float, oversampling: float):
    resolved_filter = _resolved_filter(
        "retained Fourier grid",
        family="sharp-fourier-projection",
        topology="tensor-product",
        boundary_class="periodic",
        commutation_status="commuting",
        repeated_filter_semantics="idempotent",
    )
    provenance = phx.equations.LESParameterProvenance(
        resolved_filter,
        space.prepared_id,
        "three-dimensional-periodic-unit-density",
        source_kind="user",
        evidence_ids=(),
    )
    prepared_model = _MODEL_PLANS[model](coefficient).prepare(provenance)
    return PeriodicAlgebraicLESPlan(
        prepared_model,
        PeriodicFourierGridFilterPlan(resolved_filter),
        phx.discretization.PseudospectralMethodPlan(
            dealiasing=phx.discretization.OversamplingDealiasingPlan(oversampling)
        ),
        energy_tolerance=2.0e-8,
    )


def _compile_periodic(
    space,
    model: str,
    coefficient: float,
    oversampling: float,
    /,
    *,
    forcing=None,
    forcing_id: str | None = None,
):
    problem = phx.equations.IncompressibleFlowProblem(
        3, 0.01, forcing=forcing, forcing_id=forcing_id
    )
    resolved_method = phx.discretization.PseudospectralMethodPlan(
        dealiasing=phx.discretization.PaddingDealiasingPlan(2)
    )
    dynamics = phx.equations.compile_periodic_incompressible_flow(
        problem,
        space,
        resolved_method,
        algebraic_les=_periodic_les_plan(space, model, coefficient, oversampling),
    )
    return problem, dynamics


def _positive_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    active = (numerator > 0.0) & (denominator > 0.0)
    return np.where(active, numerator / np.where(active, denominator, 1.0), 0.0)


def _independent_formula(
    model: str, coefficient: float, gradient: np.ndarray, widths: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    strain = 0.5 * (gradient + np.swapaxes(gradient, -1, -2))
    deviatoric = (
        strain - np.trace(strain, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3.0
    )
    equivalent = np.cbrt(np.prod(widths, axis=-1))
    if model == "smagorinsky":
        magnitude = np.sqrt(np.maximum(2.0 * np.sum(strain * strain, axis=(-2, -1)), 0.0))
        viscosity = (coefficient * equivalent) ** 2 * magnitude
    elif model == "wale":
        squared = np.einsum("...ik,...kj->...ij", gradient, gradient)
        symmetric = 0.5 * (squared + np.swapaxes(squared, -1, -2))
        squared_deviatoric = (
            symmetric
            - np.trace(symmetric, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3.0
        )
        strain_squared = np.sum(strain * strain, axis=(-2, -1))
        squared_invariant = np.sum(squared_deviatoric * squared_deviatoric, axis=(-2, -1))
        numerator = squared_invariant * np.sqrt(np.maximum(squared_invariant, 0.0))
        denominator = strain_squared**2 * np.sqrt(
            np.maximum(strain_squared, 0.0)
        ) + squared_invariant * np.sqrt(np.sqrt(np.maximum(squared_invariant, 0.0)))
        viscosity = (coefficient * equivalent) ** 2 * _positive_ratio(
            numerator, denominator
        )
    else:
        scaled = gradient * widths[..., None, :]
        beta = np.einsum("...ik,...jk->...ij", scaled, scaled)
        gradient_squared = np.sum(gradient * gradient, axis=(-2, -1))
        if model == "vreman":
            invariant = 0.5 * (
                np.trace(beta, axis1=-2, axis2=-1) ** 2
                - np.sum(beta * beta, axis=(-2, -1))
            )
            viscosity = coefficient * np.sqrt(
                np.maximum(_positive_ratio(invariant, gradient_squared), 0.0)
            )
        elif model == "amd":
            production = -np.sum(beta * deviatoric, axis=(-2, -1))
            viscosity = coefficient * _positive_ratio(production, gradient_squared)
        else:
            raise ValueError(f"Unsupported formula {model!r}.")
    stress = -2.0 * viscosity[..., None, None] * deviatoric
    transfer = -np.sum(stress * strain, axis=(-2, -1))
    return viscosity, stress, transfer


def _formula_measurements(model: str, coefficient: float, /) -> tuple[float, float]:
    gradient = np.asarray(
        (
            ((0.8, -0.2, 0.1), (0.3, -0.5, 0.4), (-0.1, 0.2, -0.3)),
            ((-0.2, 0.7, 0.5), (-0.4, 0.1, -0.6), (0.3, 0.8, 0.2)),
            ((0.1, -0.9, 0.4), (0.6, 0.3, 0.2), (-0.7, 0.5, -0.4)),
        ),
        dtype=np.float64,
    )
    widths = np.asarray(
        ((0.2, 0.3, 0.4), (0.4, 0.25, 0.35), (0.3, 0.5, 0.2)),
        dtype=np.float64,
    )
    scale = phx.equations.LESFilterScale(jnp.asarray(widths))
    actual = _MODEL_PLANS[model](coefficient).evaluate(
        phx.equations.AlgebraicLESInputs(jnp.asarray(gradient), scale)
    )
    expected = _independent_formula(model, coefficient, gradient, widths)
    error = max(
        float(np.max(np.abs(np.asarray(actual.kinematic_viscosity) - expected[0]))),
        float(
            np.max(np.abs(np.asarray(actual.specific_deviatoric_stress) - expected[1]))
        ),
        float(np.max(np.abs(np.asarray(actual.energy_transfer) - expected[2]))),
    )
    activity_violation = 0.0 if float(np.max(expected[0])) > 0.0 else 1.0
    return error, activity_violation


def _maximum_abs(values: Sequence[object], /) -> float:
    return max(float(jnp.max(jnp.abs(value))) for value in values)


def _tree_max_error(left: object, right: object, /) -> float:
    """Return a dtype-aware maximum discrepancy for two exact PyTrees."""
    if jax.tree.structure(left) != jax.tree.structure(right):
        return 1.0
    errors: list[float] = []
    for left_leaf, right_leaf in zip(
        jax.tree.leaves(left),
        jax.tree.leaves(right),
        strict=True,
    ):
        left_array = np.asarray(left_leaf)
        right_array = np.asarray(right_leaf)
        if left_array.shape != right_array.shape:
            errors.append(1.0)
        elif (
            np.issubdtype(left_array.dtype, np.bool_)
            or np.issubdtype(left_array.dtype, np.integer)
            or np.issubdtype(right_array.dtype, np.bool_)
            or np.issubdtype(right_array.dtype, np.integer)
        ):
            errors.append(0.0 if np.array_equal(left_array, right_array) else 1.0)
        elif np.issubdtype(left_array.dtype, np.number) and np.issubdtype(
            right_array.dtype, np.number
        ):
            errors.append(float(np.max(np.abs(left_array - right_array), initial=0.0)))
        else:
            errors.append(0.0 if np.array_equal(left_array, right_array) else 1.0)
    return max(errors, default=0.0)


def _run_periodic_static(case: Mapping[str, object], _reference):
    coefficients = _mapping(case["coefficients"], "periodic coefficients")
    grids = _mapping(case["grids"], "periodic grids")
    timesteps = _mapping(case["timesteps"], "periodic timesteps")
    parameters = _mapping(case["parameters"], "periodic parameters")
    model = str(parameters["model"])
    coefficient = float(coefficients["model"])
    oversampling = tuple(float(value) for value in parameters["oversampling_factors"])
    count = int(grids["operator"])
    space = _periodic_space(count)
    _, dynamics = _compile_periodic(space, model, coefficient, oversampling[0])
    state = dynamics.project_state(_periodic_velocity(space))
    stage = dynamics.stage(0.0, state)
    les = stage.algebraic_les
    if les is None:
        raise RuntimeError("Periodic algebraic LES stage is unavailable.")
    coordinates = space.real_coordinates(component_shape=(3,))
    work_defect = abs(
        float(jnp.real(jnp.vdot(state, les.projected_rate)) + les.modeled_dissipation)
    )
    guard = phx.solver.LESStabilityGuardedETDRKMethod(
        phx.solver.ETDRKMethod(2),
        safety_factor=float(parameters["guard_safety_factor"]),
    ).prepare(dynamics, coordinates=coordinates)
    guard_restriction = guard.step_restriction(0.0, state)
    allowed_step = guard.safety_factor * float(guard_restriction.etdrk_selected)
    accepted_guard = guard.step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        state,
        jnp.asarray(0.1 * allowed_step),
        None,
    )
    rejected_guard = guard.step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        state,
        jnp.asarray(1.1 * allowed_step),
        None,
    )

    baseline_problem = phx.equations.IncompressibleFlowProblem(3, 0.01)
    resolved_method = phx.discretization.PseudospectralMethodPlan(
        dealiasing=phx.discretization.PaddingDealiasingPlan(2)
    )
    baseline = phx.equations.compile_periodic_incompressible_flow(
        baseline_problem, space, resolved_method
    )
    _, zero = _compile_periodic(space, model, 0.0, oversampling[0])
    zero_state = baseline.project_state(_periodic_velocity(space))
    zero_error = float(
        jnp.max(jnp.abs(zero(0.0, zero_state, None) - baseline(0.0, zero_state, None)))
    )

    prepared = tuple(
        _periodic_les_plan(space, model, coefficient, factor).prepare(
            space, dynamics.projector
        )
        for factor in oversampling
    )
    rates = tuple(value.evaluate(state).projected_rate for value in prepared)
    low_error = float(jnp.linalg.norm(rates[0] - rates[-1]))
    high_error = float(jnp.linalg.norm(rates[-2] - rates[-1]))

    step_size = float(timesteps["step_size"])
    steps = int(timesteps["steps"])
    resolutions = tuple(int(value) for value in grids["campaign"])
    unforced_success = True
    forced_success = True
    maximum_divergence = 0.0
    execution_ids: list[str] = [dynamics.compilation_id]
    for resolution in resolutions:
        campaign_space = _periodic_space(resolution)
        _, unforced = _compile_periodic(
            campaign_space, model, coefficient, oversampling[0]
        )
        initial = unforced.project_state(_periodic_velocity(campaign_space))
        times = jnp.arange(steps + 1, dtype=float) * step_size
        unforced_result = phx.solver.solve_etdrk(
            phx.solver.ETDRKMethod(2), unforced.semilinear_drift, initial, times
        )
        unforced_diagnostics = unforced.diagnostics(
            float(times[-1]), unforced_result.states[-1]
        )
        maximum_divergence = max(
            maximum_divergence, float(unforced_diagnostics.divergence_norm)
        )
        unforced_success = unforced_success and bool(
            unforced_result.successful & unforced_diagnostics.finite
        )

        modal_force = 0.01 * initial

        def forcing(time, state_, args, force=modal_force):
            del time, state_, args
            return force

        _, forced = _compile_periodic(
            campaign_space,
            model,
            coefficient,
            oversampling[0],
            forcing=forcing,
            forcing_id="constant-solenoidal-modal-force",
        )
        forced_result = phx.solver.solve_etdrk(
            phx.solver.ETDRKMethod(2), forced.semilinear_drift, initial, times
        )
        forced_diagnostics = forced.diagnostics(
            float(times[-1]), forced_result.states[-1]
        )
        maximum_divergence = max(
            maximum_divergence, float(forced_diagnostics.divergence_norm)
        )
        forced_success = forced_success and bool(
            forced_result.successful & forced_diagnostics.finite
        )
        execution_ids.extend((unforced.compilation_id, forced.compilation_id))

    formula_error, formula_activity_violation = _formula_measurements(model, coefficient)
    measurements = {
        "formula_maximum_error": formula_error,
        "formula_activity_violation": formula_activity_violation,
        "backend_activity_violation": (
            0.0
            if float(jnp.max(les.model_result.kinematic_viscosity)) > 0.0
            and float(jnp.linalg.norm(les.projected_rate)) > 0.0
            else 1.0
        ),
        "coefficient_zero_operator_error": zero_error,
        "stress_symmetry_defect": float(
            jnp.max(
                jnp.abs(
                    les.model_result.specific_deviatoric_stress
                    - jnp.swapaxes(les.model_result.specific_deviatoric_stress, -1, -2)
                )
            )
        ),
        "negative_energy_transfer_violation": max(
            0.0, -float(jnp.min(les.model_result.energy_transfer))
        ),
        "operator_energy_identity_defect": work_defect,
        "hermitian_defect": float(coordinates.reality_defect(les.projected_rate)),
        "projection_divergence_norm": float(
            dynamics.projector.divergence_norm(les.projected_rate)
        ),
        "projection_energy_defect": abs(float(les.projection_energy_defect)),
        "oversampling_convergence_violation": max(0.0, high_error - low_error),
        "multiresolution_divergence_norm": maximum_divergence,
        "unforced_campaign_failure": 0.0 if unforced_success else 1.0,
        "forced_campaign_failure": 0.0 if forced_success else 1.0,
        "guard_acceptance_failure": (0.0 if bool(accepted_guard.successful) else 1.0),
        "guard_rejection_failure": (0.0 if not bool(rejected_guard.successful) else 1.0),
        "guard_rollback_error": float(
            jnp.max(jnp.abs(rejected_guard.accepted_state - state))
        ),
        "guard_coordinate_binding_failure": (
            0.0
            if guard.base_method.coordinates is not None
            and guard.base_method.coordinates.coordinate_id == coordinates.coordinate_id
            else 1.0
        ),
    }
    return measurements, {
        "formula": model,
        "prepared_model_id": dynamics.algebraic_les.model.prepared_id,
        "periodic_les_id": dynamics.algebraic_les.prepared_id,
        "compilation_ids": execution_ids,
        "oversampling_prepared_ids": [value.prepared_id for value in prepared],
        "oversampling_errors": [low_error, high_error],
        "guard_method_id": guard.method_id,
        "guard_coordinate_id": coordinates.coordinate_id,
    }


def _run_periodic_exact_filter(case: Mapping[str, object], manifest):
    if manifest is None:
        raise ValueError("Exact a-priori filtering requires a reference manifest.")
    grids = _mapping(case["grids"], "filter grids")
    source = _periodic_space(int(grids["source"]))
    resolved = _periodic_space(int(grids["resolved"]))
    resolved_filter = _resolved_filter(
        "matched a-priori Fourier projection",
        family="sharp-fourier-projection",
        topology="tensor-product",
        boundary_class="periodic",
        commutation_status="commuting",
        repeated_filter_semantics="idempotent",
    )
    context = phx.closure_data.prepare_periodic_les_analysis(
        source,
        resolved,
        resolved_filter,
        reference_manifest_id=manifest.manifest_id,
    )
    coarse = jnp.zeros(resolved.modal_shape, dtype=jnp.complex128)
    coarse = coarse.at[1, 0, 0].set(1.25 - 0.5j)
    embedded = prepare_spectral_modal_transfer(resolved, source)(coarse)
    modal_error = float(jnp.max(jnp.abs(context.filter_modal(embedded) - coarse)))
    x, y, z = jnp.meshgrid(*(axis.nodes for axis in source.axes), indexing="ij")
    left = jnp.sin(x) + 0.4 * jnp.cos(3.0 * y)
    right = jnp.cos(y) - 0.3 * jnp.sin(2.0 * z)
    expected_product = resolved.reconstruct(
        context.modal_transfer(source.project(left * right))
    )
    product_error = float(
        jnp.max(jnp.abs(context.filter_product(left, right) - expected_product))
    )
    velocity = phx.closure_data.ClosureField(
        _periodic_velocity(source),
        name="velocity",
        units="m/s",
        schema_id="periodic-apriori-velocity",
        lineage_ids=(manifest.manifest_id,),
    )
    target = phx.closure_data.les_reynolds_stress_target(
        velocity, context, convention="full"
    )
    dag = context.analysis_dag((target,))
    reference = context.bind_target(target, dag)
    binding_violation = 0.0
    if (
        reference.reference_manifest_id != manifest.manifest_id
        or reference.filter_id != resolved_filter.filter_id
        or reference.source_discretization_id != source.prepared_id
        or reference.resolved_discretization_id != resolved.prepared_id
    ):
        binding_violation = 1.0
    symmetry_defect = float(
        jnp.max(jnp.abs(target.values - jnp.swapaxes(target.values, -1, -2)))
    )
    return {
        "modal_restriction_error": modal_error,
        "product_projection_error": product_error,
        "stress_symmetry_defect": symmetry_defect,
        "reference_binding_violation": binding_violation,
    }, {
        "context_id": context.context_id,
        "analysis_dag_id": dag.dag_id,
        "target_id": target.target_id,
        "reference_id": reference.reference_id,
        "reference_manifest_id": manifest.manifest_id,
    }


def _dynamic_prepared(resolved, test, oversampling: float):
    primary_filter = _resolved_filter(
        "resolved retained Fourier projection",
        family="sharp-fourier-projection",
        topology="tensor-product",
        boundary_class="periodic",
        commutation_status="commuting",
        repeated_filter_semantics="idempotent",
    )
    test_filter = _resolved_filter(
        "coarse test Fourier projection",
        family="sharp-fourier-projection",
        topology="tensor-product",
        boundary_class="periodic",
        commutation_status="commuting",
        repeated_filter_semantics="idempotent",
    )
    provenance = phx.equations.LESParameterProvenance(
        primary_filter,
        resolved.prepared_id,
        "three-dimensional-periodic-unit-density",
        source_kind="user",
        evidence_ids=(),
    )
    dynamic_provenance = DynamicLESProvenance(provenance, test_filter, (2.0, 2.0, 2.0))
    dynamic_model = DynamicSmagorinskyPlan(
        GlobalDynamicLESAveraging(),
        ExactDenominatorRegularization(),
        AllowSignedBackscatter(),
    ).prepare(dynamic_provenance)
    return PeriodicDynamicLESPlan(
        dynamic_model,
        PeriodicFourierGridFilterPlan(primary_filter),
        PeriodicFourierTestFilterPlan(test_filter),
        phx.discretization.PseudospectralMethodPlan(
            dealiasing=phx.discretization.OversamplingDealiasingPlan(oversampling)
        ),
        energy_tolerance=2.0e-8,
    ).prepare(
        resolved,
        test,
        phx.discretization.PeriodicLerayProjector(resolved),
    )


def _run_periodic_dynamic(case: Mapping[str, object], _reference):
    grids = _mapping(case["grids"], "dynamic grids")
    parameters = _mapping(case["parameters"], "dynamic parameters")
    resolved = _periodic_space(int(grids["resolved"]))
    test = _periodic_space(int(grids["test"]))
    prepared = _dynamic_prepared(resolved, test, float(parameters["oversampling_factor"]))
    state = prepared.projector.project(resolved.project(_periodic_velocity(resolved)))
    grid_filtered = prepared.grid_filter.apply(state)
    test_filtered = prepared.test_filter.apply(grid_filtered)
    transferred = prepared.test_filter.embedding(
        prepared.test_filter.test_grid_filter.apply(
            prepared.test_filter.restriction(grid_filtered)
        )
    )
    filter_error = float(jnp.max(jnp.abs(test_filtered - transferred)))
    inputs, _, _, _ = prepared._germano_inputs(state, accepted_update_mask=True)
    expected_coefficient = float(parameters["synthetic_coefficient"])
    synthetic = DynamicLESInputs(
        expected_coefficient * inputs.modeled_tensor + 1.7 * jnp.eye(3),
        inputs.modeled_tensor,
        inputs.algebraic_inputs,
        inputs.provenance,
        accepted_update_mask=True,
    )
    recovered = prepared.dynamic_model.evaluate(synthetic)
    stage = prepared.evaluate(state)
    return {
        "test_filter_exactness_error": filter_error,
        "synthetic_coefficient_error": float(
            jnp.max(jnp.abs(recovered.coefficient - expected_coefficient))
        ),
        "synthetic_stress_magnitude": float(
            jnp.max(
                jnp.abs(recovered.prepared_algebraic_stress.specific_deviatoric_stress)
            )
        ),
        "projection_divergence_norm": float(
            prepared.projector.divergence_norm(stage.projected_rate)
        ),
        "energy_identity_defect": abs(
            float(stage.algebraic_stage.energy_identity_defect)
        ),
        "finite_evidence_failure": (
            0.0
            if bool(stage.dynamic_result.evidence.finite & stage.algebraic_stage.finite)
            else 1.0
        ),
    }, {
        "prepared_id": prepared.prepared_id,
        "dynamic_model_id": prepared.dynamic_model.prepared_id,
        "test_filter_id": prepared.test_filter.prepared_id,
    }


def _mac_core(count: int, /):
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=True) for _ in range(3)
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, 0.0), (2.0 * math.pi,) * 3)))
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(discretization).prepare()
    momentum = phx.discretization.MACMomentumPlan(operators).prepare()
    projection = phx.solver.MACPressureProjectionPlan(
        operators, solve_method="transform", tolerance=2.0e-9
    )
    return discretization, operators, momentum, projection


def _mac_velocity(discretization):
    x_faces, y_faces, z_faces = discretization.face_centers
    return (
        jnp.sin(x_faces[..., 1]),
        jnp.sin(y_faces[..., 2]),
        jnp.sin(z_faces[..., 0]),
    )


def _mac_provenance(discretization):
    resolved_filter = _resolved_filter(
        "mac-cell-volume",
        family="implicit-grid-volume",
        topology="tensor-product",
        boundary_class="periodic",
        commutation_status="unmodeled",
        repeated_filter_semantics="unmodeled",
    )
    return phx.equations.LESParameterProvenance(
        resolved_filter,
        discretization.prepared_id,
        "incompressible-unit-density",
        source_kind="user",
        evidence_ids=(),
    )


def _ksgs_coefficients(
    coefficients: Mapping[str, object], /
) -> phx.equations.KSGSCoefficients:
    return phx.equations.KSGSCoefficients(
        float(coefficients["eddy_viscosity"]),
        float(coefficients["dissipation"]),
        float(coefficients["diffusion"]),
        float(coefficients["buoyancy"]),
        float(coefficients["production_limit"]),
    )


def _mac_algebraic_plan(discretization, coefficient: float):
    return phx.equations.MACAlgebraicLESPlan(
        phx.equations.SmagorinskyLESPlan(coefficient).prepare(
            _mac_provenance(discretization)
        )
    )


def _run_mac_coupled(case: Mapping[str, object], _reference):
    grids = _mapping(case["grids"], "MAC coupled grids")
    coefficients = _mapping(case["coefficients"], "MAC coupled coefficients")
    parameters = _mapping(case["parameters"], "MAC coupled parameters")
    discretization, operators, momentum, projection = _mac_core(int(grids["count"]))
    scalar_problem = phx.discretization.MACScalarProblem(
        (
            phx.discretization.MACScalarTransport(
                "temperature",
                float(coefficients["temperature_diffusivity"]),
                advection="centered",
            ),
            phx.discretization.MACScalarTransport(
                "tracer", float(coefficients["tracer_diffusivity"]), advection="upwind"
            ),
        )
    )
    transport = scalar_problem.prepare(operators)
    scalar_sgs = phx.discretization.MACScalarSGSPlan(
        (
            phx.discretization.MACScalarSGSField(
                "temperature",
                turbulent_prandtl_number=float(coefficients["turbulent_prandtl"]),
            ),
            phx.discretization.MACScalarSGSField(
                "tracer",
                turbulent_schmidt_number=float(coefficients["turbulent_schmidt"]),
            ),
        )
    )
    buoyancy = phx.equations.MACBuoyancyLaw(
        jnp.asarray((0.0, 0.0, -1.0)),
        {"temperature": float(coefficients["buoyancy"])},
        references={"temperature": 0.0},
    )
    dynamics = phx.equations.compile_mac_scalar_buoyancy(
        phx.equations.IncompressibleFlowProblem(3, float(coefficients["viscosity"])),
        momentum,
        projection,
        scalar_problem,
        transport,
        buoyancy,
        algebraic_les=_mac_algebraic_plan(discretization, float(coefficients["model"])),
        scalar_sgs=scalar_sgs,
    )
    cells = discretization.cell_centers
    scalars = {
        "temperature": jnp.sin(cells[..., 0]) * jnp.cos(cells[..., 2]),
        "tracer": jnp.cos(cells[..., 1]),
    }
    state = dynamics.project_state(_mac_velocity(discretization), scalars)
    stage = dynamics.stage(0.0, state)
    diagnostics = dynamics.diagnostics_from_stage(stage)
    les = stage.momentum_components.les_stage
    if les is None:
        raise RuntimeError("Coupled MAC algebraic LES stage is unavailable.")
    eddy = les.model_result.kinematic_viscosity
    temperature_error = float(
        jnp.max(
            jnp.abs(
                stage.scalar_sgs_diffusivities["temperature"]
                - eddy / float(coefficients["turbulent_prandtl"])
            )
        )
    )
    tracer_error = float(
        jnp.max(
            jnp.abs(
                stage.scalar_sgs_diffusivities["tracer"]
                - eddy / float(coefficients["turbulent_schmidt"])
            )
        )
    )
    del parameters
    return {
        "momentum_projection_divergence_norm": float(
            jnp.max(jnp.abs(stage.divergence_after))
        ),
        "scalar_sgs_ratio_error": max(temperature_error, tracer_error),
        "momentum_sgs_action_magnitude": _maximum_abs(stage.momentum_components.sgs),
        "buoyancy_exchange_defect": abs(float(stage.buoyancy.exchange_defect)),
        "sgs_energy_identity_defect": abs(
            float(
                diagnostics.sgs_energy_rate
                + diagnostics.sgs_dissipation
                - diagnostics.sgs_boundary_power
            )
        ),
        "coupled_stage_failure": 0.0
        if bool(stage.success & diagnostics.success)
        else 1.0,
    }, {
        "compilation_id": dynamics.compilation_id,
        "momentum_les_id": dynamics.base_dynamics.algebraic_les.prepared_id,
        "scalar_sgs_id": dynamics.scalar_sgs.prepared_id,
        "buoyancy_id": buoyancy.law_id,
    }


def _run_mac_ksgs(case: Mapping[str, object], _reference):
    grids = _mapping(case["grids"], "MAC KSGS grids")
    coefficients = _mapping(case["coefficients"], "MAC KSGS coefficients")
    discretization, operators, momentum, projection = _mac_core(int(grids["count"]))
    viscosity = float(coefficients["viscosity"])
    scalar_problem = phx.discretization.MACScalarProblem(
        (
            phx.discretization.MACScalarTransport(
                "temperature",
                float(coefficients["temperature_diffusivity"]),
                advection="centered",
            ),
            phx.discretization.MACScalarTransport(
                "sgs_kinetic_energy", viscosity, advection="upwind"
            ),
        )
    )
    transport = scalar_problem.prepare(operators)
    scalar_sgs = phx.discretization.MACScalarSGSPlan(
        (
            phx.discretization.MACScalarSGSField(
                "temperature",
                turbulent_prandtl_number=float(coefficients["turbulent_prandtl"]),
            ),
        )
    )
    ksgs_coefficients = _ksgs_coefficients(coefficients)
    ksgs = phx.equations.StaticKSGSPlan(
        ksgs_coefficients, _mac_provenance(discretization)
    )
    buoyancy = phx.equations.MACBuoyancyLaw(
        jnp.asarray((0.0, 0.0, -1.0)),
        {"temperature": float(coefficients["boussinesq_expansion"])},
        references={"temperature": 0.0},
    )
    dynamics = phx.equations.compile_mac_scalar_buoyancy(
        phx.equations.IncompressibleFlowProblem(3, viscosity),
        momentum,
        projection,
        scalar_problem,
        transport,
        buoyancy,
        scalar_sgs=scalar_sgs,
        ksgs=ksgs,
        ksgs_field_name="sgs_kinetic_energy",
    )
    cells = discretization.cell_centers
    state = dynamics.project_state(
        _mac_velocity(discretization),
        {
            "temperature": jnp.sin(cells[..., 0]),
            "sgs_kinetic_energy": jnp.full(
                discretization.cell_shape, float(coefficients["initial_kinetic_energy"])
            ),
        },
    )
    stage = dynamics.stage(0.0, state)
    restriction = dynamics.step_restriction(0.0, state)
    if stage.ksgs is None:
        raise RuntimeError("MAC KSGS stage is unavailable.")
    result = stage.ksgs.result
    return {
        "negative_kinetic_energy_violation": max(
            0.0, -float(jnp.min(stage.ksgs.state.kinetic_energy))
        ),
        "negative_eddy_viscosity_violation": max(
            0.0, -float(jnp.min(result.eddy_viscosity))
        ),
        "ksgs_eddy_viscosity_magnitude": float(jnp.max(result.eddy_viscosity)),
        "nonfinite_rhs_violation": (
            0.0 if bool(jnp.all(jnp.isfinite(result.contributions.rhs))) else 1.0
        ),
        "step_restriction_failure": (
            0.0
            if bool(jnp.isfinite(restriction.ksgs) & (restriction.ksgs > 0.0))
            else 1.0
        ),
        "ksgs_stage_failure": 0.0 if bool(stage.success & stage.ksgs.success) else 1.0,
    }, {
        "compilation_id": dynamics.compilation_id,
        "ksgs_plan_id": ksgs.plan_id,
        "prepared_ksgs_id": dynamics.ksgs.prepared_id,
        "scalar_sgs_id": dynamics.scalar_sgs.prepared_id,
    }


def _compiled_mac_for_time(case: Mapping[str, object]):
    grids = _mapping(case["grids"], "frozen MAC grids")
    coefficients = _mapping(case["coefficients"], "frozen MAC coefficients")
    discretization, operators, momentum, projection = _mac_core(int(grids["count"]))
    dynamics = phx.equations.compile_mac_incompressible_flow(
        phx.equations.IncompressibleFlowProblem(3, float(coefficients["viscosity"])),
        momentum,
        projection,
        algebraic_les=_mac_algebraic_plan(discretization, float(coefficients["model"])),
    )
    state = dynamics.pack_velocity(_mac_velocity(discretization))
    return discretization, operators, dynamics, state


def _linear_policy():
    return LinearSolvePolicy(
        DenseLU(),
        tolerance=TolerancePolicy(relative=2.0e-8, absolute=2.0e-8, max_steps=40),
    )


def _run_frozen_imex(case: Mapping[str, object], _reference):
    _, operators, dynamics, state = _compiled_mac_for_time(case)
    timesteps = _mapping(case["timesteps"], "frozen IMEX timesteps")
    method = phx.solver.MACIMEXEulerMethod(
        dynamics,
        fixed_step_size=float(timesteps["step_size"]),
        solve_method="iterative",
        tolerance=2.0e-8,
        maximum_iterations=40,
        linear_policy=_linear_policy(),
    )
    result = method.step(0.0, state)
    return {
        "acceptance_failure": 0.0 if bool(result.accepted) else 1.0,
        "projection_divergence_norm": _maximum_abs(
            (operators.divergence(result.velocity),)
        ),
        "frozen_les_action_magnitude": _maximum_abs(result.les_stage.physical_rate),
        "inverse_identity_violation": (
            0.0 if result.predictor_inverse_id == result.projection_inverse_id else 1.0
        ),
        "frozen_stage_binding_violation": (
            0.0
            if result.les_stage is not None
            and result.les_stage.prepared_id == dynamics.algebraic_les.prepared_id
            and result.coefficient_refresh == "accepted-state-once-per-attempt"
            else 1.0
        ),
    }, {
        "compilation_id": dynamics.compilation_id,
        "method_id": method.method_id,
        "temporal_profile": method.temporal_profile,
        "stage_inverse_id": result.predictor_inverse_id,
    }


def _run_frozen_sbdf2(case: Mapping[str, object], _reference):
    _, operators, dynamics, state = _compiled_mac_for_time(case)
    timesteps = _mapping(case["timesteps"], "frozen SBDF2 timesteps")
    method = phx.solver.MACSBDF2Method(
        dynamics,
        float(timesteps["step_size"]),
        solve_method="iterative",
        tolerance=2.0e-8,
        maximum_iterations=40,
        linear_policy=_linear_policy(),
    )
    startup = method.initialize(0.0, state)
    first = method.step(startup.history)
    replay = method.step(startup.history)
    restart_error = float(jnp.max(jnp.abs(first.history.state - replay.history.state)))
    return {
        "startup_failure": 0.0 if bool(startup.accepted) else 1.0,
        "acceptance_failure": 0.0 if bool(first.accepted) else 1.0,
        "restart_replay_error": restart_error,
        "projection_divergence_norm": _maximum_abs(
            (operators.divergence(first.velocity),)
        ),
        "frozen_les_action_magnitude": _maximum_abs(
            dynamics.rate_components(0.0, state).sgs
        ),
        "inverse_identity_violation": (
            0.0 if first.predictor_inverse_id == first.projection_inverse_id else 1.0
        ),
        "g_stability_failure": (
            0.0
            if first.g_stability is not None and bool(first.g_stability.successful)
            else 1.0
        ),
    }, {
        "compilation_id": dynamics.compilation_id,
        "method_id": method.method_id,
        "coefficient_extrapolation": method.coefficient_extrapolation,
    }


def _run_channel(case: Mapping[str, object], _reference):
    grids = _mapping(case["grids"], "channel grids")
    coefficients = _mapping(case["coefficients"], "channel coefficients")
    timesteps = _mapping(case["timesteps"], "channel timesteps")
    shape = tuple(int(value) for value in grids["shape"])
    space = phx.discretization.TensorSpectralPlan(
        (
            phx.discretization.FourierBasisPlan(shape[0]),
            phx.discretization.ChebyshevBasisPlan(shape[1]),
            phx.discretization.FourierBasisPlan(shape[2]),
        ),
        axis_names=("x", "y", "z"),
        field_name="velocity",
    ).prepare(
        (
            phx.discretization.AxisDomain.periodic(0.0, 2.0 * math.pi),
            phx.discretization.AxisDomain.interval(-1.0, 1.0),
            phx.discretization.AxisDomain.periodic(0.0, 2.0 * math.pi),
        )
    )
    viscosity = float(coefficients["viscosity"])
    stokes = phx.discretization.ChannelStokesPlan(space, viscosity)
    base = phx.equations.compile_channel_flow(
        phx.equations.IncompressibleFlowProblem(3, viscosity),
        stokes,
        phx.discretization.PseudospectralMethodPlan(
            dealiasing=phx.discretization.PaddingDealiasingPlan(2)
        ),
    )
    provenance = phx.equations.LESParameterProvenance(
        phx.equations.channel_les_filter(base.discretization),
        base.discretization.prepared_id,
        "wall-resolved-channel",
        source_kind="user",
        evidence_ids=(),
    )
    model_name = str(_mapping(case["parameters"], "channel parameters")["model"])
    dynamics = phx.equations.compile_channel_les(
        base,
        _MODEL_PLANS[model_name](float(coefficients["model"])).prepare(provenance),
    )
    x = space.axes[0].nodes[:, None, None]
    y = space.axes[1].nodes[None, :, None]
    envelope = (1.0 - y**2) ** 2
    physical = jnp.zeros(space.physical_shape + (3,))
    physical = physical.at[..., 0].set(
        0.01
        * jnp.broadcast_to(-4.0 * y * (1.0 - y**2) * jnp.sin(x), space.physical_shape)
    )
    physical = physical.at[..., 1].set(
        0.01 * jnp.broadcast_to(-envelope * jnp.cos(x), space.physical_shape)
    )
    initial = dynamics.project_state(physical)
    times = jnp.asarray((0.0, float(timesteps["step_size"])))
    result = phx.solver.solve_channel_sbdf2(dynamics, initial, times)
    diagnostics = dynamics.state_diagnostics(result.velocity[-1])
    ledger = dynamics.energy_ledger(initial)
    energy_defect = abs(
        float(
            ledger.resolved_energy_rate
            - ledger.wall_power
            + ledger.molecular_dissipation
            + ledger.subgrid_transfer
        )
    )
    return {
        "solver_failure": 0.0 if bool(result.successful) else 1.0,
        "divergence_norm": float(diagnostics.divergence_norm),
        "wall_residual": float(diagnostics.wall_residual),
        "energy_ledger_defect": energy_defect,
        "negative_subgrid_transfer_violation": max(0.0, -float(ledger.subgrid_transfer)),
        "subgrid_transfer_magnitude": abs(float(ledger.subgrid_transfer)),
        "noncommutation_evidence_missing": (
            0.0 if float(dynamics.filter_geometry.noncommutation_evidence) > 0.0 else 1.0
        ),
    }, {
        "base_compilation_id": base.compilation_id,
        "les_prepared_id": dynamics.les_prepared_id,
        "filter_geometry_id": dynamics.filter_geometry.geometry_id,
    }


def _run_distributed(case: Mapping[str, object], _reference):
    grids = _mapping(case["grids"], "distributed grids")
    coefficients = _mapping(case["coefficients"], "distributed coefficients")
    parameters = _mapping(case["parameters"], "distributed parameters")
    space = _periodic_space(int(grids["count"]))
    scientific = _periodic_les_plan(
        space,
        str(parameters["model"]),
        float(coefficients["model"]),
        float(parameters["oversampling_factor"]),
    ).prepare(space, phx.discretization.PeriodicLerayProjector(space))
    topology = SpectralMeshTopology(
        (1,), devices=(jax.devices("cpu")[0],), axis_names=("spectral",)
    )
    distributed = DistributedPeriodicLESPlan(
        scientific, topology, schedule="slab"
    ).prepare()
    state = scientific.projector.project(space.project(_periodic_velocity(space)))
    evidence = distributed.parity_evidence(
        state,
        absolute_tolerance=float(parameters["absolute_tolerance"]),
        relative_tolerance=float(parameters["relative_tolerance"]),
    )
    stage = distributed.evaluate(state)
    reference_stage = scientific.evaluate(state)
    errors = (
        float(evidence.projected_rate_maximum_error),
        float(evidence.stress_maximum_error),
        float(evidence.modeled_dissipation_error),
    )
    scales = (
        float(jnp.max(jnp.abs(reference_stage.projected_rate))),
        float(jnp.max(jnp.abs(reference_stage.modal_deviatoric_specific_stress))),
        abs(float(reference_stage.modeled_dissipation)),
    )
    relative_error = max(
        error / scale if scale > 0.0 else error
        for error, scale in zip(errors, scales, strict=True)
    )
    return {
        "absolute_parity_error": max(errors),
        "relative_parity_error": relative_error,
        "parity_failure": 0.0 if bool(evidence.passed) else 1.0,
        "energy_identity_defect": abs(float(stage.energy_identity_defect)),
        "distributed_sgs_action_magnitude": float(jnp.max(jnp.abs(stage.projected_rate))),
        "host_gather_violation": 0.0 if not distributed.preparation.host_gather else 1.0,
        "qualification_inheritance_violation": (
            0.0 if not evidence.qualification_inherited else 1.0
        ),
    }, {
        "scientific_prepared_id": scientific.prepared_id,
        "distributed_prepared_id": distributed.prepared_id,
        "topology_id": topology.topology_id,
        "layout_id": distributed.execution.modal_layout.layout_id,
        "resource_report_id": distributed.preparation.resource.report_id,
    }


def _favre_model(case: Mapping[str, object]):
    coefficients = _mapping(case["coefficients"], "Favre coefficients")
    resolved_filter = _resolved_filter(
        "favre-cell-volume",
        family="implicit-grid-volume",
        topology="tensor-product",
        boundary_class="open",
        commutation_status="unmodeled",
        repeated_filter_semantics="unmodeled",
    )
    provenance = phx.equations.LESParameterProvenance(
        resolved_filter,
        "three-dimensional-compressible-cell-transport",
        "variable-density-smooth-region",
        source_kind="user",
        evidence_ids=(),
    )
    fields = phx.equations.FavreLESFieldContract("binary-mixture", ("fuel", "oxidizer"))
    return phx.equations.PreparedFavreLESModel(
        phx.equations.SmagorinskyLESPlan(float(coefficients["model"])).prepare(
            provenance
        ),
        phx.equations.LESFilterScale(jnp.asarray(tuple(coefficients["filter_widths"]))),
        fields,
        float(coefficients["turbulent_prandtl"]),
        (
            ("fuel", float(coefficients["fuel_schmidt"])),
            ("oxidizer", float(coefficients["oxidizer_schmidt"])),
        ),
        float(coefficients["viscosity_upper_bound"]),
    )


def _run_favre(case: Mapping[str, object], _reference):
    model = _favre_model(case)
    fields = model.fields
    inputs = phx.equations.FavreLESInputs(
        jnp.asarray(1.2),
        jnp.asarray(400.0),
        jnp.asarray((1.0, -0.5, 0.2)),
        jnp.asarray(((0.4, 0.1, 0.0), (-0.2, -0.1, 0.3), (0.0, 0.2, 0.05))),
        jnp.asarray((2.0, -1.0, 0.5)),
        jnp.asarray((0.3, 0.7)),
        jnp.asarray(((0.1, 0.02, 0.0), (-0.1, -0.02, 0.0))),
        jnp.asarray(1100.0),
        jnp.asarray((2.0e5, 3.0e5)),
        fields,
    )
    result = model.evaluate(inputs)
    core = model.algebraic_model.evaluate(
        phx.equations.AlgebraicLESInputs(
            inputs.favre_velocity_gradient, model.filter_scale
        )
    )
    density_reduction_error = max(
        float(
            jnp.max(
                jnp.abs(
                    result.density_weighted_deviatoric_sgs_stress / inputs.density
                    - core.specific_deviatoric_stress
                )
            )
        ),
        float(
            jnp.max(
                jnp.abs(
                    result.deviatoric_energy_transfer / inputs.density
                    - core.energy_transfer
                )
            )
        ),
    )
    species_closure = float(jnp.max(jnp.abs(jnp.sum(result.sgs_species_flux, axis=-2))))
    energy_flux_error = float(
        jnp.max(
            jnp.abs(
                result.conservative_total_energy_flux
                - result.stress_work_flux
                + result.sgs_enthalpy_flux
            )
        )
    )
    return {
        "constant_density_reduction_error": density_reduction_error,
        "species_flux_closure_error": species_closure,
        "total_energy_flux_identity_error": energy_flux_error,
        "favre_stress_magnitude": float(jnp.max(jnp.abs(result.sgs_stress))),
        "input_evidence_failure": (
            0.0 if bool(jnp.all(result.input_evidence.successful)) else 1.0
        ),
        "result_evidence_failure": (
            0.0 if bool(jnp.all(result.evidence.successful)) else 1.0
        ),
    }, {
        "closure_id": model.closure_id,
        "field_contract_id": fields.contract_id,
        "prepared_model_id": model.algebraic_model.prepared_id,
    }


def _unstructured_grid():
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0),
            (-1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, -1.0),
        )
    )
    x, y, z = vertices.T
    vertices = np.stack(
        (x + 0.13 * y + 0.04 * z, y + 0.09 * z + 0.03 * x, z + 0.07 * x),
        axis=-1,
    )
    tetrahedra = np.asarray(
        ((0, 1, 2, 3), (1, 2, 3, 4), (0, 2, 3, 5), (0, 1, 3, 6), (0, 1, 2, 7)),
        dtype=np.int32,
    )
    discretization = phx.discretization.UnstructuredFiniteVolumePlan(
        vertices, tetrahedra=tetrahedra
    ).prepare()
    gradient = phx.discretization.CellPolynomialReconstructionPlan(1).prepare(
        discretization
    )
    operators = phx.discretization.PreparedUnstructuredCollocatedOperators(
        discretization, gradient
    )
    return discretization, operators


def _run_unstructured(case: Mapping[str, object], _reference):
    coefficients = _mapping(case["coefficients"], "unstructured coefficients")
    timesteps = _mapping(case["timesteps"], "unstructured timesteps")
    parameters = _mapping(case["parameters"], "unstructured parameters")
    discretization, operators = _unstructured_grid()
    resolved_filter = _resolved_filter(
        "tetrahedral-control-volume",
        family="implicit-grid-volume",
        topology="unstructured",
        boundary_class="wall-bounded",
        commutation_status="unmodeled",
        repeated_filter_semantics="unmodeled",
    )
    provenance = phx.equations.LESParameterProvenance(
        resolved_filter,
        discretization.prepared_id,
        "variable-density-low-mach-tetrahedral-fv",
        source_kind="user",
        evidence_ids=(),
    )
    fields = phx.equations.FavreLESFieldContract("binary-mixture", ("a", "b"))
    favre = phx.equations.PreparedFavreLESModel(
        phx.equations.SmagorinskyLESPlan(float(coefficients["model"])).prepare(
            provenance
        ),
        phx.equations.LESFilterScale(discretization.directional_control_volume_widths()),
        fields,
        float(coefficients["turbulent_prandtl"]),
        (
            ("a", float(coefficients["a_schmidt"])),
            ("b", float(coefficients["b_schmidt"])),
        ),
        float(coefficients["viscosity_upper_bound"]),
    )
    prepared = phx.equations.UnstructuredLowMachLESPlan(
        favre, conservation_tolerance=float(coefficients["conservation_tolerance"])
    ).prepare(operators)
    centers = discretization.cell_centers
    density = 1.2 + 0.15 * centers[:, 0] + 0.04 * centers[:, 2]
    velocity = jnp.stack(
        (
            0.25 + 0.17 * centers[:, 0] - 0.08 * centers[:, 1],
            -0.11 + 0.06 * centers[:, 1] + 0.04 * centers[:, 2],
            0.09 - 0.05 * centers[:, 0] + 0.07 * centers[:, 2],
        ),
        axis=-1,
    )
    fraction_a = 0.42 + 0.06 * centers[:, 0] - 0.03 * centers[:, 2]
    fractions = jnp.stack((fraction_a, 1.0 - fraction_a), axis=-1)
    state = phx.equations.UnstructuredLowMachLESState(
        density, density[:, None] * velocity, density[:, None] * fractions
    )
    temperature = 295.0 + 3.0 * centers[:, 1] + centers[:, 2]
    result = prepared.semidiscrete_rate(
        state,
        2.0 + 0.2 * centers[:, 0] - 0.13 * centers[:, 1],
        temperature,
        1000.0 + 2.0 * centers[:, 0],
        jnp.stack((1005.0 * temperature, 1120.0 * temperature), axis=-1),
        0.012 + 0.002 * centers[:, 2],
        0.03 + 0.004 * centers[:, 0],
        jnp.stack(
            (0.008 + 0.001 * centers[:, 0], 0.006 + 0.001 * centers[:, 1]),
            axis=-1,
        ),
        0.08 + 0.01 * centers[:, 2],
    )
    evidence = result.evidence
    step_inputs = UnstructuredLowMachLESStepInputs(
        temperature,
        1000.0 + 2.0 * centers[:, 0],
        jnp.stack((1005.0 * temperature, 1120.0 * temperature), axis=-1),
        0.012 + 0.002 * centers[:, 2],
        0.03 + 0.004 * centers[:, 0],
        jnp.stack(
            (
                0.008 + 0.001 * centers[:, 0],
                0.006 + 0.001 * centers[:, 1],
            ),
            axis=-1,
        ),
    )
    step_size = float(timesteps["step_size"])
    method = prepared.prepare_fixed_step(
        step_size,
        pressure_tolerance=float(parameters["pressure_tolerance"]),
        pressure_iterations=int(parameters["pressure_iterations"]),
    )
    restart = method.initialize(
        state,
        2.0 + 0.2 * centers[:, 0] - 0.13 * centers[:, 1],
        step_inputs,
    )
    step_result = method.step_detailed(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        restart,
        jnp.asarray(step_size),
        step_inputs,
    )
    noncoercive_rate = eqx.tree_at(
        lambda value: value.fluxes.sgs_deviatoric_momentum_flux,
        step_result.rate,
        -step_result.rate.fluxes.sgs_deviatoric_momentum_flux,
    )
    noncoercive_evidence = method._evidence(
        restart,
        step_result.fixed_step.candidate_state,
        noncoercive_rate,
        step_result.pressure,
        step_result.restriction,
        jnp.asarray(step_size),
    )
    noncoercive_status = int(_step_status(noncoercive_evidence))
    return {
        "mass_balance_residual": float(jnp.max(jnp.abs(evidence.mass_balance_residual))),
        "momentum_balance_residual": float(
            jnp.max(jnp.abs(evidence.momentum_balance_residual))
        ),
        "scalar_balance_residual": float(
            jnp.max(jnp.abs(evidence.scalar_balance_residual))
        ),
        "enthalpy_balance_residual": float(
            jnp.max(jnp.abs(evidence.enthalpy_balance_residual))
        ),
        "negative_sgs_dissipation_violation": max(
            0.0, -float(evidence.modeled_sgs_dissipation)
        ),
        "sgs_flux_magnitude": max(
            float(jnp.max(jnp.abs(result.fluxes.sgs_momentum_flux))),
            float(jnp.max(jnp.abs(result.fluxes.sgs_scalar_flux))),
            float(jnp.max(jnp.abs(result.fluxes.sgs_enthalpy_flux))),
        ),
        "normalized_positive_sgs_work": float(
            step_result.evidence.normalized_positive_sgs_work
        ),
        "sgs_work_dissipative_failure": (
            0.0 if bool(step_result.evidence.sgs_work_dissipative) else 1.0
        ),
        "positive_work_refusal_failure": (
            0.0
            if bool(
                noncoercive_evidence.normalized_positive_sgs_work
                > float(coefficients["conservation_tolerance"])
            )
            and not bool(noncoercive_evidence.sgs_work_dissipative)
            and not bool(noncoercive_evidence.successful)
            and noncoercive_status == UNSTRUCTURED_LES_ENERGY_FAILURE
            else 1.0
        ),
        "algebraic_energy_status_failure": (
            0.0
            if bool(step_result.fixed_step.successful) and int(step_result.status) == 0
            else 1.0
        ),
        "route_failure": 0.0 if bool(evidence.successful) else 1.0,
    }, {
        "prepared_id": prepared.prepared_id,
        "mesh_id": prepared.mesh_id,
        "filter_id": prepared.filter_id,
        "model_id": prepared.model_id,
        "resource_evidence_id": evidence.resource_evidence_id,
        "algebraic_step_method_id": method.method_id,
        "noncoercive_status": noncoercive_status,
    }


def _immersed_route(
    count: int,
    coefficient: float,
    viscosity: float,
    /,
    *,
    wall_stress: bool,
):
    discretization, operators, momentum, pressure = _mac_core(count)
    pressure = phx.solver.MACPressureProjectionPlan(
        operators,
        boundaries=momentum.boundaries,
        solve_method="transform",
        tolerance=2.0e-9,
    )
    position = jnp.asarray(((1.7, 2.1, 2.4),))
    markers = phx.discretization.LagrangianMarkerSetPlan(
        jnp.asarray((7,)), position, jnp.asarray((1.0,))
    ).prepare()
    transfer = phx.discretization.MACMarkerTransferPlan(operators, markers).prepare()
    projection = phx.solver.MACImmersedBoundaryProjectionPlan(
        operators,
        transfer,
        boundaries=momentum.boundaries,
        tolerance=2.0e-7,
        maximum_iterations=300,
    )
    resolved_filter = _resolved_filter(
        "fixed-immersed-mac-cell-fluid-volume",
        family="implicit-grid-volume",
        topology="tensor-product",
        boundary_class="periodic",
        commutation_status="unmodeled",
        repeated_filter_semantics="unmodeled",
    )
    provenance = phx.equations.LESParameterProvenance(
        resolved_filter,
        discretization.prepared_id,
        "incompressible-unit-density",
        source_kind="user",
        evidence_ids=(),
    )
    algebraic = phx.equations.MACAlgebraicLESPlan(
        phx.equations.SmagorinskyLESPlan(coefficient).prepare(provenance)
    )
    kinematics = markers.kinematics(position, jnp.zeros_like(position))
    wall = VectorEquilibriumWallStressPlan().prepare(3) if wall_stress else None
    plan = FixedImmersedMACLESPlan(
        algebraic,
        projection,
        kinematics,
        jnp.ones(discretization.cell_shape),
        geometry_id="fixed-one-marker-body",
        wall_stress=wall,
        marker_wall_normal=(jnp.asarray(((1.0, 0.0, 0.0),)) if wall_stress else None),
        marker_sample_distance=(jnp.asarray((0.2,)) if wall_stress else None),
    )
    dynamics = compile_fixed_immersed_mac_les_flow(
        phx.equations.IncompressibleFlowProblem(3, viscosity),
        momentum,
        pressure,
        plan,
    )
    return discretization, plan, dynamics


def _run_immersed(case: Mapping[str, object], _reference):
    grids = _mapping(case["grids"], "immersed grids")
    coefficients = _mapping(case["coefficients"], "immersed coefficients")
    timesteps = _mapping(case["timesteps"], "immersed timesteps")
    count = int(grids["count"])
    coefficient = float(coefficients["model"])
    viscosity = float(coefficients["viscosity"])
    discretization, baseline_plan, baseline = _immersed_route(
        count, coefficient, viscosity, wall_stress=False
    )
    _, wall_plan, wall = _immersed_route(count, coefficient, viscosity, wall_stress=True)
    state = baseline.pack_velocity(
        tuple(0.05 * value for value in _mac_velocity(discretization))
    )
    baseline_method = phx.solver.MACImmersedBoundaryIMEXEulerMethod(
        baseline,
        baseline_plan.projection,
        baseline_plan.marker_motion,
        motion_id=baseline_plan.marker_motion.motion_id,
        fixed_step_size=float(timesteps["step_size"]),
        marker_constraint_normals=wall_plan.marker_wall_normal,
    )
    wall_method = wall_plan.imex_euler_method(
        wall, fixed_step_size=float(timesteps["step_size"])
    )
    baseline_step = baseline_method.step(0.0, state)
    wall_step = wall_method.step(0.0, state)
    wall_components = wall.rate_components(0.0, state)
    wall_stage = wall_components.les_stage
    if wall_stage is None or wall_stage.wall_stress is None:
        raise RuntimeError("Immersed wall-stress LES stage is unavailable.")
    ledger = wall.algebraic_les.balance_ledger(wall, wall_step)
    wall_rate_norm = _maximum_abs(wall_stage.wall_rate)
    wall_traction_magnitude = float(jnp.linalg.norm(wall_stage.wall_traction_density))
    trajectory_effect = float(jnp.linalg.norm(wall_step.state - baseline_step.state))
    normal_defect = float(
        jnp.max(
            jnp.abs(
                jnp.sum(
                    wall_stage.wall_traction_density * wall_plan.marker_wall_normal,
                    axis=-1,
                )
            )
        )
    )
    marker_slip = max(
        float(jnp.linalg.norm(baseline_step.projection.marker_slip)),
        float(jnp.linalg.norm(wall_step.projection.marker_slip)),
    )
    return {
        "normal_constraint_mode_violation": (
            0.0
            if baseline_step.projection.constraint_mode == "normal"
            and wall_step.projection.constraint_mode == "normal"
            else 1.0
        ),
        "maximum_marker_slip": marker_slip,
        "wall_traction_normal_defect": normal_defect,
        "wall_traction_magnitude": wall_traction_magnitude,
        "positive_modeled_wall_power_violation": max(
            0.0, float(wall_stage.modeled_wall_power)
        ),
        "wall_rate_missing": 0.0 if wall_rate_norm > 0.0 else 1.0,
        "immersed_sgs_action_magnitude": _maximum_abs(wall_stage.sgs_rate),
        "wall_on_off_trajectory_effect": trajectory_effect,
        "impulse_balance_residual": float(
            jnp.linalg.norm(ledger.impulse_balance_residual)
        ),
        "stress_work_balance_error": abs(
            float(ledger.fluid_stress_work - ledger.marker_stress_work)
        ),
        "execution_failure": (
            0.0
            if bool(
                baseline_step.accepted
                and wall_step.accepted
                and wall_stage.successful
                and ledger.successful
            )
            else 1.0
        ),
    }, {
        "baseline_compilation_id": baseline.compilation_id,
        "wall_compilation_id": wall.compilation_id,
        "baseline_prepared_les_id": baseline.algebraic_les.prepared_id,
        "wall_prepared_les_id": wall.algebraic_les.prepared_id,
        "geometry_id": wall.algebraic_les.geometry_id,
        "marker_id": wall.algebraic_les.marker_id,
        "solver_id": wall.algebraic_les.solver_id,
        "wall_stress_prepared_id": wall_plan.wall_stress.prepared_id,
        "admission_regime_id": wall_plan.admission_regime().plan_id,
        "temporal_method": "immersed-imex-euler-normal-constraint",
        "sbdf2_evidence": "not-claimed",
    }


def _run_lbm(case: Mapping[str, object], _reference):
    grids = _mapping(case["grids"], "LBM grids")
    timesteps = _mapping(case["timesteps"], "LBM timesteps")
    coefficients = _mapping(case["coefficients"], "LBM coefficients")
    result = lbm_smagorinsky_qualification(
        resolution=int(grids["resolution"]),
        steps=int(timesteps["steps"]),
        amplitude=float(coefficients["amplitude"]),
        base_relaxation_rate=float(coefficients["base_relaxation_rate"]),
        coefficient=float(coefficients["model"]),
    )
    molecular = _mapping(result["molecular"], "LBM molecular result")
    smagorinsky = _mapping(result["smagorinsky"], "LBM Smagorinsky result")
    reference = _mapping(result["reference"], "LBM reference result")
    return {
        "qualification_failure": 0.0 if bool(result["passed"]) else 1.0,
        "molecular_reference_relative_error": float(reference["relative_error"]),
        "maximum_mass_drift": max(
            float(molecular["global_mass_drift"]),
            float(smagorinsky["global_mass_drift"]),
        ),
        "maximum_momentum_drift": max(
            float(molecular["global_momentum_drift"]),
            float(smagorinsky["global_momentum_drift"]),
        ),
        "additional_decay_violation": max(
            0.0, -float(result["additional_amplitude_decay"])
        ),
        "backend_evidence_failure": (
            0.0
            if bool(
                smagorinsky["successful"]
                and smagorinsky["support_satisfied"]
                and smagorinsky["coefficient_active"]
            )
            else 1.0
        ),
    }, {
        "case": result["case"],
        "parameters": result["parameters"],
        "support": smagorinsky["support"],
    }


def _run_periodic_dynamic_production(case: Mapping[str, object], _reference):
    grids = _mapping(case["grids"], "dynamic production grids")
    timesteps = _mapping(case["timesteps"], "dynamic production timesteps")
    coefficients = _mapping(case["coefficients"], "dynamic production coefficients")
    resolved = _periodic_space(int(grids["resolved"]))
    test = _periodic_space(int(grids["test"]))
    primary_filter = _resolved_filter(
        "resolved retained Fourier projection",
        family="sharp-fourier-projection",
        topology="tensor-product",
        boundary_class="periodic",
        commutation_status="commuting",
        repeated_filter_semantics="idempotent",
    )
    test_filter = _resolved_filter(
        "coarse test Fourier projection",
        family="sharp-fourier-projection",
        topology="tensor-product",
        boundary_class="periodic",
        commutation_status="commuting",
        repeated_filter_semantics="idempotent",
    )
    provenance = phx.equations.LESParameterProvenance(
        primary_filter,
        resolved.prepared_id,
        "three-dimensional-periodic-unit-density",
        source_kind="user",
        evidence_ids=(),
    )
    dynamic_model = phx.equations.DynamicSmagorinskyPlan(
        phx.equations.LagrangianDynamicLESAveraging(
            float(coefficients["lagrangian_time_scale"])
        ),
        phx.equations.AdditiveDenominatorRegularization(
            float(coefficients["denominator_regularization"])
        ),
        phx.equations.NonnegativeBackscatterClip(),
    ).prepare(
        phx.equations.DynamicLESProvenance(provenance, test_filter, (2.0, 2.0, 2.0))
    )
    dynamic_plan = phx.equations.PeriodicDynamicLESPlan(
        dynamic_model,
        phx.equations.PeriodicFourierGridFilterPlan(primary_filter),
        phx.equations.PeriodicFourierTestFilterPlan(test_filter),
        phx.discretization.PseudospectralMethodPlan(
            dealiasing=phx.discretization.OversamplingDealiasingPlan(1.5)
        ),
        energy_tolerance=2.0e-8,
    )
    dynamics = phx.equations.compile_periodic_incompressible_flow(
        phx.equations.IncompressibleFlowProblem(3, float(coefficients["viscosity"])),
        resolved,
        phx.discretization.PseudospectralMethodPlan(
            dealiasing=phx.discretization.PaddingDealiasingPlan(2)
        ),
        dynamic_les=dynamic_plan,
        dynamic_test_discretization=test,
    )
    coordinates = phx.discretization.HermitianSpectralCoordinates(
        resolved, component_shape=(3,)
    )
    base_method = phx.solver.ETDRKMethod(2).prepare(
        dynamics.semilinear_drift, coordinates=coordinates
    )
    statistics = (
        phx.applications.incompressible_flow.PeriodicModalTurbulenceStatisticsPlan(
            dynamics, jnp.linspace(0.0, 7.0, 8)
        )
    )
    basis = phx.applications.incompressible_flow.SolenoidalHermitianFourierBasis(
        dynamics.projector, maximum_wavenumber=1.1
    )
    initial_velocity = basis.evaluate(jnp.linspace(0.2, 0.8, basis.coordinate_size))
    production_case = phx.applications.incompressible_flow.PeriodicSpectralProductionCase(
        dynamics,
        initial_velocity,
        case_id="periodic-dynamic-les-qualification",
    )
    step_size = float(timesteps["step_size"])
    plan = phx.applications.incompressible_flow.PeriodicSpectralProductionPlan(
        dynamics,
        base_method,
        statistics,
        production_case,
        start_time=0.0,
        end_time=step_size,
        step_size=step_size,
        checkpoint_interval=1,
    )
    with tempfile.TemporaryDirectory() as directory:
        prepared = plan.prepare(Path(directory))
        initial = prepared.initialize(initial_velocity)
        following, transition = prepared.step(initial)
        checkpointed = prepared.checkpoint(following)
        resumed = prepared.resume(checkpointed)
        snapshot = prepared.statistics_snapshot(following.time, following.accepted_state)
    velocity_restart_error = float(
        jnp.max(
            jnp.abs(resumed.accepted_state.velocity - following.accepted_state.velocity)
        )
    )
    continuation_restart_error = max(
        float(
            jnp.max(
                jnp.abs(
                    resumed.accepted_state.continuation_state.averaged_numerator
                    - following.accepted_state.continuation_state.averaged_numerator
                )
            )
        ),
        float(
            jnp.max(
                jnp.abs(
                    resumed.accepted_state.continuation_state.averaged_denominator
                    - following.accepted_state.continuation_state.averaged_denominator
                )
            )
        ),
    )
    return {
        "production_step_failure": (0.0 if bool(transition.successful) else 1.0),
        "velocity_restart_error": velocity_restart_error,
        "continuation_restart_error": continuation_restart_error,
        "continuation_update_failure": (
            0.0
            if int(following.accepted_state.continuation_state.accepted_updates)
            > int(initial.accepted_state.continuation_state.accepted_updates)
            else 1.0
        ),
        "dynamic_statistics_failure": (
            0.0
            if bool(
                snapshot.sgs_regularization_available
                & snapshot.sgs_stability_available
                & jnp.isfinite(snapshot.sgs_dynamic_coefficient_mean)
            )
            else 1.0
        ),
        "sgs_shell_balance_error": abs(
            float(snapshot.sgs_transfer_shells.total - snapshot.sgs_energy_rate)
        ),
        "dynamic_sgs_transfer_magnitude": abs(float(snapshot.sgs_energy_rate)),
    }, {
        "compilation_id": dynamics.compilation_id,
        "dynamic_les_id": dynamics.dynamic_les.prepared_id,
        "production_plan_id": plan.plan_id,
        "production_method_id": plan.method.method_id,
        "checkpoint_encoding_id": plan.checkpoint_encoding.encoding_id,
    }


def _run_dynamic_ksgs(case: Mapping[str, object], _reference):
    grids = _mapping(case["grids"], "dynamic KSGS grids")
    coefficients = _mapping(case["coefficients"], "dynamic KSGS coefficients")
    discretization, _, momentum, _ = _mac_core(int(grids["count"]))
    resolved_filter = _resolved_filter(
        "mac-dynamic-cell-volume",
        family="implicit-grid-volume",
        topology="tensor-product",
        boundary_class="periodic",
        commutation_status="commuting",
        repeated_filter_semantics="unmodeled",
    )
    test_filter = _resolved_filter(
        "mac-dynamic-binomial-test",
        family="explicit-filter",
        topology="tensor-product",
        boundary_class="periodic",
        commutation_status="commuting",
        repeated_filter_semantics="composed",
    )
    provenance = phx.equations.LESParameterProvenance(
        resolved_filter,
        discretization.prepared_id,
        "incompressible-unit-density",
        source_kind="user",
        evidence_ids=(),
    )
    plan = phx.equations.DynamicKSGSPlan(
        _ksgs_coefficients(coefficients),
        provenance,
        test_filter,
        2.0,
    )
    prepared = phx.equations.PreparedMACKSGS(plan, momentum, "sgs_kinetic_energy")
    kinetic = jnp.full(
        discretization.cell_shape,
        float(coefficients["initial_kinetic_energy"]),
    )
    viscosity = jnp.full(
        discretization.cell_shape,
        float(coefficients["molecular_viscosity"]),
    )
    initial, transport = prepared.prepare_transport(kinetic, viscosity)
    zeros = jnp.zeros(discretization.cell_shape)
    velocity = _mac_velocity(discretization)
    stage = prepared.momentum.boundaries.homogeneous_stage()
    updated = prepared.evaluate(
        velocity,
        stage,
        initial,
        transport,
        zeros,
        viscosity,
        zeros,
        accept_update=True,
    )
    continued, next_transport = prepared.prepare_transport(
        kinetic,
        viscosity,
        continuation_state=updated.result.state,
    )
    rejected = prepared.evaluate(
        tuple(jnp.roll(component, 1, axis=0) for component in velocity),
        stage,
        continued,
        next_transport,
        zeros,
        viscosity,
        zeros,
        accept_update=False,
    )
    restart_error = _tree_max_error(rejected.result.state, updated.result.state)
    coefficient_change = float(
        jnp.max(
            jnp.abs(
                updated.result.state.eddy_viscosity_coefficient
                - initial.eddy_viscosity_coefficient
            )
        )
    )
    sgs_action = _maximum_abs(updated.viscosity_result.physical_diffusive_rate)
    return {
        "accepted_update_failure": (
            0.0
            if bool(jnp.any(updated.result.state.dynamic_updates == 1) & updated.success)
            else 1.0
        ),
        "negative_coefficient_violation": max(
            0.0,
            -float(jnp.min(updated.result.state.eddy_viscosity_coefficient)),
        ),
        "dynamic_coefficient_change_magnitude": coefficient_change,
        "mac_sgs_action_magnitude": sgs_action,
        "rejected_restart_error": restart_error,
        "test_filter_binding_violation": (
            0.0
            if prepared.test_filter is not None
            and prepared.test_filter.plan.test_filter.filter_id
            == plan.test_filter.filter_id
            else 1.0
        ),
    }, {
        "plan_id": plan.plan_id,
        "prepared_id": prepared.prepared_id,
        "test_filter_id": prepared.test_filter.prepared_id,
    }


def _ocean_discretization():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(4, periodic=True),
            phx.discretization.UniformCellAxisSpec(4, periodic=True),
            phx.discretization.UniformCellAxisSpec(4),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, -1.0), (1.0, 1.0, 0.0))))
    return phx.discretization.FiniteVolumePlan(grid, component_names=("ocean",)).prepare()


def _run_low_re_ksgs(case: Mapping[str, object], _reference):
    coefficients = _mapping(case["coefficients"], "low-Re KSGS coefficients")
    discretization = _ocean_discretization()
    reference = phx.applications.ocean.LinearSeawaterReference()
    provenance = phx.equations.LESParameterProvenance(
        _resolved_filter(
            "mac-cell-volume",
            family="implicit-grid-volume",
            topology="tensor-product",
            boundary_class="wall-bounded",
            commutation_status="unmodeled",
            repeated_filter_semantics="unmodeled",
        ),
        discretization.prepared_id,
        "incompressible-unit-density",
        source_kind="user",
        evidence_ids=(),
    )
    plan = phx.equations.LowReKSGSPlan(
        _ksgs_coefficients(coefficients),
        phx.equations.LowReKSGSCoefficients(
            float(coefficients["wall_damping"]),
            float(coefficients["low_re_dissipation"]),
        ),
        provenance,
    )
    scalar_sgs = phx.discretization.MACScalarSGSPlan(
        (
            phx.discretization.MACScalarSGSField(
                reference.temperature_name,
                turbulent_prandtl_number=float(coefficients["turbulent_prandtl"]),
            ),
            phx.discretization.MACScalarSGSField(
                reference.salinity_name,
                turbulent_schmidt_number=float(coefficients["turbulent_schmidt"]),
            ),
        )
    )
    operators = phx.discretization.MACOperatorPlan(discretization).prepare()
    boundaries = phx.discretization.MACBoundaryPlan(
        operators,
        (
            phx.discretization.MACBoundarySide("z", "lower", "no-slip"),
            phx.discretization.MACBoundarySide("z", "upper", "no-slip"),
        ),
    )
    ocean = phx.applications.ocean.CartesianBoussinesqOceanPlan(
        phx.applications.ocean.OceanAxisConvention(),
        reference,
        viscosity=float(coefficients["viscosity"]),
        scalar_sgs=scalar_sgs,
        ksgs=plan,
        ksgs_field_name="sgs_kinetic_energy",
    ).prepare(discretization, boundaries=boundaries)
    z = discretization.cell_centers[..., 2]
    kinetic = (0.02 + 0.01 * (z + 1.0)) ** 2
    velocity = (
        jnp.sin(2.0 * jnp.pi * discretization.face_centers[0][..., 2]),
        jnp.zeros(discretization.face_layouts[1].shape),
        jnp.zeros(discretization.face_layouts[2].shape),
    )
    temperature = reference.reference_temperature + z
    salinity = jnp.full(discretization.cell_shape, reference.reference_salinity)
    state = ocean.initial_state(
        velocity,
        temperature,
        salinity,
        sgs_kinetic_energy=kinetic,
    )
    stage = ocean.dynamics.stage(0.0, state)
    if stage.ksgs is None or ocean.prepared_ksgs is None:
        raise RuntimeError("Low-Re KSGS stage is unavailable.")
    wall_distance = ocean.prepared_ksgs.wall_distance
    undamped = (
        plan.coefficients.eddy_viscosity
        * ocean.prepared_ksgs.filter_scale.equivalent_width
        * jnp.sqrt(kinetic)
    )
    return {
        "wall_distance_failure": (
            0.0
            if wall_distance is not None and bool(jnp.all(wall_distance > 0.0))
            else 1.0
        ),
        "damping_violation": max(
            0.0,
            float(jnp.max(stage.ksgs.result.eddy_viscosity - undamped)),
        ),
        "low_re_dissipation_missing": (
            0.0
            if bool(jnp.any(stage.ksgs.result.contributions.low_re_dissipation > 0.0))
            else 1.0
        ),
        "stage_failure": 0.0 if bool(stage.success) else 1.0,
    }, {
        "plan_id": plan.plan_id,
        "prepared_id": ocean.prepared_ksgs.prepared_id,
    }


def _learned_stress_predictor(features, args):
    gradient = features.reshape(features.shape[:-1] + (3, 3))
    strain = 0.5 * (gradient + jnp.swapaxes(gradient, -1, -2))
    trace = jnp.trace(strain, axis1=-2, axis2=-1)
    deviatoric = strain - (trace / 3.0)[..., None, None] * jnp.eye(3, dtype=strain.dtype)
    coefficient = jnp.asarray(0.2 if args is None else args, dtype=features.dtype)
    return -2.0 * coefficient * deviatoric


def _learned_binding(
    sample_shape,
    dtype,
    resolved_filter,
    discretization_id: str,
    regime: str,
    artifact_id: str,
):
    flow_schema_id = f"flow-{discretization_id}"
    schema = phx.closure_data.LearnedStressFeatureSchema(
        name=LEARNED_STRESS_FEATURE_NAME,
        component_names=LEARNED_STRESS_VELOCITY_GRADIENT_COMPONENTS,
        component_units=LEARNED_STRESS_VELOCITY_GRADIENT_UNITS,
        shape=sample_shape + (9,),
        dtype=dtype,
        flow_schema_id=flow_schema_id,
    )
    provenance = phx.equations.LESParameterProvenance(
        resolved_filter,
        discretization_id,
        regime,
        source_kind="user",
        evidence_ids=(),
    )
    output = phx.closure_data.LearnedStressOutputContract(
        shape=sample_shape + (3, 3),
        dtype=dtype,
        units="(m/s)^2",
        target_id="deviatoric-specific-stress-target",
        filter_id=resolved_filter.filter_id,
        discretization_id=discretization_id,
        regime=regime,
        symmetry_tolerance=2.0e-6,
        trace_tolerance=2.0e-6,
    )
    normalizer_provenance = phx.closure_data.NormalizerProvenance(
        partition_id="qualification-training-partition",
        training_assignment_ids=("qualification-assignment",),
        training_sample_ids=("qualification-sample",),
        feature_name=LEARNED_STRESS_FEATURE_NAME,
        schema_id=flow_schema_id,
    )
    normalizer = phx.closure_data.TrainOnlyNormalizer(
        jnp.zeros((9,), dtype=dtype),
        jnp.ones((9,), dtype=dtype),
        normalizer_provenance,
        epsilon=1.0e-12,
    )
    plan = phx.closure_data.LearnedStressBindingPlan(
        schema,
        output,
        resolved_filter,
        provenance,
        model_artifact_id=artifact_id,
        normalizer_id=normalizer.normalizer_id,
        energy_policy="dissipative",
    )
    return plan.prepare(
        _learned_stress_predictor,
        normalizer,
        model_artifact_id=artifact_id,
        target_id=output.target_id,
        output_units=output.units,
    )


def _run_learned_stress(case: Mapping[str, object], _reference):
    backend = str(_mapping(case["parameters"], "learned parameters")["backend"])
    count = int(_mapping(case["grids"], "learned grids")["count"])
    coefficients = _mapping(case["coefficients"], "learned coefficients")
    predictor_viscosity = float(coefficients["predictor_viscosity"])
    if backend == "periodic":
        space = _periodic_space(count)
        resolved_filter = _resolved_filter(
            "retained Fourier grid",
            family="sharp-fourier-projection",
            topology="tensor-product",
            boundary_class="periodic",
            commutation_status="commuting",
            repeated_filter_semantics="idempotent",
        )
        binding = _learned_binding(
            space.physical_shape,
            jnp.dtype(space.plan.precision.physical_dtype),
            resolved_filter,
            space.prepared_id,
            "three-dimensional-periodic-unit-density",
            "qualification-periodic-learned-stress",
        )
        projector = phx.discretization.PeriodicLerayProjector(space)
        prepared = PeriodicLearnedStressPlan(binding).prepare(space, projector)
        state = projector.project(space.project(_periodic_velocity(space)))
        result = prepared(state, predictor_viscosity)
        conservation = float(result.momentum_conservation_defect)
        divergence = float(result.divergence_norm)
        work_defect = max(
            abs(float(result.energy_identity_defect)),
            abs(float(result.projection_work_defect)),
        )
        projected_rate_magnitude = float(jnp.max(jnp.abs(result.projected_rate)))
    elif backend == "mac":
        discretization, operators, momentum, projection = _mac_core(count)
        resolved_filter = _resolved_filter(
            "mac-cell-volume",
            family="implicit-grid-volume",
            topology="tensor-product",
            boundary_class="periodic",
            commutation_status="unmodeled",
            repeated_filter_semantics="unmodeled",
        )
        binding = _learned_binding(
            discretization.cell_shape,
            operators.pressure_space.dtype,
            resolved_filter,
            discretization.prepared_id,
            "incompressible-unit-density",
            "qualification-mac-learned-stress",
        )
        prepared = MACLearnedStressPlan(binding).prepare(momentum, projection)
        result = prepared(
            _mac_velocity(discretization),
            momentum.boundaries.homogeneous_stage(),
            predictor_viscosity,
        )
        conservation = float(jnp.max(jnp.abs(result.momentum_conservation_defect)))
        divergence = float(jnp.max(jnp.abs(result.projection.divergence_after)))
        work_defect = abs(float(result.energy_identity_defect))
        projected_rate_magnitude = _maximum_abs(result.projected_rate)
    else:
        raise ValueError("Learned-stress backend must be periodic or mac.")
    symmetry = float(
        jnp.max(
            jnp.abs(
                result.learned_result.stress
                - jnp.swapaxes(result.learned_result.stress, -1, -2)
            )
        )
    )
    trace = float(
        jnp.max(jnp.abs(jnp.trace(result.learned_result.stress, axis1=-2, axis2=-1)))
    )
    return {
        "momentum_conservation_defect": conservation,
        "projection_divergence_norm": divergence,
        "work_identity_defect": work_defect,
        "stress_symmetry_defect": symmetry,
        "stress_trace_defect": trace,
        "learned_stress_magnitude": float(jnp.max(jnp.abs(result.learned_result.stress))),
        "projected_rate_magnitude": projected_rate_magnitude,
        "energy_policy_failure": (
            0.0
            if bool(result.energy_policy_satisfied & (result.integrated_transfer >= 0.0))
            else 1.0
        ),
        "backend_failure": 0.0 if bool(result.successful) else 1.0,
    }, {
        "backend": backend,
        "binding_id": binding.plan.plan_id,
        "prepared_id": prepared.prepared_id,
        "model_artifact_id": binding.plan.model_artifact_id,
    }


def _boundary_channel(
    tangential_boundary: str,
    viscosity: float,
    model_coefficient: float,
    perturbation_amplitude: float,
    shape: tuple[int, int, int],
):
    space = phx.discretization.TensorSpectralPlan(
        (
            phx.discretization.FourierBasisPlan(shape[0]),
            phx.discretization.ChebyshevBasisPlan(shape[1]),
            phx.discretization.FourierBasisPlan(shape[2]),
        ),
        axis_names=("x", "y", "z"),
        field_name="velocity",
    ).prepare(
        (
            phx.discretization.AxisDomain.periodic(0.0, 2.0 * math.pi),
            phx.discretization.AxisDomain.interval(-1.0, 1.0),
            phx.discretization.AxisDomain.periodic(0.0, 2.0 * math.pi),
        )
    )
    stokes = phx.discretization.ChannelStokesPlan(
        space,
        viscosity,
        tangential_boundary=tangential_boundary,
        mean_constraint=phx.discretization.ChannelMeanConstraint(
            "pressure_gradient", (0.0, 0.0)
        ),
    )
    base = phx.equations.compile_channel_flow(
        phx.equations.IncompressibleFlowProblem(3, viscosity),
        stokes,
        phx.discretization.PseudospectralMethodPlan(
            dealiasing=phx.discretization.PaddingDealiasingPlan(2)
        ),
    )
    provenance = phx.equations.LESParameterProvenance(
        phx.equations.channel_les_filter(space),
        space.prepared_id,
        "wall-resolved-channel",
        source_kind="user",
        evidence_ids=(),
    )
    dynamics = phx.equations.compile_channel_les(
        base,
        phx.equations.WALELESPlan(model_coefficient).prepare(provenance),
    )
    x = space.axes[0].nodes[:, None, None]
    y = space.axes[1].nodes[None, :, None]
    envelope = (1.0 - y**2) ** 2
    perturbation = perturbation_amplitude
    velocity = jnp.zeros(space.physical_shape + (3,))
    velocity = velocity.at[..., 0].set(
        jnp.broadcast_to(
            0.1 * (1.0 - y**2) - 4.0 * perturbation * y * (1.0 - y**2) * jnp.sin(x),
            space.physical_shape,
        )
    )
    velocity = velocity.at[..., 1].set(
        jnp.broadcast_to(
            -perturbation * envelope * jnp.cos(x),
            space.physical_shape,
        )
    )
    return space, dynamics, dynamics.project_state(velocity)


def _run_channel_wall_owner(case: Mapping[str, object], _reference):
    timesteps = _mapping(case["timesteps"], "channel wall timesteps")
    coefficients = _mapping(case["coefficients"], "channel wall coefficients")
    parameters = _mapping(case["parameters"], "channel wall parameters")
    grids = _mapping(case["grids"], "channel wall grids")
    shape = tuple(int(value) for value in grids["shape"])
    _, off, initial_off = _boundary_channel(
        "velocity",
        float(coefficients["viscosity"]),
        float(coefficients["model"]),
        float(parameters["perturbation_amplitude"]),
        shape,
    )
    _, on, initial_on = _boundary_channel(
        "traction",
        float(coefficients["viscosity"]),
        float(coefficients["model"]),
        float(parameters["perturbation_amplitude"]),
        shape,
    )
    maximum_step = min(
        float(off.explicit_restriction(initial_off).maximum_step),
        float(on.explicit_restriction(initial_on).maximum_step),
    )
    requested = float(timesteps["step_size"])
    step = min(requested, 0.1 * maximum_step)
    off_solution = phx.solver.solve_channel_sbdf2(
        off, initial_off, jnp.asarray((0.0, step, 2.0 * step))
    )
    owner = VectorEquilibriumWallStressPlan().prepare_channel(
        on,
        step,
        density=float(coefficients["density"]),
        sample_distance=(
            float(coefficients["sample_distance"]),
            float(coefficients["sample_distance"]),
        ),
    )
    initial = owner.initialize(initial_on, 0.0, None)
    first = owner.step(0, 0.0, initial, step, None)
    second = owner.step(1, step, first.accepted_state, step, None)
    trajectory = float(
        jnp.linalg.norm(
            off_solution.velocity[-1] - second.accepted_state.channel.current_velocity
        )
    )
    return {
        "wall_on_off_trajectory_effect": trajectory,
        "boundary_work_defect": abs(float(second.evidence.energy_boundary_work_defect)),
        "wall_power_identity_error": abs(
            float(
                second.evidence.energy_ledger.wall_power
                - second.evidence.stokes.boundary_power
            )
        ),
        "dissipation_failure": (0.0 if bool(second.evidence.dissipative) else 1.0),
        "channel_les_viscosity_magnitude": float(
            on.explicit_restriction(
                second.accepted_state.channel.current_velocity
            ).maximum_kinematic_viscosity
        ),
        "owner_execution_failure": (
            0.0
            if bool(off_solution.successful & first.successful & second.successful)
            else 1.0
        ),
    }, {
        "off_compilation_id": off.compilation_id,
        "on_compilation_id": on.compilation_id,
        "owner_id": owner.prepared_id,
        "temporal_method": "wall-owned-channel-sbdf2",
    }


def _run_channel_restriction(case: Mapping[str, object], _reference):
    coefficients = _mapping(case["coefficients"], "channel restriction coefficients")
    parameters = _mapping(case["parameters"], "channel restriction parameters")
    grids = _mapping(case["grids"], "channel restriction grids")
    shape = tuple(int(value) for value in grids["shape"])
    _, dynamics, initial = _boundary_channel(
        "velocity",
        float(coefficients["viscosity"]),
        float(coefficients["model"]),
        float(parameters["perturbation_amplitude"]),
        shape,
    )
    restriction = dynamics.explicit_restriction(initial)
    unsafe = 2.0 * float(restriction.maximum_step)
    solution = phx.solver.solve_channel_sbdf2(
        dynamics, initial, jnp.asarray((0.0, unsafe))
    )
    unchanged = float(jnp.max(jnp.abs(solution.velocity[-1] - solution.velocity[0])))
    return {
        "advective_rate_missing": (
            0.0 if float(restriction.advective_rate) > 0.0 else 1.0
        ),
        "wall_derivative_missing": (
            0.0 if float(restriction.wall_normal_derivative_norm) > 0.0 else 1.0
        ),
        "channel_les_viscosity_magnitude": float(restriction.maximum_kinematic_viscosity),
        "unsafe_step_permitted": (0.0 if not bool(restriction.permits(unsafe)) else 1.0),
        "unsafe_step_state_change": unchanged,
        "unsafe_step_accepted": (0.0 if not bool(solution.successful) else 1.0),
    }, {
        "compilation_id": dynamics.compilation_id,
        "temporal_method": restriction.temporal_method,
        "restriction_status": int(solution.diagnostics.status[0]),
    }


def _run_stochastic_mac_inflow(case: Mapping[str, object], _reference):
    coefficients = _mapping(case["coefficients"], "inflow coefficients")
    timesteps = _mapping(case["timesteps"], "inflow timesteps")
    parameters = _mapping(case["parameters"], "inflow parameters")
    angles = 0.5 * jnp.pi * jnp.arange(4)
    coordinates = jnp.stack((jnp.zeros_like(angles), angles), axis=-1)
    owner = StochasticTurbulentInflowPlan("spectral").prepare_mac_boundary(
        coordinates,
        jnp.asarray((1.0, 0.0)),
        jnp.ones((4,)),
        jnp.asarray(
            (
                (float(coefficients["velocity_variance"]), 0.0),
                (0.0, 0.0),
            )
        ),
        axis="x",
        side="lower",
        boundary_shape=(4,),
        spectral_wavevectors=jnp.asarray(((0.0, 1.0),)),
    )
    initial = owner.initialize(
        jax.random.key(int(parameters["seed"])),
        0.0,
        mean_velocity=jnp.asarray((float(coefficients["mean_normal_velocity"]), 0.0)),
    )
    first = owner.advance(
        initial.state,
        float(timesteps["step_size"]),
        mean_velocity=jnp.asarray((float(coefficients["mean_normal_velocity"]), 0.0)),
    )
    restored = StochasticTurbulentInflowMACBoundaryState(
        inflow_state=initial.state.inflow_state,
        velocity=initial.state.velocity,
        scalars=initial.state.scalars,
        time=initial.state.time,
        accepted_steps=initial.state.accepted_steps,
        prepared_id=initial.state.prepared_id,
    )
    replay = owner.advance(
        restored,
        float(timesteps["step_size"]),
        mean_velocity=jnp.asarray((float(coefficients["mean_normal_velocity"]), 0.0)),
    )
    replay_error = max(
        float(jnp.max(jnp.abs(first.provider.value - replay.provider.value))),
        float(jnp.max(jnp.abs(first.provider.rate - replay.provider.rate))),
    )
    return {
        "restart_replay_error": replay_error,
        "rate_closure_error": float(first.evidence.rate_closure_error),
        "fluctuation_volume_flux": abs(float(first.evidence.fluctuation_volume_flux)),
        "maximum_divergence_residual": float(first.evidence.maximum_divergence_residual),
        "owner_failure": (
            0.0
            if bool(
                first.evidence.successful & (first.boundary.kind == "velocity-inflow")
            )
            else 1.0
        ),
    }, {
        "prepared_id": owner.prepared_id,
        "inflow_id": owner.inflow.prepared_id,
        "boundary_kind": first.boundary.kind,
        "commit_semantics": "accepted-fluid-step-only",
    }


def _binary_species_schema():
    return ChemicalSpeciesSchema.from_unique_species(
        ("fuel", "oxidizer"),
        (ChemicalPhaseKind.GAS, ChemicalPhaseKind.GAS),
        jnp.asarray((0.020, 0.032)),
        ("F", "O"),
        jnp.eye(2, dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.int32),
        gas_standard_pressure=1.0e5,
    )


def _binary_thermodynamics(schema):
    calorics = PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.full((2, 1), 2.5 * UNIVERSAL_GAS_CONSTANT),
        jnp.asarray((1.0e3, 2.0e3)),
        reference_molar_entropy=jnp.asarray((100.0, 110.0)),
        reference_temperature=300.0,
        minimum_temperature=120.0,
        maximum_temperature=1500.0,
    )
    return HomogeneousHelmholtzPlan(
        IdealGasReferenceHelmholtzTerm(schema, calorics),
        ZeroResidualHelmholtzTerm(schema),
    )


def _transported_favre_model(schema, coefficients):
    resolved_filter = _resolved_filter(
        "favre-cell-volume",
        family="implicit-grid-volume",
        topology="tensor-product",
        boundary_class="open",
        commutation_status="unmodeled",
        repeated_filter_semantics="unmodeled",
    )
    provenance = phx.equations.LESParameterProvenance(
        resolved_filter,
        "three-dimensional-compressible-cell-transport",
        "variable-density-smooth-region",
        source_kind="user",
        evidence_ids=(),
    )
    fields = phx.equations.FavreLESFieldContract(schema.schema_id, schema.species_names)
    return phx.equations.PreparedFavreLESModel(
        phx.equations.SmagorinskyLESPlan(float(coefficients["model"])).prepare(
            provenance
        ),
        phx.equations.LESFilterScale(jnp.asarray((0.05, 0.06, 0.07))),
        fields,
        float(coefficients["turbulent_prandtl"]),
        (
            ("fuel", float(coefficients["fuel_schmidt"])),
            ("oxidizer", float(coefficients["oxidizer_schmidt"])),
        ),
        float(coefficients["viscosity_upper_bound"]),
        isotropic_trace_policy="provided-sgs-kinetic-energy",
        sgs_kinetic_energy_dissipation_coefficient=float(coefficients["sgs_dissipation"]),
        sgs_kinetic_energy_turbulent_schmidt_number=float(
            coefficients["kinetic_schmidt"]
        ),
    )


def _run_favre_dg_energy(case: Mapping[str, object], _reference):
    coefficients = _mapping(case["coefficients"], "Favre DG coefficients")
    schema = _binary_species_schema()
    closure = _transported_favre_model(schema, coefficients)
    system = HomogeneousMixtureCompressibleNavierStokesSystem(
        _binary_thermodynamics(schema),
        ConstantTransport(0.0, 0.0),
        3,
        favre_les=closure,
    )
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    mesh = CellMesh(
        points,
        (
            CellBlock(
                "cells",
                "tetrahedron",
                np.arange(4, dtype=np.int32)[None, :],
            ),
        ),
    )
    discretization = FiniteElementPlan(
        mesh,
        FiniteElementFieldSpec(
            "state",
            discontinuous_element("tetrahedron", 1),
            component_shape=(system.component_count,),
        ),
    ).prepare()
    exterior = tuple(
        int(value)
        for value in np.asarray(discretization.exterior_facet_domain.entity_indices)
    )
    boundaries = FiniteElementBoundarySet(
        discretization,
        {"outflow": (exterior, ExtrapolationBoundary())},
    )
    compiled = compile_conservation_problem(
        ConservationProblemIR(
            "transported-favre-sgs-energy-qualification",
            "state",
            system,
            boundaries,
        ),
        discretization,
        NodalDGConservationMethodPlan(
            RusanovFluxPlan(),
            viscous=ViscousDGPlan(
                formulation="ldg",
                boundary_closures=(
                    ViscousBoundaryClosure(boundaries.patches[0].boundary.boundary_id),
                ),
            ),
        ),
    )
    point_state = system.primitive_to_conserved(
        jnp.asarray((0.36, 0.84, 0.0, 0.0, 0.0, 400.0, 0.25))
    )
    state = jnp.broadcast_to(
        point_state, discretization.field_spaces[0].vector_space.shape
    )
    rate = compiled(0.0, state)
    expected = system.favre_les_coupled_rate(
        point_state, jnp.zeros(point_state.shape + (3,))
    ).conserved_source
    source_error = float(jnp.max(jnp.abs(rate - jnp.broadcast_to(expected, rate.shape))))
    return {
        "dg_source_error": source_error,
        "total_energy_exchange_error": float(
            jnp.max(jnp.abs(rate[..., system.energy_index]))
        ),
        "negative_sgs_source_missing": (
            0.0
            if bool(jnp.all(rate[..., system.sgs_kinetic_energy_index] < 0.0))
            else 1.0
        ),
        "transported_slot_missing": (
            0.0 if system.sgs_kinetic_energy_index is not None else 1.0
        ),
    }, {
        "closure_id": closure.closure_id,
        "system_id": system.system_id,
        "compilation_id": compiled.compilation_id,
        "discretization_id": discretization.prepared_id,
        "sgs_kinetic_energy_index": system.sgs_kinetic_energy_index,
    }


def _pressure_stepped_unstructured(coefficients):
    discretization, operators = _unstructured_grid()
    resolved_filter = _resolved_filter(
        "tetrahedral-control-volume",
        family="implicit-grid-volume",
        topology="unstructured",
        boundary_class="wall-bounded",
        commutation_status="unmodeled",
        repeated_filter_semantics="unmodeled",
    )
    provenance = phx.equations.LESParameterProvenance(
        resolved_filter,
        discretization.prepared_id,
        "variable-density-low-mach-tetrahedral-fv",
        source_kind="user",
        evidence_ids=(),
    )
    ksgs_plan = phx.equations.StaticKSGSPlan(_ksgs_coefficients(coefficients), provenance)
    favre = phx.equations.PreparedFavreLESModel(
        phx.equations.SmagorinskyLESPlan(float(coefficients["model"])).prepare(
            provenance
        ),
        phx.equations.LESFilterScale(discretization.directional_control_volume_widths()),
        phx.equations.FavreLESFieldContract("binary-mixture", ("a", "b")),
        float(coefficients["turbulent_prandtl"]),
        (
            ("a", float(coefficients["a_schmidt"])),
            ("b", float(coefficients["b_schmidt"])),
        ),
        float(coefficients["viscosity_upper_bound"]),
        isotropic_trace_policy="provided-sgs-kinetic-energy",
    )
    prepared = phx.equations.UnstructuredLowMachLESPlan(
        favre,
        ksgs_plan=ksgs_plan,
        conservation_tolerance=float(coefficients["conservation_tolerance"]),
    ).prepare(operators)
    centers = discretization.cell_centers
    density = jnp.full((centers.shape[0],), 1.3)
    velocity = centers @ jnp.diag(jnp.asarray((1.0, 1.0, -2.0))) + jnp.asarray(
        (0.1, -0.03, 0.02)
    )
    fraction_a = 0.45 + 0.04 * centers[:, 0] - 0.02 * centers[:, 2]
    fractions = jnp.stack((fraction_a, 1.0 - fraction_a), axis=-1)
    state = phx.equations.UnstructuredLowMachLESState(
        density,
        density[:, None] * velocity,
        density[:, None] * fractions,
        ksgs=ksgs_plan.initialize_state(0.02 + 0.003 * centers[:, 1]),
    )
    pressure = 0.2 * centers[:, 0] - 0.13 * centers[:, 1]
    temperature = 295.0 + 2.0 * centers[:, 1] + centers[:, 2]
    inputs = UnstructuredLowMachLESStepInputs(
        temperature,
        1000.0 + 2.0 * centers[:, 0],
        jnp.stack((1005.0 * temperature, 1120.0 * temperature), axis=-1),
        0.009 + 0.001 * centers[:, 2],
        0.03 + 0.003 * centers[:, 0],
        jnp.stack(
            (
                0.006 + 0.001 * centers[:, 0],
                0.005 + 0.001 * centers[:, 1],
            ),
            axis=-1,
        ),
    )
    return prepared, state, pressure, inputs


def _run_unstructured_pressure(case: Mapping[str, object], _reference):
    timesteps = _mapping(case["timesteps"], "unstructured timesteps")
    coefficients = _mapping(case["coefficients"], "unstructured coefficients")
    prepared, state, pressure, inputs = _pressure_stepped_unstructured(coefficients)
    step = float(timesteps["step_size"])
    method = prepared.prepare_fixed_step(
        step, pressure_tolerance=2.0e-8, pressure_iterations=300
    )
    restart = method.initialize(state, pressure, inputs)
    first = method.step_detailed(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        restart,
        jnp.asarray(step),
        inputs,
    )
    replay = method.step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        restart,
        jnp.asarray(step),
        inputs,
    )
    second = method.step_detailed(
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(step),
        first.fixed_step.accepted_state,
        jnp.asarray(step),
        inputs,
    )
    replay_error = _tree_max_error(first.fixed_step.accepted_state, replay.accepted_state)
    if first.rate.ksgs is None or second.rate.ksgs is None:
        raise RuntimeError("Unstructured pressure KSGS evidence is unavailable.")
    production_policy = "conservative-face-work-equal-cell-split"
    expected_sgs_transport_id = content_address(
        {
            "kind": "unstructured-low-mach-sgs-transport",
            "favre": prepared.plan.favre_model.closure_id,
            "algebraic_model": (prepared.plan.favre_model.algebraic_model.prepared_id),
            "ksgs": prepared.plan.ksgs_plan.plan_id,
            "eddy_viscosity_owner": "static-ksgs",
            "ksgs_production_discretization": production_policy,
            "ksgs_production_limit_disposition": ("modeled-enthalpy-density-source"),
            "nonorthogonal_correction": (prepared.nonorthogonal_correction_id),
        }
    )
    deviatoric_face_stress_magnitude = max(
        float(jnp.max(jnp.abs(result.rate.fluxes.sgs_deviatoric_momentum_flux)))
        for result in (first, second)
    )
    minimum_raw_production = min(
        float(jnp.min(result.rate.ksgs_raw_production_density))
        for result in (first, second)
    )
    normalized_energy_balance = max(
        float(first.evidence.normalized_resolved_sgs_energy_balance),
        float(second.evidence.normalized_resolved_sgs_energy_balance),
    )
    normalized_modeled_transfer = max(
        float(first.evidence.normalized_modeled_transfer_residual),
        float(second.evidence.normalized_modeled_transfer_residual),
    )
    viscosity_owner_error = max(
        float(
            jnp.max(
                jnp.abs(
                    result.rate.kinematic_eddy_viscosity - result.rate.ksgs.eddy_viscosity
                )
            )
        )
        for result in (first, second)
    )
    production_limit_reduction_magnitude = max(
        float(jnp.max(result.rate.ksgs_production_limit_reduction_density))
        for result in (first, second)
    )
    modeled_enthalpy_source_magnitude = max(
        float(jnp.max(result.rate.modeled_enthalpy_source_density))
        for result in (first, second)
    )
    enthalpy_source_identity_error = max(
        float(
            jnp.max(
                jnp.abs(
                    result.rate.modeled_enthalpy_source_density
                    - result.rate.ksgs_production_limit_reduction_density
                )
            )
        )
        for result in (first, second)
    )
    thermalization_rate_magnitude = min(
        float(result.evidence.production_limit_thermalization_rate)
        for result in (first, second)
    )
    enthalpy_thermalization_balance_error = max(
        abs(
            float(
                jnp.sum(
                    prepared.operators.discretization.cell_volumes
                    * (
                        result.fixed_step.candidate_state.enthalpy_density
                        - current.enthalpy_density
                    )
                )
                - step * result.evidence.production_limit_thermalization_rate
            )
        )
        for result, current in (
            (first, restart),
            (second, first.fixed_step.accepted_state),
        )
    )
    modeled_energy_split_residual = max(
        abs(float(result.rate.evidence.modeled_energy_split_residual))
        for result in (first, second)
    )
    rhs_rebuild_error = 0.0
    production_rebuild_error = 0.0
    for result, current in (
        (first, restart),
        (second, first.fixed_step.accepted_state),
    ):
        expected_rhs = (
            result.rate.ksgs_production_density / current.conservative.density
            - result.rate.ksgs.contributions.dissipation
            + result.rate.ksgs.contributions.diffusion
            + result.rate.ksgs.contributions.buoyancy
            - result.rate.ksgs.contributions.low_re_dissipation
        )
        rhs_rebuild_error = max(
            rhs_rebuild_error,
            float(jnp.max(jnp.abs(result.rate.ksgs.contributions.rhs - expected_rhs))),
        )
        density = current.conservative.density
        production_rebuild_error = max(
            production_rebuild_error,
            float(
                jnp.max(
                    jnp.abs(
                        result.rate.ksgs_raw_production_density
                        - density * result.rate.ksgs.contributions.raw_production
                    )
                )
            ),
            float(
                jnp.max(
                    jnp.abs(
                        result.rate.ksgs_production_density
                        - density * result.rate.ksgs.contributions.production
                    )
                )
            ),
            float(
                jnp.max(
                    jnp.abs(
                        result.rate.ksgs_production_limit_reduction_density
                        - (
                            result.rate.ksgs_raw_production_density
                            - result.rate.ksgs_production_density
                        )
                    )
                )
            ),
        )
    negative_raw_production = -jnp.maximum(
        jnp.abs(first.rate.ksgs_raw_production_density),
        jnp.asarray(1.0e-12),
    )
    negative_rate = eqx.tree_at(
        lambda value: (
            value.ksgs_raw_production_density,
            value.ksgs.evidence.production_nonnegative,
        ),
        first.rate,
        (
            negative_raw_production,
            jnp.zeros_like(
                first.rate.ksgs.evidence.production_nonnegative,
                dtype=bool,
            ),
        ),
    )
    negative_evidence = method._evidence(
        restart,
        first.fixed_step.candidate_state,
        negative_rate,
        first.pressure,
        first.restriction,
        jnp.asarray(step),
    )
    negative_status = int(_step_status(negative_evidence))
    return {
        "divergence_after_norm": float(first.evidence.divergence_after_norm),
        "pressure_residual_norm": float(first.evidence.pressure_residual_norm),
        "mass_flux_identity_error": float(
            jnp.max(
                jnp.abs(
                    first.fixed_step.accepted_state.mass_flux
                    - first.rate.fluxes.mass_flux
                )
            )
        ),
        "sgs_flux_magnitude": max(
            float(jnp.max(jnp.abs(first.rate.fluxes.sgs_momentum_flux))),
            float(jnp.max(jnp.abs(first.rate.fluxes.sgs_scalar_flux))),
            float(jnp.max(jnp.abs(first.rate.fluxes.sgs_enthalpy_flux))),
        ),
        "restart_replay_error": replay_error,
        "normalized_sgs_energy_balance": normalized_energy_balance,
        "normalized_modeled_transfer_residual": normalized_modeled_transfer,
        "production_limit_reduction_magnitude": (production_limit_reduction_magnitude),
        "modeled_enthalpy_source_magnitude": (modeled_enthalpy_source_magnitude),
        "enthalpy_source_identity_error": (enthalpy_source_identity_error),
        "thermalization_rate_magnitude": thermalization_rate_magnitude,
        "enthalpy_thermalization_balance_error": (enthalpy_thermalization_balance_error),
        "modeled_energy_split_residual": modeled_energy_split_residual,
        "total_energy_balance_failure": (
            0.0
            if bool(first.evidence.energy_balanced & second.evidence.energy_balanced)
            else 1.0
        ),
        "executed_rhs_rebuild_error": rhs_rebuild_error,
        "production_evidence_rebuild_error": production_rebuild_error,
        "ksgs_production_policy_failure": (
            0.0 if prepared.sgs_transport_id == expected_sgs_transport_id else 1.0
        ),
        "deviatoric_face_stress_magnitude": (deviatoric_face_stress_magnitude),
        "minimum_ksgs_raw_production_density": minimum_raw_production,
        "negative_local_production_violation": max(0.0, -minimum_raw_production),
        "negative_work_refusal_failure": (
            0.0
            if bool(
                jnp.any(negative_raw_production < 0.0)
                & ~negative_evidence.modeled_transfer_balanced
                & ~negative_evidence.energy_balanced
                & ~negative_evidence.successful
                & (negative_status == UNSTRUCTURED_LES_ENERGY_FAILURE)
            )
            else 1.0
        ),
        "energy_balanced_failure": (
            0.0
            if bool(first.evidence.energy_balanced & second.evidence.energy_balanced)
            else 1.0
        ),
        "energy_status_failure": (
            0.0
            if int(first.status) == 0
            and int(second.status) == 0
            and bool(first.evidence.successful & second.evidence.successful)
            else 1.0
        ),
        "viscosity_owner_identity_error": viscosity_owner_error,
        "ksgs_state_failure": (
            0.0
            if bool(
                jnp.all(
                    second.fixed_step.accepted_state.conservative.ksgs.kinetic_energy
                    >= 0.0
                )
            )
            else 1.0
        ),
        "continuation_failure": (
            0.0
            if bool(
                first.fixed_step.successful
                & second.fixed_step.successful
                & first.evidence.pressure_converged
                & first.evidence.shared_mass_flux
                & first.evidence.conservative
                & first.evidence.energy_balanced
                & second.evidence.energy_balanced
            )
            else 1.0
        ),
    }, {
        "prepared_id": prepared.prepared_id,
        "method_id": method.method_id,
        "ksgs_plan_id": prepared.plan.ksgs_plan.plan_id,
        "viscosity_owner": "ksgs",
        "ksgs_production_policy": production_policy,
        "sgs_transport_id": prepared.sgs_transport_id,
        "production_limit_disposition": ("modeled-enthalpy-density-source"),
        "negative_work_status": negative_status,
        "accepted_steps": int(second.fixed_step.accepted_state.accepted_steps),
        "step_statuses": [int(first.status), int(second.status)],
    }


def _run_immersed_sbdf2(case: Mapping[str, object], _reference):
    grids = _mapping(case["grids"], "immersed SBDF2 grids")
    coefficients = _mapping(case["coefficients"], "immersed SBDF2 coefficients")
    timesteps = _mapping(case["timesteps"], "immersed SBDF2 timesteps")
    discretization, plan, dynamics = _immersed_route(
        int(grids["count"]),
        float(coefficients["model"]),
        float(coefficients["viscosity"]),
        wall_stress=False,
    )
    velocity = tuple(0.05 * value for value in _mac_velocity(discretization))
    state = dynamics.pack_velocity(velocity)
    method = plan.sbdf2_method(dynamics, float(timesteps["step_size"]))
    startup = method.initialize(0.0, state)
    startup_ledger = dynamics.algebraic_les.balance_ledger(dynamics, startup)
    payload = io.BytesIO()
    eqx.tree_serialise_leaves(payload, startup.history)
    payload.seek(0)
    restored = eqx.tree_deserialise_leaves(payload, startup.history)
    advanced = method.step(restored)
    advanced_ledger = dynamics.algebraic_les.balance_ledger(
        dynamics, advanced, history=restored
    )
    current_stage = dynamics.rate_components(restored.time, restored.state).les_stage
    previous_stage = dynamics.rate_components(
        restored.time - float(timesteps["step_size"]),
        restored.previous_state,
    ).les_stage
    if current_stage is None or previous_stage is None:
        raise RuntimeError("Immersed SBDF2 LES stages are unavailable.")
    extrapolated_sgs_rate = tuple(
        2.0 * current - previous
        for current, previous in zip(
            current_stage.sgs_rate,
            previous_stage.sgs_rate,
            strict=True,
        )
    )
    expected_bulk_work = float(timesteps["step_size"]) * jnp.real(
        dynamics.momentum.operators.velocity_space.inner(
            advanced.velocity, extrapolated_sgs_rate
        )
    )
    extrapolated_action = _maximum_abs(extrapolated_sgs_rate)
    restart_error = _tree_max_error(restored, startup.history)
    return {
        "restart_roundtrip_error": restart_error,
        "projection_divergence_norm": float(
            jnp.linalg.norm(advanced.projection.divergence_after)
        ),
        "marker_slip_norm": float(jnp.linalg.norm(advanced.projection.marker_slip)),
        "constraint_mode_violation": (
            0.0 if advanced.projection.constraint_mode == "full-vector" else 1.0
        ),
        "advanced_impulse_balance_residual": float(
            jnp.linalg.norm(advanced_ledger.impulse_balance_residual)
        ),
        "advanced_transfer_work_residual": abs(
            float(advanced_ledger.transfer_work_residual)
        ),
        "sgs_extrapolated_action_magnitude": extrapolated_action,
        "sgs_bulk_work_magnitude": abs(float(advanced_ledger.sgs_bulk_work)),
        "sgs_extrapolated_work_error": abs(
            float(advanced_ledger.sgs_bulk_work - expected_bulk_work)
        ),
        "startup_balance_failure": (0.0 if bool(startup_ledger.successful) else 1.0),
        "advanced_balance_failure": (0.0 if bool(advanced_ledger.successful) else 1.0),
        "continuation_failure": (
            0.0
            if bool(
                startup.accepted
                and advanced.accepted
                and advanced.history.accepted_steps == 2
                and advanced.history.method_id == method.method_id
            )
            else 1.0
        ),
    }, {
        "compilation_id": dynamics.compilation_id,
        "prepared_les_id": dynamics.algebraic_les.prepared_id,
        "method_id": method.method_id,
        "constraint_mode": advanced.projection.constraint_mode,
    }


def _run_distributed_production(case: Mapping[str, object], _reference):
    grids = _mapping(case["grids"], "distributed production grids")
    timesteps = _mapping(case["timesteps"], "distributed production timesteps")
    coefficients = _mapping(case["coefficients"], "distributed production coefficients")
    space = _periodic_space(int(grids["count"]))
    scientific = _periodic_les_plan(
        space,
        "smagorinsky",
        float(coefficients["model"]),
        1.5,
    ).prepare(space, phx.discretization.PeriodicLerayProjector(space))
    topology = SpectralMeshTopology(
        (1,),
        devices=(jax.devices("cpu")[0],),
        axis_names=("spectral",),
    )
    source = DistributedPeriodicLESPlan(
        scientific,
        topology,
        schedule="slab",
        checkpoint_count=1,
    )
    step = float(timesteps["step_size"])
    problem = phx.equations.IncompressibleFlowProblem(3, float(coefficients["viscosity"]))
    dynamics = phx.applications.incompressible_flow.compile_distributed_periodic_les(
        problem,
        source,
    )
    state = dynamics.project_state(space.project(_periodic_velocity(space)))
    production_case = (
        phx.applications.incompressible_flow.DistributedPeriodicLESProductionCase(
            dynamics,
            state,
            case_id=str(case["case_id"]),
        )
    )
    production = (
        phx.applications.incompressible_flow.DistributedPeriodicLESProductionPlan(
            problem,
            source,
            phx.applications.incompressible_flow.DistributedPeriodicLESMethodPlan(
                "etdrk2",
                safety_factor=float(coefficients["safety_factor"]),
            ),
            production_case,
            start_time=0.0,
            end_time=2.0 * step,
            step_size=step,
            checkpoint_interval=1,
            segment_steps=1,
            checkpoint_retention=2,
        )
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        profile = HPCFilesystemProfile(
            "distributed-les-posix",
            "qualification-filesystem",
            atomic_rename_same_filesystem=True,
            file_fsync=True,
            directory_fsync=True,
            advisory_locking=True,
            attempt_private_staging=True,
        )
        repository = POSIXArtifactRepository(
            root / "artifacts",
            POSIXRepositoryPolicy(
                profile,
                maximum_chunk_bytes=256,
                maximum_metadata_bytes=1024 * 1024,
            ),
        )
        checkpoint_policy = phx.solver.CheckpointGenerationPolicy(
            production.checkpoint_retention
        )
        dependency = phx.qualification.SupportDependency(
            "repository-profile", repository.support_tuple.support_tuple_id
        )
        resolved = phx.lifecycle.ResolvedRunSpec(
            (),
            (dependency,),
            release_index_id="unresolved-release-index",
            profile_ids=(dependency.profile_id,),
            trust_policy_id="unresolved-release-trust-policy",
            valid_at=10,
            valid_from=0,
            valid_until=20,
            prepared_configuration_id=production.plan_id,
            precision_policy_id=production.manifest.precision_id,
            resource_policy_id=(
                production.dynamics.backend.preparation.resource.report_id
            ),
            checkpoint_policy_id=checkpoint_policy.policy_id,
            output_policy_id="distributed-output-policy",
            repository_id=repository.provider_id,
            scheduler_id="distributed-scheduler",
            auth_policy_id="unsigned-candidate-only",
        )
        store = ArtifactCheckpointStore(
            repository,
            production.manifest,
            checkpoint_policy,
            resolved,
            writer_id="distributed-les-qualification-worker",
            encoding_plan=production.checkpoint_encoding,
        )
        prepared = production.prepare(store)
        initial = prepared.initialize(state)
        following, transition = prepared.step(initial)
        checkpointed = prepared.checkpoint(following)
        resumed = prepared.resume(checkpointed)
        restart = prepared.restart_evidence(resumed)
    restart_error = float(
        jnp.max(jnp.abs(resumed.accepted_state - following.accepted_state))
    )
    expected = production.dynamics.backend.execution.modal_layout.sharding(
        production.dynamics.backend.execution.topology
    )
    production_stage = production.dynamics.stage(0.0, state)
    return {
        "production_step_failure": (0.0 if bool(transition.successful) else 1.0),
        "restart_state_error": restart_error,
        "distributed_sgs_action_magnitude": float(
            jnp.max(jnp.abs(production_stage.rates.algebraic_les_rate))
        ),
        "sharding_failure": (
            0.0
            if restart.sharding_preserved and resumed.accepted_state.sharding == expected
            else 1.0
        ),
        "qualification_inheritance_violation": (
            0.0
            if not production.qualification_inherited
            and not production.dynamics.qualification_inherited
            else 1.0
        ),
        "device_residency_failure": (
            0.0 if production.runtime_plan.device_resident else 1.0
        ),
    }, {
        "source_plan_id": source.plan_id,
        "production_case_id": production.case.identity_id,
        "initial_condition_id": production.case.initial_condition_id,
        "production_plan_id": production.plan_id,
        "compilation_id": production.dynamics.compilation_id,
        "topology_id": production.manifest.topology_id,
        "layout_id": production.manifest.geometry_layout_id,
    }


_PRODUCERS = {
    "periodic-static": _run_periodic_static,
    "periodic-exact-filter": _run_periodic_exact_filter,
    "periodic-dynamic": _run_periodic_dynamic,
    "periodic-dynamic-production": _run_periodic_dynamic_production,
    "mac-coupled": _run_mac_coupled,
    "mac-ksgs": _run_mac_ksgs,
    "dynamic-ksgs": _run_dynamic_ksgs,
    "low-re-ksgs": _run_low_re_ksgs,
    "frozen-imex": _run_frozen_imex,
    "frozen-sbdf2": _run_frozen_sbdf2,
    "channel": _run_channel,
    "channel-wall-owner": _run_channel_wall_owner,
    "channel-restriction": _run_channel_restriction,
    "stochastic-mac-inflow": _run_stochastic_mac_inflow,
    "distributed": _run_distributed,
    "distributed-production": _run_distributed_production,
    "favre": _run_favre,
    "favre-dg-energy": _run_favre_dg_energy,
    "unstructured": _run_unstructured,
    "unstructured-pressure": _run_unstructured_pressure,
    "lbm": _run_lbm,
    "immersed": _run_immersed,
    "immersed-sbdf2": _run_immersed_sbdf2,
    "learned-stress": _run_learned_stress,
}


def _base_profiles(campaign: Mapping[str, object], /):
    result: dict[str, tuple[phx.qualification.CapabilityProfile, str]] = {}
    for value in _sequence(campaign["base_profiles"], "base_profiles"):
        record = _mapping(value, "base profile")
        support = phx.qualification.SupportTuple.from_record(
            _mapping(record["support"], "base support tuple")
        )
        profile = phx.qualification.CapabilityProfile(
            str(record["name"]),
            str(record["provider"]),
            "candidate",
            (support,),
            required_gates=("independent-review", "release-signature"),
            release_evidence=(),
            released=False,
        )
        result[str(record["key"])] = (profile, str(record["scope"]))
    return result


def _case_dependencies(
    case: Mapping[str, object],
    base_profiles: Mapping[str, tuple[phx.qualification.CapabilityProfile, str]],
    /,
):
    scientific: list[phx.qualification.SupportDependency] = []
    deployment: list[phx.qualification.SupportDependency] = []
    for key in _sequence(case["dependencies"], "case dependencies"):
        profile, scope = base_profiles[str(key)]
        dependency = phx.qualification.SupportDependency(
            profile.profile_id, profile.support_tuples[0].support_tuple_id
        )
        (scientific if scope == "scientific" else deployment).append(dependency)
    return tuple(scientific), tuple(deployment)


def _run_spec(
    campaign: Mapping[str, object],
    case: Mapping[str, object],
    scientific: Sequence[phx.qualification.SupportDependency],
    deployment: Sequence[phx.qualification.SupportDependency],
    /,
):
    dependencies = tuple(scientific) + tuple(deployment)
    return phx.lifecycle.ResolvedRunSpec(
        scientific,
        deployment,
        release_index_id=str(campaign["release_index_id"]),
        profile_ids=tuple(value.profile_id for value in dependencies),
        trust_policy_id=str(campaign["trust_policy_id"]),
        valid_at=int(campaign["issued_at"]),
        valid_from=int(campaign["issued_at"]),
        valid_until=int(campaign["expires_at"]),
        prepared_configuration_id=str(case["case_id"]),
        precision_policy_id=str(campaign["precision"]),
        resource_policy_id="finite-default-campaign",
        checkpoint_policy_id="route-native-restart",
        output_policy_id="content-addressed-json",
        repository_id=str(campaign["build_id"]),
        scheduler_id="local-finite-campaign",
        auth_policy_id="unsigned-candidate-only",
    )


def _metric_result(
    metric: Mapping[str, object], value: object, /
) -> tuple[float | None, str, str]:
    numeric = None
    if not isinstance(value, bool) and isinstance(value, (int, float, np.number)):
        candidate = float(value)
        if math.isfinite(candidate):
            numeric = candidate
    threshold = float(metric["threshold"])
    if numeric is None:
        return None, "failed", "measurement is missing or non-finite"
    passed = (
        numeric <= threshold
        if metric["comparison"] == "less-than-or-equal"
        else numeric >= threshold
    )
    relation = "satisfied" if passed else "violated"
    reason = (
        f"measured {numeric:.17g} {relation} preregistered "
        f"{metric['comparison']} threshold {threshold:.17g} {metric['units']}"
    )
    return numeric, "passed" if passed else "failed", reason


def _write_json(path: Path, payload: Mapping[str, object], /) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def execute_campaign(
    campaign: Mapping[str, object],
    matrix: phx.qualification.QualificationMatrix,
    output: Path,
    /,
) -> Mapping[str, object]:
    """Execute every admitted case and emit measured generic evidence candidates."""
    validated = validate_campaign(campaign, matrix)
    cases = tuple(
        _mapping(value, "campaign case")
        for value in _sequence(validated["cases"], "campaign cases")
    )
    admitted_references: dict[
        str, tuple[phx.qualification.ReferenceArtifactManifest, ...]
    ] = {}
    for case in cases:
        required = case["producer"] == "periodic-exact-filter"
        admitted: list[phx.qualification.ReferenceArtifactManifest] = []
        for value in _sequence(case["references"], "case references"):
            reference = admit_reference(
                _mapping(value, "case reference"), required=required
            )
            if reference is not None:
                admitted.append(reference)
        if required and len(admitted) != 1:
            raise ValueError(
                "Exact a-priori filtering requires exactly one admitted reference."
            )
        admitted_references[str(case["case_id"])] = tuple(admitted)
    reference_manifests = {
        value.manifest_id: value
        for values in admitted_references.values()
        for value in values
    }

    base_profiles = _base_profiles(validated)
    evidence: list[phx.qualification.QualificationEvidence] = []
    supports: list[phx.qualification.SupportTuple] = []
    profiles: list[phx.qualification.CapabilityProfile] = []
    run_specs: list[phx.lifecycle.ResolvedRunSpec] = []
    raw_artifacts: list[dict[str, object]] = []
    for case in cases:
        producer_name = str(case["producer"])
        if producer_name not in _PRODUCERS:
            raise ValueError(f"Unknown LES qualification producer {producer_name!r}.")
        support = phx.qualification.SupportTuple.from_record(
            _mapping(case["support"], "case support")
        )
        supports.append(support)
        scientific, deployment = _case_dependencies(case, base_profiles)
        run_spec = _run_spec(validated, case, scientific, deployment)
        run_specs.append(run_spec)
        references = admitted_references[str(case["case_id"])]
        reference = None if not references else references[0]
        measurements, execution = _PRODUCERS[producer_name](case, reference)
        declared = tuple(
            _mapping(value, "metric declaration")
            for value in _sequence(case["metrics"], "case metrics")
        )
        if set(measurements) != {str(value["name"]) for value in declared}:
            raise ValueError(
                f"Producer {producer_name!r} did not return exactly its preregistered metrics."
            )
        evaluated: dict[str, dict[str, object]] = {}
        outcomes: dict[str, tuple[str, str]] = {}
        for metric in declared:
            name = str(metric["name"])
            numeric, outcome, reason = _metric_result(metric, measurements[name])
            evaluated[name] = {
                "value": numeric,
                "units": metric["units"],
                "criterion_id": metric["criterion_id"],
            }
            outcomes[name] = (outcome, reason)
        raw_core = {
            "kind": _MEASUREMENT_KIND,
            "campaign_id": validated["campaign_id"],
            "case_id": case["case_id"],
            "producer": producer_name,
            "support_tuple_id": support.support_tuple_id,
            "resolved_run_spec_id": run_spec.spec_id,
            "reference_manifest_ids": [value.manifest_id for value in references],
            "measurements": evaluated,
            "execution": execution,
        }
        raw = {**raw_core, "artifact_id": content_address(raw_core)}
        raw_artifacts.append(raw)
        for metric in declared:
            name = str(metric["name"])
            outcome, reason = outcomes[name]
            evidence.append(
                phx.qualification.QualificationEvidence(
                    str(metric["evidence_kind"]),
                    outcome,
                    (str(case["case_id"]), support.support_tuple_id, run_spec.spec_id),
                    build_id=str(validated["build_id"]),
                    environment_id=str(validated["environment_id"]),
                    backend=str(validated["backend"]),
                    topology=str(dict(support.attributes)["topology"]),
                    precision=str(validated["precision"]),
                    reduction=str(validated["reduction"]),
                    replay_id=run_spec.spec_id,
                    criteria_ids=(str(metric["criterion_id"]),),
                    raw_artifact_ids=(str(raw["artifact_id"]),),
                    reviewer_id=str(validated["reviewer_id"]),
                    issued_at=int(validated["issued_at"]),
                    expires_at=int(validated["expires_at"]),
                    reason=reason,
                    requalification_triggers=(
                        "base-dependency-change",
                        "build-change",
                        "campaign-change",
                    ),
                )
            )
        dependencies = tuple(scientific) + tuple(deployment)
        profiles.append(
            phx.qualification.CapabilityProfile(
                f"large-eddy-simulation.{case['name']}",
                str(validated["provider"]),
                "candidate",
                (support,),
                dependencies=dependencies,
                required_gates=tuple(str(value) for value in case["predicates"]),
                release_evidence=(),
                released=False,
            )
        )

    coverage = matrix.evaluate(evidence, at_time=int(validated["issued_at"]))
    candidate_core = {
        "kind": _CANDIDATE_KIND,
        "campaign_id": validated["campaign_id"],
        "matrix_id": matrix.matrix_id,
        "coverage_report_id": coverage.report_id,
        "status": "unreleased-candidate",
        "qualification_outcome": coverage.outcome,
        "released": False,
        "signed": False,
        "base_candidate_profile_ids": sorted(
            profile.profile_id for profile, _ in base_profiles.values()
        ),
        "reference_manifest_ids": sorted(reference_manifests),
        "support_tuple_ids": sorted(value.support_tuple_id for value in supports),
        "resolved_run_spec_ids": sorted(value.spec_id for value in run_specs),
        "raw_artifact_ids": sorted(str(value["artifact_id"]) for value in raw_artifacts),
        "evidence_ids": sorted(value.evidence_id for value in evidence),
        "candidate_profile_ids": sorted(value.profile_id for value in profiles),
        "unresolved_release_requirements": [
            "base-candidate-release-admission",
            "independent-review",
            "release-gate-binding",
            "trusted-release-index-signature",
        ],
    }
    candidate = {**candidate_core, "candidate_id": content_address(candidate_core)}

    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "campaign.json", validated)
    _write_json(output / "matrix.json", matrix.to_record())
    _write_json(output / "coverage.json", coverage.to_record())
    for raw in raw_artifacts:
        _write_json(output / "raw" / f"{raw['artifact_id']}.json", raw)
    for value in evidence:
        _write_json(output / "evidence" / f"{value.evidence_id}.json", value.to_record())
    for value in supports:
        _write_json(
            output / "support" / f"{value.support_tuple_id}.json", value.to_record()
        )
    for value in run_specs:
        _write_json(output / "run-specs" / f"{value.spec_id}.json", value.to_record())
    for value in profiles:
        _write_json(output / "profiles" / f"{value.profile_id}.json", value.to_record())
    for value, _scope in base_profiles.values():
        _write_json(
            output / "base-profiles" / f"{value.profile_id}.json", value.to_record()
        )
    for value in reference_manifests.values():
        _write_json(
            output / "references" / f"{value.manifest_id}.json",
            value.to_record(),
        )
    _write_json(output / "candidate.json", candidate)
    return candidate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--campaign",
        type=Path,
        default=root / "benchmarks" / "large_eddy_simulation_qualification_campaign.json",
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=root / "benchmarks" / "large_eddy_simulation_qualification_matrix.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    matrix = load_matrix(arguments.matrix)
    campaign = load_campaign(arguments.campaign, matrix)
    candidate = execute_campaign(campaign, matrix, arguments.output)
    print(json.dumps(candidate, indent=2, sort_keys=True, allow_nan=False))
    outcome = candidate.get("qualification_outcome")
    exit_codes = {"passed": 0, "failed": 1, "inconclusive": 2}
    if outcome not in exit_codes:
        raise ValueError("Candidate has an invalid qualification_outcome.")
    return exit_codes[outcome]


if __name__ == "__main__":
    raise SystemExit(main())
