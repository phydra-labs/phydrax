#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Truncated SL(2,C) kernels, native boosters/EPRL vertices, and finite foams."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from scipy.linalg import expm

from phydrax import ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...operators.quantum.lattice import (
    prepare_su2_sector_basis,
    SU2CouplingTreePlan,
    SU2SectorResourcePolicy,
)
from ._eprl import EPRL_TRIANGLES, EPRLVertexPlan


def _half(value: int, /) -> float:
    return 0.5 * int(value)


def _triangle_key(first: int, second: int, /) -> tuple[int, int]:
    return (first, second) if first < second else (second, first)


class SL2CPrincipalSeriesPlan(StrictModule):
    """Finite-j truncation of one unitary SL(2,C) principal-series representation."""

    rho: float = eqx.field(static=True)
    twice_k: int = eqx.field(static=True)
    maximum_twice_j: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        rho: float,
        twice_k: int,
        maximum_twice_j: int,
        /,
        *,
        tolerance: float = 1e-11,
    ):
        rho_ = float(rho)
        k = int(twice_k)
        maximum = int(maximum_twice_j)
        tolerance_ = float(tolerance)
        if not math.isfinite(rho_) or k < 0 or maximum < k or (maximum - k) % 2:
            raise ValueError("SL(2,C) principal-series labels or cutoff are invalid.")
        if tolerance_ <= 0.0:
            raise ValueError("SL(2,C) tolerance must be positive.")
        self.rho = rho_
        self.twice_k = k
        self.maximum_twice_j = maximum
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "sl2c-principal-series-plan",
                "rho": rho_,
                "twice_k": k,
                "maximum_twice_j": maximum,
                "tolerance": tolerance_,
                "generator_convention": "canonical-k3-j-basis-truncation",
            }
        )

    @property
    def twice_j_values(self) -> tuple[int, ...]:
        return tuple(range(self.twice_k, self.maximum_twice_j + 1, 2))


class SL2CBoostEvidence(StrictModule):
    matrix: Array
    generator_hermiticity_residual: Array
    unitarity_residual: Array
    composition_residual: Array
    cutoff_boundary_weight: Array
    accepted: Array
    claim: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


def _boost_generator(plan: SL2CPrincipalSeriesPlan, twice_m: int, /):
    m = _half(twice_m)
    values = tuple(value for value in plan.twice_j_values if value >= abs(twice_m))
    if not values:
        raise ValueError(
            "Magnetic projection lies outside the truncated principal series."
        )
    generator = np.zeros((len(values), len(values)), dtype=np.complex128)
    k = _half(plan.twice_k)
    for index, twice_j in enumerate(values):
        spin = _half(twice_j)
        beta = 0.0 if spin == 0.0 else k * plan.rho / (spin * (spin + 1.0))
        generator[index, index] = beta * m
        if index > 0:
            alpha = (
                math.sqrt(
                    max(0.0, (spin * spin - k * k) * (spin * spin + plan.rho**2))
                    / max(spin * spin * (4.0 * spin * spin - 1.0), 1e-300)
                )
                if spin > 0.5
                else 0.0
            )
            coupling = alpha * math.sqrt(max(0.0, spin * spin - m * m))
            generator[index, index - 1] = coupling
            generator[index - 1, index] = coupling
    return values, generator


def evaluate_sl2c_boost(
    plan: SL2CPrincipalSeriesPlan,
    twice_m: int,
    rapidity: float,
    /,
) -> SL2CBoostEvidence:
    """Exponentiate the bounded Hermitian boost generator and audit composition."""

    if not isinstance(plan, SL2CPrincipalSeriesPlan):
        raise TypeError("plan must be SL2CPrincipalSeriesPlan.")
    radius = float(rapidity)
    if not math.isfinite(radius):
        raise ValueError("Boost rapidity must be finite.")
    _, generator = _boost_generator(plan, int(twice_m))
    matrix = expm(-1j * radius * generator)
    half_matrix = expm(-0.5j * radius * generator)
    hermiticity = float(np.linalg.norm(generator - generator.conj().T))
    unitarity = float(np.linalg.norm(matrix.conj().T @ matrix - np.eye(matrix.shape[0])))
    composition = float(np.linalg.norm(half_matrix @ half_matrix - matrix))
    boundary = float(np.max(np.abs(matrix[-1])))
    accepted = max(hermiticity, unitarity, composition) <= plan.tolerance
    evidence_id = canonical_fingerprint(
        {
            "kind": "sl2c-boost-evidence",
            "plan": plan.plan_id,
            "twice_m": int(twice_m),
            "rapidity": radius,
            "matrix": array_tree_fingerprint(matrix),
        }
    )
    return SL2CBoostEvidence(
        matrix=jnp.asarray(matrix),
        generator_hermiticity_residual=jnp.asarray(hermiticity),
        unitarity_residual=jnp.asarray(unitarity),
        composition_residual=jnp.asarray(composition),
        cutoff_boundary_weight=jnp.asarray(boundary),
        accepted=jnp.asarray(accepted),
        claim="finite-j-principal-series-kernel-with-cutoff-evidence",
        evidence_id=evidence_id,
    )


def _intertwiner_tensor(
    spins: tuple[int, int, int, int],
    twice_intertwiner: int,
    resources: SU2SectorResourcePolicy,
    /,
) -> np.ndarray:
    basis = prepare_su2_sector_basis(
        SU2CouplingTreePlan(
            ("edge-0", "edge-1", "edge-2", "edge-3"),
            spins,
            0,
            resources,
        )
    )
    matching = tuple(
        index
        for index, path in enumerate(basis.plan.coupling_paths)
        if path[0] == int(twice_intertwiner)
    )
    if len(matching) != 1:
        raise ValueError("Declared four-valent intertwiner is absent or ambiguous.")
    return np.asarray(basis.transform[matching[0]]).reshape(
        tuple(value + 1 for value in spins)
    )


class NativeB4BoosterPlan(StrictModule):
    boundary_twice_spins: tuple[int, int, int, int] = eqx.field(static=True)
    internal_twice_spins: tuple[int, int, int, int] = eqx.field(static=True)
    boundary_twice_intertwiner: int = eqx.field(static=True)
    internal_twice_intertwiner: int = eqx.field(static=True)
    immirzi_parameter: float = eqx.field(static=True)
    radial_nodes: Array
    radial_weights: Array
    resources: SU2SectorResourcePolicy = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        boundary_twice_spins: Sequence[int],
        internal_twice_spins: Sequence[int],
        boundary_twice_intertwiner: int,
        internal_twice_intertwiner: int,
        immirzi_parameter: float,
        radial_nodes: ArrayLike,
        radial_weights: ArrayLike,
        resources: SU2SectorResourcePolicy,
        /,
        *,
        tolerance: float = 1e-8,
    ):
        boundary = tuple(int(value) for value in boundary_twice_spins)
        internal = tuple(int(value) for value in internal_twice_spins)
        nodes = np.asarray(radial_nodes, dtype=float)
        weights = np.asarray(radial_weights, dtype=float)
        gamma = float(immirzi_parameter)
        tolerance_ = float(tolerance)
        if len(boundary) != 4 or len(internal) != 4:
            raise ValueError("B4 boosters require four boundary and internal spins.")
        if any(value < 0 for value in boundary) or any(
            internal[index] < boundary[index] or (internal[index] - boundary[index]) % 2
            for index in range(4)
        ):
            raise ValueError("B4 internal spins must be admissible EPRL shells.")
        if nodes.ndim != 1 or nodes.size < 4 or weights.shape != nodes.shape:
            raise ValueError("B4 radial quadrature arrays are invalid.")
        if (
            np.any(nodes < 0.0)
            or np.any(weights <= 0.0)
            or gamma <= 0.0
            or tolerance_ <= 0.0
        ):
            raise ValueError("B4 quadrature, Immirzi parameter, or tolerance is invalid.")
        if not isinstance(resources, SU2SectorResourcePolicy):
            raise TypeError("resources must be SU2SectorResourcePolicy.")
        _intertwiner_tensor(boundary, int(boundary_twice_intertwiner), resources)
        _intertwiner_tensor(internal, int(internal_twice_intertwiner), resources)
        content = {
            "kind": "native-b4-booster-plan",
            "boundary_twice_spins": boundary,
            "internal_twice_spins": internal,
            "boundary_twice_intertwiner": int(boundary_twice_intertwiner),
            "internal_twice_intertwiner": int(internal_twice_intertwiner),
            "immirzi_parameter": gamma,
            "radial_nodes": array_tree_fingerprint(nodes),
            "radial_weights": array_tree_fingerprint(weights),
            "resources": resources.policy_id,
            "tolerance": tolerance_,
        }
        self.boundary_twice_spins = boundary  # type: ignore[assignment]
        self.internal_twice_spins = internal  # type: ignore[assignment]
        self.boundary_twice_intertwiner = int(boundary_twice_intertwiner)
        self.internal_twice_intertwiner = int(internal_twice_intertwiner)
        self.immirzi_parameter = gamma
        self.radial_nodes = jnp.asarray(nodes)
        self.radial_weights = jnp.asarray(weights)
        self.resources = resources
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(content)


class NativeB4BoosterEvidence(StrictModule):
    value: Array
    reduced_quadrature_value: Array
    quadrature_residual: Array
    cutoff_boundary_weight: Array
    finite: Array
    accepted: Array
    plan_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


def evaluate_native_b4_booster(
    plan: NativeB4BoosterPlan,
    /,
) -> NativeB4BoosterEvidence:
    """Evaluate a nonzero-spin B4 booster from truncated principal-series kernels."""

    if not isinstance(plan, NativeB4BoosterPlan):
        raise TypeError("plan must be NativeB4BoosterPlan.")
    boundary_tensor = _intertwiner_tensor(
        plan.boundary_twice_spins,
        plan.boundary_twice_intertwiner,
        plan.resources,
    )
    internal_tensor = _intertwiner_tensor(
        plan.internal_twice_spins,
        plan.internal_twice_intertwiner,
        plan.resources,
    )
    magnetic_values = tuple(
        tuple(range(-spin, spin + 1, 2)) for spin in plan.boundary_twice_spins
    )
    principal = tuple(
        SL2CPrincipalSeriesPlan(
            plan.immirzi_parameter * _half(spin),
            spin,
            internal,
            tolerance=plan.tolerance,
        )
        for spin, internal in zip(
            plan.boundary_twice_spins,
            plan.internal_twice_spins,
            strict=True,
        )
    )
    integrand_values = []
    boundary_weights = []
    for radius in np.asarray(plan.radial_nodes):
        kernels: list[dict[int, complex]] = []
        local_boundary = 0.0
        for leg in range(4):
            values: dict[int, complex] = {}
            j_index = principal[leg].twice_j_values.index(plan.boundary_twice_spins[leg])
            l_index = principal[leg].twice_j_values.index(plan.internal_twice_spins[leg])
            allowed = tuple(
                value
                for value in magnetic_values[leg]
                if abs(value) <= plan.internal_twice_spins[leg]
            )
            for magnetic in allowed:
                evidence = evaluate_sl2c_boost(principal[leg], magnetic, float(radius))
                values[magnetic] = complex(np.asarray(evidence.matrix)[l_index, j_index])
                local_boundary = max(
                    local_boundary, float(evidence.cutoff_boundary_weight)
                )
            kernels.append(values)
        total = 0.0 + 0.0j
        for indices in np.ndindex(*boundary_tensor.shape):
            magnetic = tuple(
                -plan.boundary_twice_spins[leg] + 2 * indices[leg] for leg in range(4)
            )
            internal_indices = tuple(
                (magnetic[leg] + plan.internal_twice_spins[leg]) // 2 for leg in range(4)
            )
            total += (
                np.conj(boundary_tensor[indices])
                * internal_tensor[internal_indices]
                * math.prod(kernels[leg][magnetic[leg]] for leg in range(4))
            )
        integrand_values.append(np.sinh(radius) ** 2 * total / (4.0 * np.pi))
        boundary_weights.append(local_boundary)
    integrand = np.asarray(integrand_values)
    weights = np.asarray(plan.radial_weights)
    value = np.sum(weights * integrand)
    reduced = np.sum(weights[::2] * integrand[::2])
    residual = abs(value - reduced) / max(1.0, abs(value))
    finite = bool(np.isfinite(value) and np.isfinite(residual))
    accepted = finite and residual <= plan.tolerance
    evidence_id = canonical_fingerprint(
        {
            "kind": "native-b4-booster-evidence",
            "plan": plan.plan_id,
            "integrand": array_tree_fingerprint(integrand),
            "value": (value.real, value.imag),
        }
    )
    return NativeB4BoosterEvidence(
        value=jnp.asarray(value),
        reduced_quadrature_value=jnp.asarray(reduced),
        quadrature_residual=jnp.asarray(residual),
        cutoff_boundary_weight=jnp.asarray(max(boundary_weights, default=0.0)),
        finite=jnp.asarray(finite),
        accepted=jnp.asarray(accepted),
        plan_id=plan.plan_id,
        claim="finite-cutoff-native-nonzero-spin-b4-booster",
        evidence_id=evidence_id,
    )


def _duality_matrix(twice_spin: int, /) -> np.ndarray:
    dimension = twice_spin + 1
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    for index, twice_m in enumerate(range(-twice_spin, twice_spin + 1, 2)):
        opposite = (-twice_m + twice_spin) // 2
        exponent = (twice_spin - twice_m) // 2
        matrix[index, opposite] = (-1.0) ** exponent
    return matrix


def _tetrahedron_tensor(
    spins: tuple[int, int, int, int],
    intertwiner: int,
    resources: SU2SectorResourcePolicy,
    dual_axes: Sequence[int],
    /,
) -> np.ndarray:
    tensor = _intertwiner_tensor(spins, intertwiner, resources)
    for axis in dual_axes:
        dual = _duality_matrix(spins[axis])
        tensor = np.tensordot(dual, tensor, axes=((1,), (axis,)))
        tensor = np.moveaxis(tensor, 0, axis)
    return tensor


def su2_15j_symbol(
    boundary_twice_spins: Sequence[int],
    boundary_twice_intertwiners: Sequence[int],
    resources: SU2SectorResourcePolicy,
    /,
) -> complex:
    """Contract five oriented four-valent intertwiners in the 4-simplex graph."""

    spins = tuple(int(value) for value in boundary_twice_spins)
    intertwiners = tuple(int(value) for value in boundary_twice_intertwiners)
    if len(spins) != 10 or len(intertwiners) != 5:
        raise ValueError("The SU(2) 15j symbol requires ten spins and five intertwiners.")
    spin_lookup = {
        triangle: spins[index] for index, triangle in enumerate(EPRL_TRIANGLES)
    }
    triangle_symbol = {
        triangle: chr(ord("a") + index) for index, triangle in enumerate(EPRL_TRIANGLES)
    }
    tensors = []
    subscripts = []
    for tetrahedron in range(5):
        others = tuple(value for value in range(5) if value != tetrahedron)
        incident = cast(
            tuple[int, int, int, int],
            tuple(spin_lookup[_triangle_key(tetrahedron, other)] for other in others),
        )
        dual_axes = tuple(
            axis for axis, other in enumerate(others) if other < tetrahedron
        )
        tensors.append(
            _tetrahedron_tensor(
                incident,
                intertwiners[tetrahedron],
                resources,
                dual_axes,
            )
        )
        subscripts.append(
            "".join(
                triangle_symbol[_triangle_key(tetrahedron, other)] for other in others
            )
        )
    equation = ",".join(subscripts) + "->"
    return complex(np.asarray(ein.contract(equation, *tensors)))


class NativeEPRLVertexData(StrictModule):
    plan: EPRLVertexPlan = eqx.field(static=True)
    support_twice_spins: Array
    internal_twice_intertwiners: Array
    booster_values: Array
    booster_error_bounds: Array
    data_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: EPRLVertexPlan,
        support_twice_spins: ArrayLike,
        internal_twice_intertwiners: ArrayLike,
        booster_values: ArrayLike,
        booster_error_bounds: ArrayLike,
        /,
    ):
        if not isinstance(plan, EPRLVertexPlan):
            raise TypeError("plan must be EPRLVertexPlan.")
        support = np.asarray(support_twice_spins, dtype=np.int32)
        intertwiners = np.asarray(internal_twice_intertwiners, dtype=np.int32)
        boosters = np.asarray(booster_values, dtype=np.complex128)
        errors = np.asarray(booster_error_bounds, dtype=float)
        if support.ndim != 2 or support.shape[1] != 10:
            raise ValueError("EPRL support table must have shape (support, 10).")
        count = support.shape[0]
        if (
            intertwiners.shape != (count, 5)
            or boosters.shape != (count, 5)
            or errors.shape != (count, 5)
        ):
            raise ValueError(
                "EPRL internal intertwiners and boosters must have shape (support, 5)."
            )
        admitted = set(itertools.product(*plan.internal_twice_spin_support))
        if any(tuple(row) not in admitted for row in support.tolist()):
            raise ValueError(
                "Native EPRL support includes a spin outside the semantic plan."
            )
        if count > plan.maximum_support_tuples or np.any(errors < 0.0):
            raise ValueError("Native EPRL support or error bounds exceed admission.")
        content = {
            "kind": "native-eprl-vertex-data",
            "plan": plan.plan_id,
            "support": array_tree_fingerprint(support),
            "intertwiners": array_tree_fingerprint(intertwiners),
            "boosters": array_tree_fingerprint(boosters),
            "errors": array_tree_fingerprint(errors),
        }
        self.plan = plan
        self.support_twice_spins = jnp.asarray(support)
        self.internal_twice_intertwiners = jnp.asarray(intertwiners)
        self.booster_values = jnp.asarray(boosters)
        self.booster_error_bounds = jnp.asarray(errors)
        self.data_id = canonical_fingerprint(content)


class NativeEPRLVertexEvidence(StrictModule):
    amplitude: Array
    absolute_error_bound: Array
    support_contributions: Array
    delta_l_shell_norms: Array
    finite: Array
    claim: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


def evaluate_native_eprl_vertex(
    data: NativeEPRLVertexData,
    resources: SU2SectorResourcePolicy,
    /,
) -> NativeEPRLVertexEvidence:
    if not isinstance(data, NativeEPRLVertexData):
        raise TypeError("data must be NativeEPRLVertexData.")
    contributions = []
    error = 0.0
    shells: dict[int, float] = {}
    for support, intertwiners, boosters, booster_errors in zip(
        np.asarray(data.support_twice_spins),
        np.asarray(data.internal_twice_intertwiners),
        np.asarray(data.booster_values),
        np.asarray(data.booster_error_bounds),
        strict=True,
    ):
        symbol = su2_15j_symbol(tuple(support), tuple(intertwiners), resources)
        face_weight = (
            math.prod(int(value) + 1 for value in support)
            if data.plan.face_amplitude == "dimension"
            else 1.0
        )
        booster_product = np.prod(boosters)
        contribution = face_weight * symbol * booster_product
        contributions.append(contribution)
        relative = sum(
            booster_errors[index] / max(abs(boosters[index]), 1e-300)
            for index in range(5)
        )
        error += abs(contribution) * relative
        shell = max(
            (int(support[index]) - data.plan.boundary_twice_spins[index]) // 2
            for index in range(10)
        )
        shells[shell] = shells.get(shell, 0.0) + abs(contribution) ** 2
    contribution_table = np.asarray(contributions, dtype=np.complex128)
    amplitude = np.sum(contribution_table)
    shell_table = np.asarray(
        [math.sqrt(shells.get(index, 0.0)) for index in range(data.plan.delta_l + 1)]
    )
    finite = bool(np.isfinite(amplitude) and np.isfinite(error))
    evidence_id = canonical_fingerprint(
        {
            "kind": "native-eprl-vertex-evidence",
            "data": data.data_id,
            "contributions": array_tree_fingerprint(contribution_table),
            "absolute_error_bound": error,
            "shell_norms": array_tree_fingerprint(shell_table),
        }
    )
    return NativeEPRLVertexEvidence(
        amplitude=jnp.asarray(amplitude),
        absolute_error_bound=jnp.asarray(error),
        support_contributions=jnp.asarray(contribution_table),
        delta_l_shell_norms=jnp.asarray(shell_table),
        finite=jnp.asarray(finite),
        claim="finite-cutoff-native-lorentzian-eprl-vertex-no-continuum-claim",
        evidence_id=evidence_id,
    )


@dataclass(frozen=True, slots=True)
class SpinFoamVertexTensor:
    label: str
    face_labels: tuple[str, ...]
    amplitudes: np.ndarray
    vertex_id: str

    def __init__(
        self,
        label: str,
        face_labels: Sequence[str],
        amplitudes: ArrayLike,
        /,
    ):
        label_ = str(label).strip()
        faces = tuple(str(value).strip() for value in face_labels)
        values = np.asarray(amplitudes, dtype=np.complex128)
        if (
            not label_
            or not faces
            or len(set(faces)) != len(faces)
            or values.ndim != len(faces)
        ):
            raise ValueError("Spin-foam vertex tensor identity or rank is invalid.")
        content = {
            "kind": "spin-foam-vertex-tensor",
            "label": label_,
            "face_labels": faces,
            "amplitudes": array_tree_fingerprint(values),
        }
        object.__setattr__(self, "label", label_)
        object.__setattr__(self, "face_labels", faces)
        object.__setattr__(self, "amplitudes", values)
        object.__setattr__(self, "vertex_id", canonical_fingerprint(content))


class FiniteSpinFoamComplexPlan(StrictModule):
    face_support: tuple[tuple[str, tuple[int, ...]], ...] = eqx.field(static=True)
    vertices: tuple[SpinFoamVertexTensor, ...] = eqx.field(static=True)
    boundary_faces: tuple[str, ...] = eqx.field(static=True)
    face_amplitude_power: float = eqx.field(static=True)
    maximum_assignments: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        face_support: Mapping[str, Sequence[int]],
        vertices: Sequence[SpinFoamVertexTensor],
        /,
        *,
        boundary_faces: Sequence[str] = (),
        face_amplitude_power: float = 1.0,
        maximum_assignments: int = 1_000_000,
    ):
        support = tuple(
            sorted(
                (
                    str(label),
                    tuple(int(value) for value in values),
                )
                for label, values in face_support.items()
            )
        )
        vertices_ = tuple(vertices)
        boundary = tuple(str(value) for value in boundary_faces)
        labels = {label for label, _ in support}
        if not support or any(not values for _, values in support):
            raise ValueError("Every spin-foam face requires non-empty finite support.")
        if any(not isinstance(value, SpinFoamVertexTensor) for value in vertices_):
            raise TypeError("vertices must contain SpinFoamVertexTensor values.")
        if any(not set(value.face_labels) <= labels for value in vertices_):
            raise ValueError("A spin-foam vertex references an unknown face.")
        support_map = dict(support)
        if any(
            value.amplitudes.shape
            != tuple(len(support_map[label]) for label in value.face_labels)
            for value in vertices_
        ):
            raise ValueError(
                "A spin-foam vertex tensor shape does not match face support."
            )
        if not set(boundary) <= labels:
            raise ValueError("A boundary face is outside the finite complex.")
        assignment_count = math.prod(len(values) for _, values in support)
        maximum = int(maximum_assignments)
        if assignment_count > maximum:
            raise ValueError("Finite spin-foam assignments exceed maximum_assignments.")
        self.face_support = support
        self.vertices = vertices_
        self.boundary_faces = boundary
        self.face_amplitude_power = float(face_amplitude_power)
        self.maximum_assignments = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-spin-foam-complex-plan",
                "face_support": support,
                "vertices": [value.vertex_id for value in vertices_],
                "boundary_faces": boundary,
                "face_amplitude_power": float(face_amplitude_power),
                "maximum_assignments": maximum,
            }
        )


class FiniteSpinFoamComplexEvidence(StrictModule):
    amplitude: Array
    assignment_count: int = eqx.field(static=True)
    absolute_sum: Array
    cancellation_ratio: Array
    finite: Array
    claim: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


def contract_finite_spin_foam(
    plan: FiniteSpinFoamComplexPlan,
    /,
) -> FiniteSpinFoamComplexEvidence:
    """Sum a finite 2-complex with explicit face support and local tensors."""

    if not isinstance(plan, FiniteSpinFoamComplexPlan):
        raise TypeError("plan must be FiniteSpinFoamComplexPlan.")
    support = dict(plan.face_support)
    internal = tuple(
        label for label, _ in plan.face_support if label not in plan.boundary_faces
    )
    boundary = plan.boundary_faces
    output_shape = tuple(len(support[label]) for label in boundary)
    amplitude = np.zeros(output_shape or (), dtype=np.complex128)
    absolute = np.zeros_like(amplitude, dtype=float)
    face_labels = tuple(label for label, _ in plan.face_support)
    assignment_count = 0
    for assignment_indices in np.ndindex(*(len(support[label]) for label in face_labels)):
        assignment_count += 1
        assignment = dict(zip(face_labels, assignment_indices, strict=True))
        value = 1.0 + 0.0j
        for label, index in assignment.items():
            twice_spin = support[label][index]
            value *= (twice_spin + 1) ** plan.face_amplitude_power
        for vertex in plan.vertices:
            value *= vertex.amplitudes[
                tuple(assignment[label] for label in vertex.face_labels)
            ]
        output_index = tuple(assignment[label] for label in boundary)
        amplitude[output_index] += value
        absolute[output_index] += abs(value)
    ratio = np.abs(amplitude) / np.maximum(absolute, 1e-300)
    finite = bool(np.all(np.isfinite(amplitude)))
    evidence_id = canonical_fingerprint(
        {
            "kind": "finite-spin-foam-complex-evidence",
            "plan": plan.plan_id,
            "amplitude": array_tree_fingerprint(amplitude),
            "absolute_sum": array_tree_fingerprint(absolute),
            "internal_faces": internal,
        }
    )
    return FiniteSpinFoamComplexEvidence(
        amplitude=jnp.asarray(amplitude),
        assignment_count=assignment_count,
        absolute_sum=jnp.asarray(absolute),
        cancellation_ratio=jnp.asarray(ratio),
        finite=jnp.asarray(finite),
        claim="finite-cutoff-two-complex-amplitude-no-continuum-quantum-gravity-claim",
        evidence_id=evidence_id,
    )


class SpinFoamCoarseGrainingEvidence(StrictModule):
    coarse_tensor: Array
    retained_rank: int = eqx.field(static=True)
    discarded_weight: Array
    relative_error: Array
    evidence_id: str = eqx.field(static=True)


def coarse_grain_spin_foam_tensor(
    tensor: ArrayLike,
    left_axes: Sequence[int],
    /,
    *,
    maximum_rank: int,
) -> SpinFoamCoarseGrainingEvidence:
    """SVD-coarse-grain one local amplitude tensor with an explicit error ledger."""

    values = np.asarray(tensor, dtype=np.complex128)
    left = tuple(int(value) for value in left_axes)
    if values.ndim < 2 or not left or not set(left) < set(range(values.ndim)):
        raise ValueError("Coarse-graining axes must be a proper non-empty tensor subset.")
    right = tuple(value for value in range(values.ndim) if value not in left)
    permutation = left + right
    left_dimension = math.prod(values.shape[value] for value in left)
    right_dimension = math.prod(values.shape[value] for value in right)
    matrix = np.transpose(values, permutation).reshape((left_dimension, right_dimension))
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    rank = min(int(maximum_rank), singular.size)
    if rank < 1:
        raise ValueError("maximum_rank must retain at least one singular value.")
    coarse = (u[:, :rank] * singular[:rank]) @ vh[:rank]
    discarded = float(np.sum(singular[rank:] ** 2))
    relative = float(np.linalg.norm(matrix - coarse) / max(1.0, np.linalg.norm(matrix)))
    coarse_tensor = np.transpose(
        coarse.reshape(tuple(values.shape[value] for value in permutation)),
        np.argsort(permutation),
    )
    evidence_id = canonical_fingerprint(
        {
            "kind": "spin-foam-coarse-graining-evidence",
            "tensor": array_tree_fingerprint(values),
            "left_axes": left,
            "maximum_rank": int(maximum_rank),
        }
    )
    return SpinFoamCoarseGrainingEvidence(
        coarse_tensor=jnp.asarray(coarse_tensor),
        retained_rank=rank,
        discarded_weight=jnp.asarray(discarded),
        relative_error=jnp.asarray(relative),
        evidence_id=evidence_id,
    )


class EPRLSemiclassicalEvidence(StrictModule):
    scales: Array
    amplitudes: Array
    regge_actions: Array
    phase_residuals: Array
    fitted_power: Array
    expected_power: Array
    power_residual: Array
    geometric: Array
    accepted: Array
    claim: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


def assess_eprl_semiclassical_asymptotics(
    scales: ArrayLike,
    amplitudes: ArrayLike,
    base_twice_spins: ArrayLike,
    dihedral_angles: ArrayLike,
    /,
    *,
    expected_power: float,
    phase_tolerance: float,
    power_tolerance: float,
    geometric: bool,
) -> EPRLSemiclassicalEvidence:
    scales_ = np.asarray(scales, dtype=float)
    amplitudes_ = np.asarray(amplitudes, dtype=np.complex128)
    spins = np.asarray(base_twice_spins, dtype=float)
    angles = np.asarray(dihedral_angles, dtype=float)
    if scales_.ndim != 1 or scales_.size < 3 or amplitudes_.shape != scales_.shape:
        raise ValueError(
            "Semiclassical evidence requires three aligned scale amplitudes."
        )
    if spins.shape != angles.shape or spins.ndim != 1:
        raise ValueError("Regge spins and dihedral angles must align.")
    action = 0.5 * np.sum(spins * angles) * scales_
    phases = np.unwrap(np.angle(amplitudes_))
    branch = np.round((phases - action) / (2.0 * np.pi))
    phase_residual = np.abs(phases - action - 2.0 * np.pi * branch)
    fitted = float(
        np.polyfit(np.log(scales_), np.log(np.maximum(np.abs(amplitudes_), 1e-300)), 1)[0]
    )
    power_residual = abs(fitted - float(expected_power))
    accepted = bool(
        geometric
        and np.max(phase_residual) <= float(phase_tolerance)
        and power_residual <= float(power_tolerance)
    )
    evidence_id = canonical_fingerprint(
        {
            "kind": "eprl-semiclassical-evidence",
            "scales": array_tree_fingerprint(scales_),
            "amplitudes": array_tree_fingerprint(amplitudes_),
            "spins": array_tree_fingerprint(spins),
            "angles": array_tree_fingerprint(angles),
            "expected_power": float(expected_power),
            "geometric": bool(geometric),
        }
    )
    return EPRLSemiclassicalEvidence(
        scales=jnp.asarray(scales_),
        amplitudes=jnp.asarray(amplitudes_),
        regge_actions=jnp.asarray(action),
        phase_residuals=jnp.asarray(phase_residual),
        fitted_power=jnp.asarray(fitted),
        expected_power=jnp.asarray(expected_power),
        power_residual=jnp.asarray(power_residual),
        geometric=jnp.asarray(geometric),
        accepted=jnp.asarray(accepted),
        claim="finite-large-spin-regge-evidence-no-continuum-quantum-gravity-claim",
        evidence_id=evidence_id,
    )


__all__ = [
    "EPRLSemiclassicalEvidence",
    "FiniteSpinFoamComplexEvidence",
    "FiniteSpinFoamComplexPlan",
    "NativeB4BoosterEvidence",
    "NativeB4BoosterPlan",
    "NativeEPRLVertexData",
    "NativeEPRLVertexEvidence",
    "SL2CBoostEvidence",
    "SL2CPrincipalSeriesPlan",
    "SpinFoamCoarseGrainingEvidence",
    "SpinFoamVertexTensor",
    "assess_eprl_semiclassical_asymptotics",
    "coarse_grain_spin_foam_tensor",
    "contract_finite_spin_foam",
    "evaluate_native_b4_booster",
    "evaluate_native_eprl_vertex",
    "evaluate_sl2c_boost",
    "su2_15j_symbol",
]
