#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
import math
from math import prod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from ... import ein
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...discretization._lattice_boundary import (
    CheckerboardEntityLayout,
    LatticeBoundaryPhasePlan,
)
from ...graph._gauge_transport import GaugeCovariantShiftPlan
from ...linalg._operators import AbstractLinearOperator
from ...linalg._properties import (
    LinearCapabilityError,
    OperatorCapabilities,
    OperatorProperties,
)
from ...linalg._rational_functions import (
    PartialFractionRationalFunction,
    plan_rational_function_action,
    prepare_rational_function_action,
    rational_function_action,
    RationalFunctionDiagnostics,
    RationalFunctionPlan,
    RationalFunctionPolicy,
)
from ...linalg._spaces import ArraySpace
from ...metrix._gauge_representation import AbstractGaugeRepresentation
from ._pseudofermion_operator import AbstractPseudofermionDiracOperator


class LatticeFermionResourcePolicy(StrictModule):
    """Hard fixed-shape admission limits for lattice-fermion plans."""

    maximum_sites: int = eqx.field(static=True)
    maximum_spin_components: int = eqx.field(static=True)
    maximum_color_components: int = eqx.field(static=True)
    maximum_degrees_of_freedom: int = eqx.field(static=True)
    maximum_local_inverse_bytes: int = eqx.field(static=True)
    maximum_fifth_extent: int = eqx.field(static=True)
    maximum_rational_poles: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_sites: int = 1_048_576,
        maximum_spin_components: int = 16,
        maximum_color_components: int = 64,
        maximum_degrees_of_freedom: int = 67_108_864,
        maximum_local_inverse_bytes: int = 1_073_741_824,
        maximum_fifth_extent: int = 256,
        maximum_rational_poles: int = 256,
    ):
        values = tuple(
            int(value)
            for value in (
                maximum_sites,
                maximum_spin_components,
                maximum_color_components,
                maximum_degrees_of_freedom,
                maximum_local_inverse_bytes,
                maximum_fifth_extent,
                maximum_rational_poles,
            )
        )
        if any(value < 1 for value in values):
            raise ValueError("Lattice-fermion resource limits must be positive.")
        (
            self.maximum_sites,
            self.maximum_spin_components,
            self.maximum_color_components,
            self.maximum_degrees_of_freedom,
            self.maximum_local_inverse_bytes,
            self.maximum_fifth_extent,
            self.maximum_rational_poles,
        ) = values
        self.policy_id = canonical_fingerprint(
            {
                "kind": "lattice-fermion-resource-policy",
                "maximum_sites": values[0],
                "maximum_spin_components": values[1],
                "maximum_color_components": values[2],
                "maximum_degrees_of_freedom": values[3],
                "maximum_local_inverse_bytes": values[4],
                "maximum_fifth_extent": values[5],
                "maximum_rational_poles": values[6],
            }
        )

    def admit(
        self,
        site_count: int,
        spin_components: int,
        color_components: int,
        /,
        *,
        fifth_extent: int = 1,
    ) -> None:
        sites = int(site_count)
        spin = int(spin_components)
        color = int(color_components)
        fifth = int(fifth_extent)
        dofs = sites * spin * color * fifth
        if sites < 1 or sites > self.maximum_sites:
            raise ValueError("Lattice site count exceeds the resource policy.")
        if spin < 1 or spin > self.maximum_spin_components:
            raise ValueError("Spin-component count exceeds the resource policy.")
        if color < 1 or color > self.maximum_color_components:
            raise ValueError("Color-component count exceeds the resource policy.")
        if fifth < 1 or fifth > self.maximum_fifth_extent:
            raise ValueError("Fifth-dimensional extent exceeds the resource policy.")
        if dofs > self.maximum_degrees_of_freedom:
            raise ValueError(
                "Lattice-fermion degrees of freedom exceed the resource policy."
            )


class LatticeFermionResourceEvidence(StrictModule):
    """Realized immutable dimensions and storage for one admitted operator."""

    site_count: int = eqx.field(static=True)
    spin_components: int = eqx.field(static=True)
    color_components: int = eqx.field(static=True)
    fifth_extent: int = eqx.field(static=True)
    degrees_of_freedom: int = eqx.field(static=True)
    field_storage_bytes: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        site_count: int,
        spin_components: int,
        color_components: int,
        dtype: Any,
        policy: LatticeFermionResourcePolicy,
        /,
        *,
        fifth_extent: int = 1,
    ):
        if not isinstance(policy, LatticeFermionResourcePolicy):
            raise TypeError("policy must be LatticeFermionResourcePolicy.")
        policy.admit(
            site_count,
            spin_components,
            color_components,
            fifth_extent=fifth_extent,
        )
        sites = int(site_count)
        spin = int(spin_components)
        color = int(color_components)
        fifth = int(fifth_extent)
        dofs = sites * spin * color * fifth
        storage = dofs * np.dtype(dtype).itemsize
        self.site_count = sites
        self.spin_components = spin
        self.color_components = color
        self.fifth_extent = fifth
        self.degrees_of_freedom = dofs
        self.field_storage_bytes = storage
        self.policy_id = policy.policy_id
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "lattice-fermion-resource-evidence",
                "site_count": sites,
                "spin_components": spin,
                "color_components": color,
                "fifth_extent": fifth,
                "degrees_of_freedom": dofs,
                "field_storage_bytes": storage,
                "policy": policy.policy_id,
            }
        )


def canonical_site_spinor_space(
    site_count: int,
    spin_components: int,
    color_components: int,
    /,
    *,
    dtype: Any = jnp.complex64,
    boundary_id: str,
    representation_id: str,
) -> ArraySpace:
    """Return the canonical ``(site, spin, color)`` complex fermion space.

    Color is the final payload axis, matching gauge representations constructed
    with their canonical ``color_axis=-1`` convention.
    """

    sites = int(site_count)
    spin = int(spin_components)
    color = int(color_components)
    boundary = str(boundary_id)
    representation = str(representation_id)
    dtype_ = np.dtype(jax.dtypes.canonicalize_dtype(np.dtype(dtype)))
    if sites < 1 or spin < 1 or color < 1:
        raise ValueError("Site, spin, and color counts must be positive.")
    if not boundary or not representation:
        raise ValueError("Boundary and representation identities must be non-empty.")
    if not jnp.issubdtype(dtype_, jnp.complexfloating):
        raise TypeError("Canonical site spinors require a complex dtype.")
    return ArraySpace(
        (sites, spin, color),
        dtype=dtype_,
        space_id=canonical_fingerprint(
            {
                "kind": "canonical-site-spinor-space",
                "shape": [sites, spin, color],
                "dtype": dtype_.str,
                "boundary": boundary,
                "representation": representation,
                "axis_order": ["site", "spin", "color"],
            }
        ),
    )


def canonical_euclidean_gamma_matrices(
    dimension: int,
    /,
    *,
    dtype: Any = jnp.complex64,
) -> tuple[Array, Array]:
    """Construct a deterministic Hermitian Euclidean Clifford representation."""

    dimension_ = int(dimension)
    if dimension_ < 2 or dimension_ % 2 or dimension_ > 8:
        raise ValueError(
            "Canonical Euclidean gamma matrices require even dimension 2..8."
        )
    dtype_ = np.dtype(jax.dtypes.canonicalize_dtype(np.dtype(dtype)))
    if not jnp.issubdtype(dtype_, jnp.complexfloating):
        raise TypeError("Euclidean gamma matrices require a complex dtype.")
    sigma_1 = np.asarray(((0, 1), (1, 0)), dtype=dtype_)
    sigma_2 = np.asarray(((0, -1j), (1j, 0)), dtype=dtype_)
    sigma_3 = np.asarray(((1, 0), (0, -1)), dtype=dtype_)
    identity = np.eye(2, dtype=dtype_)
    half = dimension_ // 2
    matrices: list[np.ndarray] = []
    for position in range(half):
        left = np.eye(1, dtype=dtype_)
        for _ in range(position):
            left = np.kron(left, identity)
        right = np.eye(1, dtype=dtype_)
        for _ in range(half - position - 1):
            right = np.kron(right, sigma_3)
        matrices.append(np.kron(np.kron(left, sigma_1), right))
        matrices.append(np.kron(np.kron(left, sigma_2), right))
    gamma = np.stack(matrices, axis=0)
    gamma5 = np.eye(gamma.shape[-1], dtype=dtype_)
    for matrix in gamma:
        gamma5 = gamma5 @ matrix
    gamma5 = ((-1j) ** half) * gamma5
    return jnp.asarray(gamma), jnp.asarray(gamma5)


def _validated_gamma(
    dimension: int,
    gamma_matrices: ArrayLike | None,
    gamma5: ArrayLike | None,
    dtype: np.dtype,
    /,
) -> tuple[Array, Array, int]:
    if gamma_matrices is None and gamma5 is None:
        gamma, chirality = canonical_euclidean_gamma_matrices(dimension, dtype=dtype)
    elif gamma_matrices is None or gamma5 is None:
        raise ValueError("gamma_matrices and gamma5 must be supplied together.")
    else:
        gamma = jnp.asarray(gamma_matrices, dtype=dtype)
        chirality = jnp.asarray(gamma5, dtype=dtype)
    if gamma.ndim != 3 or gamma.shape[0] != dimension or gamma.shape[1] != gamma.shape[2]:
        raise ValueError("gamma_matrices must have shape (dimension, spin, spin).")
    spin = int(gamma.shape[1])
    if chirality.shape != (spin, spin):
        raise ValueError("gamma5 must have shape (spin, spin).")
    host_gamma = np.asarray(gamma)
    host_gamma5 = np.asarray(chirality)
    identity = np.eye(spin, dtype=host_gamma.dtype)
    tolerance = 200.0 * np.finfo(host_gamma.real.dtype).eps * max(1, spin)
    clifford_defect = 0.0
    for mu in range(dimension):
        for nu in range(dimension):
            expected = 2.0 * identity if mu == nu else np.zeros_like(identity)
            defect = np.max(
                np.abs(
                    host_gamma[mu] @ host_gamma[nu]
                    + host_gamma[nu] @ host_gamma[mu]
                    - expected
                )
            )
            clifford_defect = max(clifford_defect, float(defect))
    hermitian_defect = float(
        max(np.max(np.abs(matrix.conj().T - matrix)) for matrix in host_gamma)
    )
    chirality_defect = max(
        float(np.max(np.abs(host_gamma5.conj().T - host_gamma5))),
        float(np.max(np.abs(host_gamma5 @ host_gamma5 - identity))),
        max(
            float(np.max(np.abs(host_gamma5 @ matrix + matrix @ host_gamma5)))
            for matrix in host_gamma
        ),
    )
    if max(clifford_defect, hermitian_defect, chirality_defect) > tolerance:
        raise ValueError(
            "Gamma matrices do not satisfy the Hermitian Euclidean Clifford contract."
        )
    return gamma, chirality, spin


def _finite_real(value: float, name: str, /) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _spin_action(matrix: Array, field: Array, /) -> Array:
    return ein.contract("ab,...bc->...ac", matrix, field, backend="jax")


def _transpose_action(
    action: Any,
    source: ArraySpace,
    target: ArraySpace,
    vector: PyTree[Any],
    /,
) -> Array:
    value = target.validate(vector)
    result = jax.linear_transpose(action, source.zeros())(value)[0]
    return source.validate(result)


def _adjoint_action(
    action: Any,
    source: ArraySpace,
    target: ArraySpace,
    vector: PyTree[Any],
    /,
) -> Array:
    value = target.validate(vector)
    transposed = jax.linear_transpose(action, source.zeros())(jnp.conj(value))[0]
    return source.validate(jnp.conj(transposed))


def _matrix_free_materialization() -> Array:
    raise LinearCapabilityError(
        "Lattice Dirac operators are intentionally matrix-free; apply them to "
        "canonical basis vectors for bounded reference checks."
    )


def _operator_capabilities() -> OperatorCapabilities:
    return OperatorCapabilities(
        transpose=True,
        adjoint=True,
        materialize=False,
        diagonal_assembly=False,
    )


def _operator_properties(*, self_adjoint: bool = False) -> OperatorProperties:
    return OperatorProperties(
        self_adjoint=self_adjoint,
        evidence={"self_adjoint": "construction"} if self_adjoint else None,
    )


def _free_directional_shifts(
    boundary: LatticeBoundaryPhasePlan,
    field: Array,
    /,
) -> tuple[Array, Array]:
    payload = field.shape[1:]
    lattice = field.reshape(boundary.topology.axis_sizes + payload)
    forward = jnp.stack(
        tuple(
            boundary.forward(lattice, axis).reshape((boundary.site_count,) + payload)
            for axis in range(boundary.dimension)
        ),
        axis=1,
    )
    backward = jnp.stack(
        tuple(
            boundary.backward(lattice, axis).reshape((boundary.site_count,) + payload)
            for axis in range(boundary.dimension)
        ),
        axis=1,
    )
    return forward, backward


def _wilson_action(
    field: Array,
    forward: Array,
    backward: Array,
    gamma_matrices: Array,
    mass: float,
    wilson_parameter: float,
    lattice_spacing: float,
    /,
) -> Array:
    diagonal = mass + gamma_matrices.shape[0] * wilson_parameter / lattice_spacing
    result = jnp.asarray(diagonal, dtype=field.dtype) * field
    hopping_scale = jnp.asarray(0.5 / lattice_spacing, dtype=field.dtype)
    for axis in range(gamma_matrices.shape[0]):
        gamma_forward = _spin_action(gamma_matrices[axis], forward[:, axis])
        gamma_backward = _spin_action(gamma_matrices[axis], backward[:, axis])
        result = result - hopping_scale * (
            wilson_parameter * (forward[:, axis] + backward[:, axis])
            - gamma_forward
            + gamma_backward
        )
    return result


class AbstractLatticeDiracOperator(AbstractPseudofermionDiracOperator):
    """Matrix-free lattice Dirac contract on canonical site-major spinors."""

    source: ArraySpace
    target: ArraySpace
    boundary: LatticeBoundaryPhasePlan
    links: Array
    gamma_matrices: Array
    resource_policy: LatticeFermionResourcePolicy = eqx.field(static=True)
    gamma5: Array
    resources: LatticeFermionResourceEvidence
    lattice_shape: tuple[int, ...] = eqx.field(static=True)
    site_count: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    spin_components: int = eqx.field(static=True)
    color_components: int = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)
    representation_id: str = eqx.field(static=True)
    gauge_field_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def _apply(self, vector: Array, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def _adjoint_apply(self, vector: Array, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def _gamma5_apply(self, vector: Array, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def diagonal_mv(self, vector: PyTree[Any], /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def diagonal_inverse_mv(self, vector: PyTree[Any], /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def with_links(self, links: ArrayLike, /) -> "AbstractLatticeDiracOperator":
        raise NotImplementedError

    def mv(self, vector: PyTree[Any], /) -> Array:
        value = self.source.validate(vector)
        return self.target.validate(self._apply(value))

    def transpose_mv(self, vector: PyTree[Any], /) -> Array:
        value = self.target.validate(vector)
        return self.source.validate(jnp.conj(self._adjoint_apply(jnp.conj(value))))

    def adjoint_mv(self, vector: PyTree[Any], /) -> Array:
        return self.source.validate(self._adjoint_apply(self.target.validate(vector)))

    def gamma5_mv(self, vector: PyTree[Any], /) -> Array:
        return self.source.validate(self._gamma5_apply(self.source.validate(vector)))

    def gamma5_hermitian_mv(self, vector: PyTree[Any], /) -> Array:
        return self.gamma5_mv(self.mv(vector))

    def gamma5_adjoint_mv(self, vector: PyTree[Any], /) -> Array:
        """Apply the adjoint of the Hermitian-kernel action ``gamma5 D``."""

        return self.adjoint_mv(self.gamma5_mv(vector))

    def gamma5_conjugate_mv(self, vector: PyTree[Any], /) -> Array:
        return self.gamma5_mv(self.mv(self.gamma5_mv(vector)))

    def _materialize(self, /) -> Array:
        return _matrix_free_materialization()


def _initialize_lattice_operator(
    operator: AbstractLatticeDiracOperator,
    boundary: LatticeBoundaryPhasePlan,
    links: Array,
    gamma_matrices: Array,
    gamma5: Array,
    color_components: int,
    representation_id: str,
    gauge_field_id: str,
    resource_policy: LatticeFermionResourcePolicy,
    operator_id: str,
    /,
) -> None:
    site_count = boundary.site_count
    spin = int(gamma_matrices.shape[-1])
    dtype = np.dtype(gamma_matrices.dtype)
    space = canonical_site_spinor_space(
        site_count,
        spin,
        color_components,
        dtype=dtype,
        boundary_id=boundary.plan_id,
        representation_id=representation_id,
    )
    operator.source = space
    operator.target = space
    operator.properties = _operator_properties()
    operator.capabilities = _operator_capabilities()
    operator.batch_shape = ()
    operator.operator_id = operator_id
    operator.boundary = boundary
    operator.links = links
    operator.gamma_matrices = gamma_matrices
    operator.gamma5 = gamma5
    operator.resources = LatticeFermionResourceEvidence(
        site_count,
        spin,
        color_components,
        dtype,
        resource_policy,
    )
    operator.resource_policy = resource_policy
    operator.lattice_shape = boundary.topology.axis_sizes
    operator.site_count = site_count
    operator.dimension = boundary.dimension
    operator.spin_components = spin
    operator.color_components = int(color_components)
    operator.boundary_id = boundary.plan_id
    operator.representation_id = representation_id
    operator.gauge_field_id = gauge_field_id


class FreeWilsonDiracOperator(AbstractLatticeDiracOperator):
    """Free Wilson operator with exact boundary phases and no gauge storage."""

    mass: float = eqx.field(static=True)
    wilson_parameter: float = eqx.field(static=True)
    lattice_spacing: float = eqx.field(static=True)
    resource_policy: LatticeFermionResourcePolicy = eqx.field(static=True)

    def __init__(
        self,
        boundary: LatticeBoundaryPhasePlan,
        /,
        *,
        mass: float,
        color_components: int = 1,
        wilson_parameter: float = 1.0,
        lattice_spacing: float = 1.0,
        gamma_matrices: ArrayLike | None = None,
        gamma5: ArrayLike | None = None,
        dtype: Any = jnp.complex64,
        resources: LatticeFermionResourcePolicy | None = None,
    ):
        if not isinstance(boundary, LatticeBoundaryPhasePlan):
            raise TypeError("boundary must be LatticeBoundaryPhasePlan.")
        policy = LatticeFermionResourcePolicy() if resources is None else resources
        if not isinstance(policy, LatticeFermionResourcePolicy):
            raise TypeError("resources must be LatticeFermionResourcePolicy or None.")
        dtype_ = np.dtype(jax.dtypes.canonicalize_dtype(np.dtype(dtype)))
        gamma, chirality, _ = _validated_gamma(
            boundary.dimension, gamma_matrices, gamma5, dtype_
        )
        mass_ = _finite_real(mass, "mass")
        wilson = _finite_real(wilson_parameter, "wilson_parameter")
        spacing = _finite_real(lattice_spacing, "lattice_spacing")
        if wilson <= 0.0 or spacing <= 0.0:
            raise ValueError("wilson_parameter and lattice_spacing must be positive.")
        color = int(color_components)
        representation_id = canonical_fingerprint(
            {"kind": "trivial-gauge-representation", "dimension": color}
        )
        gauge_field_id = canonical_fingerprint(
            {
                "kind": "trivial-gauge-field",
                "boundary": boundary.plan_id,
                "representation": representation_id,
            }
        )
        operator_id = canonical_fingerprint(
            {
                "kind": "free-wilson-dirac-operator",
                "boundary": boundary.plan_id,
                "representation": representation_id,
                "gauge_field": gauge_field_id,
                "mass": mass_,
                "wilson_parameter": wilson,
                "lattice_spacing": spacing,
                "gamma": array_tree_fingerprint(
                    (np.asarray(gamma), np.asarray(chirality))
                ),
            }
        )
        _initialize_lattice_operator(
            self,
            boundary,
            jnp.zeros((0, 0, 0), dtype=dtype_),
            gamma,
            chirality,
            color,
            representation_id,
            gauge_field_id,
            policy,
            operator_id,
        )
        self.mass = mass_
        self.wilson_parameter = wilson
        self.lattice_spacing = spacing
        self.resource_policy = policy

    def _apply(self, vector: Array, /) -> Array:
        forward, backward = _free_directional_shifts(self.boundary, vector)
        return _wilson_action(
            vector,
            forward,
            backward,
            self.gamma_matrices,
            self.mass,
            self.wilson_parameter,
            self.lattice_spacing,
        )

    def _adjoint_apply(self, vector: Array, /) -> Array:
        return self._gamma5_apply(self._apply(self._gamma5_apply(vector)))

    def _gamma5_apply(self, vector: Array, /) -> Array:
        return _spin_action(self.gamma5, vector)

    def diagonal_mv(self, vector: PyTree[Any], /) -> Array:
        value = self.source.validate(vector)
        diagonal = (
            self.mass + self.dimension * self.wilson_parameter / self.lattice_spacing
        )
        return jnp.asarray(diagonal, dtype=value.dtype) * value

    def diagonal_inverse_mv(self, vector: PyTree[Any], /) -> Array:
        value = self.source.validate(vector)
        diagonal = (
            self.mass + self.dimension * self.wilson_parameter / self.lattice_spacing
        )
        if diagonal == 0.0:
            raise ValueError("The Wilson site-diagonal block is singular.")
        return value / jnp.asarray(diagonal, dtype=value.dtype)

    def with_links(self, links: ArrayLike, /) -> "FreeWilsonDiracOperator":
        values = jnp.asarray(links)
        if values.shape != (0, 0, 0):
            raise ValueError(
                "FreeWilsonDiracOperator accepts only its empty gauge field."
            )
        return FreeWilsonDiracOperator(
            self.boundary,
            mass=self.mass,
            color_components=self.color_components,
            wilson_parameter=self.wilson_parameter,
            lattice_spacing=self.lattice_spacing,
            gamma_matrices=self.gamma_matrices,
            gamma5=self.gamma5,
            dtype=self.source.dtype,
            resources=self.resource_policy,
        )


class WilsonDiracOperator(AbstractLatticeDiracOperator):
    """Gauge-covariant Wilson operator in one explicit representation."""

    representation: AbstractGaugeRepresentation
    transport: GaugeCovariantShiftPlan
    mass: float = eqx.field(static=True)
    wilson_parameter: float = eqx.field(static=True)
    lattice_spacing: float = eqx.field(static=True)
    resource_policy: LatticeFermionResourcePolicy = eqx.field(static=True)

    def __init__(
        self,
        boundary: LatticeBoundaryPhasePlan,
        representation: AbstractGaugeRepresentation,
        transport: GaugeCovariantShiftPlan,
        links: ArrayLike,
        /,
        *,
        mass: float,
        wilson_parameter: float = 1.0,
        lattice_spacing: float = 1.0,
        gamma_matrices: ArrayLike | None = None,
        gamma5: ArrayLike | None = None,
        dtype: Any = jnp.complex64,
        gauge_field_id: str | None = None,
        resources: LatticeFermionResourcePolicy | None = None,
    ):
        if not isinstance(boundary, LatticeBoundaryPhasePlan):
            raise TypeError("boundary must be LatticeBoundaryPhasePlan.")
        if not isinstance(representation, AbstractGaugeRepresentation):
            raise TypeError("representation must be AbstractGaugeRepresentation.")
        if not isinstance(transport, GaugeCovariantShiftPlan):
            raise TypeError("transport must be GaugeCovariantShiftPlan.")
        if (
            transport.site_count != boundary.site_count
            or transport.dimension != boundary.dimension
        ):
            raise ValueError("Transport and boundary lattice dimensions must agree.")
        if transport.representation_id != representation.representation_id:
            raise ValueError("Transport and Wilson representation identities must agree.")
        if transport.boundary_id != boundary.plan_id:
            raise ValueError("Transport and Wilson boundary identities must agree.")
        if representation.color_axis != -1:
            raise ValueError(
                "Canonical Wilson spinors require representation color_axis=-1."
            )
        policy = LatticeFermionResourcePolicy() if resources is None else resources
        if not isinstance(policy, LatticeFermionResourcePolicy):
            raise TypeError("resources must be LatticeFermionResourcePolicy or None.")
        dtype_ = np.dtype(jax.dtypes.canonicalize_dtype(np.dtype(dtype)))
        gamma, chirality, spin = _validated_gamma(
            boundary.dimension, gamma_matrices, gamma5, dtype_
        )
        values = jnp.asarray(links)
        if values.ndim != 3 or values.shape[-2:] != representation.group.point_shape:
            raise ValueError(
                "links must have shape (edge_count, group_matrix_rows, group_matrix_columns)."
            )
        probe = jnp.zeros(
            (boundary.site_count, spin, representation.dimension), dtype=dtype_
        )
        shifted = jax.eval_shape(transport.forward, values, probe)
        expected = (
            boundary.site_count,
            boundary.dimension,
            spin,
            representation.dimension,
        )
        if shifted.shape != expected:
            raise ValueError(
                f"Gauge transport returned shape {shifted.shape}; expected {expected}."
            )
        mass_ = _finite_real(mass, "mass")
        wilson = _finite_real(wilson_parameter, "wilson_parameter")
        spacing = _finite_real(lattice_spacing, "lattice_spacing")
        if wilson <= 0.0 or spacing <= 0.0:
            raise ValueError("wilson_parameter and lattice_spacing must be positive.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "gauge-field",
                    "link_space": transport.link_space_id,
                    "links": array_tree_fingerprint(np.asarray(values)),
                }
            )
            if gauge_field_id is None
            else str(gauge_field_id)
        )
        if not identifier:
            raise ValueError("gauge_field_id must be non-empty.")
        operator_id = canonical_fingerprint(
            {
                "kind": "wilson-dirac-operator",
                "transport": transport.plan_id,
                "boundary": boundary.plan_id,
                "representation": representation.representation_id,
                "gauge_field": identifier,
                "mass": mass_,
                "wilson_parameter": wilson,
                "lattice_spacing": spacing,
                "gamma": array_tree_fingerprint(
                    (np.asarray(gamma), np.asarray(chirality))
                ),
            }
        )
        _initialize_lattice_operator(
            self,
            boundary,
            values,
            gamma,
            chirality,
            representation.dimension,
            representation.representation_id,
            identifier,
            policy,
            operator_id,
        )
        self.representation = representation
        self.transport = transport
        self.mass = mass_
        self.wilson_parameter = wilson
        self.lattice_spacing = spacing
        self.resource_policy = policy

    def _apply(self, vector: Array, /) -> Array:
        forward = self.transport.forward(self.links, vector)
        backward = self.transport.backward(self.links, vector)
        return _wilson_action(
            vector,
            forward,
            backward,
            self.gamma_matrices,
            self.mass,
            self.wilson_parameter,
            self.lattice_spacing,
        )

    def _adjoint_apply(self, vector: Array, /) -> Array:
        return self._gamma5_apply(self._apply(self._gamma5_apply(vector)))

    def _gamma5_apply(self, vector: Array, /) -> Array:
        return _spin_action(self.gamma5, vector)

    def diagonal_mv(self, vector: PyTree[Any], /) -> Array:
        value = self.source.validate(vector)
        diagonal = (
            self.mass + self.dimension * self.wilson_parameter / self.lattice_spacing
        )
        return jnp.asarray(diagonal, dtype=value.dtype) * value

    def diagonal_inverse_mv(self, vector: PyTree[Any], /) -> Array:
        value = self.source.validate(vector)
        diagonal = (
            self.mass + self.dimension * self.wilson_parameter / self.lattice_spacing
        )
        if diagonal == 0.0:
            raise ValueError("The Wilson site-diagonal block is singular.")
        return value / jnp.asarray(diagonal, dtype=value.dtype)

    def with_links(self, links: ArrayLike, /) -> "WilsonDiracOperator":
        return WilsonDiracOperator(
            self.boundary,
            self.representation,
            self.transport,
            links,
            mass=self.mass,
            wilson_parameter=self.wilson_parameter,
            lattice_spacing=self.lattice_spacing,
            gamma_matrices=self.gamma_matrices,
            gamma5=self.gamma5,
            dtype=self.source.dtype,
            gauge_field_id=self.gauge_field_id,
            resources=self.resource_policy,
        )


class CloverInverseEvidence(StrictModule):
    """Condition and two-sided residual evidence for all local clover inverses."""

    condition_estimate: Array
    left_residual: Array
    right_residual: Array
    finite: Array
    tolerance: float = eqx.field(static=True)
    storage_bytes: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class CloverWilsonDiracOperator(AbstractLatticeDiracOperator):
    """Wilson operator plus a gamma5-Hermitian site-local clover block."""

    wilson: WilsonDiracOperator
    clover_term: Array
    local_inverse: Array
    inverse_evidence: CloverInverseEvidence
    inverse_tolerance: float = eqx.field(static=True)
    resource_policy: LatticeFermionResourcePolicy = eqx.field(static=True)

    def __init__(
        self,
        wilson: WilsonDiracOperator,
        clover_term: ArrayLike,
        /,
        *,
        inverse_tolerance: float = 1e-6,
        resources: LatticeFermionResourcePolicy | None = None,
    ):
        if not isinstance(wilson, WilsonDiracOperator):
            raise TypeError("wilson must be WilsonDiracOperator.")
        policy = wilson.resource_policy if resources is None else resources
        if not isinstance(policy, LatticeFermionResourcePolicy):
            raise TypeError("resources must be LatticeFermionResourcePolicy or None.")
        tolerance = _finite_real(inverse_tolerance, "inverse_tolerance")
        if tolerance <= 0.0:
            raise ValueError("inverse_tolerance must be positive.")
        spin = wilson.spin_components
        color = wilson.color_components
        expected = (wilson.site_count, spin, color, spin, color)
        clover = jnp.asarray(clover_term, dtype=wilson.source.dtype)
        if clover.shape != expected:
            raise ValueError(
                f"clover_term must have shape {expected}; got {clover.shape}."
            )
        block_size = spin * color
        storage_bytes = (
            wilson.site_count * block_size * block_size * clover.dtype.itemsize
        )
        if storage_bytes > policy.maximum_local_inverse_bytes:
            raise ValueError("Clover local inverses exceed the resource policy.")
        matrices = clover.reshape((wilson.site_count, block_size, block_size))
        gamma5_color = jnp.kron(wilson.gamma5, jnp.eye(color, dtype=clover.dtype))
        host_matrices = np.asarray(matrices)
        host_gamma5 = np.asarray(gamma5_color)
        hermiticity_defect = float(
            np.max(
                np.abs(
                    np.swapaxes(host_matrices.conj(), -1, -2)
                    - host_gamma5[None] @ host_matrices @ host_gamma5[None]
                )
            )
        )
        if hermiticity_defect > tolerance:
            raise ValueError("clover_term must satisfy local gamma5-Hermiticity.")
        diagonal = (
            wilson.mass
            + wilson.dimension * wilson.wilson_parameter / wilson.lattice_spacing
        )
        local = matrices + diagonal * jnp.eye(block_size, dtype=clover.dtype)[None]
        inverse = jnp.linalg.inv(local)
        identity = jnp.eye(block_size, dtype=clover.dtype)[None]
        left_residual = jnp.max(jnp.abs(local @ inverse - identity), axis=(-2, -1))
        right_residual = jnp.max(jnp.abs(inverse @ local - identity), axis=(-2, -1))
        singular_values = jnp.linalg.svd(local, compute_uv=False)
        condition = singular_values[:, 0] / singular_values[:, -1]
        finite = (
            jnp.all(jnp.isfinite(inverse), axis=(-2, -1))
            & jnp.isfinite(condition)
            & (left_residual <= tolerance)
            & (right_residual <= tolerance)
        )
        inverse = eqx.error_if(
            inverse,
            ~jnp.all(finite),
            "Clover local block inversion failed its residual contract.",
        )
        evidence_id = canonical_fingerprint(
            {
                "kind": "clover-local-inverse-evidence",
                "wilson": wilson.operator_id,
                "clover": array_tree_fingerprint(np.asarray(clover)),
                "tolerance": tolerance,
                "storage_bytes": storage_bytes,
            }
        )
        operator_id = canonical_fingerprint(
            {
                "kind": "clover-wilson-dirac-operator",
                "wilson": wilson.operator_id,
                "clover": array_tree_fingerprint(np.asarray(clover)),
                "inverse_evidence": evidence_id,
            }
        )
        _initialize_lattice_operator(
            self,
            wilson.boundary,
            wilson.links,
            wilson.gamma_matrices,
            wilson.gamma5,
            color,
            wilson.representation_id,
            wilson.gauge_field_id,
            policy,
            operator_id,
        )
        self.wilson = wilson
        self.clover_term = clover
        self.local_inverse = inverse.reshape(expected)
        self.inverse_evidence = CloverInverseEvidence(
            condition_estimate=condition,
            left_residual=left_residual,
            right_residual=right_residual,
            finite=finite,
            tolerance=tolerance,
            storage_bytes=storage_bytes,
            evidence_id=evidence_id,
        )
        self.inverse_tolerance = tolerance
        self.resource_policy = policy

    def _local_action(self, blocks: Array, vector: Array, /) -> Array:
        return ein.contract("sacbd,sbd->sac", blocks, vector, backend="jax")

    def _apply(self, vector: Array, /) -> Array:
        return self.wilson.mv(vector) + self._local_action(self.clover_term, vector)

    def _adjoint_apply(self, vector: Array, /) -> Array:
        return self._gamma5_apply(self._apply(self._gamma5_apply(vector)))

    def _gamma5_apply(self, vector: Array, /) -> Array:
        return _spin_action(self.gamma5, vector)

    def diagonal_mv(self, vector: PyTree[Any], /) -> Array:
        value = self.source.validate(vector)
        return self.wilson.diagonal_mv(value) + self._local_action(
            self.clover_term, value
        )

    def diagonal_inverse_mv(self, vector: PyTree[Any], /) -> Array:
        return self._local_action(self.local_inverse, self.source.validate(vector))

    def with_links(self, links: ArrayLike, /) -> "CloverWilsonDiracOperator":
        return CloverWilsonDiracOperator(
            self.wilson.with_links(links),
            self.clover_term,
            inverse_tolerance=self.inverse_tolerance,
            resources=self.resource_policy,
        )


class EvenOddSchurPlan(StrictModule):
    """Immutable checkerboard block-elimination structure."""

    layout: CheckerboardEntityLayout
    reduced_space: ArraySpace
    eliminated_space: ArraySpace
    reduced_parity: int = eqx.field(static=True)
    eliminated_parity: int = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)
    representation_id: str = eqx.field(static=True)
    reduced_dofs: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class EvenOddSchurOperator(AbstractLinearOperator):
    """Matrix-free reduced checkerboard Schur complement."""

    operator: AbstractLatticeDiracOperator
    plan: EvenOddSchurPlan
    boundary_id: str = eqx.field(static=True)
    representation_id: str = eqx.field(static=True)
    gauge_field_id: str = eqx.field(static=True)

    def __init__(
        self,
        operator: AbstractLatticeDiracOperator,
        plan: EvenOddSchurPlan,
        /,
    ):
        if not isinstance(operator, AbstractLatticeDiracOperator):
            raise TypeError("operator must be AbstractLatticeDiracOperator.")
        if not isinstance(plan, EvenOddSchurPlan):
            raise TypeError("plan must be EvenOddSchurPlan.")
        if operator.operator_id != plan.operator_id:
            raise ValueError("Schur plan and operator identities do not match.")
        self.operator = operator
        self.plan = plan
        self.boundary_id = operator.boundary_id
        self.representation_id = operator.representation_id
        self.gauge_field_id = operator.gauge_field_id
        self.source = plan.reduced_space
        self.target = plan.reduced_space
        self.properties = _operator_properties()
        self.capabilities = _operator_capabilities()
        self.batch_shape = ()
        self.operator_id = canonical_fingerprint(
            {
                "kind": "even-odd-schur-operator",
                "operator": operator.operator_id,
                "plan": plan.plan_id,
            }
        )

    def _embedded(self, values: Array, parity: int, /) -> Array:
        layout = self.plan.layout
        even_shape = (layout.even_count,) + self.operator.source.shape[1:]
        odd_shape = (layout.odd_count,) + self.operator.source.shape[1:]
        even = jnp.zeros(even_shape, dtype=values.dtype)
        odd = jnp.zeros(odd_shape, dtype=values.dtype)
        if parity == 0:
            even = values
        else:
            odd = values
        return layout.merge(even, odd).reshape(self.operator.source.shape)

    def block(self, values: Array, source_parity: int, target_parity: int, /) -> Array:
        full = self._embedded(values, source_parity)
        return self.plan.layout.gather(self.operator.mv(full), target_parity)

    def inverse_diagonal_block(self, values: Array, parity: int, /) -> Array:
        full = self._embedded(values, parity)
        inverted = self.operator.diagonal_inverse_mv(full)
        return self.plan.layout.gather(inverted, parity)

    def mv(self, vector: PyTree[Any], /) -> Array:
        value = self.source.validate(vector)
        reduced = self.plan.reduced_parity
        eliminated = self.plan.eliminated_parity
        direct = self.block(value, reduced, reduced)
        coupled = self.block(value, reduced, eliminated)
        solved = self.inverse_diagonal_block(coupled, eliminated)
        return self.target.validate(direct - self.block(solved, eliminated, reduced))

    def transpose_mv(self, vector: PyTree[Any], /) -> Array:
        return _transpose_action(self.mv, self.source, self.target, vector)

    def adjoint_mv(self, vector: PyTree[Any], /) -> Array:
        return _adjoint_action(self.mv, self.source, self.target, vector)

    def _materialize(self, /) -> Array:
        return _matrix_free_materialization()


class PreparedEvenOddSchur(StrictModule):
    """Numeric Dirac epoch bound to one immutable checkerboard plan."""

    operator: AbstractLatticeDiracOperator
    schur: EvenOddSchurOperator
    plan: EvenOddSchurPlan
    prepared_id: str = eqx.field(static=True)

    def reduce_rhs(self, rhs: PyTree[Any], /) -> Array:
        value = self.operator.target.validate(rhs)
        reduced = self.plan.layout.gather(value, self.plan.reduced_parity)
        eliminated = self.plan.layout.gather(value, self.plan.eliminated_parity)
        solved = self.schur.inverse_diagonal_block(
            eliminated, self.plan.eliminated_parity
        )
        return reduced - self.schur.block(
            solved,
            self.plan.eliminated_parity,
            self.plan.reduced_parity,
        )

    def reconstruct(self, reduced_solution: ArrayLike, rhs: PyTree[Any], /) -> Array:
        reduced = self.plan.reduced_space.validate(reduced_solution)
        right = self.operator.target.validate(rhs)
        eliminated_rhs = self.plan.layout.gather(right, self.plan.eliminated_parity)
        coupling = self.schur.block(
            reduced,
            self.plan.reduced_parity,
            self.plan.eliminated_parity,
        )
        eliminated = self.schur.inverse_diagonal_block(
            eliminated_rhs - coupling,
            self.plan.eliminated_parity,
        )
        if self.plan.reduced_parity == 0:
            merged = self.plan.layout.merge(reduced, eliminated)
        else:
            merged = self.plan.layout.merge(eliminated, reduced)
        return merged.reshape(self.operator.source.shape)


def plan_even_odd_schur(
    operator: AbstractLatticeDiracOperator,
    layout: CheckerboardEntityLayout,
    /,
    *,
    reduced_parity: int = 1,
    maximum_reduced_dofs: int = 33_554_432,
) -> EvenOddSchurPlan:
    if not isinstance(operator, AbstractLatticeDiracOperator):
        raise TypeError("operator must be AbstractLatticeDiracOperator.")
    if not isinstance(layout, CheckerboardEntityLayout):
        raise TypeError("layout must be CheckerboardEntityLayout.")
    if layout.topology.topology_id != operator.boundary.topology.topology_id:
        raise ValueError("Checkerboard and Dirac topologies must agree.")
    parity = int(reduced_parity)
    if parity not in (0, 1):
        raise ValueError("reduced_parity must be zero or one.")
    reduced_count = layout.even_count if parity == 0 else layout.odd_count
    eliminated_count = layout.odd_count if parity == 0 else layout.even_count
    payload = operator.source.shape[1:]
    reduced_dofs = reduced_count * prod(payload)
    if reduced_dofs > int(maximum_reduced_dofs):
        raise ValueError("Reduced checkerboard degrees of freedom exceed the plan limit.")
    reduced_space = ArraySpace(
        (reduced_count,) + payload,
        dtype=operator.source.dtype,
        space_id=canonical_fingerprint(
            {
                "kind": "checkerboard-spinor-space",
                "operator_space": operator.source.space_id,
                "layout": layout.layout_id,
                "parity": parity,
            }
        ),
    )
    eliminated_space = ArraySpace(
        (eliminated_count,) + payload,
        dtype=operator.source.dtype,
        space_id=canonical_fingerprint(
            {
                "kind": "checkerboard-spinor-space",
                "operator_space": operator.source.space_id,
                "layout": layout.layout_id,
                "parity": 1 - parity,
            }
        ),
    )
    plan_id = canonical_fingerprint(
        {
            "kind": "even-odd-schur-plan",
            "operator": operator.operator_id,
            "layout": layout.layout_id,
            "reduced_parity": parity,
            "maximum_reduced_dofs": int(maximum_reduced_dofs),
        }
    )
    return EvenOddSchurPlan(
        layout=layout,
        reduced_space=reduced_space,
        eliminated_space=eliminated_space,
        reduced_parity=parity,
        eliminated_parity=1 - parity,
        operator_id=operator.operator_id,
        boundary_id=operator.boundary_id,
        representation_id=operator.representation_id,
        reduced_dofs=reduced_dofs,
        plan_id=plan_id,
    )


def prepare_even_odd_schur(
    operator: AbstractLatticeDiracOperator,
    plan: EvenOddSchurPlan,
    /,
) -> PreparedEvenOddSchur:
    schur = EvenOddSchurOperator(operator, plan)
    return PreparedEvenOddSchur(
        operator=operator,
        schur=schur,
        plan=plan,
        prepared_id=canonical_fingerprint(
            {
                "kind": "prepared-even-odd-schur",
                "operator": operator.operator_id,
                "plan": plan.plan_id,
                "gauge_field": operator.gauge_field_id,
            }
        ),
    )


class TwistedMassDoubletOperator(AbstractLinearOperator):
    """Two-flavor Wilson doublet ``D_W + i mu gamma5 tau3``."""

    wilson: AbstractLatticeDiracOperator
    source: ArraySpace
    target: ArraySpace
    twisted_mass: float = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)
    representation_id: str = eqx.field(static=True)
    gauge_field_id: str = eqx.field(static=True)
    resource_policy: LatticeFermionResourcePolicy = eqx.field(static=True)
    degrees_of_freedom: int = eqx.field(static=True)

    def __init__(
        self,
        wilson: AbstractLatticeDiracOperator,
        /,
        *,
        twisted_mass: float,
        resources: LatticeFermionResourcePolicy | None = None,
    ):
        if not isinstance(wilson, AbstractLatticeDiracOperator):
            raise TypeError("wilson must be AbstractLatticeDiracOperator.")
        mu = _finite_real(twisted_mass, "twisted_mass")
        policy = wilson.resource_policy if resources is None else resources
        if not isinstance(policy, LatticeFermionResourcePolicy):
            raise TypeError("resources must be LatticeFermionResourcePolicy or None.")
        policy.admit(
            wilson.site_count,
            wilson.spin_components,
            wilson.color_components,
            fifth_extent=2,
        )
        degrees_of_freedom = 2 * wilson.resources.degrees_of_freedom
        shape = (wilson.site_count, 2, wilson.spin_components, wilson.color_components)
        space = ArraySpace(
            shape,
            dtype=wilson.source.dtype,
            space_id=canonical_fingerprint(
                {
                    "kind": "twisted-mass-doublet-space",
                    "single_flavor": wilson.source.space_id,
                    "flavor_order": ["plus", "minus"],
                }
            ),
        )
        self.wilson = wilson
        self.source = space
        self.target = space
        self.twisted_mass = mu
        self.resource_policy = policy
        self.degrees_of_freedom = degrees_of_freedom
        self.properties = _operator_properties()
        self.capabilities = _operator_capabilities()
        self.batch_shape = ()
        self.boundary_id = wilson.boundary_id
        self.representation_id = wilson.representation_id
        self.gauge_field_id = wilson.gauge_field_id
        self.operator_id = canonical_fingerprint(
            {
                "kind": "twisted-mass-doublet-operator",
                "wilson": wilson.operator_id,
                "twisted_mass": mu,
                "tau3_order": [1, -1],
                "resources": policy.policy_id,
            }
        )

    def _apply(self, vector: Array, /) -> Array:
        base = jax.vmap(self.wilson.mv, in_axes=1, out_axes=1)(vector)
        chiral = jax.vmap(self.wilson.gamma5_mv, in_axes=1, out_axes=1)(vector)
        signs = jnp.asarray((1.0, -1.0), dtype=vector.dtype)[None, :, None, None]
        return base + 1j * self.twisted_mass * signs * chiral

    def mv(self, vector: PyTree[Any], /) -> Array:
        return self.target.validate(self._apply(self.source.validate(vector)))

    def transpose_mv(self, vector: PyTree[Any], /) -> Array:
        value = self.target.validate(vector)
        return self.source.validate(jnp.conj(self._adjoint_apply(jnp.conj(value))))

    def _adjoint_apply(self, vector: Array, /) -> Array:
        base = jax.vmap(self.wilson.adjoint_mv, in_axes=1, out_axes=1)(vector)
        chiral = jax.vmap(self.wilson.gamma5_mv, in_axes=1, out_axes=1)(vector)
        signs = jnp.asarray((1.0, -1.0), dtype=vector.dtype)[None, :, None, None]
        return base - 1j * self.twisted_mass * signs * chiral

    def adjoint_mv(self, vector: PyTree[Any], /) -> Array:
        return self.source.validate(self._adjoint_apply(self.target.validate(vector)))

    def flavor_gamma5_mv(self, vector: PyTree[Any], /) -> Array:
        value = self.source.validate(vector)
        chiral = jax.vmap(self.wilson.gamma5_mv, in_axes=1, out_axes=1)(value)
        signs = jnp.asarray((1.0, -1.0), dtype=value.dtype)[None, :, None, None]
        return signs * chiral

    def with_links(self, links: ArrayLike, /) -> "TwistedMassDoubletOperator":
        return TwistedMassDoubletOperator(
            self.wilson.with_links(links),
            twisted_mass=self.twisted_mass,
            resources=self.resource_policy,
        )

    def _materialize(self, /) -> Array:
        return _matrix_free_materialization()


def _staggered_phases(
    boundary: LatticeBoundaryPhasePlan, dtype: Any, /
) -> tuple[Array, Array]:
    coordinates = np.indices(boundary.topology.axis_sizes, dtype=np.int64)
    flattened = coordinates.reshape((boundary.dimension, boundary.site_count)).T
    eta = np.ones((boundary.site_count, boundary.dimension), dtype=np.int8)
    for axis in range(boundary.dimension):
        eta[:, axis] = 1 - 2 * (np.sum(flattened[:, :axis], axis=1) % 2)
    epsilon = 1 - 2 * (np.sum(flattened, axis=1) % 2)
    return jnp.asarray(eta, dtype=dtype), jnp.asarray(epsilon, dtype=dtype)


def _directional_transport(
    transport: GaugeCovariantShiftPlan,
    links: Array,
    field: Array,
    axis: int,
    displacement: int,
    /,
) -> Array:
    result = field
    action = transport.forward if displacement > 0 else transport.backward
    for _ in range(abs(displacement)):
        result = action(links, result)[:, axis]
    return result


class StaggeredDiracOperator(AbstractLatticeDiracOperator):
    """One-link Kogut--Susskind staggered Dirac operator."""

    representation: AbstractGaugeRepresentation
    transport: GaugeCovariantShiftPlan
    eta: Array
    epsilon: Array
    mass: float = eqx.field(static=True)
    lattice_spacing: float = eqx.field(static=True)
    resource_policy: LatticeFermionResourcePolicy = eqx.field(static=True)

    def __init__(
        self,
        boundary: LatticeBoundaryPhasePlan,
        representation: AbstractGaugeRepresentation,
        transport: GaugeCovariantShiftPlan,
        links: ArrayLike,
        /,
        *,
        mass: float,
        lattice_spacing: float = 1.0,
        dtype: Any = jnp.complex64,
        gauge_field_id: str | None = None,
        resources: LatticeFermionResourcePolicy | None = None,
    ):
        if not isinstance(boundary, LatticeBoundaryPhasePlan):
            raise TypeError("boundary must be LatticeBoundaryPhasePlan.")
        if not isinstance(representation, AbstractGaugeRepresentation):
            raise TypeError("representation must be AbstractGaugeRepresentation.")
        if not isinstance(transport, GaugeCovariantShiftPlan):
            raise TypeError("transport must be GaugeCovariantShiftPlan.")
        if (
            transport.site_count != boundary.site_count
            or transport.dimension != boundary.dimension
            or transport.representation_id != representation.representation_id
            or transport.boundary_id != boundary.plan_id
        ):
            raise ValueError(
                "Staggered transport, representation, and boundary contracts disagree."
            )
        if representation.color_axis != -1:
            raise ValueError(
                "Canonical staggered spinors require representation color_axis=-1."
            )
        policy = LatticeFermionResourcePolicy() if resources is None else resources
        if not isinstance(policy, LatticeFermionResourcePolicy):
            raise TypeError("resources must be LatticeFermionResourcePolicy or None.")
        dtype_ = np.dtype(jax.dtypes.canonicalize_dtype(np.dtype(dtype)))
        if not jnp.issubdtype(dtype_, jnp.complexfloating):
            raise TypeError("Staggered fermions require a complex dtype.")
        values = jnp.asarray(links)
        if values.ndim != 3 or values.shape[-2:] != representation.group.point_shape:
            raise ValueError("links have an incompatible gauge-matrix shape.")
        mass_ = _finite_real(mass, "mass")
        spacing = _finite_real(lattice_spacing, "lattice_spacing")
        if spacing <= 0.0:
            raise ValueError("lattice_spacing must be positive.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "gauge-field",
                    "link_space": transport.link_space_id,
                    "links": array_tree_fingerprint(np.asarray(values)),
                }
            )
            if gauge_field_id is None
            else str(gauge_field_id)
        )
        if not identifier:
            raise ValueError("gauge_field_id must be non-empty.")
        eta, epsilon = _staggered_phases(boundary, dtype_)
        gamma = jnp.zeros((boundary.dimension, 1, 1), dtype=dtype_)
        chirality = jnp.ones((1, 1), dtype=dtype_)
        operator_id = canonical_fingerprint(
            {
                "kind": "staggered-dirac-operator",
                "transport": transport.plan_id,
                "gauge_field": identifier,
                "mass": mass_,
                "lattice_spacing": spacing,
                "phase_convention": "eta-mu-preceding-coordinate-parity",
            }
        )
        _initialize_lattice_operator(
            self,
            boundary,
            values,
            gamma,
            chirality,
            representation.dimension,
            representation.representation_id,
            identifier,
            policy,
            operator_id,
        )
        self.representation = representation
        self.transport = transport
        self.eta = eta
        self.epsilon = epsilon
        self.mass = mass_
        self.lattice_spacing = spacing
        self.resource_policy = policy

    def _apply(self, vector: Array, /) -> Array:
        forward = self.transport.forward(self.links, vector)
        backward = self.transport.backward(self.links, vector)
        difference = forward - backward
        derivative = jnp.sum(
            self.eta[..., None, None] * difference,
            axis=1,
        ) / (2.0 * self.lattice_spacing)
        return self.mass * vector + derivative

    def _adjoint_apply(self, vector: Array, /) -> Array:
        return self._gamma5_apply(self._apply(self._gamma5_apply(vector)))

    def _gamma5_apply(self, vector: Array, /) -> Array:
        return self.epsilon[:, None, None] * vector

    def diagonal_mv(self, vector: PyTree[Any], /) -> Array:
        return self.mass * self.source.validate(vector)

    def diagonal_inverse_mv(self, vector: PyTree[Any], /) -> Array:
        if self.mass == 0.0:
            raise ValueError("The massless staggered diagonal block is singular.")
        return self.source.validate(vector) / self.mass

    def with_links(self, links: ArrayLike, /) -> "StaggeredDiracOperator":
        return StaggeredDiracOperator(
            self.boundary,
            self.representation,
            self.transport,
            links,
            mass=self.mass,
            lattice_spacing=self.lattice_spacing,
            dtype=self.source.dtype,
            gauge_field_id=self.gauge_field_id,
            resources=self.resource_policy,
        )


class NaikStaggeredDiracOperator(AbstractLatticeDiracOperator):
    """Staggered Naik/HISQ path operator with distinct one- and three-link fields."""

    one_link: StaggeredDiracOperator
    three_link_links: Array
    one_link_coefficient: float = eqx.field(static=True)
    three_link_coefficient: float = eqx.field(static=True)
    three_link_gauge_field_id: str = eqx.field(static=True)

    def __init__(
        self,
        one_link: StaggeredDiracOperator,
        /,
        *,
        three_link_links: ArrayLike | None = None,
        one_link_coefficient: float = 9.0 / 8.0,
        three_link_coefficient: float = -1.0 / 24.0,
        three_link_gauge_field_id: str | None = None,
    ):
        if not isinstance(one_link, StaggeredDiracOperator):
            raise TypeError("one_link must be StaggeredDiracOperator.")
        first = _finite_real(one_link_coefficient, "one_link_coefficient")
        third = _finite_real(three_link_coefficient, "three_link_coefficient")
        three_links = (
            one_link.links if three_link_links is None else jnp.asarray(three_link_links)
        )
        if three_links.shape != one_link.links.shape:
            raise ValueError(
                "three_link_links must use the one-link gauge storage shape."
            )
        identifier = (
            one_link.gauge_field_id
            if three_link_links is None
            else (
                canonical_fingerprint(
                    {
                        "kind": "naik-three-link-gauge-field",
                        "link_space": one_link.transport.link_space_id,
                        "links": array_tree_fingerprint(np.asarray(three_links)),
                    }
                )
                if three_link_gauge_field_id is None
                else str(three_link_gauge_field_id)
            )
        )
        if not identifier:
            raise ValueError("three_link_gauge_field_id must be non-empty.")
        operator_id = canonical_fingerprint(
            {
                "kind": "naik-hisq-staggered-dirac-operator",
                "one_link": one_link.operator_id,
                "three_link_gauge_field": identifier,
                "one_link_coefficient": first,
                "three_link_coefficient": third,
                "path_lengths": [1, 3],
            }
        )
        _initialize_lattice_operator(
            self,
            one_link.boundary,
            one_link.links,
            one_link.gamma_matrices,
            one_link.gamma5,
            one_link.color_components,
            one_link.representation_id,
            one_link.gauge_field_id,
            one_link.resource_policy,
            operator_id,
        )
        self.one_link = one_link
        self.three_link_links = three_links
        self.one_link_coefficient = first
        self.three_link_coefficient = third
        self.three_link_gauge_field_id = identifier

    def _apply(self, vector: Array, /) -> Array:
        first_forward = self.one_link.transport.forward(self.links, vector)
        first_backward = self.one_link.transport.backward(self.links, vector)
        derivative = jnp.zeros_like(vector)
        for axis in range(self.dimension):
            third_forward = _directional_transport(
                self.one_link.transport,
                self.three_link_links,
                vector,
                axis,
                3,
            )
            third_backward = _directional_transport(
                self.one_link.transport,
                self.three_link_links,
                vector,
                axis,
                -3,
            )
            directional = self.one_link_coefficient * (
                first_forward[:, axis] - first_backward[:, axis]
            ) + self.three_link_coefficient * (third_forward - third_backward)
            derivative = derivative + self.one_link.eta[:, axis, None, None] * directional
        return self.one_link.mass * vector + derivative / (
            2.0 * self.one_link.lattice_spacing
        )

    def _adjoint_apply(self, vector: Array, /) -> Array:
        return self._gamma5_apply(self._apply(self._gamma5_apply(vector)))

    def _gamma5_apply(self, vector: Array, /) -> Array:
        return self.one_link._gamma5_apply(vector)

    def diagonal_mv(self, vector: PyTree[Any], /) -> Array:
        return self.one_link.diagonal_mv(vector)

    def diagonal_inverse_mv(self, vector: PyTree[Any], /) -> Array:
        return self.one_link.diagonal_inverse_mv(vector)

    def with_link_fields(
        self,
        one_link_links: ArrayLike,
        three_link_links: ArrayLike,
        /,
    ) -> "NaikStaggeredDiracOperator":
        return NaikStaggeredDiracOperator(
            self.one_link.with_links(one_link_links),
            three_link_links=three_link_links,
            one_link_coefficient=self.one_link_coefficient,
            three_link_coefficient=self.three_link_coefficient,
            three_link_gauge_field_id=self.three_link_gauge_field_id,
        )

    def with_links(self, links: ArrayLike, /) -> "NaikStaggeredDiracOperator":
        return NaikStaggeredDiracOperator(
            self.one_link.with_links(links),
            three_link_links=links,
            one_link_coefficient=self.one_link_coefficient,
            three_link_coefficient=self.three_link_coefficient,
            three_link_gauge_field_id=self.three_link_gauge_field_id,
        )


class DomainWallDiracOperator(AbstractLinearOperator):
    """Finite Shamir domain-wall operator with explicit fifth-axis closure."""

    kernel: AbstractLatticeDiracOperator
    source: ArraySpace
    target: ArraySpace
    fifth_extent: int = eqx.field(static=True)
    domain_wall_height: float = eqx.field(static=True)
    fermion_mass: float = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)
    representation_id: str = eqx.field(static=True)
    gauge_field_id: str = eqx.field(static=True)
    resources: LatticeFermionResourceEvidence
    resource_policy: LatticeFermionResourcePolicy = eqx.field(static=True)

    def __init__(
        self,
        kernel: AbstractLatticeDiracOperator,
        /,
        *,
        fifth_extent: int,
        domain_wall_height: float,
        fermion_mass: float,
        resources: LatticeFermionResourcePolicy | None = None,
    ):
        if not isinstance(kernel, AbstractLatticeDiracOperator):
            raise TypeError("kernel must be AbstractLatticeDiracOperator.")
        extent = int(fifth_extent)
        height = _finite_real(domain_wall_height, "domain_wall_height")
        mass = _finite_real(fermion_mass, "fermion_mass")
        if not 0.0 <= mass <= 1.0:
            raise ValueError("fermion_mass must lie in [0, 1].")
        policy = LatticeFermionResourcePolicy() if resources is None else resources
        if not isinstance(policy, LatticeFermionResourcePolicy):
            raise TypeError("resources must be LatticeFermionResourcePolicy or None.")
        evidence = LatticeFermionResourceEvidence(
            kernel.site_count,
            kernel.spin_components,
            kernel.color_components,
            kernel.source.dtype,
            policy,
            fifth_extent=extent,
        )
        shape = (extent,) + kernel.source.shape
        space = ArraySpace(
            shape,
            dtype=kernel.source.dtype,
            space_id=canonical_fingerprint(
                {
                    "kind": "domain-wall-spinor-space",
                    "kernel_space": kernel.source.space_id,
                    "fifth_extent": extent,
                    "axis_order": ["fifth", "site", "spin", "color"],
                }
            ),
        )
        self.kernel = kernel
        self.source = space
        self.target = space
        self.fifth_extent = extent
        self.domain_wall_height = height
        self.fermion_mass = mass
        self.properties = _operator_properties()
        self.capabilities = _operator_capabilities()
        self.batch_shape = ()
        self.boundary_id = kernel.boundary_id
        self.representation_id = kernel.representation_id
        self.gauge_field_id = kernel.gauge_field_id
        self.resources = evidence
        self.resource_policy = policy
        self.operator_id = canonical_fingerprint(
            {
                "kind": "finite-domain-wall-dirac-operator",
                "kernel": kernel.operator_id,
                "fifth_extent": extent,
                "domain_wall_height": height,
                "fermion_mass": mass,
                "fifth_boundary": "mass-coupled-open-wall",
            }
        )

    def _project_plus(self, value: Array, /) -> Array:
        return 0.5 * (value + self.kernel.gamma5_mv(value))

    def _project_minus(self, value: Array, /) -> Array:
        return 0.5 * (value - self.kernel.gamma5_mv(value))

    def _apply(self, vector: Array, /) -> Array:
        diagonal = (
            jax.vmap(self.kernel.mv)(vector) + (1.0 - self.domain_wall_height) * vector
        )
        forward = jnp.roll(vector, -1, axis=0).at[-1].set(-self.fermion_mass * vector[0])
        backward = jnp.roll(vector, 1, axis=0).at[0].set(-self.fermion_mass * vector[-1])
        return (
            diagonal
            - jax.vmap(self._project_minus)(forward)
            - jax.vmap(self._project_plus)(backward)
        )

    def mv(self, vector: PyTree[Any], /) -> Array:
        return self.target.validate(self._apply(self.source.validate(vector)))

    def transpose_mv(self, vector: PyTree[Any], /) -> Array:
        return _transpose_action(self._apply, self.source, self.target, vector)

    def adjoint_mv(self, vector: PyTree[Any], /) -> Array:
        return _adjoint_action(self._apply, self.source, self.target, vector)

    def gamma5_reflection_mv(self, vector: PyTree[Any], /) -> Array:
        value = self.source.validate(vector)
        return jax.vmap(self.kernel.gamma5_mv)(value[::-1])

    def with_links(self, links: ArrayLike, /) -> "DomainWallDiracOperator":
        return DomainWallDiracOperator(
            self.kernel.with_links(links),
            fifth_extent=self.fifth_extent,
            domain_wall_height=self.domain_wall_height,
            fermion_mass=self.fermion_mass,
            resources=self.resource_policy,
        )

    def _materialize(self, /) -> Array:
        return _matrix_free_materialization()


class MobiusDomainWallDiracOperator(AbstractLinearOperator):
    """Finite Möbius domain-wall operator with independent b and c kernels."""

    kernel: AbstractLatticeDiracOperator
    source: ArraySpace
    target: ArraySpace
    fifth_extent: int = eqx.field(static=True)
    domain_wall_height: float = eqx.field(static=True)
    fermion_mass: float = eqx.field(static=True)
    b_coefficient: float = eqx.field(static=True)
    c_coefficient: float = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)
    representation_id: str = eqx.field(static=True)
    gauge_field_id: str = eqx.field(static=True)
    resources: LatticeFermionResourceEvidence
    resource_policy: LatticeFermionResourcePolicy = eqx.field(static=True)

    def __init__(
        self,
        kernel: AbstractLatticeDiracOperator,
        /,
        *,
        fifth_extent: int,
        domain_wall_height: float,
        fermion_mass: float,
        b_coefficient: float,
        c_coefficient: float,
        resources: LatticeFermionResourcePolicy | None = None,
    ):
        if not isinstance(kernel, AbstractLatticeDiracOperator):
            raise TypeError("kernel must be AbstractLatticeDiracOperator.")
        extent = int(fifth_extent)
        height = _finite_real(domain_wall_height, "domain_wall_height")
        mass = _finite_real(fermion_mass, "fermion_mass")
        b = _finite_real(b_coefficient, "b_coefficient")
        c = _finite_real(c_coefficient, "c_coefficient")
        if not 0.0 <= mass <= 1.0:
            raise ValueError("fermion_mass must lie in [0, 1].")
        if b + c <= 0.0:
            raise ValueError("Möbius coefficients must satisfy b + c > 0.")
        policy = LatticeFermionResourcePolicy() if resources is None else resources
        if not isinstance(policy, LatticeFermionResourcePolicy):
            raise TypeError("resources must be LatticeFermionResourcePolicy or None.")
        evidence = LatticeFermionResourceEvidence(
            kernel.site_count,
            kernel.spin_components,
            kernel.color_components,
            kernel.source.dtype,
            policy,
            fifth_extent=extent,
        )
        shape = (extent,) + kernel.source.shape
        space = ArraySpace(
            shape,
            dtype=kernel.source.dtype,
            space_id=canonical_fingerprint(
                {
                    "kind": "mobius-domain-wall-spinor-space",
                    "kernel_space": kernel.source.space_id,
                    "fifth_extent": extent,
                }
            ),
        )
        self.kernel = kernel
        self.source = space
        self.target = space
        self.fifth_extent = extent
        self.domain_wall_height = height
        self.fermion_mass = mass
        self.b_coefficient = b
        self.c_coefficient = c
        self.properties = _operator_properties()
        self.capabilities = _operator_capabilities()
        self.batch_shape = ()
        self.boundary_id = kernel.boundary_id
        self.representation_id = kernel.representation_id
        self.gauge_field_id = kernel.gauge_field_id
        self.resources = evidence
        self.resource_policy = policy
        self.operator_id = canonical_fingerprint(
            {
                "kind": "mobius-domain-wall-dirac-operator",
                "kernel": kernel.operator_id,
                "fifth_extent": extent,
                "domain_wall_height": height,
                "fermion_mass": mass,
                "b": b,
                "c": c,
            }
        )

    def _d_minus(self, value: Array, /) -> Array:
        return (
            self.c_coefficient * (self.kernel.mv(value) - self.domain_wall_height * value)
            - value
        )

    def _apply(self, vector: Array, /) -> Array:
        kernel = jax.vmap(self.kernel.mv)(vector) - self.domain_wall_height * vector
        diagonal = self.b_coefficient * kernel + vector
        forward = jnp.roll(vector, -1, axis=0).at[-1].set(-self.fermion_mass * vector[0])
        backward = jnp.roll(vector, 1, axis=0).at[0].set(-self.fermion_mass * vector[-1])
        projected_forward = 0.5 * (forward - jax.vmap(self.kernel.gamma5_mv)(forward))
        projected_backward = 0.5 * (backward + jax.vmap(self.kernel.gamma5_mv)(backward))
        return diagonal + jax.vmap(self._d_minus)(projected_forward + projected_backward)

    def mv(self, vector: PyTree[Any], /) -> Array:
        return self.target.validate(self._apply(self.source.validate(vector)))

    def transpose_mv(self, vector: PyTree[Any], /) -> Array:
        return _transpose_action(self._apply, self.source, self.target, vector)

    def adjoint_mv(self, vector: PyTree[Any], /) -> Array:
        return _adjoint_action(self._apply, self.source, self.target, vector)

    def gamma5_reflection_mv(self, vector: PyTree[Any], /) -> Array:
        value = self.source.validate(vector)
        return jax.vmap(self.kernel.gamma5_mv)(value[::-1])

    def with_links(self, links: ArrayLike, /) -> "MobiusDomainWallDiracOperator":
        return MobiusDomainWallDiracOperator(
            self.kernel.with_links(links),
            fifth_extent=self.fifth_extent,
            domain_wall_height=self.domain_wall_height,
            fermion_mass=self.fermion_mass,
            b_coefficient=self.b_coefficient,
            c_coefficient=self.c_coefficient,
            resources=self.resource_policy,
        )

    def _materialize(self, /) -> Array:
        return _matrix_free_materialization()


class RationalSignApproximation(StrictModule):
    """Positive-shift inverse-square-root approximation for ``sign(H)``."""

    shifts: Array
    weights: Array
    constant: float = eqx.field(static=True)
    spectral_lower_bound: float = eqx.field(static=True)
    spectral_upper_bound: float = eqx.field(static=True)
    maximum_error: float = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)

    def __init__(
        self,
        shifts: ArrayLike,
        weights: ArrayLike,
        /,
        *,
        constant: float = 0.0,
        spectral_lower_bound: float,
        spectral_upper_bound: float,
        maximum_error: float,
    ):
        shifts_ = np.asarray(shifts, dtype=float)
        weights_ = np.asarray(weights, dtype=float)
        if shifts_.ndim != 1 or shifts_.size < 1 or shifts_.shape != weights_.shape:
            raise ValueError("shifts and weights must be equal nonempty vectors.")
        if np.any(~np.isfinite(shifts_)) or np.any(shifts_ <= 0.0):
            raise ValueError("Rational sign shifts must be finite and positive.")
        if np.any(~np.isfinite(weights_)):
            raise ValueError("Rational sign weights must be finite.")
        constant_ = _finite_real(constant, "constant")
        lower = _finite_real(spectral_lower_bound, "spectral_lower_bound")
        upper = _finite_real(spectral_upper_bound, "spectral_upper_bound")
        error = _finite_real(maximum_error, "maximum_error")
        if lower <= 0.0 or upper <= lower or error < 0.0:
            raise ValueError("Rational sign spectral bounds/error are invalid.")
        self.shifts = jnp.asarray(shifts_)
        self.weights = jnp.asarray(weights_)
        self.constant = constant_
        self.spectral_lower_bound = lower
        self.spectral_upper_bound = upper
        self.maximum_error = error
        self.approximation_id = canonical_fingerprint(
            {
                "kind": "rational-sign-approximation",
                "coefficients": array_tree_fingerprint((shifts_, weights_)),
                "constant": constant_,
                "spectral_lower_bound": lower,
                "spectral_upper_bound": upper,
                "maximum_error": error,
                "convention": "H-times-inverse-square-root-of-H-squared",
            }
        )

    @property
    def pole_count(self) -> int:
        return int(self.shifts.size)

    def inverse_square_root_function(
        self, dtype: Any, /
    ) -> PartialFractionRationalFunction:
        return PartialFractionRationalFunction(
            -self.shifts.astype(dtype),
            -self.weights.astype(dtype),
            polynomial_coefficients=jnp.asarray((self.constant,), dtype=dtype),
            function_id=canonical_fingerprint(
                {
                    "kind": "overlap-inverse-square-root-function",
                    "approximation": self.approximation_id,
                    "dtype": np.dtype(dtype).str,
                }
            ),
        )


class _Gamma5HermitianKernel(AbstractLinearOperator):
    kernel: AbstractLatticeDiracOperator
    boundary_id: str = eqx.field(static=True)
    representation_id: str = eqx.field(static=True)
    gauge_field_id: str = eqx.field(static=True)

    def __init__(self, kernel: AbstractLatticeDiracOperator, /):
        self.kernel = kernel
        self.source = kernel.source
        self.boundary_id = kernel.boundary_id
        self.representation_id = kernel.representation_id
        self.gauge_field_id = kernel.gauge_field_id
        self.target = kernel.target
        self.properties = _operator_properties(self_adjoint=True)
        self.capabilities = _operator_capabilities()
        self.batch_shape = ()
        self.operator_id = canonical_fingerprint(
            {"kind": "gamma5-hermitian-kernel", "kernel": kernel.operator_id}
        )

    def mv(self, vector: PyTree[Any], /) -> Array:
        return self.kernel.gamma5_hermitian_mv(vector)

    def transpose_mv(self, vector: PyTree[Any], /) -> Array:
        return _transpose_action(self.mv, self.source, self.target, vector)

    def adjoint_mv(self, vector: PyTree[Any], /) -> Array:
        return self.mv(vector)

    def _materialize(self, /) -> Array:
        return _matrix_free_materialization()


class _SquaredHermitianKernel(AbstractLinearOperator):
    kernel: _Gamma5HermitianKernel
    boundary_id: str = eqx.field(static=True)
    representation_id: str = eqx.field(static=True)
    gauge_field_id: str = eqx.field(static=True)

    def __init__(self, kernel: _Gamma5HermitianKernel, /):
        self.kernel = kernel
        self.boundary_id = kernel.boundary_id
        self.representation_id = kernel.representation_id
        self.gauge_field_id = kernel.gauge_field_id
        self.source = kernel.source
        self.target = kernel.target
        self.properties = OperatorProperties(
            self_adjoint=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_semidefinite": "construction",
            },
        )
        self.capabilities = _operator_capabilities()
        self.batch_shape = ()
        self.operator_id = canonical_fingerprint(
            {"kind": "squared-hermitian-kernel", "kernel": kernel.operator_id}
        )

    def mv(self, vector: PyTree[Any], /) -> Array:
        return self.kernel.mv(self.kernel.mv(vector))

    def transpose_mv(self, vector: PyTree[Any], /) -> Array:
        return _transpose_action(self.mv, self.source, self.target, vector)

    def adjoint_mv(self, vector: PyTree[Any], /) -> Array:
        return self.mv(vector)

    def _materialize(self, /) -> Array:
        return _matrix_free_materialization()


class OverlapActionEvidence(StrictModule):
    """Per-action rational solve status plus declared spectral cutoff/error."""

    status: Array
    diagnostics: RationalFunctionDiagnostics
    finite: Array
    successful: Array
    spectral_lower_bound: float = eqx.field(static=True)
    spectral_upper_bound: float = eqx.field(static=True)
    maximum_approximation_error: float = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class OverlapActionResult(StrictModule):
    value: Array
    sign_value: Array
    evidence: OverlapActionEvidence


class OverlapDiracOperator(AbstractLatticeDiracOperator):
    """Overlap operator using a planned native shifted-Krylov rational sign action."""

    kernel: AbstractLatticeDiracOperator
    hermitian_kernel: _Gamma5HermitianKernel
    squared_kernel: _SquaredHermitianKernel
    approximation: RationalSignApproximation
    rational_function: PartialFractionRationalFunction
    rational_plan: RationalFunctionPlan = eqx.field(static=True)
    rational_policy: RationalFunctionPolicy = eqx.field(static=True)
    fermion_mass: float = eqx.field(static=True)
    resource_policy: LatticeFermionResourcePolicy = eqx.field(static=True)

    def __init__(
        self,
        kernel: AbstractLatticeDiracOperator,
        approximation: RationalSignApproximation,
        /,
        *,
        fermion_mass: float = 0.0,
        rational_policy: RationalFunctionPolicy | None = None,
        resources: LatticeFermionResourcePolicy | None = None,
    ):
        if not isinstance(kernel, AbstractLatticeDiracOperator):
            raise TypeError("kernel must be AbstractLatticeDiracOperator.")
        if not isinstance(approximation, RationalSignApproximation):
            raise TypeError("approximation must be RationalSignApproximation.")
        policy = LatticeFermionResourcePolicy() if resources is None else resources
        if not isinstance(policy, LatticeFermionResourcePolicy):
            raise TypeError("resources must be LatticeFermionResourcePolicy or None.")
        if approximation.pole_count > policy.maximum_rational_poles:
            raise ValueError("Rational sign pole count exceeds the resource policy.")
        selected = (
            RationalFunctionPolicy() if rational_policy is None else rational_policy
        )
        if not isinstance(selected, RationalFunctionPolicy):
            raise TypeError("rational_policy must be RationalFunctionPolicy or None.")
        mass = _finite_real(fermion_mass, "fermion_mass")
        if not -1.0 <= mass <= 1.0:
            raise ValueError("fermion_mass must lie in [-1, 1].")
        hermitian = _Gamma5HermitianKernel(kernel)
        squared = _SquaredHermitianKernel(hermitian)
        function = approximation.inverse_square_root_function(kernel.source.dtype)
        rational_plan = plan_rational_function_action(squared, function, selected)
        operator_id = canonical_fingerprint(
            {
                "kind": "overlap-dirac-operator",
                "kernel": kernel.operator_id,
                "approximation": approximation.approximation_id,
                "rational_plan": rational_plan.plan_id,
                "fermion_mass": mass,
                "normalization": "half-one-plus-mass",
            }
        )
        _initialize_lattice_operator(
            self,
            kernel.boundary,
            kernel.links,
            kernel.gamma_matrices,
            kernel.gamma5,
            kernel.color_components,
            kernel.representation_id,
            kernel.gauge_field_id,
            policy,
            operator_id,
        )
        self.kernel = kernel
        self.hermitian_kernel = hermitian
        self.squared_kernel = squared
        self.approximation = approximation
        self.rational_function = function
        self.rational_plan = rational_plan
        self.rational_policy = selected
        self.fermion_mass = mass
        self.resource_policy = policy

    def sign_action(self, vector: PyTree[Any], /) -> tuple[Array, OverlapActionEvidence]:
        value = self.source.validate(vector)
        prepared = prepare_rational_function_action(
            self.squared_kernel,
            value,
            self.rational_function,
            self.rational_plan,
        )
        rational = rational_function_action(prepared)
        sign_value = self.hermitian_kernel.mv(rational.value)
        finite = jnp.all(jnp.isfinite(sign_value))
        successful = rational.successful & finite
        evidence = OverlapActionEvidence(
            status=rational.status,
            diagnostics=rational.diagnostics,
            finite=finite,
            successful=successful,
            spectral_lower_bound=self.approximation.spectral_lower_bound,
            spectral_upper_bound=self.approximation.spectral_upper_bound,
            maximum_approximation_error=self.approximation.maximum_error,
            approximation_id=self.approximation.approximation_id,
            plan_id=self.rational_plan.plan_id,
        )
        return sign_value, evidence

    def action_with_evidence(self, vector: PyTree[Any], /) -> OverlapActionResult:
        value = self.source.validate(vector)
        sign_value, evidence = self.sign_action(value)
        left = 0.5 * (1.0 + self.fermion_mass)
        right = 0.5 * (1.0 - self.fermion_mass)
        result = left * value + right * self.gamma5_mv(sign_value)
        return OverlapActionResult(value=result, sign_value=sign_value, evidence=evidence)

    def _apply(self, vector: Array, /) -> Array:
        return self.action_with_evidence(vector).value

    def _adjoint_apply(self, vector: Array, /) -> Array:
        return self._gamma5_apply(self._apply(self._gamma5_apply(vector)))

    def _gamma5_apply(self, vector: Array, /) -> Array:
        return self.kernel.gamma5_mv(vector)

    def diagonal_mv(self, vector: PyTree[Any], /) -> Array:
        raise LinearCapabilityError(
            "Overlap operators have no site-local diagonal split."
        )

    def diagonal_inverse_mv(self, vector: PyTree[Any], /) -> Array:
        raise LinearCapabilityError(
            "Overlap operators have no site-local diagonal inverse."
        )

    def with_links(self, links: ArrayLike, /) -> "OverlapDiracOperator":
        return OverlapDiracOperator(
            self.kernel.with_links(links),
            self.approximation,
            fermion_mass=self.fermion_mass,
            rational_policy=self.rational_policy,
            resources=self.resource_policy,
        )


__all__ = [
    "AbstractLatticeDiracOperator",
    "CloverInverseEvidence",
    "CloverWilsonDiracOperator",
    "DomainWallDiracOperator",
    "EvenOddSchurOperator",
    "EvenOddSchurPlan",
    "FreeWilsonDiracOperator",
    "LatticeFermionResourceEvidence",
    "LatticeFermionResourcePolicy",
    "MobiusDomainWallDiracOperator",
    "NaikStaggeredDiracOperator",
    "OverlapActionEvidence",
    "OverlapActionResult",
    "OverlapDiracOperator",
    "PreparedEvenOddSchur",
    "RationalSignApproximation",
    "StaggeredDiracOperator",
    "TwistedMassDoubletOperator",
    "WilsonDiracOperator",
    "canonical_euclidean_gamma_matrices",
    "canonical_site_spinor_space",
    "plan_even_odd_schur",
    "prepare_even_odd_schur",
]
