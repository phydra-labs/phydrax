#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Matrix-free geometric Kähler–Dirac fermions for regulated 2D twisted SYM."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein
from phydrax.linalg import (
    ArraySpace,
    LinearCapabilityError,
    OperatorCapabilities,
    OperatorProperties,
)
from phydrax.operators.path_integral import AbstractPseudofermionDiracOperator

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ._actions import TwistedSYMConfiguration
from ._twisted_n2 import TwistedN2SYMPlan, TwistedSYMCoordinateLayout


def _mul(left: Array, right: Array, /) -> Array:
    return ein.contract("...ij,...jk->...ik", left, right)


def _periodic_shift(value: Array, axis: int, amount: int, /) -> Array:
    return jnp.roll(value, -int(amount), axis=axis)


def _fermion_shift(
    value: Array,
    axis: int,
    amount: int,
    /,
    *,
    temporal_axis: int,
    boundary_phase: complex,
) -> Array:
    shifted = _periodic_shift(value, axis, amount)
    if axis != temporal_axis or amount == 0:
        return shifted
    extent = value.shape[axis]
    phase = boundary_phase if amount > 0 else np.conj(boundary_phase)
    selector = -1 if amount > 0 else 0
    factors = jnp.ones((extent,), dtype=value.dtype).at[selector].set(phase)
    shape = [1] * value.ndim
    shape[axis] = extent
    return shifted * factors.reshape(tuple(shape))


class TwistedKahlerDiracOperator(AbstractPseudofermionDiracOperator):
    """Antisymmetric η/ψ₀/ψ₁/χ₀₁ geometric fermion operator."""

    theory: TwistedN2SYMPlan = eqx.field(static=True)
    layout: TwistedSYMCoordinateLayout
    links: Array
    configuration: TwistedSYMConfiguration
    fermion_space: ArraySpace
    component_labels: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        theory: TwistedN2SYMPlan,
        layout: TwistedSYMCoordinateLayout,
        links: ArrayLike,
        /,
    ):
        if not isinstance(theory, TwistedN2SYMPlan):
            raise TypeError("theory must be TwistedN2SYMPlan.")
        if not isinstance(layout, TwistedSYMCoordinateLayout):
            raise TypeError("layout must be TwistedSYMCoordinateLayout.")
        if layout.prepared.plan.plan_id != theory.bosonic_plan.plan_id:
            raise ValueError("Fermion coordinate layout does not match the theory plan.")
        coordinates = jnp.asarray(links)
        configuration = layout.unpack(coordinates)
        field_shape = theory.bosonic_plan.lattice_shape + (
            4,
            theory.bosonic_plan.matrix_rank,
            theory.bosonic_plan.matrix_rank,
        )
        space = ArraySpace(
            field_shape,
            dtype=np.complex128,
            space_id=f"twisted-n2-kahler-dirac:{theory.plan_id}",
        )
        self.theory = theory
        self.layout = layout
        self.links = coordinates
        self.configuration = configuration
        self.component_labels = ("eta", "psi-0", "psi-1", "chi-01")
        self.fermion_space = space
        self.source = space
        self.target = space
        self.properties = OperatorProperties()
        self.capabilities = OperatorCapabilities(
            transpose=True,
            adjoint=True,
            materialize=False,
        )
        self.batch_shape = ()
        self.operator_id = canonical_fingerprint(
            {
                "kind": "twisted-n2-kahler-dirac-operator",
                "theory": theory.plan_id,
                "layout": layout.layout_id,
                "component_labels": self.component_labels,
                "boundary_phase": (
                    theory.fermion_boundary_phase.real,
                    theory.fermion_boundary_phase.imag,
                ),
                "convention": "eta-divergence-plus-chi-covariant-curl",
            }
        )

    def with_links(self, links: ArrayLike, /) -> TwistedKahlerDiracOperator:
        return TwistedKahlerDiracOperator(self.theory, self.layout, links)

    def _forward_fermion_shift(self, value: Array, axis: int, /) -> Array:
        return _fermion_shift(
            value,
            axis,
            1,
            temporal_axis=self.theory.temporal_axis,
            boundary_phase=self.theory.fermion_boundary_phase,
        )

    def _backward_fermion_shift(self, value: Array, axis: int, /) -> Array:
        return _fermion_shift(
            value,
            axis,
            -1,
            temporal_axis=self.theory.temporal_axis,
            boundary_phase=self.theory.fermion_boundary_phase,
        )

    def _divergence_component(self, value: Array, axis: int, /) -> Array:
        reverse = self.configuration.reverse_links.values[..., axis, :, :]
        local = _mul(value, reverse)
        backward_reverse = _periodic_shift(reverse, axis, -1)
        backward_value = self._backward_fermion_shift(value, axis)
        return (
            local - _mul(backward_reverse, backward_value)
        ) / self.theory.bosonic_plan.lattice_spacing

    def _curl_component(
        self, value: Array, derivative_axis: int, form_axis: int, /
    ) -> Array:
        forward = self.configuration.links.values
        derivative_link = forward[..., derivative_axis, :, :]
        shifted_value = self._forward_fermion_shift(value, derivative_axis)
        endpoint_link = _periodic_shift(derivative_link, form_axis, 1)
        return (
            _mul(derivative_link, shifted_value) - _mul(value, endpoint_link)
        ) / self.theory.bosonic_plan.lattice_spacing

    @staticmethod
    def _transpose_action(function, template: Array, cotangent: Array, /) -> Array:
        return jax.linear_transpose(function, template)(cotangent)[0]

    def _apply(self, vector: Array, /) -> Array:
        eta = vector[..., 0, :, :]
        psi_0 = vector[..., 1, :, :]
        psi_1 = vector[..., 2, :, :]
        chi = vector[..., 3, :, :]
        zero = jnp.zeros_like(eta)

        divergence_0 = lambda value: self._divergence_component(value, 0)
        divergence_1 = lambda value: self._divergence_component(value, 1)
        curl_0 = lambda value: -self._curl_component(value, 1, 0)
        curl_1 = lambda value: self._curl_component(value, 0, 1)

        output_eta = divergence_0(psi_0) + divergence_1(psi_1)
        output_psi_0 = -self._transpose_action(divergence_0, zero, eta)
        output_psi_1 = -self._transpose_action(divergence_1, zero, eta)
        output_chi = curl_0(psi_0) + curl_1(psi_1)
        output_psi_0 = output_psi_0 - self._transpose_action(curl_0, zero, chi)
        output_psi_1 = output_psi_1 - self._transpose_action(curl_1, zero, chi)
        return jnp.stack((output_eta, output_psi_0, output_psi_1, output_chi), axis=-3)

    def mv(self, vector: ArrayLike, /) -> Array:
        return self.target.validate(self._apply(self.source.validate(vector)))

    def transpose_mv(self, vector: ArrayLike, /) -> Array:
        value = self.target.validate(vector)
        zero = jnp.zeros(self.fermion_space.shape, dtype=self.fermion_space.dtype)
        return self.source.validate(jax.linear_transpose(self._apply, zero)(value)[0])

    def adjoint_mv(self, vector: ArrayLike, /) -> Array:
        value = self.target.validate(vector)
        return self.source.validate(jnp.conj(self.transpose_mv(jnp.conj(value))))

    def _materialize(self, /) -> Array:
        raise LinearCapabilityError(
            "TwistedKahlerDiracOperator forbids production dense materialization."
        )


class RegulatedTwistedDiracOperator(AbstractPseudofermionDiracOperator):
    """Rectangular operator whose normal form is ``M†M + mu² I``."""

    kahler_dirac: TwistedKahlerDiracOperator
    links: Array
    regulator_mass: float = eqx.field(static=True)

    def __init__(
        self,
        kahler_dirac: TwistedKahlerDiracOperator,
        regulator_mass: float,
        /,
    ):
        if not isinstance(kahler_dirac, TwistedKahlerDiracOperator):
            raise TypeError("kahler_dirac must be TwistedKahlerDiracOperator.")
        mass = float(regulator_mass)
        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError("regulator_mass must be positive and finite.")
        target = ArraySpace(
            (2,) + kahler_dirac.fermion_space.shape,
            dtype=kahler_dirac.fermion_space.dtype,
            space_id=f"regulated-twisted-dirac-target:{kahler_dirac.operator_id}",
        )
        self.kahler_dirac = kahler_dirac
        self.links = kahler_dirac.links
        self.regulator_mass = mass
        self.source = kahler_dirac.source
        self.target = target
        self.properties = OperatorProperties()
        self.capabilities = OperatorCapabilities(
            transpose=True,
            adjoint=True,
            materialize=False,
        )
        self.batch_shape = ()
        self.operator_id = canonical_fingerprint(
            {
                "kind": "regulated-twisted-n2-dirac-operator",
                "kahler_dirac": kahler_dirac.operator_id,
                "regulator_mass": mass,
                "normal_form": "M-adjoint-M-plus-mu-squared-identity",
            }
        )

    def with_links(self, links: ArrayLike, /) -> RegulatedTwistedDiracOperator:
        return RegulatedTwistedDiracOperator(
            self.kahler_dirac.with_links(links), self.regulator_mass
        )

    def mv(self, vector: ArrayLike, /) -> Array:
        value = self.source.validate(vector)
        return self.target.validate(
            jnp.stack((self.kahler_dirac.mv(value), self.regulator_mass * value))
        )

    def transpose_mv(self, vector: ArrayLike, /) -> Array:
        value = self.target.validate(vector)
        return self.source.validate(
            self.kahler_dirac.transpose_mv(value[0]) + self.regulator_mass * value[1]
        )

    def adjoint_mv(self, vector: ArrayLike, /) -> Array:
        value = self.target.validate(vector)
        return self.source.validate(
            self.kahler_dirac.adjoint_mv(value[0]) + self.regulator_mass * value[1]
        )

    def _materialize(self, /) -> Array:
        raise LinearCapabilityError(
            "RegulatedTwistedDiracOperator forbids production dense materialization."
        )


class TwistedFermionAlgebraEvidence(StrictModule):
    antisymmetry_residual: Array
    adjoint_residual: Array
    normal_minimum_eigenvalue: Array
    normal_maximum_eigenvalue: Array
    structural_lower_bound: Array
    structural_upper_bound: Array
    interval_contains_spectrum: Array
    finite: Array
    accepted: Array
    operator_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def materialize_twisted_fermion_reference(
    operator: TwistedKahlerDiracOperator,
    /,
    *,
    maximum_elements: int,
) -> Array:
    """Guarded dense matrix for tiny independent algebra/Pfaffian controls."""
    if not isinstance(operator, TwistedKahlerDiracOperator):
        raise TypeError("operator must be TwistedKahlerDiracOperator.")
    dimension = operator.source.size
    maximum = int(maximum_elements)
    if maximum < 1 or dimension * dimension > maximum:
        raise ValueError("Dense twisted-fermion reference exceeds maximum_elements.")
    identity = jnp.eye(dimension, dtype=operator.fermion_space.dtype).reshape(
        (dimension,) + operator.fermion_space.shape
    )
    columns = tuple(
        operator.mv(identity[index]).reshape((-1,)) for index in range(dimension)
    )
    return jnp.stack(columns, axis=1)


def assess_twisted_fermion_algebra(
    operator: TwistedKahlerDiracOperator,
    /,
    *,
    regulator_mass: float,
    structural_lower_bound: float,
    structural_upper_bound: float,
    maximum_dense_elements: int,
    tolerance: float = 1e-9,
) -> TwistedFermionAlgebraEvidence:
    if not isinstance(operator, TwistedKahlerDiracOperator):
        raise TypeError("operator must be TwistedKahlerDiracOperator.")
    matrix = materialize_twisted_fermion_reference(
        operator, maximum_elements=maximum_dense_elements
    )
    scale = jnp.maximum(1.0, jnp.linalg.norm(matrix))
    antisymmetry = jnp.linalg.norm(matrix + jnp.swapaxes(matrix, 0, 1)) / scale
    adjoint_columns = tuple(
        operator.adjoint_mv(
            jnp.eye(
                operator.fermion_space.size,
                dtype=operator.fermion_space.dtype,
            )[index].reshape(operator.fermion_space.shape)
        ).reshape((-1,))
        for index in range(operator.fermion_space.size)
    )
    adjoint = jnp.stack(adjoint_columns, axis=1)
    adjoint_residual = jnp.linalg.norm(adjoint - jnp.conj(matrix.T)) / scale
    normal = jnp.conj(matrix.T) @ matrix + float(regulator_mass) ** 2 * jnp.eye(
        matrix.shape[1], dtype=matrix.dtype
    )
    eigenvalues = jnp.linalg.eigvalsh(normal)
    minimum = jnp.min(eigenvalues)
    maximum = jnp.max(eigenvalues)
    lower = jnp.asarray(structural_lower_bound, dtype=minimum.dtype)
    upper = jnp.asarray(structural_upper_bound, dtype=maximum.dtype)
    interval = (minimum >= lower - tolerance) & (maximum <= upper + tolerance)
    finite = (
        jnp.isfinite(antisymmetry)
        & jnp.isfinite(adjoint_residual)
        & jnp.all(jnp.isfinite(eigenvalues))
    )
    accepted = (
        finite & (antisymmetry <= tolerance) & (adjoint_residual <= tolerance) & interval
    )
    return TwistedFermionAlgebraEvidence(
        antisymmetry_residual=antisymmetry,
        adjoint_residual=adjoint_residual,
        normal_minimum_eigenvalue=minimum,
        normal_maximum_eigenvalue=maximum,
        structural_lower_bound=lower,
        structural_upper_bound=upper,
        interval_contains_spectrum=interval,
        finite=finite,
        accepted=accepted,
        operator_id=operator.operator_id,
        claim="tiny-dense-regulated-kahler-dirac-reference-only",
    )


__all__ = [
    "RegulatedTwistedDiracOperator",
    "TwistedFermionAlgebraEvidence",
    "TwistedKahlerDiracOperator",
    "assess_twisted_fermion_algebra",
    "materialize_twisted_fermion_reference",
]
