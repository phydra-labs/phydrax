#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._holomorphic import ComplexAffineNormalization, HolomorphicJet
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...continuation import ParameterContinuationProblem
from ...optim import variable_projection, VariableProjectionProblem
from ._holomorphic_constraints import HolomorphicAffineCoefficientMap
from ._holomorphic_frame import HolomorphicPolynomialFrame


class PoleSet(StrictModule, NonTrainableState):
    """Fixed scalar complex poles with positive static orders."""

    locations: Array
    orders: tuple[int, ...] = eqx.field(static=True)
    pole_set_id: str = eqx.field(static=True)

    def __init__(self, locations: ArrayLike, orders: Sequence[int], /):
        locations_raw = np.asarray(locations, dtype=np.complex128)
        orders_ = tuple(int(value) for value in orders)
        if locations_raw.ndim != 1 or locations_raw.size == 0:
            raise ValueError("Pole locations must be one nonempty vector.")
        if len(orders_) != locations_raw.size or any(value <= 0 for value in orders_):
            raise ValueError("Pole orders must be positive and match pole locations.")
        if not np.all(np.isfinite(locations_raw)):
            raise ValueError("Pole locations must be finite.")
        if len(set(complex(value) for value in locations_raw)) != locations_raw.size:
            raise ValueError("Fixed pole locations must be distinct.")
        locations_ = jnp.asarray(locations_raw)
        self.locations = locations_
        self.orders = orders_
        self.pole_set_id = canonical_fingerprint(
            {
                "kind": "meromorphic-pole-set",
                "locations": array_tree_fingerprint(locations_),
                "orders": list(orders_),
            }
        )

    @property
    def term_count(self) -> int:
        return sum(self.orders)


class TrainablePoleSet(StrictModule):
    """Real Cartesian trainable pole locations with fixed positive orders."""

    location_real: Array
    location_imag: Array
    orders: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, locations: ArrayLike, orders: Sequence[int], /):
        fixed = PoleSet(locations, orders)
        self.location_real = jnp.real(fixed.locations)
        self.location_imag = jnp.imag(fixed.locations)
        self.orders = fixed.orders

    @property
    def locations(self) -> Array:
        return self.location_real + 1j * self.location_imag

    def fixed(self) -> PoleSet:
        return PoleSet(self.locations, self.orders)


class MeromorphicLinearFrameCertificate(StrictModule, NonTrainableState):
    """Finite real-coordinate frame with explicit pole-set analyticity boundary."""

    complex_input_size: int = eqx.field(static=True)
    complex_output_size: int = eqx.field(static=True)
    real_coefficient_count: int = eqx.field(static=True)
    maximum_derivative_order: int = eqx.field(static=True)
    normalization_id: str = eqx.field(static=True)
    basis_construction: str = eqx.field(static=True)
    coefficient_mode: str = eqx.field(static=True)
    pole_set_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        complex_output_size: int,
        real_coefficient_count: int,
        maximum_derivative_order: int,
        normalization_id: str,
        pole_set_id: str,
    ):
        output = int(complex_output_size)
        coefficient_count = int(real_coefficient_count)
        derivative = int(maximum_derivative_order)
        if min(output, coefficient_count) <= 0 or derivative < 0:
            raise ValueError("Meromorphic frame dimensions are invalid.")
        self.complex_input_size = 1
        self.complex_output_size = output
        self.real_coefficient_count = coefficient_count
        self.maximum_derivative_order = derivative
        self.normalization_id = str(normalization_id)
        self.basis_construction = "polynomial-plus-fixed-principal-parts"
        self.coefficient_mode = "real-cartesian-meromorphic-frame"
        self.pole_set_id = str(pole_set_id)
        self.frame_id = canonical_fingerprint(
            {
                "kind": "meromorphic-linear-frame-certificate",
                "complex_output_size": output,
                "real_coefficient_count": coefficient_count,
                "maximum_derivative_order": derivative,
                "normalization_id": str(normalization_id),
                "pole_set_id": str(pole_set_id),
            }
        )


class MeromorphicMapCertificate(StrictModule, NonTrainableState):
    """Construction evidence for a finite meromorphic potential family."""

    complex_output_size: int = eqx.field(static=True)
    maximum_derivative_order: int = eqx.field(static=True)
    pole_set_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    parameter_mode: str = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def __init__(
        self,
        frame: MeromorphicLinearFrameCertificate,
        /,
        *,
        parameter_mode: str,
        construction_dependency: str,
    ):
        mode = str(parameter_mode)
        dependency = str(construction_dependency)
        if not mode or not dependency:
            raise ValueError("Meromorphic certificate identifiers must be nonempty.")
        self.complex_output_size = frame.complex_output_size
        self.maximum_derivative_order = frame.maximum_derivative_order
        self.pole_set_id = frame.pole_set_id
        self.frame_id = frame.frame_id
        self.parameter_mode = mode
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "meromorphic-map-certificate",
                "complex_output_size": frame.complex_output_size,
                "maximum_derivative_order": frame.maximum_derivative_order,
                "pole_set_id": frame.pole_set_id,
                "frame_id": frame.frame_id,
                "parameter_mode": mode,
                "construction_dependency": dependency,
            }
        )


class PoleClearanceReport(StrictModule, NonTrainableState):
    """Pole separation from one closed physical disk and from other poles."""

    valid: Array
    minimum_domain_clearance: Array
    minimum_pair_separation: Array
    required_clearance: Array
    domain_id: str = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        poles: PoleSet,
        /,
        *,
        center: complex,
        radius: float,
        required_clearance: float = 0.0,
    ):
        if not isinstance(poles, PoleSet):
            raise TypeError("poles must be PoleSet.")
        center_ = complex(center)
        radius_ = float(radius)
        clearance_ = float(required_clearance)
        if (
            not math.isfinite(abs(center_))
            or not math.isfinite(radius_)
            or radius_ <= 0.0
            or not math.isfinite(clearance_)
            or clearance_ < 0.0
        ):
            raise ValueError("Disk pole-clearance geometry is invalid.")
        distance = jnp.abs(poles.locations - center_) - radius_
        minimum_domain = jnp.min(distance)
        if poles.locations.size == 1:
            minimum_pair = jnp.asarray(jnp.inf, dtype=distance.dtype)
        else:
            differences = jnp.abs(poles.locations[:, None] - poles.locations[None, :])
            differences = jnp.where(
                jnp.eye(poles.locations.size, dtype=bool),
                jnp.asarray(jnp.inf, dtype=differences.dtype),
                differences,
            )
            minimum_pair = jnp.min(differences)
        required = jnp.asarray(clearance_, dtype=distance.dtype)
        valid = (
            jnp.isfinite(minimum_domain)
            & (minimum_domain >= required)
            & (minimum_pair > required)
        )
        domain_id = canonical_fingerprint(
            {
                "kind": "closed-complex-disk-domain",
                "center": array_tree_fingerprint(jnp.asarray(center_)),
                "radius": radius_,
            }
        )
        self.valid = valid
        self.minimum_domain_clearance = minimum_domain
        self.minimum_pair_separation = minimum_pair
        self.required_clearance = required
        self.domain_id = domain_id
        self.report_id = canonical_fingerprint(
            {
                "kind": "pole-clearance-report",
                "pole_set": poles.pole_set_id,
                "domain": domain_id,
                "required_clearance": clearance_,
            }
        )


class DomainHolomorphicCertificate(StrictModule, NonTrainableState):
    """Evidence that a meromorphic construction has no poles on one domain."""

    meromorphic_certificate_id: str = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)
    clearance_report_id: str = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def __init__(
        self,
        meromorphic: MeromorphicMapCertificate,
        clearance: PoleClearanceReport,
        /,
    ):
        if not isinstance(meromorphic, MeromorphicMapCertificate):
            raise TypeError("meromorphic must be MeromorphicMapCertificate.")
        if not isinstance(clearance, PoleClearanceReport):
            raise TypeError("clearance must be PoleClearanceReport.")
        if not bool(clearance.valid):
            raise ValueError("Pole clearance is insufficient for domain holomorphy.")
        self.meromorphic_certificate_id = meromorphic.certificate_id
        self.domain_id = clearance.domain_id
        self.clearance_report_id = clearance.report_id
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "domain-holomorphic-certificate",
                "meromorphic": meromorphic.certificate_id,
                "domain": clearance.domain_id,
                "clearance": clearance.report_id,
            }
        )


def _rising_factorial(start: int, count: int, /) -> int:
    result = 1
    for value in range(count):
        result *= start + value
    return result


class MeromorphicLinearFrame(StrictModule, NonTrainableState):
    """Polynomial regular part plus fixed finite principal parts."""

    poles: PoleSet
    regular_frame: HolomorphicPolynomialFrame
    regular_degree: int = eqx.field(static=True)
    complex_output_size: int = eqx.field(static=True)
    _certificate: MeromorphicLinearFrameCertificate

    def __init__(
        self,
        regular_degree: int,
        poles: PoleSet,
        complex_output_size: int = 1,
        /,
        *,
        normalization: ComplexAffineNormalization | None = None,
        maximum_derivative_order: int = 4,
    ):
        degree = int(regular_degree)
        output = int(complex_output_size)
        derivative = int(maximum_derivative_order)
        if degree < 0 or output <= 0 or derivative < 0:
            raise ValueError("Meromorphic frame dimensions are invalid.")
        if not isinstance(poles, PoleSet):
            raise TypeError("poles must be PoleSet.")
        regular = HolomorphicPolynomialFrame.one_variable(
            degree,
            1,
            normalization=normalization,
        )
        feature_count = degree + 1 + poles.term_count
        self.poles = poles
        self.regular_frame = regular
        self.regular_degree = degree
        self.complex_output_size = output
        self._certificate = MeromorphicLinearFrameCertificate(
            complex_output_size=output,
            real_coefficient_count=2 * output * feature_count,
            maximum_derivative_order=max(derivative, degree),
            normalization_id=regular.normalization.normalization_id,
            pole_set_id=poles.pole_set_id,
        )

    @property
    def feature_count(self) -> int:
        return self.regular_degree + 1 + self.poles.term_count

    def linear_frame_certificate(self) -> MeromorphicLinearFrameCertificate:
        return self._certificate

    def _features(self, coordinate: ArrayLike, order: int, /) -> Array:
        scalar = jnp.asarray(coordinate).reshape(())
        regular_basis = self.regular_frame.basis_derivative(scalar, (order,))[0]
        regular = regular_basis[: self.regular_degree + 1]
        principal = []
        for location, maximum_order in zip(
            self.poles.locations,
            self.poles.orders,
            strict=True,
        ):
            for pole_order in range(1, maximum_order + 1):
                factor = (
                    (-1) ** order
                    * _rising_factorial(pole_order, order)
                    * (scalar - location) ** (-(pole_order + order))
                )
                principal.append(factor)
        return jnp.concatenate((regular, jnp.stack(tuple(principal))))

    def basis_derivative(
        self,
        coordinates: ArrayLike,
        multi_index: Sequence[int],
        /,
    ) -> Array:
        derivative = tuple(int(value) for value in multi_index)
        if len(derivative) != 1 or derivative[0] < 0:
            raise ValueError("Meromorphic frame derivative multi-index is invalid.")
        if derivative[0] > self._certificate.maximum_derivative_order:
            raise ValueError("Meromorphic frame derivative order is unavailable.")
        features = self._features(coordinates, derivative[0])
        result = jnp.zeros(
            (
                self.complex_output_size,
                self._certificate.real_coefficient_count,
            ),
            dtype=features.dtype,
        )
        count = self.feature_count
        for output in range(self.complex_output_size):
            start = 2 * output * count
            result = result.at[output, start : start + count].set(features)
            result = result.at[output, start + count : start + 2 * count].set(
                1j * features
            )
        return result


class ConstrainedMeromorphicPotential(StrictModule):
    """Meromorphic frame parameterized by one prepared affine coefficient map."""

    __hash__ = object.__hash__

    free_coordinates: Array
    coefficient_map: HolomorphicAffineCoefficientMap
    _certificate: MeromorphicMapCertificate

    def __init__(
        self,
        coefficient_map: HolomorphicAffineCoefficientMap,
        /,
        *,
        initial_free_coordinates: ArrayLike | None = None,
    ):
        frame = coefficient_map.operator.plan.frame
        certificate = frame.linear_frame_certificate()
        if not isinstance(certificate, MeromorphicLinearFrameCertificate):
            raise TypeError(
                "Constrained meromorphic potential requires a meromorphic frame."
            )
        free = (
            jnp.zeros(
                (coefficient_map.nullity,),
                dtype=coefficient_map.particular_coefficients.dtype,
            )
            if initial_free_coordinates is None
            else jnp.asarray(initial_free_coordinates)
        )
        if free.shape != (coefficient_map.nullity,) or jnp.iscomplexobj(free):
            raise ValueError("Meromorphic free coordinates must be one real vector.")
        self.free_coordinates = free
        self.coefficient_map = coefficient_map
        self._certificate = MeromorphicMapCertificate(
            certificate,
            parameter_mode="real-cartesian-nullspace",
            construction_dependency=coefficient_map.map_id,
        )

    @property
    def frame(self) -> MeromorphicLinearFrame:
        return self.coefficient_map.operator.plan.frame

    @property
    def coefficient_vector(self) -> Array:
        return self.coefficient_map.coefficient_vector(self.free_coordinates)

    def __call__(self, coordinate: ArrayLike, /) -> Array:
        return self.frame.basis_derivative(coordinate, (0,)) @ self.coefficient_vector

    def jet(self, coordinate: ArrayLike, order: int, /) -> HolomorphicJet:
        order_ = int(order)
        value = self(coordinate)
        derivatives = tuple(
            self.frame.basis_derivative(coordinate, (current,)) @ self.coefficient_vector
            for current in range(1, order_ + 1)
        )
        return HolomorphicJet(value, derivatives)

    def meromorphic_certificate(self) -> MeromorphicMapCertificate:
        return self._certificate

    def certify_on_disk(
        self,
        /,
        *,
        center: complex,
        radius: float,
        required_clearance: float = 0.0,
    ) -> DomainHolomorphicCertificate:
        report = PoleClearanceReport(
            self.frame.poles,
            center=center,
            radius=radius,
            required_clearance=required_clearance,
        )
        return DomainHolomorphicCertificate(self._certificate, report)


class MeromorphicVariableProjectionPlan(StrictModule, NonTrainableState):
    """Reduced fitting problem with nonlinear poles and linear real residues."""

    coordinates: Array
    observations: Array
    regular_degree: int = eqx.field(static=True)
    pole_orders: tuple[int, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        coordinates: ArrayLike,
        observations: ArrayLike,
        regular_degree: int,
        pole_orders: Sequence[int],
        /,
    ):
        coordinates_ = jnp.asarray(coordinates)
        observations_ = jnp.asarray(observations)
        degree = int(regular_degree)
        orders = tuple(int(value) for value in pole_orders)
        if (
            coordinates_.ndim != 1
            or observations_.shape != coordinates_.shape
            or not jnp.iscomplexobj(coordinates_)
            or not jnp.iscomplexobj(observations_)
        ):
            raise ValueError("Meromorphic fitting data must be matching complex vectors.")
        if degree < 0 or not orders or any(value <= 0 for value in orders):
            raise ValueError("Meromorphic fitting frame dimensions are invalid.")
        self.coordinates = coordinates_
        self.observations = observations_
        self.regular_degree = degree
        self.pole_orders = orders
        self.plan_id = canonical_fingerprint(
            {
                "kind": "meromorphic-variable-projection-plan",
                "coordinates": array_tree_fingerprint(coordinates_),
                "observations": array_tree_fingerprint(observations_),
                "regular_degree": degree,
                "pole_orders": list(orders),
            }
        )

    @property
    def nonlinear_size(self) -> int:
        return 2 * len(self.pole_orders)

    def _complex_design(self, parameters: Array, /) -> Array:
        values = jnp.asarray(parameters)
        pole_count = len(self.pole_orders)
        if values.shape != (2 * pole_count,):
            raise ValueError("Pole parameter vector has invalid shape.")
        poles = values[:pole_count] + 1j * values[pole_count:]
        regular = jnp.stack(
            tuple(self.coordinates**power for power in range(self.regular_degree + 1)),
            axis=1,
        )
        principal = []
        for pole, maximum_order in zip(poles, self.pole_orders, strict=True):
            for order in range(1, maximum_order + 1):
                principal.append((self.coordinates - pole) ** (-order))
        return jnp.concatenate((regular, jnp.stack(tuple(principal), axis=1)), axis=1)

    def problem(
        self,
        observations: ArrayLike | None = None,
        /,
    ) -> VariableProjectionProblem:
        observations_complex = (
            self.observations if observations is None else jnp.asarray(observations)
        )
        if observations_complex.shape != self.observations.shape or not jnp.iscomplexobj(
            observations_complex
        ):
            raise ValueError("Meromorphic observations must match the plan data shape.")
        observations_real = jnp.concatenate(
            (jnp.real(observations_complex), jnp.imag(observations_complex))
        )

        def design(parameters, args):
            del args
            complex_design = self._complex_design(parameters)
            real = jnp.real(complex_design)
            imaginary = jnp.imag(complex_design)
            return jnp.block([[real, -imaginary], [imaginary, real]])

        return VariableProjectionProblem(
            design,
            observations_real,
            problem_id=self.plan_id,
        )

    def fit(self, initial_poles: ArrayLike, /, **kwargs):
        """Fit nonlinear pole locations and the optimal linear coefficient block."""
        locations = jnp.asarray(initial_poles)
        pole_count = len(self.pole_orders)
        if locations.shape != (pole_count,) or not jnp.iscomplexobj(locations):
            raise ValueError("initial_poles must be one complex value per pole order.")
        parameters = jnp.concatenate((jnp.real(locations), jnp.imag(locations)))
        return variable_projection(self.problem(), parameters, **kwargs)

    def continuation_problem(
        self,
        final_observations: ArrayLike,
        /,
    ) -> ParameterContinuationProblem:
        """Track stationary reduced pole fits along a linear observation path."""
        final = jnp.asarray(final_observations)
        if final.shape != self.observations.shape or not jnp.iscomplexobj(final):
            raise ValueError("final_observations must match the complex fitting data.")
        initial = self.observations

        def stationarity(parameters, coordinate, args):
            observations = (1.0 - coordinate) * initial + coordinate * final
            problem = self.problem(observations)

            def objective(values):
                residual = problem.linear_solution(values, args)[1]
                return 0.5 * jnp.real(jnp.vdot(residual, residual))

            return jax.grad(objective)(parameters)

        return ParameterContinuationProblem(
            stationarity,
            parameter_lower=0.0,
            parameter_upper=1.0,
            problem_id=f"{self.plan_id}/observation-path",
        )


__all__ = [
    "ConstrainedMeromorphicPotential",
    "DomainHolomorphicCertificate",
    "MeromorphicLinearFrame",
    "MeromorphicLinearFrameCertificate",
    "MeromorphicMapCertificate",
    "MeromorphicVariableProjectionPlan",
    "PoleClearanceReport",
    "PoleSet",
    "TrainablePoleSet",
]
