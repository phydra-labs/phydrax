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
from ..._spectral._spherical import (
    SphericalExecution,
    SphericalHarmonicPlan,
    SphericalSampling,
)
from ...linalg import (
    ArraySpace,
    DiagonalPairing,
    FunctionLinearOperator,
    OperatorProperties,
)
from .._core import (
    DiscretizationCapability,
    DiscretizationKey,
    DiscretizationRole,
    PreparationReport,
)
from .._lifecycle import (
    AbstractDiscretizationPlan,
    validate_prepared_metadata,
)
from .._measure import DiscreteMeasure
from .._spaces import DiscreteFieldSpace, TensorDofLayout
from .._support import DiscreteSupport
from .._tensor import AbstractStrongFormDiscretization
from .._topology import EntitySet, PointTopology
from ._precision import SpectralPrecisionPolicy
from ._spherical_layout import SphericalModeLayout


_DEFAULT_EXPLICIT_BYTES = 512 * 1024**2


def _positive_limit(value: int, name: str, /) -> int:
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be positive.")
    return resolved


def _physical_points(theta: Array, phi: Array, radius: float, /) -> Array:
    colatitude, longitude = jnp.meshgrid(theta, phi, indexing="ij")
    sine = jnp.sin(colatitude)
    return radius * jnp.stack(
        (
            sine * jnp.cos(longitude),
            sine * jnp.sin(longitude),
            jnp.cos(colatitude),
        ),
        axis=-1,
    )


class SphericalSpectralPlan(AbstractDiscretizationPlan):
    """Symbolic exact-sampling spectral plan for spin fields on a round sphere."""

    precision: SpectralPrecisionPolicy
    key: DiscretizationKey
    bandlimit: int = eqx.field(static=True)
    sampling: SphericalSampling = eqx.field(static=True)
    spin: int = eqx.field(static=True)
    reality: bool = eqx.field(static=True)
    execution: SphericalExecution = eqx.field(static=True)
    field_name: str = eqx.field(static=True)
    max_precompute_bytes: int = eqx.field(static=True)
    max_explicit_eigenbasis_bytes: int = eqx.field(static=True)
    max_dense_operator_bytes: int = eqx.field(static=True)
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        bandlimit: int,
        /,
        *,
        sampling: SphericalSampling = "mw",
        spin: int = 0,
        reality: bool = True,
        execution: SphericalExecution = "recursive",
        field_name: str = "state",
        precision: SpectralPrecisionPolicy | None = None,
        key: DiscretizationKey | None = None,
        max_precompute_bytes: int = 512 * 1024**2,
        max_explicit_eigenbasis_bytes: int = _DEFAULT_EXPLICIT_BYTES,
        max_dense_operator_bytes: int = _DEFAULT_EXPLICIT_BYTES,
        plan_id: str | None = None,
    ):
        layout = SphericalModeLayout(bandlimit, spin=spin, reality=reality)
        sampling_ = str(sampling).lower()
        execution_ = str(execution).lower()
        if sampling_ not in ("mw", "mwss", "dh", "gl"):
            raise ValueError("sampling must be 'mw', 'mwss', 'dh', or 'gl'.")
        if execution_ not in ("recursive", "precomputed"):
            raise ValueError("execution must be 'recursive' or 'precomputed'.")
        field = str(field_name)
        if not field:
            raise ValueError("field_name must be non-empty.")
        precision_ = SpectralPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, SpectralPrecisionPolicy):
            raise TypeError("precision must be a SpectralPrecisionPolicy or None.")
        physical_complex = precision_.physical_dtype.startswith("complex")
        if layout.reality and physical_complex:
            raise ValueError("Reality-accelerated spherical fields require real storage.")
        if not layout.reality and not physical_complex:
            raise ValueError(
                "Complex spherical transforms require complex physical storage."
            )
        if (
            precision_.transform_dtype != "complex128"
            or precision_.coefficient_dtype != "complex128"
        ):
            raise ValueError(
                "Spherical S2FFT execution currently requires complex128 transform "
                "and coefficient precision."
            )
        if precision_.physical_dtype not in (
            "float64",
            "complex128",
        ) or precision_.output_dtype not in ("float64", "complex128"):
            raise ValueError(
                "Spherical S2FFT execution currently requires float64 or complex128 "
                "physical/output precision."
            )
        key_ = (
            DiscretizationKey(
                "spherical_spectral",
                DiscretizationRole.PHYSICAL,
                domain_labels=("theta", "phi"),
            )
            if key is None
            else key
        )
        if not isinstance(key_, DiscretizationKey):
            raise TypeError("key must be a DiscretizationKey.")
        capabilities = [
            DiscretizationCapability.PROJECTION,
            DiscretizationCapability.RECONSTRUCTION,
            DiscretizationCapability.SPECTRAL_TRANSFORM,
            DiscretizationCapability.MATRIX_FREE,
            DiscretizationCapability.DIFFERENTIABLE_GEOMETRY,
        ]
        if layout.spin == 0:
            capabilities.append(DiscretizationCapability.STRONG_DERIVATIVE)
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "spherical-spectral-plan-v1",
                    "layout": layout.layout_id,
                    "sampling": sampling_,
                    "execution": execution_,
                    "field": field,
                    "precision": precision_.policy_id,
                    "key": key_.key_id,
                    "max_precompute_bytes": int(max_precompute_bytes),
                    "max_explicit_eigenbasis_bytes": int(max_explicit_eigenbasis_bytes),
                    "max_dense_operator_bytes": int(max_dense_operator_bytes),
                }
            )
            if plan_id is None
            else str(plan_id)
        )
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.precision = precision_
        self.key = key_
        self.bandlimit = layout.bandlimit
        self.sampling = sampling_
        self.spin = layout.spin
        self.reality = layout.reality
        self.execution = execution_
        self.field_name = field
        self.max_precompute_bytes = _positive_limit(
            max_precompute_bytes, "max_precompute_bytes"
        )
        self.max_explicit_eigenbasis_bytes = _positive_limit(
            max_explicit_eigenbasis_bytes, "max_explicit_eigenbasis_bytes"
        )
        self.max_dense_operator_bytes = _positive_limit(
            max_dense_operator_bytes, "max_dense_operator_bytes"
        )
        self.capabilities = tuple(capabilities)
        self.plan_id = identifier

    def prepare(
        self,
        /,
        *,
        radius: float = 1.0,
        numeric_version: str = "0",
    ) -> SphericalSpectralDiscretization:
        radius_ = float(radius)
        if not np.isfinite(radius_) or radius_ <= 0.0:
            raise ValueError("radius must be finite and positive.")
        transform = SphericalHarmonicPlan(
            self.bandlimit,
            sampling=self.sampling,
            spin=self.spin,
            reality=self.reality,
            execution=self.execution,
            max_precompute_bytes=self.max_precompute_bytes,
        )
        return SphericalSpectralDiscretization(
            self,
            transform,
            radius=radius_,
            numeric_version=numeric_version,
        )


class SphericalSpectralDiscretization(AbstractStrongFormDiscretization):
    """Prepared exact-sampling S2 spectral discretization with physical primary state."""

    plan: SphericalSpectralPlan
    transform: SphericalHarmonicPlan
    layout: SphericalModeLayout
    physical_space: DiscreteFieldSpace
    modal_space: DiscreteFieldSpace
    key: DiscretizationKey
    support: DiscreteSupport
    field_spaces: tuple[DiscreteFieldSpace, ...]
    measures: tuple[DiscreteMeasure, ...]
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    preparation: PreparationReport
    radius: float = eqx.field(static=True)
    sample_shape: tuple[int, int] = eqx.field(static=True)
    coefficient_shape: tuple[int, int] = eqx.field(static=True)
    _points: Array
    _quadrature_weights: Array

    def __init__(
        self,
        plan: SphericalSpectralPlan,
        transform: SphericalHarmonicPlan,
        /,
        *,
        radius: float,
        numeric_version: str,
    ):
        if not isinstance(plan, SphericalSpectralPlan):
            raise TypeError("plan must be a SphericalSpectralPlan.")
        if not isinstance(transform, SphericalHarmonicPlan):
            raise TypeError("transform must be a SphericalHarmonicPlan.")
        layout = SphericalModeLayout(
            transform.bandlimit,
            spin=transform.spin,
            reality=transform.reality,
        )
        if (
            transform.bandlimit != plan.bandlimit
            or transform.sampling != plan.sampling
            or transform.spin != plan.spin
            or transform.reality != plan.reality
            or transform.execution != plan.execution
        ):
            raise ValueError("Prepared transform does not match the spherical plan.")
        radius_ = float(radius)
        geometry_dtype = jnp.float64
        weights = (
            transform.theta_quadrature_weights[:, None]
            * transform.phi_quadrature_weights[None, :]
            * radius_**2
        ).astype(geometry_dtype)
        points = _physical_points(transform.theta, transform.phi, radius_).astype(
            geometry_dtype
        )
        count = int(math.prod(transform.sample_shape))
        entities = EntitySet("spherical_samples", 0, np.arange(count, dtype=np.int64))
        topology = PointTopology(entities)
        embedding_id = canonical_fingerprint(
            {
                "kind": "round-sphere-sampling-v1",
                "radius": radius_,
                "transform": transform.transform_id,
                "points": array_tree_fingerprint(np.asarray(points)),
            }
        )
        support = DiscreteSupport(topology, 3, embedding_id)
        measure = DiscreteMeasure(
            "spherical_area",
            support.support_id,
            entities.entity_set_id,
            np.asarray(weights).reshape((-1,)),
            normalization="physical",
        )
        pairing = DiagonalPairing(weights)
        physical_layout = TensorDofLayout(
            ("theta", "phi"),
            transform.sample_shape,
            location_id=entities.entity_set_id,
        )
        projection_id = canonical_fingerprint(
            {
                "kind": "spherical-spectral-projection-v1",
                "transform": transform.transform_id,
                "radius": radius_,
            }
        )
        reconstruction_id = canonical_fingerprint(
            {
                "kind": "spherical-spectral-reconstruction-v1",
                "transform": transform.transform_id,
                "radius": radius_,
            }
        )
        physical_space = DiscreteFieldSpace(
            plan.field_name,
            support.support_id,
            physical_layout,
            ArraySpace(
                transform.sample_shape,
                dtype=jnp.dtype(plan.precision.physical_dtype),
                pairing=pairing,
            ),
            representation="point_value",
            projection_id=projection_id,
            reconstruction_id=reconstruction_id,
        )
        modal_layout = TensorDofLayout(
            ("ell", "m"),
            transform.coefficient_shape,
            location_id=layout.layout_id,
        )
        modal_space = DiscreteFieldSpace(
            plan.field_name,
            support.support_id,
            modal_layout,
            ArraySpace(
                transform.coefficient_shape,
                dtype=jnp.dtype(plan.precision.coefficient_dtype),
            ),
            representation="modal_coefficient",
            projection_id=projection_id,
            reconstruction_id=reconstruction_id,
        )
        preparation = PreparationReport(
            capabilities=plan.capabilities,
            diagnostics=(
                f"sampling:{transform.sampling}",
                f"spin:{transform.spin}",
                f"reality:{int(transform.reality)}",
                f"execution:{transform.execution}",
            ),
            resource_counts={
                "physical_points": count,
                "padded_coefficients": int(math.prod(transform.coefficient_shape)),
                "logical_modes": layout.logical_mode_count,
                "precompute_bytes": transform.precompute_bytes,
                "physical_bytes": count
                * np.dtype(plan.precision.physical_dtype).itemsize,
                "coefficient_bytes": int(math.prod(transform.coefficient_shape))
                * np.dtype(plan.precision.coefficient_dtype).itemsize,
                "dense_transform_entries": 0,
            },
        )
        spaces, measures, capabilities = validate_prepared_metadata(
            key=plan.key,
            support=support,
            field_spaces=(physical_space,),
            measures=(measure,),
            capabilities=plan.capabilities,
            preparation=preparation,
        )
        version = str(numeric_version)
        if not version:
            raise ValueError("numeric_version must be non-empty.")
        prepared_id = canonical_fingerprint(
            {
                "kind": "spherical-spectral-discretization-v1",
                "plan": plan.plan_id,
                "transform": transform.transform_id,
                "execution": transform.execution_id,
                "layout": layout.layout_id,
                "radius": radius_,
                "support": support.support_id,
                "physical_space": physical_space.field_space_id,
                "measure": measure.measure_id,
                "modal_space": modal_space.field_space_id,
                "numeric_version": version,
            }
        )
        self.plan = plan
        self.transform = transform
        self.layout = layout
        self.physical_space = physical_space
        self.modal_space = modal_space
        self.key = plan.key
        self.support = support
        self.field_spaces = spaces
        self.measures = measures
        self.capabilities = capabilities
        self.plan_id = plan.plan_id
        self.prepared_id = prepared_id
        self.numeric_version = version
        self.preparation = preparation
        self.radius = radius_
        self.sample_shape = transform.sample_shape
        self.coefficient_shape = transform.coefficient_shape
        self._points = points.reshape((-1, 3))
        self._quadrature_weights = weights

    @property
    def state_shape(self) -> tuple[int, ...]:
        return self.sample_shape

    @property
    def physical_shape(self) -> tuple[int, ...]:
        return self.sample_shape

    @property
    def modal_shape(self) -> tuple[int, ...]:
        return self.coefficient_shape

    @property
    def spatial_dimension(self) -> int:
        return 2

    @property
    def quadrature_weights(self) -> Array:
        return self._quadrature_weights

    @property
    def points(self) -> Array:
        return self._points

    @property
    def discretization_id(self) -> str:
        return self.prepared_id

    @property
    def precision_evidence(self):
        return self.plan.precision.evidence()

    @property
    def resource_evidence_id(self) -> str:
        return self.preparation.report_id

    def _validate_physical(self, values: ArrayLike, name: str, /) -> Array:
        array = jnp.asarray(values)
        if array.ndim < 2 or tuple(array.shape[:2]) != self.sample_shape:
            raise ValueError(
                f"{name} must begin with shape {self.sample_shape}; got {array.shape}."
            )
        return array

    def project(self, values: ArrayLike, /) -> Array:
        physical = self._validate_physical(values, "Spherical physical values")
        transform_values = (
            self.plan.precision.physical(physical)
            if self.layout.reality
            else self.plan.precision.transform(physical)
        )
        coefficients = self.transform.analysis(transform_values)
        coefficients = self.layout.mask_invalid(coefficients)
        if self.layout.reality:
            coefficients = self.layout.canonicalize_reality(coefficients)
        return self.plan.precision.coefficients(coefficients)

    def reconstruct(self, coefficients: ArrayLike, /) -> Array:
        modal = self.plan.precision.coefficients(coefficients)
        modal = self.layout.mask_invalid(modal)
        if self.layout.reality:
            modal = self.layout.canonicalize_reality(modal)
        return self.plan.precision.output(self.transform.synthesis(modal))

    def invalid_storage_defect(self, coefficients: ArrayLike, /) -> Array:
        array = jnp.asarray(coefficients)
        masked = self.layout.mask_invalid(array)
        return jnp.max(jnp.abs(array - masked), initial=0.0)

    def conjugacy_defect(self, coefficients: ArrayLike, /) -> Array:
        return self.layout.conjugacy_defect(coefficients)

    def negative_laplacian_levels(self, /) -> Array:
        if self.layout.spin != 0:
            raise ValueError("Scalar Laplace-Beltrami levels require spin zero.")
        degree = jnp.arange(self.layout.bandlimit, dtype=self._quadrature_weights.dtype)
        return degree * (degree + 1.0) / self.radius**2

    def laplacian_multiplier(self, /) -> Array:
        return self.layout.level_values(-self.negative_laplacian_levels())

    def modal_laplacian(self, coefficients: ArrayLike, /) -> Array:
        modal = self.layout.mask_invalid(coefficients)
        if self.layout.reality:
            modal = self.layout.canonicalize_reality(modal)
        if tuple(modal.shape[-2:]) == self.coefficient_shape:
            multiplier = self.laplacian_multiplier()
        elif modal.ndim >= 3 and tuple(modal.shape[-3:-1]) == self.coefficient_shape:
            multiplier = self.laplacian_multiplier()[..., None]
        else:
            raise ValueError(
                "Spherical modal state has an incompatible coefficient shape."
            )
        return self.layout.mask_invalid(modal * multiplier.astype(modal.dtype))

    def modal_integral(self, coefficients: ArrayLike, /) -> Array:
        """Integrate scalar spin-zero coefficients without reconstruction."""
        if self.layout.spin != 0:
            raise ValueError("Spherical modal integration requires spin zero.")
        modal = self.layout.mask_invalid(coefficients)
        if self.layout.reality:
            modal = self.layout.canonicalize_reality(modal)
        degree_axis, order_axis, _ = self.layout._coefficient_axes(modal)
        constant = jnp.take(modal, 0, axis=degree_axis)
        adjusted_order_axis = order_axis - int(order_axis > degree_axis)
        constant = jnp.take(
            constant,
            self.layout.bandlimit - 1,
            axis=adjusted_order_axis,
        )
        scale = jnp.asarray(
            math.sqrt(4.0 * math.pi) * self.radius**2,
            dtype=jnp.real(modal).dtype,
        )
        return scale.astype(modal.dtype) * constant

    def coordinate_derivative(
        self,
        coefficients: ArrayLike,
        /,
        *,
        coordinate: str,
        representation: str = "physical",
        require_all_valid: bool = True,
        polar_tolerance: float | None = None,
    ):
        """Return a chart-valued coordinate derivative with pole evidence."""
        from ._spherical_operators import spherical_coordinate_derivative

        return spherical_coordinate_derivative(
            self,
            coefficients,
            coordinate=coordinate,
            representation=representation,
            require_all_valid=require_all_valid,
            polar_tolerance=polar_tolerance,
        )

    def laplacian(
        self,
        state: ArrayLike,
        /,
        *,
        axes: int | Sequence[int] | None = None,
    ) -> Array:
        selected = (
            (0, 1) if axes is None else (axes,) if isinstance(axes, int) else tuple(axes)
        )
        if selected != (0, 1):
            raise ValueError("Spherical Laplace-Beltrami acts on both intrinsic axes.")
        physical = self._validate_physical(state, "Spherical state")
        return self.reconstruct(self.modal_laplacian(self.project(physical)))

    def integral(
        self,
        state: ArrayLike,
        /,
        *,
        axes: int | Sequence[int] | None = None,
    ) -> Array:
        selected = (
            (0, 1) if axes is None else (axes,) if isinstance(axes, int) else tuple(axes)
        )
        if selected != (0, 1):
            raise ValueError("Spherical integration acts on both intrinsic axes.")
        values = self._validate_physical(state, "Spherical state")
        weights = self._quadrature_weights.reshape(
            self.sample_shape + (1,) * (values.ndim - 2)
        )
        return jnp.sum(weights * values, axis=(0, 1))

    def partial_derivative(
        self,
        state: ArrayLike,
        /,
        *,
        axis: int,
        order: int = 1,
    ) -> Array:
        del state, axis, order
        raise NotImplementedError(
            "SphericalSpectralDiscretization has no global coordinate derivative frame."
        )

    def gradient(
        self,
        state: ArrayLike,
        /,
        *,
        axes: int | Sequence[int] | None = None,
    ) -> Array:
        del state, axes
        raise NotImplementedError(
            "SphericalSpectralDiscretization has no global coordinate gradient frame."
        )

    def divergence(
        self,
        state: ArrayLike,
        /,
        *,
        axes: int | Sequence[int] | None = None,
        dual: bool = False,
    ) -> Array:
        del state, axes, dual
        raise NotImplementedError(
            "SphericalSpectralDiscretization has no global coordinate divergence frame."
        )

    def flatten(self, state: ArrayLike, /) -> Array:
        values = self._validate_physical(state, "Spherical state")
        return values.reshape((math.prod(self.sample_shape),) + values.shape[2:])

    def unflatten(self, state: ArrayLike, /) -> Array:
        values = jnp.asarray(state)
        count = math.prod(self.sample_shape)
        if values.ndim < 1 or int(values.shape[0]) != count:
            raise ValueError(f"Flattened spherical state must begin with ({count},).")
        return values.reshape(self.sample_shape + values.shape[1:])

    def _real_mode_specification(self, rank: int, /) -> tuple[Array, tuple[str, ...]]:
        coefficients = []
        mode_ids = []
        offset = self.layout.bandlimit - 1
        inverse_root_two = 1.0 / math.sqrt(2.0)
        retained = 0
        for degree in range(self.layout.bandlimit):
            block = 2 * degree + 1
            if retained + block > rank:
                break
            zero = np.zeros(self.coefficient_shape, dtype=np.complex128)
            zero[degree, offset] = 1.0
            coefficients.append(zero)
            mode_ids.append(f"sphere-real:ell:{degree}:m:0")
            for order in range(1, degree + 1):
                cosine = np.zeros(self.coefficient_shape, dtype=np.complex128)
                cosine[degree, offset + order] = inverse_root_two
                cosine[degree, offset - order] = (-1) ** order * inverse_root_two
                sine = np.zeros(self.coefficient_shape, dtype=np.complex128)
                sine[degree, offset + order] = -1j * inverse_root_two
                sine[degree, offset - order] = 1j * (-1) ** order * inverse_root_two
                coefficients.extend((cosine, sine))
                mode_ids.extend(
                    (
                        f"sphere-real:ell:{degree}:m:{order}:cos",
                        f"sphere-real:ell:{degree}:m:{order}:sin",
                    )
                )
            retained += block
        return jnp.asarray(np.stack(coefficients, axis=0)), tuple(mode_ids)

    def eigenpairs(self, *, rank: int | None = None) -> tuple[Array, Array]:
        if self.layout.spin != 0 or not self.layout.reality:
            raise ValueError("Real spherical eigenpairs require a real spin-zero space.")
        maximum = self.layout.logical_mode_count
        retained = maximum if rank is None else int(rank)
        degree_count = math.isqrt(retained)
        if retained <= 0 or retained > maximum or degree_count**2 != retained:
            raise ValueError(
                "Spherical eigenpair rank must be a positive complete-degree square "
                f"not exceeding {maximum}."
            )
        point_count = math.prod(self.sample_shape)
        estimate = retained * (
            point_count * np.dtype(self.plan.precision.physical_dtype).itemsize
            + math.prod(self.coefficient_shape)
            * np.dtype(self.plan.precision.coefficient_dtype).itemsize
        )
        if estimate > self.plan.max_explicit_eigenbasis_bytes:
            raise ValueError(
                "Spherical eigenbasis construction exceeds "
                f"max_explicit_eigenbasis_bytes; estimated {estimate} bytes."
            )
        coefficients, _ = self._real_mode_specification(retained)
        modes = self.transform.synthesis(coefficients)
        modes = jnp.moveaxis(jnp.real(modes), 0, -1).astype(
            jnp.dtype(self.plan.precision.physical_dtype)
        )
        weights = self._quadrature_weights[..., None]
        norms = jnp.sqrt(jnp.sum(weights * modes**2, axis=(0, 1)))
        modes = modes / norms.reshape((1, 1, retained))
        values = []
        for degree in range(degree_count):
            values.extend([degree * (degree + 1.0) / self.radius**2] * (2 * degree + 1))
        return jnp.asarray(values, dtype=modes.dtype), modes

    def eigenmode_ids(self, *, rank: int | None = None) -> tuple[str, ...]:
        maximum = self.layout.logical_mode_count
        retained = maximum if rank is None else int(rank)
        degree_count = math.isqrt(retained)
        if retained <= 0 or retained > maximum or degree_count**2 != retained:
            raise ValueError("Spherical mode IDs require a complete-degree square rank.")
        _, mode_ids = self._real_mode_specification(retained)
        return mode_ids

    def laplacian_matrix(self) -> Array:
        count = math.prod(self.sample_shape)
        itemsize = np.dtype(self.plan.precision.physical_dtype).itemsize
        estimate = count * count * itemsize
        if estimate > self.plan.max_dense_operator_bytes:
            raise ValueError(
                "Spherical dense Laplacian exceeds max_dense_operator_bytes; "
                f"estimated {estimate} bytes."
            )
        identity = jnp.eye(count, dtype=jnp.dtype(self.plan.precision.physical_dtype))
        columns = jax.vmap(
            lambda vector: self.laplacian(vector.reshape(self.sample_shape)).reshape(
                (-1,)
            )
        )(identity)
        return columns.T


def spherical_laplacian_operator(
    discretization: SphericalSpectralDiscretization,
    /,
) -> FunctionLinearOperator:
    """Return the scalar physical-space Laplace-Beltrami operator."""
    if not isinstance(discretization, SphericalSpectralDiscretization):
        raise TypeError("discretization must be a SphericalSpectralDiscretization.")
    if discretization.layout.spin != 0:
        raise ValueError("Spherical Laplace-Beltrami operators require spin zero.")
    space = discretization.physical_space.vector_space
    return FunctionLinearOperator(
        discretization.laplacian,
        source=space,
        target=space,
        properties=OperatorProperties(
            self_adjoint=True,
            evidence={"self_adjoint": "construction"},
        ),
        operator_id=canonical_fingerprint(
            {
                "kind": "spherical-laplace-beltrami-operator-v1",
                "discretization": discretization.prepared_id,
                "sign": "negative-semidefinite",
            }
        ),
    )


__all__ = [
    "SphericalSpectralDiscretization",
    "SphericalSpectralPlan",
    "spherical_laplacian_operator",
]
