#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._polynomial._chebyshev import chebyshev_lobatto_data
from ..._polynomial._orthogonal import (
    legendre_rule_data,
    standard_derivative_matrix,
    standard_vandermonde,
)
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    AbstractLinearTransform,
    DenseLinearOperator,
    DenseLinearTransform,
    FactorizationPolicy,
    factorize,
    FFTLinearTransform,
    RealTrigonometricTransform,
    SimilarityScaledLinearTransform,
)
from .._axis import AxisDiscretization
from .._axis_domain import AxisDomain
from .._spectral import ModalTransform
from ._precision import SpectralPrecisionPolicy


SpectralBasisFamily: TypeAlias = Literal[
    "fourier",
    "sine",
    "cosine",
    "chebyshev",
    "legendre",
    "rational_chebyshev_line",
    "rational_chebyshev_half_line",
]
SpectralBoundaryKind: TypeAlias = Literal[
    "periodic",
    "homogeneous_dirichlet",
    "homogeneous_neumann",
    "homogeneous_trace",
    "decay",
    "unconstrained",
]


def _analysis_from_synthesis(
    synthesis: ArrayLike,
    dtype: str,
    /,
) -> np.ndarray:
    """Construct a left inverse through the canonical dense SVD substrate."""
    synthesis_ = jnp.asarray(synthesis, dtype=jnp.dtype(dtype))
    operator = DenseLinearOperator(synthesis_)
    factorization = factorize(operator, FactorizationPolicy("svd"))
    identity = jnp.eye(synthesis_.shape[0], dtype=synthesis_.dtype)
    columns = tuple(
        factorization.solve(identity[:, index]).value
        for index in range(identity.shape[1])
    )
    return np.asarray(jnp.stack(columns, axis=1))


class SpectralModeLayout(StrictModule, NonTrainableState):
    """Stable one-dimensional spectral modes and storage correspondences."""

    mode_numbers: Array
    conjugate_indices: Array
    zero_mask: Array
    nyquist_mask: Array
    mode_ids: tuple[str, ...] = eqx.field(static=True)
    family: SpectralBasisFamily = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        family: SpectralBasisFamily,
        mode_numbers: ArrayLike,
        /,
        *,
        conjugate_indices: ArrayLike | None = None,
        nyquist_mask: ArrayLike | None = None,
        mode_ids: tuple[str, ...] | None = None,
    ):
        numbers_host = np.asarray(mode_numbers, dtype=np.int64).reshape((-1,))
        if numbers_host.size == 0 or len(set(numbers_host.tolist())) != numbers_host.size:
            raise ValueError("Spectral mode numbers must be non-empty and unique.")
        count = int(numbers_host.size)
        conjugates_host = (
            np.arange(count, dtype=np.int64)
            if conjugate_indices is None
            else np.asarray(conjugate_indices, dtype=np.int64).reshape((-1,))
        )
        if conjugates_host.shape != (count,) or np.any(
            (conjugates_host < 0) | (conjugates_host >= count)
        ):
            raise ValueError("conjugate_indices must reference one valid mode per mode.")
        if np.any(conjugates_host[conjugates_host] != np.arange(count)):
            raise ValueError("Spectral conjugate indices must be involutive.")
        nyquist_host = (
            np.zeros((count,), dtype=bool)
            if nyquist_mask is None
            else np.asarray(nyquist_mask, dtype=bool).reshape((-1,))
        )
        if nyquist_host.shape != (count,):
            raise ValueError("nyquist_mask must contain one value per mode.")
        ids = (
            tuple(f"{family}:{int(value)}" for value in numbers_host)
            if mode_ids is None
            else tuple(str(value) for value in mode_ids)
        )
        if len(ids) != count or any(not value for value in ids) or len(set(ids)) != count:
            raise ValueError("mode_ids must contain one unique non-empty ID per mode.")
        self.mode_numbers = jnp.asarray(numbers_host, dtype=jnp.int32)
        self.conjugate_indices = jnp.asarray(conjugates_host, dtype=jnp.int32)
        self.zero_mask = jnp.asarray(numbers_host == 0)
        self.nyquist_mask = jnp.asarray(nyquist_host)
        self.mode_ids = ids
        self.family = family
        self.layout_id = canonical_fingerprint(
            {
                "kind": "spectral-mode-layout",
                "family": family,
                "numbers": array_tree_fingerprint(numbers_host),
                "conjugates": array_tree_fingerprint(conjugates_host),
                "nyquist": array_tree_fingerprint(nyquist_host),
                "mode_ids": list(ids),
            }
        )

    @property
    def count(self) -> int:
        return int(self.mode_numbers.size)


class AbstractSpectralBasisPlan(StrictModule, NonTrainableState):
    """Symbolic one-dimensional global spectral basis plan."""

    mode_count: int = eqx.field(static=True)
    family: SpectralBasisFamily = eqx.field(static=True)
    periodic: bool = eqx.field(static=True)
    boundary: SpectralBoundaryKind = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def prepare(
        self,
        domain: AxisDomain,
        /,
        *,
        precision: SpectralPrecisionPolicy,
    ) -> "PreparedSpectralAxis":
        raise NotImplementedError

    @abc.abstractmethod
    def resized(self, mode_count: int, /) -> "AbstractSpectralBasisPlan":
        raise NotImplementedError


def _finite_domain(
    domain: AxisDomain,
    /,
    *,
    periodic: bool,
    basis: str,
) -> tuple[Array, Array]:
    if not isinstance(domain, AxisDomain):
        raise TypeError("domain must be an AxisDomain.")
    expected = "periodic" if periodic else "bounded"
    if domain.kind != expected:
        raise ValueError(f"{basis} bases require an {expected} axis domain.")
    return domain.lower, domain.upper


class PreparedSpectralAxis(StrictModule, NonTrainableState):
    """Prepared physical nodes, reference modes, measure, and transform."""

    plan: AbstractSpectralBasisPlan
    nodes: Array
    reference_nodes: Array
    quadrature_weights: Array
    domain: AxisDomain
    modes: SpectralModeLayout
    execution_transform: AbstractLinearTransform
    precision: SpectralPrecisionPolicy
    derivative_matrix: Array | None
    modal_transform: ModalTransform | None
    family: SpectralBasisFamily = eqx.field(static=True)
    periodic: bool = eqx.field(static=True)
    boundary: SpectralBoundaryKind = eqx.field(static=True)
    lower_endpoint_included: bool = eqx.field(static=True)
    upper_endpoint_included: bool = eqx.field(static=True)
    derivative_exact: bool = eqx.field(static=True)
    derivative_residual: float = eqx.field(static=True)
    axis_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: AbstractSpectralBasisPlan,
        nodes: ArrayLike,
        reference_nodes: ArrayLike,
        quadrature_weights: ArrayLike,
        domain: AxisDomain,
        modes: SpectralModeLayout,
        execution_transform: AbstractLinearTransform,
        precision: SpectralPrecisionPolicy,
        /,
        *,
        lower_endpoint_included: bool,
        upper_endpoint_included: bool,
        derivative_matrix: ArrayLike | None = None,
        derivative_exact: bool = True,
        derivative_residual: float = 0.0,
        modal_transform: ModalTransform | None = None,
    ):
        if not isinstance(plan, AbstractSpectralBasisPlan):
            raise TypeError("plan must be an AbstractSpectralBasisPlan.")
        if not isinstance(domain, AxisDomain):
            raise TypeError("domain must be an AxisDomain.")
        if plan.periodic != domain.periodic_axis:
            raise ValueError("Basis periodicity must match its prepared axis domain.")
        if not isinstance(modes, SpectralModeLayout):
            raise TypeError("modes must be a SpectralModeLayout.")
        if not isinstance(execution_transform, AbstractLinearTransform):
            raise TypeError("execution_transform must be an AbstractLinearTransform.")
        if not isinstance(precision, SpectralPrecisionPolicy):
            raise TypeError("precision must be a SpectralPrecisionPolicy.")
        nodes_ = jnp.asarray(nodes, dtype=jnp.dtype(precision.physical_dtype)).reshape(
            (-1,)
        )
        reference = jnp.asarray(
            reference_nodes,
            dtype=jnp.dtype(precision.physical_dtype),
        ).reshape((-1,))
        weights = jnp.asarray(
            quadrature_weights,
            dtype=jnp.dtype(precision.physical_dtype),
        ).reshape((-1,))
        if (
            nodes_.shape != reference.shape
            or nodes_.shape != weights.shape
            or nodes_.size == 0
        ):
            raise ValueError(
                "Prepared spectral nodes, reference nodes, and weights must align."
            )
        finite_positive = (
            jnp.all(jnp.isfinite(nodes_))
            & jnp.all(jnp.isfinite(reference))
            & jnp.all(jnp.isfinite(weights))
            & jnp.all(weights > 0.0)
        )
        weights = eqx.error_if(
            weights,
            ~finite_positive,
            "Prepared spectral nodes and weights must be finite with positive weights.",
        )
        if modes.count != plan.mode_count:
            raise ValueError("Prepared spectral mode layout must match the plan count.")
        if execution_transform.physical_space.shape != nodes_.shape:
            raise ValueError(
                "Execution-transform physical shape must match spectral nodes."
            )
        if execution_transform.modal_space.shape != (modes.count,):
            raise ValueError("Execution-transform modal shape must match spectral modes.")
        derivative = (
            None
            if derivative_matrix is None
            else jnp.asarray(
                derivative_matrix,
                dtype=jnp.dtype(precision.coefficient_dtype),
            )
        )
        if derivative is not None and derivative.shape != (
            modes.count,
            modes.count,
        ):
            raise ValueError("derivative_matrix must have square modal shape or be None.")
        if modal_transform is not None and not isinstance(
            modal_transform, ModalTransform
        ):
            raise TypeError("modal_transform must be a ModalTransform or None.")
        lower = bool(lower_endpoint_included)
        upper = bool(upper_endpoint_included)
        if domain.periodic_axis and (lower or upper):
            raise ValueError("Periodic spectral axes cannot include boundary endpoints.")
        derivative_residual_ = float(derivative_residual)
        if (
            not np.isfinite(derivative_residual_)
            or derivative_residual_ < 0.0
            or (derivative_matrix is None and not derivative_exact)
        ):
            raise ValueError("Derivative exactness evidence is inconsistent.")
        self.plan = plan
        self.nodes = nodes_
        self.reference_nodes = reference
        self.quadrature_weights = weights
        self.domain = domain
        self.modes = modes
        self.execution_transform = execution_transform
        self.derivative_matrix = derivative
        self.modal_transform = modal_transform
        self.precision = precision
        self.family = plan.family
        self.periodic = plan.periodic
        self.boundary = plan.boundary
        self.lower_endpoint_included = lower
        self.upper_endpoint_included = upper
        self.derivative_exact = bool(derivative_exact)
        self.derivative_residual = derivative_residual_
        self.axis_id = canonical_fingerprint(
            {
                "kind": "prepared-spectral-axis",
                "plan": plan.plan_id,
                "domain": domain.domain_id,
                "reference_nodes": array_tree_fingerprint(np.asarray(reference)),
                "modes": modes.layout_id,
                "transform": execution_transform.transform_id,
                "precision": precision.policy_id,
                "lower_endpoint_included": lower,
                "upper_endpoint_included": upper,
                "derivative": (
                    None
                    if derivative is None
                    else array_tree_fingerprint(np.asarray(derivative))
                ),
                "derivative_exact": bool(derivative_exact),
                "derivative_residual": derivative_residual_,
                "modal_transform": (
                    None if modal_transform is None else modal_transform.transform_id
                ),
            }
        )

    @property
    def mode_count(self) -> int:
        return self.modes.count

    @property
    def physical_count(self) -> int:
        return int(self.nodes.size)

    @property
    def bounds(self) -> Array | None:
        return self.domain.finite_bounds

    @property
    def length(self) -> Array:
        return self.domain.length

    def analyze(self, values: ArrayLike, /) -> Array:
        return self.execution_transform.analyze(self.precision.transform(values))

    def synthesize(self, coefficients: ArrayLike, /) -> Array:
        return self.execution_transform.synthesize(
            self.precision.coefficients(coefficients)
        )

    def at_count(self, mode_count: int, /) -> "PreparedSpectralAxis":
        return self.plan.resized(mode_count).prepare(
            self.domain,
            precision=self.precision,
        )

    def axis_discretization(self) -> AxisDiscretization:
        primary = "interval" if self.family == "sine" else "point"
        return AxisDiscretization(
            nodes=self.nodes,
            quad_weights=self.quadrature_weights,
            basis=self.family,
            domain=self.domain,
            primary_entity=primary,
            lower_endpoint_included=self.lower_endpoint_included,
            upper_endpoint_included=self.upper_endpoint_included,
        )

    def laplacian_eigenvalues(self) -> Array:
        if self.family in (
            "chebyshev",
            "legendre",
            "rational_chebyshev_line",
            "rational_chebyshev_half_line",
        ):
            raise ValueError(
                "Polynomial and rational Laplacians are not diagonal eigenoperators."
            )
        values = self.modes.mode_numbers.astype(jnp.dtype(self.precision.physical_dtype))
        scale = jnp.asarray(jnp.pi, dtype=values.dtype) / self.length
        if self.family == "fourier":
            scale = 2.0 * scale
        return (scale * values) ** 2

    def derivative_multiplier(self, order: int, /) -> Array:
        derivative_order = int(order)
        if derivative_order < 0:
            raise ValueError("Spectral derivative order must be non-negative.")
        if self.derivative_matrix is not None:
            raise ValueError(
                "This basis uses a prepared modal derivative action, not a multiplier."
            )
        values = self.modes.mode_numbers.astype(
            jnp.dtype(self.precision.coefficient_dtype)
        )
        if derivative_order == 0:
            return jnp.ones_like(values)
        if self.family not in ("fourier", "sine", "cosine"):
            raise ValueError(
                "This basis does not define a closed modal derivative multiplier."
            )
        if self.family == "fourier":
            scale = 2.0 * jnp.asarray(jnp.pi, dtype=values.dtype) / self.length
            return (1j * scale * values) ** derivative_order
        if derivative_order % 2:
            raise ValueError(
                "Odd sine/cosine derivatives map to a dual parity basis; request "
                "physical derivative values instead of an endomorphism multiplier."
            )
        scale = jnp.asarray(jnp.pi, dtype=values.dtype) / self.length
        return (1j * scale * values) ** derivative_order


class FourierBasisPlan(AbstractSpectralBasisPlan):
    """Complex exponential basis on a periodic interval."""

    def __init__(self, mode_count: int):
        count = int(mode_count)
        if count < 2:
            raise ValueError("Fourier bases require at least two modes.")
        self.mode_count = count
        self.family = "fourier"
        self.periodic = True
        self.boundary = "periodic"
        self.plan_id = canonical_fingerprint(
            {"kind": "fourier-basis-plan", "mode_count": count, "packing": "complex"}
        )

    def resized(self, mode_count: int, /) -> "FourierBasisPlan":
        return FourierBasisPlan(mode_count)

    def prepare(
        self,
        domain: AxisDomain,
        /,
        *,
        precision: SpectralPrecisionPolicy,
    ) -> PreparedSpectralAxis:
        lower_, upper_ = _finite_domain(
            domain,
            periodic=True,
            basis="Fourier",
        )
        lower_ = lower_.astype(jnp.dtype(precision.physical_dtype))
        upper_ = upper_.astype(jnp.dtype(precision.physical_dtype))
        count = self.mode_count
        reference = jnp.arange(count, dtype=lower_.dtype) / float(count)
        nodes = lower_ + (upper_ - lower_) * reference
        weights = jnp.full((count,), (upper_ - lower_) / float(count), dtype=nodes.dtype)
        numbers_host = np.rint(np.fft.fftfreq(count) * count).astype(np.int64)
        lookup = {int(value): index for index, value in enumerate(numbers_host)}
        conjugates = np.asarray(
            [lookup.get(int(-value), index) for index, value in enumerate(numbers_host)],
            dtype=np.int64,
        )
        nyquist = np.zeros((count,), dtype=bool)
        if count % 2 == 0:
            nyquist[count // 2] = True
        modes = SpectralModeLayout(
            "fourier",
            numbers_host,
            conjugate_indices=conjugates,
            nyquist_mask=nyquist,
        )
        base = FFTLinearTransform(
            count,
            dtype=jnp.dtype(precision.transform_dtype),
        )
        transform = SimilarityScaledLinearTransform(
            base,
            jnp.sqrt(weights).astype(jnp.dtype(precision.transform_dtype)),
        )
        return PreparedSpectralAxis(
            self,
            nodes,
            reference,
            weights,
            domain,
            modes,
            transform,
            precision,
            lower_endpoint_included=False,
            upper_endpoint_included=False,
        )


class SineBasisPlan(AbstractSpectralBasisPlan):
    """Cell-centered sine basis with homogeneous Dirichlet endpoint semantics."""

    def __init__(self, mode_count: int):
        count = int(mode_count)
        if count < 2:
            raise ValueError("Sine bases require at least two modes.")
        self.mode_count = count
        self.family = "sine"
        self.periodic = False
        self.boundary = "homogeneous_dirichlet"
        self.plan_id = canonical_fingerprint(
            {"kind": "sine-basis-plan", "mode_count": count, "transform_type": 2}
        )

    def resized(self, mode_count: int, /) -> "SineBasisPlan":
        return SineBasisPlan(mode_count)

    def prepare(
        self,
        domain: AxisDomain,
        /,
        *,
        precision: SpectralPrecisionPolicy,
    ) -> PreparedSpectralAxis:
        lower_, upper_ = _finite_domain(
            domain,
            periodic=False,
            basis="Sine",
        )
        lower_ = lower_.astype(jnp.dtype(precision.physical_dtype))
        upper_ = upper_.astype(jnp.dtype(precision.physical_dtype))
        count = self.mode_count
        unit = (jnp.arange(count, dtype=lower_.dtype) + 0.5) / float(count)
        reference = -1.0 + 2.0 * unit
        nodes = lower_ + (upper_ - lower_) * unit
        weights = jnp.full((count,), (upper_ - lower_) / float(count), dtype=nodes.dtype)
        modes = SpectralModeLayout("sine", np.arange(1, count + 1, dtype=np.int64))
        base = RealTrigonometricTransform(
            "dst",
            2,
            count,
            dtype=jnp.dtype(precision.physical_dtype).type,
        )
        transform = SimilarityScaledLinearTransform(base, jnp.sqrt(weights))
        return PreparedSpectralAxis(
            self,
            nodes,
            reference,
            weights,
            domain,
            modes,
            transform,
            precision,
            lower_endpoint_included=False,
            upper_endpoint_included=False,
        )


class CosineBasisPlan(AbstractSpectralBasisPlan):
    """Endpoint-including cosine basis with homogeneous Neumann semantics."""

    def __init__(self, mode_count: int):
        count = int(mode_count)
        if count < 2:
            raise ValueError("Cosine bases require at least two modes.")
        self.mode_count = count
        self.family = "cosine"
        self.periodic = False
        self.boundary = "homogeneous_neumann"
        self.plan_id = canonical_fingerprint(
            {"kind": "cosine-basis-plan", "mode_count": count, "transform_type": 1}
        )

    def resized(self, mode_count: int, /) -> "CosineBasisPlan":
        return CosineBasisPlan(mode_count)

    def prepare(
        self,
        domain: AxisDomain,
        /,
        *,
        precision: SpectralPrecisionPolicy,
    ) -> PreparedSpectralAxis:
        lower_, upper_ = _finite_domain(
            domain,
            periodic=False,
            basis="Cosine",
        )
        lower_ = lower_.astype(jnp.dtype(precision.physical_dtype))
        upper_ = upper_.astype(jnp.dtype(precision.physical_dtype))
        count = self.mode_count
        reference = jnp.linspace(-1.0, 1.0, count, endpoint=True)
        nodes = lower_ + 0.5 * (reference + 1.0) * (upper_ - lower_)
        spacing = (upper_ - lower_) / float(count - 1)
        weights = jnp.full((count,), spacing, dtype=nodes.dtype)
        weights = weights.at[0].set(0.5 * spacing)
        weights = weights.at[-1].set(0.5 * spacing)
        modes = SpectralModeLayout("cosine", np.arange(count, dtype=np.int64))
        base = RealTrigonometricTransform(
            "dct",
            1,
            count,
            dtype=jnp.dtype(precision.physical_dtype).type,
        )
        transform = SimilarityScaledLinearTransform(base, jnp.sqrt(weights))
        return PreparedSpectralAxis(
            self,
            nodes,
            reference,
            weights,
            domain,
            modes,
            transform,
            precision,
            lower_endpoint_included=True,
            upper_endpoint_included=True,
        )


class ChebyshevBasisPlan(AbstractSpectralBasisPlan):
    """Global Chebyshev--Lobatto interpolation basis with dense budget."""

    maximum_construction_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        mode_count: int,
        /,
        *,
        maximum_construction_bytes: int = 512 * 1024**2,
    ):
        count = int(mode_count)
        maximum = int(maximum_construction_bytes)
        if count < 2 or maximum <= 0:
            raise ValueError(
                "Chebyshev mode_count and maximum_construction_bytes must be positive."
            )
        self.mode_count = count
        self.family = "chebyshev"
        self.periodic = False
        self.boundary = "unconstrained"
        self.maximum_construction_bytes = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "chebyshev-basis-plan",
                "mode_count": count,
                "node_rule": "lobatto",
                "maximum_construction_bytes": maximum,
            }
        )

    def resized(self, mode_count: int, /) -> "ChebyshevBasisPlan":
        return ChebyshevBasisPlan(
            mode_count,
            maximum_construction_bytes=self.maximum_construction_bytes,
        )

    def prepare(
        self,
        domain: AxisDomain,
        /,
        *,
        precision: SpectralPrecisionPolicy,
    ) -> PreparedSpectralAxis:
        lower, upper = _finite_domain(
            domain,
            periodic=False,
            basis="Chebyshev",
        )
        lower_value = float(np.asarray(lower))
        upper_value = float(np.asarray(upper))
        count = self.mode_count
        itemsize = np.dtype(precision.coefficient_dtype).itemsize
        estimate = 5 * count * count * itemsize
        if estimate > self.maximum_construction_bytes:
            raise ValueError("Chebyshev transform exceeds maximum_construction_bytes.")
        data = chebyshev_lobatto_data(
            count,
            maximum_derivative_order=0,
            dtype=precision.physical_dtype,
            maximum_construction_bytes=self.maximum_construction_bytes,
        )
        reference_nodes = np.asarray(data.nodes, dtype=float)
        reference_weights = np.asarray(data.quadrature_weights, dtype=float)
        half = 0.5 * (upper_value - lower_value)
        midpoint = 0.5 * (upper_value + lower_value)
        nodes = midpoint + half * reference_nodes
        weights = half * reference_weights
        synthesis = np.asarray(
            standard_vandermonde(
                "chebyshev",
                reference_nodes,
                count - 1,
            )
        )
        analysis = _analysis_from_synthesis(
            synthesis,
            precision.coefficient_dtype,
        )
        derivative = np.asarray(
            standard_derivative_matrix(
                "chebyshev",
                count,
                scale=2.0 / (upper_value - lower_value),
                dtype=precision.physical_dtype,
            )
        )
        modal = ModalTransform(
            analysis,
            synthesis,
            weights,
            mode_ids=tuple(f"chebyshev:{degree}" for degree in range(count)),
        )
        execution = DenseLinearTransform(
            np.asarray(analysis, dtype=precision.coefficient_dtype),
            np.asarray(synthesis, dtype=precision.coefficient_dtype),
            transform_id=modal.transform_id,
        )
        return PreparedSpectralAxis(
            self,
            nodes,
            reference_nodes,
            weights,
            domain,
            SpectralModeLayout(
                "chebyshev",
                np.arange(count),
                mode_ids=tuple(f"chebyshev:{degree}" for degree in range(count)),
            ),
            execution,
            precision,
            lower_endpoint_included=True,
            upper_endpoint_included=True,
            derivative_matrix=derivative,
            modal_transform=modal,
        )


class LegendreBasisPlan(AbstractSpectralBasisPlan):
    """Global Legendre basis on a Gauss, Radau, or Lobatto grid."""

    node_rule: Literal["gauss", "radau", "lobatto"] = eqx.field(static=True)
    maximum_construction_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        mode_count: int,
        /,
        *,
        node_rule: Literal["gauss", "radau", "lobatto"] = "gauss",
        maximum_construction_bytes: int = 512 * 1024**2,
    ):
        count = int(mode_count)
        maximum = int(maximum_construction_bytes)
        if count < 2 or maximum <= 0:
            raise ValueError(
                "Legendre mode_count and maximum_construction_bytes must be positive."
            )
        if node_rule not in ("gauss", "radau", "lobatto"):
            raise ValueError("Unknown Legendre node rule.")
        self.mode_count = count
        self.family = "legendre"
        self.periodic = False
        self.boundary = "unconstrained"
        self.node_rule = node_rule
        self.maximum_construction_bytes = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "legendre-basis-plan",
                "mode_count": count,
                "node_rule": node_rule,
                "maximum_construction_bytes": maximum,
            }
        )

    def resized(self, mode_count: int, /) -> "LegendreBasisPlan":
        return LegendreBasisPlan(
            mode_count,
            node_rule=self.node_rule,
            maximum_construction_bytes=self.maximum_construction_bytes,
        )

    def prepare(
        self,
        domain: AxisDomain,
        /,
        *,
        precision: SpectralPrecisionPolicy,
    ) -> PreparedSpectralAxis:
        lower, upper = _finite_domain(
            domain,
            periodic=False,
            basis="Legendre",
        )
        lower_value = float(np.asarray(lower))
        upper_value = float(np.asarray(upper))
        count = self.mode_count
        itemsize = np.dtype(precision.coefficient_dtype).itemsize
        estimate = 5 * count * count * itemsize
        if estimate > self.maximum_construction_bytes:
            raise ValueError("Legendre transform exceeds maximum_construction_bytes.")
        rule = legendre_rule_data(count, self.node_rule)
        reference_nodes = np.asarray(rule.nodes, dtype=float)
        reference_weights = np.asarray(rule.weights, dtype=float)
        half = 0.5 * (upper_value - lower_value)
        midpoint = 0.5 * (upper_value + lower_value)
        nodes = midpoint + half * reference_nodes
        weights = half * reference_weights
        standard = np.asarray(
            standard_vandermonde("legendre", reference_nodes, count - 1)
        )
        normalizers = np.sqrt(
            (2.0 * np.arange(count, dtype=float) + 1.0) / (upper_value - lower_value)
        )
        synthesis = standard * normalizers[None, :]
        analysis = _analysis_from_synthesis(
            synthesis,
            precision.coefficient_dtype,
        )
        derivative_standard = np.asarray(
            standard_derivative_matrix(
                "legendre",
                count,
                dtype=precision.physical_dtype,
            )
        )
        derivative = (
            derivative_standard
            * normalizers[None, :]
            / normalizers[:, None]
            * (2.0 / (upper_value - lower_value))
        )
        modal = ModalTransform(
            analysis,
            synthesis,
            weights,
            mode_ids=tuple(f"legendre:{degree}" for degree in range(count)),
        )
        execution = DenseLinearTransform(
            np.asarray(analysis, dtype=precision.coefficient_dtype),
            np.asarray(synthesis, dtype=precision.coefficient_dtype),
            transform_id=modal.transform_id,
        )
        return PreparedSpectralAxis(
            self,
            nodes,
            reference_nodes,
            weights,
            domain,
            SpectralModeLayout(
                "legendre",
                np.arange(count),
                mode_ids=tuple(f"legendre:{degree}" for degree in range(count)),
            ),
            execution,
            precision,
            lower_endpoint_included=self.node_rule in ("radau", "lobatto"),
            upper_endpoint_included=self.node_rule == "lobatto",
            derivative_matrix=derivative,
            modal_transform=modal,
        )


__all__ = [
    "AbstractSpectralBasisPlan",
    "ChebyshevBasisPlan",
    "CosineBasisPlan",
    "FourierBasisPlan",
    "LegendreBasisPlan",
    "PreparedSpectralAxis",
    "SineBasisPlan",
    "SpectralBasisFamily",
    "SpectralBoundaryKind",
    "SpectralModeLayout",
]
