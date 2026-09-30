#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, assert_never, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ..discretization import StructuredCochainBridge
from ..discretization._cell_de_rham import AbstractCellDeRhamComplex
from ..exterior._complex import AbstractDeRhamComplex
from ..linalg import (
    AbstractVectorSpace,
    ArraySpace,
    DenseLinearOperator,
    DifferentiationPolicy,
    eigen as eigen_linalg,
    FailurePolicy,
    FunctionLinearOperator,
    GMRES,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
    PropertyEvidence,
    solve,
    SparseLU,
    TolerancePolicy,
)
from ..sparse import compile_sparse_jacobian, SparseColoring, SparseDerivativePlan
from ..typing import parse
from ._maxwell import (
    _apply_hodge_metric,
    AbstractMaxwellFrequencyResponse,
    AbstractPreparedMaxwellConstitutive,
    CompatibleMaxwellState,
    MaxwellCochainLayout,
    PreparedCompatibleMaxwell,
)
from ._maxwell_boundaries import MaxwellBoundaryPlan
from ._maxwell_pml import MaxwellCPMLPlan, PreparedMaxwellCPML, PreparedMaxwellCPMLTerm
from ._maxwell_sources import MaxwellSourceForcing


FrequencyMaxwellSolveMethod: TypeAlias = Literal["krylov", "direct"]


def _paired_matrix(metric: Array | AbstractVectorSpace, matrix: Array, /) -> Array:
    if isinstance(metric, AbstractVectorSpace):

        def pair(column: Array) -> Array:
            return _apply_hodge_metric(metric, column)

        return jax.vmap(pair, in_axes=1, out_axes=1)(matrix)
    return metric[:, None] * matrix if metric.ndim == 1 else metric @ matrix


def _verified_dense_operator(
    matrix: Array,
    name: str,
    /,
    *,
    positive_definite: bool = False,
) -> DenseLinearOperator:
    host = np.asarray(matrix)
    tolerance = (
        64.0
        * max(host.shape[0], 1)
        * np.finfo(host.real.dtype).eps
        * max(1.0, float(np.linalg.norm(host)))
    )
    if not np.allclose(host, host.conj().T, rtol=1e-10, atol=tolerance):
        raise ValueError(f"{name} must be Hermitian.")
    if (
        positive_definite
        and np.linalg.eigvalsh(0.5 * (host + host.conj().T))[0] <= tolerance
    ):
        raise ValueError(f"{name} must be positive definite.")
    evidence: dict[str, PropertyEvidence] = {"self_adjoint": "verified"}
    if positive_definite:
        evidence["positive_definite"] = "verified"
    return DenseLinearOperator(
        matrix,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=positive_definite,
            positive_semidefinite=not positive_definite,
            evidence=evidence,
        ),
        operator_id=canonical_fingerprint(
            {
                "kind": "verified-maxwell-eigen-operator",
                "name": name,
                "matrix": array_tree_fingerprint(matrix),
            }
        ),
    )


class FrequencyMaxwellSolveResult(StrictModule):
    electric: Array
    residual_norm: Array
    converged: Array
    iterations: Array
    status: Array
    diagnostics: Any


class FrequencyMaxwellEigenResult(StrictModule):
    angular_frequencies: Array
    modes: Array
    residuals: Array
    status: Array
    diagnostics: eigen_linalg.EigenSolveDiagnostics
    result_id: str = eqx.field(static=True)


class FrequencyMaxwellPowerLedger(StrictModule):
    """Time-averaged ``exp(-iωt)`` power balance of one frequency-domain field.

    ``source_power = -½Re⟨E, J⟩`` is delivered by the impressed current,
    ``electric_material`` and ``magnetic_material`` are ``½ω Im⟨E, ε(ω)E⟩`` and
    ``½ω Im⟨H, μ(ω)H⟩``, ``absorbed_power`` is the work of the equivalent
    stretched-coordinate currents, and ``boundary_power = ½Re⟨E, Y E⟩`` the
    impedance-boundary dissipation. ``residual`` is source minus sinks; it
    vanishes to the solve residual when perfect-conductor rows carry ``E = 0``.
    """

    source_power: Array
    electric_material: Array
    magnetic_material: Array
    absorbed_power: Array
    boundary_power: Array
    residual: Array
    relative_residual: Array


class _FrequencyStretching(StrictModule):
    """Per-axis inverse CFS stretching ``1/s = 1/(κ + σ/(α − iω))`` on cochains."""

    bridge: StructuredCochainBridge
    cpml: PreparedMaxwellCPML
    electric_inverse: Array
    magnetic_inverse: Array

    def __init__(
        self,
        bridge: StructuredCochainBridge,
        layout: MaxwellCochainLayout,
        plan: MaxwellCPMLPlan,
        omega: Array,
        wave_speed: Array,
        /,
    ) -> None:
        cpml = plan.prepare(bridge, layout, wave_speed)
        self.bridge = bridge
        self.cpml = cpml
        self.electric_inverse = self._inverse(
            cpml.electric_terms, layout.electric_count, bridge.dimension, omega
        )
        self.magnetic_inverse = self._inverse(
            cpml.magnetic_terms, layout.magnetic_count, bridge.dimension, omega
        )

    @staticmethod
    def _inverse(
        terms: tuple[PreparedMaxwellCPMLTerm, ...],
        size: int,
        dimension: int,
        omega: Array,
        /,
    ) -> Array:
        inverse = jnp.ones((dimension, size), dtype=jnp.complex128)
        for term in terms:
            stretch = term.kappa + term.sigma / (term.alpha - 1j * omega)
            inverse = inverse.at[term.axis, term.indices].set(1.0 / stretch)
        return inverse

    def magnetic_curl(self, degree: int, electric: Array, /) -> Array:
        return sum(
            (
                self.magnetic_inverse[axis]
                * self.bridge.directional_exterior_derivative(degree, electric, axis)
                for axis in range(self.bridge.dimension)
            ),
            start=jnp.zeros(self.magnetic_inverse.shape[1], dtype=jnp.complex128),
        )

    def electric_curl(self, degree: int, magnetic: Array, /) -> Array:
        return sum(
            (
                self.electric_inverse[axis]
                * self.bridge.directional_codifferential(degree, magnetic, axis)
                for axis in range(self.bridge.dimension)
            ),
            start=jnp.zeros(self.electric_inverse.shape[1], dtype=jnp.complex128),
        )

    def equivalent_currents(
        self,
        electric_degree: int,
        magnetic_degree: int,
        electric: Array,
        magnetic: Array,
        /,
    ) -> tuple[Array, Array]:
        """``J_s = -Σ(1/s - 1)δₐH`` and ``M_s = Σ(1/s - 1)dₐE`` of the layer."""
        electric_current = jnp.zeros(self.electric_inverse.shape[1], dtype=jnp.complex128)
        magnetic_current = jnp.zeros(self.magnetic_inverse.shape[1], dtype=jnp.complex128)
        for axis in range(self.bridge.dimension):
            electric_current = electric_current - (
                self.electric_inverse[axis] - 1.0
            ) * self.bridge.directional_codifferential(magnetic_degree, magnetic, axis)
            magnetic_current = magnetic_current + (
                self.magnetic_inverse[axis] - 1.0
            ) * self.bridge.directional_exterior_derivative(
                electric_degree, electric, axis
            )
        return electric_current, magnetic_current


class FrequencyMaxwellOperator(StrictModule):
    """Matrix-free ``exp(-iωt)`` curl-curl operator on compatible cochains.

    The constitutive law contributes its continuous ``frequency_response(ω)``;
    ``stretching`` applies CFS complex coordinate stretching with the same graded
    ``(σ, κ, α)`` profile as the time-domain `MaxwellCPMLPlan`. ``boundaries``
    are the time-domain `MaxwellBoundaryPlan` traces (domain boundary or explicit
    ``support``): perfect-conductor entries become identity rows with ``E = 0``,
    perfect-magnetic-conductor entries zero ``H``, and impedance entries add the
    surface conduction current ``Y E``. Without boundaries the natural trace of
    the absolute cochain complex applies.
    """

    cochain: AbstractDeRhamComplex
    constitutive: AbstractPreparedMaxwellConstitutive
    response: AbstractMaxwellFrequencyResponse
    stretching: _FrequencyStretching | None
    angular_frequency: Array
    layout: MaxwellCochainLayout
    conductor: Array
    magnetic_wall: Array
    admittance: Array
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: AbstractCellDeRhamComplex,
        layout: MaxwellCochainLayout,
        constitutive: AbstractPreparedMaxwellConstitutive,
        angular_frequency: ArrayLike,
        /,
        *,
        stretching: MaxwellCPMLPlan | None = None,
        boundaries: Sequence[MaxwellBoundaryPlan] = (),
    ) -> None:
        bridge = (
            discretization
            if isinstance(discretization, StructuredCochainBridge)
            else None
        )
        if bridge is not None:
            cochain = bridge.cochain
        elif isinstance(discretization, AbstractCellDeRhamComplex):
            cochain = discretization
        else:
            raise TypeError("Frequency Maxwell requires an AbstractCellDeRhamComplex.")
        if not isinstance(layout, MaxwellCochainLayout):
            raise TypeError("Frequency Maxwell requires a MaxwellCochainLayout.")
        if not isinstance(constitutive, AbstractPreparedMaxwellConstitutive):
            raise TypeError("constitutive must be prepared Maxwell material data.")
        if not constitutive.capabilities.frequency_domain:
            raise ValueError("Constitutive law does not support frequency-domain use.")
        if constitutive.layout_id != layout.layout_id:
            raise ValueError("Frequency material and Maxwell layout do not match.")
        frequency = jnp.asarray(angular_frequency)
        if jnp.iscomplexobj(frequency):
            raise TypeError("angular_frequency must be real.")
        if (
            frequency.shape != ()
            or not jnp.issubdtype(frequency.dtype, jnp.inexact)
            or not bool(jnp.isfinite(frequency))
            or not bool(frequency > 0.0)
        ):
            raise ValueError("angular_frequency must be a finite positive scalar.")
        if stretching is not None:
            if not isinstance(stretching, MaxwellCPMLPlan):
                raise TypeError("stretching must be MaxwellCPMLPlan or None.")
            if bridge is None:
                raise ValueError(
                    "Coordinate stretching requires a StructuredCochainBridge."
                )
        boundary_plans = tuple(boundaries)
        if not all(isinstance(plan, MaxwellBoundaryPlan) for plan in boundary_plans):
            raise TypeError("boundaries must contain MaxwellBoundaryPlan values.")
        prepared_boundaries = tuple(
            plan.prepare(discretization, layout) for plan in boundary_plans
        )
        conductor = jnp.zeros((layout.electric_count,), dtype=jnp.bool_)
        magnetic_wall = jnp.zeros((layout.magnetic_count,), dtype=jnp.bool_)
        admittance = jnp.zeros((layout.electric_count,), dtype=jnp.complex128)
        for boundary in prepared_boundaries:
            match boundary.kind:
                case "pec":
                    conductor = conductor | boundary.electric_boundary
                case "pmc":
                    magnetic_wall = magnetic_wall | boundary.magnetic_boundary
                case "impedance":
                    admittance = admittance + boundary.impedance_current(
                        jnp.ones((layout.electric_count,), dtype=jnp.complex128)
                    )
                case _:
                    assert_never(boundary.kind)
        self.cochain = cochain
        self.constitutive = constitutive
        self.response = constitutive.frequency_response(frequency)
        self.stretching = (
            None
            if stretching is None or bridge is None
            else _FrequencyStretching(
                bridge,
                layout,
                stretching,
                frequency,
                constitutive.wave_speed_bound(),
            )
        )
        self.layout = layout
        self.angular_frequency = frequency
        self.conductor = conductor
        self.magnetic_wall = magnetic_wall
        self.admittance = admittance
        self.operator_id = canonical_fingerprint(
            {
                "kind": "frequency-maxwell-operator",
                "cochain": cochain.realization_id,
                "layout": layout.layout_id,
                "constitutive": constitutive.prepared_id,
                "angular_frequency": float(np.asarray(frequency)),
                "stretching": None
                if self.stretching is None
                else self.stretching.cpml.prepared_id,
                "boundaries": [boundary.prepared_id for boundary in prepared_boundaries],
            }
        )

    @property
    def size(self) -> int:
        return self.layout.electric_count

    def _stretched_curl(self, electric: Array, /) -> Array:
        if self.stretching is None:
            return self.cochain.exterior_derivative(
                self.layout.electric_degree, electric, boundary="absolute"
            )
        return self.stretching.magnetic_curl(self.layout.electric_degree, electric)

    def _stretched_curl_adjoint(self, magnetic: Array, /) -> Array:
        if self.stretching is None:
            return self.cochain.codifferential(
                self.layout.magnetic_degree, magnetic, boundary="absolute"
            )
        return self.stretching.electric_curl(self.layout.magnetic_degree, magnetic)

    def _magnetic(self, flux: Array, /) -> Array:
        """``H = μ(ω)⁻¹ B`` with perfect-magnetic-conductor entries held at zero."""
        return jnp.where(self.magnetic_wall, 0, self.response.magnetic_field(flux))

    def _curl_curl(self, electric: Array, /) -> Array:
        return self._stretched_curl_adjoint(
            self._magnetic(self._stretched_curl(electric))
        )

    def mv(self, electric: ArrayLike, /) -> Array:
        electric_ = jnp.asarray(electric)
        if electric_.shape != (self.size,):
            raise ValueError("Frequency Maxwell electric field has wrong shape.")
        # exp(-i*omega*t): curl_s(mu^-1 curl_s E) - omega^2 eps E - i omega Y E = i omega J.
        free = jnp.where(self.conductor, 0, electric_)
        applied = (
            self._curl_curl(free)
            - self.angular_frequency**2 * self.response.electric_displacement(free)
            - 1j * self.angular_frequency * self.admittance * free
        )
        # Perfect-conductor rows are the identity, so E = source there.
        return jnp.where(self.conductor, electric_, applied)

    def defect(
        self, electric: ArrayLike, source: ArrayLike, /
    ) -> MaxwellHarmonicDefectReport:
        electric_, source_ = jnp.asarray(electric), jnp.asarray(source)
        if electric_.shape != (self.size,) or source_.shape != (self.size,):
            raise ValueError("Frequency Maxwell defect vectors have the wrong shape.")
        applied = self.mv(electric_)
        residual = applied - source_
        metric = self.cochain.hilbert_complex(boundary="absolute").space(
            self.layout.electric_degree
        )
        absolute = _hodge_norm(metric, residual)
        denominator = _hodge_norm(metric, applied) + _hodge_norm(metric, source_)
        relative = absolute / jnp.maximum(denominator, jnp.finfo(absolute.dtype).tiny)
        zero = jnp.asarray(0.0, dtype=absolute.dtype)
        return MaxwellHarmonicDefectReport(
            absolute,
            relative,
            absolute,
            zero,
            zero,
            zero,
            jnp.asarray(True),
            "exp(-i*omega*t)",
        )

    def adjoint_mv(self, electric: ArrayLike, /) -> Array:
        electric_ = jnp.asarray(electric)
        _, pullback = jax.vjp(self.mv, jnp.zeros_like(electric_))
        return pullback(electric_)[0]

    def linear_operator(self, /, *, adjoint: bool = False) -> FunctionLinearOperator:
        dtype = jnp.result_type(self.angular_frequency.dtype, jnp.complex64)
        space = ArraySpace((self.size,), dtype=dtype)
        action = self.adjoint_mv if adjoint else self.mv
        return FunctionLinearOperator(
            action,
            source=space,
            target=space,
            operator_id=canonical_fingerprint(
                {
                    "kind": "frequency-maxwell-native-operator",
                    "operator": self.operator_id,
                    "adjoint": bool(adjoint),
                }
            ),
        )

    def power_ledger(
        self, electric: ArrayLike, source: ArrayLike, /
    ) -> FrequencyMaxwellPowerLedger:
        """Power balance for a field solving ``mv(E) = source`` with ``source = iωJ``.

        Perfect-conductor rows carry prescribed values rather than currents and are
        excluded from the source, material, and boundary pairings.
        """
        electric_ = jnp.asarray(electric).astype(jnp.complex128)
        source_ = jnp.asarray(source).astype(jnp.complex128)
        if electric_.shape != (self.size,) or source_.shape != (self.size,):
            raise ValueError("Frequency Maxwell ledger vectors have the wrong shape.")
        omega = self.angular_frequency
        electric_star = self.cochain.hilbert_complex(boundary="absolute").space(
            self.layout.electric_degree
        )
        magnetic_star = self.cochain.hilbert_complex(boundary="absolute").space(
            self.layout.magnetic_degree
        )
        free = jnp.where(self.conductor, 0, electric_)
        current = jnp.where(self.conductor, 0, source_ / (1j * omega))
        flux = self._stretched_curl(electric_) / (1j * omega)
        magnetic = self._magnetic(flux)
        source_power = -0.5 * jnp.real(
            jnp.vdot(free, _apply_hodge_metric(electric_star, current))
        )
        electric_material = (
            0.5
            * omega
            * jnp.imag(
                jnp.vdot(
                    free,
                    _apply_hodge_metric(
                        electric_star, self.response.electric_displacement(free)
                    ),
                )
            )
        )
        magnetic_material = (
            0.5
            * omega
            * jnp.imag(jnp.vdot(magnetic, _apply_hodge_metric(magnetic_star, flux)))
        )
        if self.stretching is None:
            absorbed = jnp.zeros_like(source_power)
        else:
            electric_current, magnetic_current = self.stretching.equivalent_currents(
                self.layout.electric_degree,
                self.layout.magnetic_degree,
                electric_,
                magnetic,
            )
            absorbed = 0.5 * jnp.real(
                jnp.vdot(electric_, _apply_hodge_metric(electric_star, electric_current))
                + jnp.vdot(magnetic, _apply_hodge_metric(magnetic_star, magnetic_current))
            )
        boundary_power = 0.5 * jnp.real(
            jnp.vdot(free, _apply_hodge_metric(electric_star, self.admittance * free))
        )
        residual = (
            source_power
            - electric_material
            - magnetic_material
            - absorbed
            - boundary_power
        )
        scale = (
            jnp.abs(source_power)
            + jnp.abs(electric_material)
            + jnp.abs(magnetic_material)
            + jnp.abs(absorbed)
            + jnp.abs(boundary_power)
        )
        return FrequencyMaxwellPowerLedger(
            source_power,
            electric_material,
            magnetic_material,
            absorbed,
            boundary_power,
            residual,
            jnp.abs(residual) / jnp.maximum(scale, jnp.finfo(scale.dtype).tiny),
        )

    def sparse_coloring(self, /) -> SparseColoring:
        """Structural coloring of the traced operator pattern.

        The pattern is fixed by the layout, stretching, and boundary masks;
        frequency enters only coefficient values, so the coloring is reusable at
        every frequency.
        """
        return self._sparse_plan(None).coloring

    def _sparse_plan(self, coloring: SparseColoring | None, /) -> SparseDerivativePlan:
        space = ArraySpace((self.size,), dtype=jnp.complex128)
        return compile_sparse_jacobian(
            lambda electric, operator: operator.mv(electric),
            jnp.zeros((self.size,), dtype=jnp.complex128),
            source=space,
            target=space,
            sample_args=self,
            structure=coloring,
            complex_semantics="holomorphic",
        )

    def solve(
        self,
        source: ArrayLike,
        /,
        *,
        method: FrequencyMaxwellSolveMethod = "krylov",
        tolerance: float = 1e-9,
        restart: int = 40,
        maxiter: int = 400,
        policy: LinearSolvePolicy | None = None,
        coloring: SparseColoring | None = None,
    ) -> FrequencyMaxwellSolveResult:
        """Solve ``mv(E) = source`` natively.

        ``"krylov"`` applies the matrix-free operator in restarted GMRES;
        ``"direct"`` assembles the exact sparse operator by structural coloring
        (``coloring`` reuses a pattern from `sparse_coloring`) and factors it with
        native sparse LU. ``policy`` replaces the default policy of either route.
        """
        method = parse(method, FrequencyMaxwellSolveMethod, "method")
        source_ = jnp.asarray(source, dtype=jnp.result_type(source, jnp.complex64))
        if source_.shape != (self.size,):
            raise ValueError("Frequency Maxwell source has wrong shape.")
        if policy is not None and not isinstance(policy, LinearSolvePolicy):
            raise TypeError("Frequency Maxwell solve policy must be LinearSolvePolicy.")
        match method:
            case "krylov":
                selected = (
                    LinearSolvePolicy(
                        GMRES(
                            restart=int(restart),
                            stagnation_iterations=int(restart),
                        ),
                        tolerance=TolerancePolicy(
                            relative=float(tolerance),
                            absolute=0.0,
                            max_steps=int(maxiter),
                        ),
                        failure=FailurePolicy("status"),
                    )
                    if policy is None
                    else policy
                )
                operator = self.linear_operator()
            case "direct":
                selected = (
                    LinearSolvePolicy(
                        SparseLU(),
                        differentiation=DifferentiationPolicy("none"),
                        failure=FailurePolicy("status"),
                    )
                    if policy is None
                    else policy
                )
                operator = self._sparse_plan(coloring).operator(
                    jnp.zeros((self.size,), dtype=jnp.complex128)
                )
            case _:
                assert_never(method)
        result = solve(
            LinearSystem(operator),
            operator.target.unflatten(source_),
            policy=selected,
        )
        solution = operator.source.flatten(result.value)
        residual = self.mv(solution) - source_
        residual_norm = jnp.sqrt(jnp.real(jnp.vdot(residual, residual)))
        return FrequencyMaxwellSolveResult(
            solution,
            residual_norm,
            result.successful,
            jnp.max(result.diagnostics.iterations),
            result.status,
            result.diagnostics,
        )

    def adjoint_solve(
        self,
        cotangent: ArrayLike,
        /,
        *,
        tolerance: float = 1e-9,
        restart: int = 40,
        maxiter: int = 400,
        policy: LinearSolvePolicy | None = None,
    ) -> FrequencyMaxwellSolveResult:
        cotangent_ = jnp.asarray(
            cotangent, dtype=jnp.result_type(cotangent, jnp.complex64)
        )
        if cotangent_.shape != (self.size,):
            raise ValueError("Frequency Maxwell adjoint source has wrong shape.")
        selected = (
            LinearSolvePolicy(
                GMRES(
                    restart=int(restart),
                    stagnation_iterations=int(restart),
                ),
                tolerance=TolerancePolicy(
                    relative=float(tolerance),
                    absolute=0.0,
                    max_steps=int(maxiter),
                ),
                failure=FailurePolicy("status"),
            )
            if policy is None
            else policy
        )
        if not isinstance(selected, LinearSolvePolicy):
            raise TypeError("Frequency Maxwell solve policy must be LinearSolvePolicy.")
        operator = self.linear_operator(adjoint=True)
        result = solve(
            LinearSystem(operator),
            operator.target.unflatten(cotangent_),
            policy=selected,
        )
        solution = operator.source.flatten(result.value)
        residual = self.adjoint_mv(solution) - cotangent_
        residual_norm = jnp.sqrt(jnp.real(jnp.vdot(residual, residual)))
        return FrequencyMaxwellSolveResult(
            solution,
            residual_norm,
            result.successful,
            jnp.max(result.diagnostics.iterations),
            result.status,
            result.diagnostics,
        )

    def materialize(self, /, *, maximum_dofs: int = 4096) -> Array:
        if self.size > int(maximum_dofs):
            raise ValueError("Frequency Maxwell materialization exceeds maximum_dofs.")
        basis = jnp.eye(self.size, dtype=jnp.complex128)
        return jax.vmap(self.mv, in_axes=1, out_axes=1)(basis)

    def eigensystem(
        self,
        mode_count: int,
        /,
        *,
        maximum_dofs: int = 4096,
    ) -> FrequencyMaxwellEigenResult:
        """Hermitian generalized eigenpairs of a lossless nondispersive operator."""
        count = int(mode_count)
        if count <= 0 or count > self.size:
            raise ValueError("mode_count is outside the operator dimension.")
        if self.stretching is not None:
            raise ValueError(
                "Complex coordinate stretching makes the Maxwell pencil non-Hermitian."
            )
        if self.response.dispersive or not self.response.lossless:
            raise ValueError(
                "The Hermitian Maxwell eigen path requires a lossless nondispersive response."
            )
        if bool(jnp.any(self.conductor)) or bool(jnp.any(self.admittance != 0.0)):
            raise ValueError(
                "The Hermitian Maxwell eigen path does not accept perfect-conductor "
                "or impedance boundaries."
            )
        if self.size > int(maximum_dofs):
            raise ValueError("Frequency Maxwell materialization exceeds maximum_dofs.")
        identity = jnp.eye(self.size, dtype=jnp.complex128)
        stiffness = jax.vmap(self._curl_curl, in_axes=1, out_axes=1)(identity)
        mass = jax.vmap(self.response.electric_displacement, in_axes=1, out_axes=1)(
            identity
        )
        hodge = self.cochain.hilbert_complex(boundary="absolute").space(
            self.layout.electric_degree
        )
        paired_stiffness = _paired_matrix(hodge, stiffness)
        paired_mass = _paired_matrix(hodge, mass)
        problem = eigen_linalg.GeneralizedEigenproblem(
            _verified_dense_operator(
                paired_stiffness,
                "paired Maxwell stiffness",
            ),
            _verified_dense_operator(
                paired_mass,
                "paired Maxwell mass",
                positive_definite=True,
            ),
        )
        solved = eigen_linalg.eigensolve(
            problem,
            policy=eigen_linalg.EigenSolvePolicy(
                eigen_linalg.DenseEigh(),
                count=count,
                which="smallest-algebraic",
            ),
        )
        values = jnp.real(solved.eigenvalues)
        return FrequencyMaxwellEigenResult(
            angular_frequencies=jnp.sqrt(jnp.maximum(values, 0.0)),
            modes=solved.eigenvectors,
            residuals=solved.diagnostics.residual_norms,
            status=solved.status,
            diagnostics=solved.diagnostics,
            result_id=canonical_fingerprint(
                {
                    "kind": "frequency-maxwell-eigensystem",
                    "operator": self.operator_id,
                    "eigen_plan": solved.provenance.plan_id,
                }
            ),
        )


class MaxwellHarmonicDefectReport(StrictModule):
    absolute_norm: Array
    relative_norm: Array
    electric_defect: Array
    magnetic_defect: Array
    charge_defect: Array
    auxiliary_defect: Array
    eligible: Array
    convention: str = eqx.field(static=True)


class MaxwellHarmonicSource(StrictModule):
    electric_current: Array
    magnetic_current: Array
    convention: str = eqx.field(static=True)

    def __init__(
        self,
        electric_current: ArrayLike,
        magnetic_current: ArrayLike,
        /,
        *,
        convention: str = "exp(-i*omega*t)",
    ) -> None:
        if convention != "exp(-i*omega*t)":
            raise ValueError(
                "Maxwell harmonic source convention must be exp(-i*omega*t)."
            )
        self.electric_current = jnp.asarray(electric_current)
        self.magnetic_current = jnp.asarray(magnetic_current)
        self.convention = convention


def _hodge_norm(metric: Array | AbstractVectorSpace, value: Array, /) -> Array:
    paired = _apply_hodge_metric(metric, value)
    return jnp.sqrt(jnp.maximum(jnp.real(jnp.vdot(value, paired)), 0.0))


def _tree_norm(tree: Any, /) -> Array:
    leaves = tuple(
        leaf for leaf in jax.tree_util.tree_leaves(tree) if isinstance(leaf, jax.Array)
    )
    if not leaves:
        return jnp.asarray(0.0)
    return jnp.sqrt(sum(jnp.real(jnp.vdot(leaf, leaf)) for leaf in leaves))


def compatible_maxwell_harmonic_defect(
    runtime: PreparedCompatibleMaxwell,
    state_phasor: CompatibleMaxwellState,
    source_phasor: MaxwellHarmonicSource,
    angular_frequency: ArrayLike,
    step_size: ArrayLike,
    /,
) -> MaxwellHarmonicDefectReport:
    """Defect of the complete affine leapfrog map for exp(-i*omega*t)."""

    if not isinstance(runtime, PreparedCompatibleMaxwell):
        raise TypeError("Complete Maxwell harmonic defect requires a prepared runtime.")
    if not runtime.capabilities.linear_time_invariant or runtime.capabilities.nonlinear:
        raise ValueError(
            "Complete harmonic defect requires linear time-invariant dynamics."
        )
    state = runtime._state(state_phasor)
    if not isinstance(source_phasor, MaxwellHarmonicSource):
        raise TypeError("source_phasor must be MaxwellHarmonicSource.")
    if source_phasor.electric_current.shape != (runtime.layout.electric_count,):
        raise ValueError("Harmonic electric source has the wrong retained shape.")
    if source_phasor.magnetic_current.shape != (runtime.layout.magnetic_count,):
        raise ValueError("Harmonic magnetic source has the wrong retained shape.")
    omega, dt = jnp.asarray(angular_frequency), runtime._step_size(step_size)
    if omega.shape != () or jnp.iscomplexobj(omega):
        raise ValueError("Harmonic angular frequency must be a real scalar.")
    phase_half = jnp.exp(-0.5j * omega * dt)
    phase_full = phase_half**2
    base = MaxwellSourceForcing(
        source_phasor.electric_current,
        source_phasor.magnetic_current,
    )
    samples = (
        base,
        MaxwellSourceForcing(
            phase_half * source_phasor.electric_current,
            phase_half * source_phasor.magnetic_current,
        ),
        MaxwellSourceForcing(
            phase_full * source_phasor.electric_current,
            phase_full * source_phasor.magnetic_current,
        ),
    )
    coefficients = None if runtime.pml is None else runtime.pml.bind_coefficients(dt)
    stepped = runtime._step_core(
        jnp.asarray(0.0),
        state,
        dt,
        None,
        cpml_coefficients=coefficients,
        source_samples=samples,
    )
    target_material = jax.tree_util.tree_map(
        lambda value: phase_full * value,
        state.auxiliary.material,
    )
    target_boundary = jax.tree_util.tree_map(
        lambda value: phase_full * value,
        state.auxiliary.boundary,
    )
    d_defect = (
        stepped.primary.electric_displacement
        - phase_full * state.primary.electric_displacement
    )
    b_defect = stepped.primary.magnetic_flux - phase_full * state.primary.magnetic_flux
    q_defect = stepped.primary.charge - phase_full * state.primary.charge
    material_defect = jax.tree_util.tree_map(
        lambda left, right: left - right,
        stepped.auxiliary.material,
        target_material,
    )
    boundary_defect = jax.tree_util.tree_map(
        lambda left, right: left - right,
        stepped.auxiliary.boundary,
        target_boundary,
    )
    electric_norm = _hodge_norm(
        runtime.cochain.hilbert_complex(boundary="absolute").space(
            runtime.layout.electric_degree
        ),
        d_defect,
    )
    magnetic_norm = _hodge_norm(
        runtime.cochain.hilbert_complex(boundary="absolute").space(
            runtime.layout.magnetic_degree
        ),
        b_defect,
    )
    charge_norm = jnp.linalg.norm(q_defect)
    auxiliary_norm = jnp.sqrt(
        _tree_norm(material_defect) ** 2 + _tree_norm(boundary_defect) ** 2
    )
    absolute = jnp.sqrt(
        electric_norm**2 + magnetic_norm**2 + charge_norm**2 + auxiliary_norm**2
    )
    state_scale = (
        _hodge_norm(
            runtime.cochain.hilbert_complex(boundary="absolute").space(
                runtime.layout.electric_degree
            ),
            stepped.primary.electric_displacement,
        )
        + _hodge_norm(
            runtime.cochain.hilbert_complex(boundary="absolute").space(
                runtime.layout.magnetic_degree
            ),
            stepped.primary.magnetic_flux,
        )
        + jnp.linalg.norm(stepped.primary.charge)
        + _tree_norm(stepped.auxiliary)
        + _hodge_norm(
            runtime.cochain.hilbert_complex(boundary="absolute").space(
                runtime.layout.electric_degree
            ),
            state.primary.electric_displacement,
        )
        + _hodge_norm(
            runtime.cochain.hilbert_complex(boundary="absolute").space(
                runtime.layout.magnetic_degree
            ),
            state.primary.magnetic_flux,
        )
    )
    relative = absolute / jnp.maximum(state_scale, jnp.finfo(absolute.dtype).tiny)
    return MaxwellHarmonicDefectReport(
        absolute,
        relative,
        electric_norm,
        magnetic_norm,
        charge_norm,
        auxiliary_norm,
        jnp.asarray(True),
        source_phasor.convention,
    )


class FrequencyMaxwellAdjointResult(StrictModule):
    solution: Array
    adjoint: Array
    objective: Array
    source_gradient: Array
    primal_result: FrequencyMaxwellSolveResult
    adjoint_result: FrequencyMaxwellSolveResult
    valid: Array


def frequency_maxwell_adjoint(
    operator: FrequencyMaxwellOperator,
    source: ArrayLike,
    objective: Any,
    /,
) -> FrequencyMaxwellAdjointResult:
    if not callable(objective):
        raise TypeError("objective must be callable.")
    solved = operator.solve(source)
    value, pullback = jax.vjp(
        lambda field: jnp.asarray(objective(field)), solved.electric
    )
    if value.shape != () or jnp.iscomplexobj(value):
        raise ValueError("Frequency Maxwell objective must be a real scalar.")
    cotangent = pullback(jnp.asarray(1.0))[0]
    adjoint_result = operator.adjoint_solve(cotangent)
    adjoint = adjoint_result.electric
    valid = (
        solved.converged
        & adjoint_result.converged
        & jnp.isfinite(value)
        & jnp.all(jnp.isfinite(solved.electric))
        & jnp.all(jnp.isfinite(adjoint))
    )
    return FrequencyMaxwellAdjointResult(
        solved.electric,
        adjoint,
        value,
        adjoint,
        solved,
        adjoint_result,
        valid,
    )


def eigenspace_directional_derivative(
    spectrum: eigen_linalg.PreparedSelfAdjointSpectrum,
    selection: eigen_linalg.SpectralSelection,
    perturbation: ArrayLike,
    metric_perturbation: ArrayLike | None = None,
    /,
    *,
    policy: eigen_linalg.SelfAdjointSpectralSubspacePolicy | None = None,
) -> eigen_linalg.SelfAdjointSpectralDerivativeResult:
    """Differentiate an isolated Maxwell eigenspace as a basis-invariant projector."""
    return eigen_linalg.self_adjoint_spectral_projector_derivative(
        spectrum,
        selection,
        perturbation,
        metric_perturbation,
        policy=policy,
    )


__all__ = [
    "FrequencyMaxwellAdjointResult",
    "FrequencyMaxwellEigenResult",
    "FrequencyMaxwellOperator",
    "FrequencyMaxwellPowerLedger",
    "FrequencyMaxwellSolveMethod",
    "FrequencyMaxwellSolveResult",
    "MaxwellHarmonicDefectReport",
    "MaxwellHarmonicSource",
    "compatible_maxwell_harmonic_defect",
    "eigenspace_directional_derivative",
    "frequency_maxwell_adjoint",
]
