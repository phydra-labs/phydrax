#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ..discretization._cell_de_rham import AbstractCellDeRhamComplex
from ..discretization._gram import gram_solve_policy
from ..discretization.fem import FiniteElementDeRhamComplex
from ..exterior._complex import AbstractDeRhamComplex, ComplexBoundary
from ..linalg import (
    AbstractLinearOperator,
    AbstractVectorSpace,
    ArraySpace,
    hodge_laplacian,
    LinearSolvePolicy,
    LinearSystem,
    OperatorPairing,
    plan as plan_linear_solve,
    prepare,
    PreparedLinearSolve,
    solve,
    stiffness_form,
)
from ..linalg._algebra_operators import apply_real_map_componentwise
from ..linalg.eigen import (
    Eigenproblem,
    eigensolve,
    EigenSolvePolicy,
    EigenSolveResult,
    RestartedLanczos,
)
from ..meshing._quality import CellQualityEvaluation, evaluate_cell_quality
from ..typing import checked, parse
from ._maxwell import (
    _apply_hodge_metric,
    AbstractMaxwellConstitutivePlan,
    AbstractMaxwellFrequencyResponse,
    AbstractPreparedMaxwellConstitutive,
    CompatibleMaxwellState,
    InstantaneousMaxwellFrequencyResponse,
    MaxwellAuxiliaryState,
    MaxwellCapabilities,
    MaxwellCochainLayout,
    MaxwellPrimaryState,
)


def _spectral_bound(
    complex_: AbstractCellDeRhamComplex,
    /,
    *,
    boundary: ComplexBoundary,
) -> tuple[Array, EigenSolveResult]:
    """Prepare a sparse Lanczos pencil and a global trace upper certificate.

    Ritz values alone do not bound unseen modes. For a PSD pencil, its complete
    coordinate trace bounds every eigenvalue. Computing that trace uses one
    native Riesz solve per electric coordinate, linear coordinate workspace, and
    no materialization. An explicit caller bound bypasses this admission cost.
    """
    hilbert = complex_.hilbert_complex(boundary=boundary)
    space = hilbert.space(1)
    if not isinstance(space, ArraySpace):
        raise TypeError(
            "Maxwell spectral preparation requires an array coordinate space."
        )
    stiffness = stiffness_form(hilbert, 1)
    # M₁⁻¹K₁ is self-adjoint in the M₁ pairing, not in Euclidean coordinates.
    # This Riesz realization makes native Lanczos apply the actual pencil action.
    pencil_action = hodge_laplacian(hilbert, 1, part="upper")
    size = space.size
    if size < 2:
        raise ValueError(
            "Automatic Maxwell spectral preparation needs at least two electric coordinates."
        )
    result = eigensolve(
        Eigenproblem(
            pencil_action,
            problem_id=f"{complex_.realization_id}:maxwell-pencil:{boundary}",
        ),
        policy=EigenSolvePolicy(
            RestartedLanczos(subspace_dimension=min(size, 32)),
            count=1,
            which="largest-algebraic",
            max_steps=max(100, size),
            key=jax.random.key(0),
        ),
    )

    def diagonal(index: Array, total: Array) -> Array:
        basis = jnp.zeros((size,), dtype=jnp.float64).at[index].set(1.0)
        covector = space.validate(stiffness.mv(basis))
        image = space.flatten(space.inverse_riesz(space.unflatten(covector)))
        return total + jnp.real(image[index])

    trace = jax.lax.fori_loop(0, size, diagonal, jnp.asarray(0.0, dtype=jnp.float64))
    bound = trace * (1.0 + 1.0e-8)
    bound = eqx.error_if(
        bound,
        ~jnp.isfinite(bound)
        | (bound <= 0.0)
        | ~jnp.all(result.converged)
        | (jnp.max(result.eigenvalues) > bound),
        "Maxwell sparse spectral preparation failed its global upper-bound certificate.",
    )
    return bound, result


def _material_coefficient(value: ArrayLike, name: str, /) -> tuple[Array, Array, Array]:
    coefficient = jnp.asarray(value)
    if jnp.iscomplexobj(coefficient):
        raise TypeError(f"{name} must be a real lossless coefficient.")
    coefficient = coefficient.astype(jnp.float64)
    if coefficient.size == 0:
        raise ValueError(f"{name} must have nonempty coefficient support.")
    if coefficient.ndim in (0, 1):
        eigenvalues = coefficient
    elif coefficient.ndim in (2, 3) and coefficient.shape[-2:] == (3, 3):
        coefficient = eqx.error_if(
            coefficient,
            jnp.any(jnp.abs(coefficient - jnp.swapaxes(coefficient, -1, -2)) > 1e-12),
            f"{name} tensors must be symmetric.",
        )
        eigenvalues = jnp.linalg.eigvalsh(coefficient)
    else:
        raise ValueError(
            f"{name} must be scalar, cell scalars, or three-dimensional cell tensors."
        )
    coefficient = eqx.error_if(
        coefficient,
        jnp.any(~jnp.isfinite(coefficient)) | jnp.any(eigenvalues <= 0.0),
        f"{name} must be finite and positive definite.",
    )
    return coefficient, jnp.min(eigenvalues), jnp.max(eigenvalues)


def _prepare_material_solve(
    operator: AbstractLinearOperator,
    policy: LinearSolvePolicy | None,
    identifier: str,
    /,
) -> PreparedLinearSolve:
    selected = gram_solve_policy(operator.source.size) if policy is None else policy
    system = LinearSystem(operator, problem_id=identifier)
    selected_plan = plan_linear_solve(system, selected)
    if selected_plan.backend == "host-sparse":
        raise ValueError(
            "FE Maxwell material solves require a native device LinearSolvePolicy."
        )
    return prepare(system, selected_plan)


@final
class FiniteElementMaxwellConstitutivePlan(AbstractMaxwellConstitutivePlan):
    """Material-weighted FE Gram forms, separate from metric Riesz pairings."""

    permittivity: Array
    inverse_permeability: Array
    solve_policy: LinearSolvePolicy | None
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        permittivity: ArrayLike = 1.0,
        inverse_permeability: ArrayLike = 1.0,
        solve_policy: LinearSolvePolicy | None = None,
        plan_id: str | None = None,
    ) -> None:
        epsilon, _, _ = _material_coefficient(permittivity, "permittivity")
        inverse_mu, _, _ = _material_coefficient(
            inverse_permeability, "inverse_permeability"
        )
        if solve_policy is not None and not isinstance(solve_policy, LinearSolvePolicy):
            raise TypeError("solve_policy must be a LinearSolvePolicy or None.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "finite-element-maxwell-constitutive-plan",
                    "epsilon": array_tree_fingerprint(epsilon),
                    "inverse_mu": array_tree_fingerprint(inverse_mu),
                }
            )
            if plan_id is None
            else str(plan_id)
        )
        if not identifier:
            raise ValueError("plan_id must be nonempty.")
        self.permittivity = epsilon
        self.inverse_permeability = inverse_mu
        self.solve_policy = solve_policy
        self.plan_id = identifier

    def prepare(
        self,
        cochain: AbstractDeRhamComplex,
        layout: MaxwellCochainLayout,
        /,
        *,
        boundary: ComplexBoundary = "absolute",
    ) -> PreparedFiniteElementMaxwellConstitutive:
        if not isinstance(cochain, FiniteElementDeRhamComplex):
            raise TypeError(
                "FE Maxwell material forms require a FiniteElementDeRhamComplex."
            )
        return PreparedFiniteElementMaxwellConstitutive(
            self, cochain, layout, boundary=boundary
        )


@final
class PreparedFiniteElementMaxwellConstitutive(AbstractPreparedMaxwellConstitutive):
    complex: FiniteElementDeRhamComplex
    permittivity: Array
    inverse_permeability: Array
    electric_form: AbstractLinearOperator
    magnetic_form: AbstractLinearOperator
    electric_solve: PreparedLinearSolve | None
    magnetic_solve: PreparedLinearSolve | None
    inverse_permittivity: Array | None
    permeability: Array | None
    electric_space: ArraySpace
    magnetic_space: ArraySpace
    electric_indices: Array
    magnetic_indices: Array
    boundary: ComplexBoundary = eqx.field(static=True)
    material_speed: Array
    capabilities: MaxwellCapabilities
    layout_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        plan: FiniteElementMaxwellConstitutivePlan,
        complex_: FiniteElementDeRhamComplex,
        layout: MaxwellCochainLayout,
        /,
        *,
        boundary: ComplexBoundary = "absolute",
    ) -> None:
        if layout.layout_id != MaxwellCochainLayout(complex_).layout_id:
            raise ValueError("Material layout must belong to the exact FE complex.")
        if (
            complex_.dimension != 3
            or layout.electric_degree != 1
            or layout.magnetic_degree != 2
        ):
            raise ValueError(
                "FE Maxwell material forms require full three-dimensional Maxwell roles."
            )
        boundary_ = parse(boundary, ComplexBoundary, "boundary")
        epsilon, epsilon_min, _ = _material_coefficient(plan.permittivity, "permittivity")
        inverse_mu, _, inverse_mu_max = _material_coefficient(
            plan.inverse_permeability, "inverse_permeability"
        )
        cells = sum(block.cell_count for block in complex_.mesh.blocks)
        if epsilon.shape == (3, 3):
            epsilon = jnp.broadcast_to(epsilon, (cells, 3, 3))
        if inverse_mu.shape == (3, 3):
            inverse_mu = jnp.broadcast_to(inverse_mu, (cells, 3, 3))
        electric = complex_.constitutive_operator(1, epsilon, boundary=boundary_)
        magnetic = complex_.constitutive_operator(2, inverse_mu, boundary=boundary_)
        electric_solve = (
            None
            if epsilon.ndim == 0
            else _prepare_material_solve(
                electric,
                plan.solve_policy,
                f"{plan.plan_id}:{complex_.realization_id}:epsilon:{boundary_}",
            )
        )
        magnetic_solve = (
            None
            if inverse_mu.ndim == 0
            else _prepare_material_solve(
                magnetic,
                plan.solve_policy,
                f"{plan.plan_id}:{complex_.realization_id}:inverse-mu:{boundary_}",
            )
        )
        hilbert = complex_.hilbert_complex(boundary=boundary_)
        electric_space = hilbert.space(1)
        magnetic_space = hilbert.space(2)
        if not isinstance(electric_space, ArraySpace) or not isinstance(
            magnetic_space, ArraySpace
        ):
            raise TypeError("FE material forms require array coordinate spaces.")
        electric_indices = complex_.active_indices(1, boundary=boundary_)
        magnetic_indices = complex_.active_indices(2, boundary=boundary_)
        self.complex = complex_
        self.permittivity = epsilon
        self.inverse_permeability = inverse_mu
        self.electric_form = electric
        self.magnetic_form = magnetic
        self.electric_solve = electric_solve
        self.magnetic_solve = magnetic_solve
        self.inverse_permittivity = 1.0 / epsilon if epsilon.ndim == 0 else None
        self.permeability = 1.0 / inverse_mu if inverse_mu.ndim == 0 else None
        self.electric_space = electric_space
        self.magnetic_space = magnetic_space
        self.electric_indices = electric_indices
        self.magnetic_indices = magnetic_indices
        self.boundary = boundary_
        self.material_speed = jnp.sqrt(inverse_mu_max / epsilon_min)
        self.capabilities = MaxwellCapabilities(
            lossless=True,
            passive=True,
            reversible=True,
            structured_only=False,
            frequency_domain=True,
        )
        self.layout_id = layout.layout_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-fe-maxwell-material",
                "plan": plan.plan_id,
                "complex": complex_.realization_id,
                "layout": layout.layout_id,
                "boundary": boundary_,
                "electric_solve": None
                if electric_solve is None
                else electric_solve.template.plan.plan_id,
                "magnetic_solve": None
                if magnetic_solve is None
                else magnetic_solve.template.plan.plan_id,
            }
        )

    def initialize_state(self, /) -> None:
        return None

    def validate_state(self, state: Any, /) -> None:
        if state is not None:
            raise ValueError("Instantaneous FE material state must be None.")

    def _scalar_map(self, values: Array, scale: Array, indices: Array, /) -> Array:
        if self.boundary == "absolute":
            return scale * values
        return jnp.zeros_like(values).at[indices].set(scale * values[indices])

    def electric_field(self, displacement: Array, state: Any, /) -> Array:
        self.validate_state(state)
        if self.inverse_permittivity is not None:
            return self._scalar_map(
                displacement, self.inverse_permittivity, self.electric_indices
            )
        if self.electric_solve is None:
            raise ValueError(
                "A nonscalar FE electric material must own its prepared solve."
            )
        if jnp.iscomplexobj(displacement):
            return apply_real_map_componentwise(
                lambda values: self.electric_field(values, state), displacement
            )
        rhs = self.electric_space.riesz(displacement[self.electric_indices])
        result = solve(self.electric_solve, rhs)
        active = eqx.error_if(
            self.electric_space.validate(result.value),
            ~result.successful,
            "FE permittivity solve failed.",
        )
        return jnp.zeros_like(displacement).at[self.electric_indices].set(active)

    def electric_displacement(self, electric: Array, state: Any, /) -> Array:
        self.validate_state(state)
        if self.permittivity.ndim == 0:
            return self._scalar_map(electric, self.permittivity, self.electric_indices)
        if jnp.iscomplexobj(electric):
            return apply_real_map_componentwise(
                lambda values: self.electric_displacement(values, state), electric
            )
        covector = self.electric_space.validate(
            self.electric_form.mv(electric[self.electric_indices])
        )
        active = self.electric_space.inverse_riesz(covector)
        return jnp.zeros_like(electric).at[self.electric_indices].set(active)

    def magnetic_field(self, flux: Array, state: Any, /) -> Array:
        self.validate_state(state)
        if self.inverse_permeability.ndim == 0:
            return self._scalar_map(
                flux, self.inverse_permeability, self.magnetic_indices
            )
        if jnp.iscomplexobj(flux):
            return apply_real_map_componentwise(
                lambda values: self.magnetic_field(values, state), flux
            )
        covector = self.magnetic_space.validate(
            self.magnetic_form.mv(flux[self.magnetic_indices])
        )
        active = self.magnetic_space.inverse_riesz(covector)
        return jnp.zeros_like(flux).at[self.magnetic_indices].set(active)

    def magnetic_flux(self, magnetic: Array, state: Any, /) -> Array:
        self.validate_state(state)
        if self.permeability is not None:
            return self._scalar_map(magnetic, self.permeability, self.magnetic_indices)
        if self.magnetic_solve is None:
            raise ValueError(
                "A nonscalar FE magnetic material must own its prepared solve."
            )
        if jnp.iscomplexobj(magnetic):
            return apply_real_map_componentwise(
                lambda values: self.magnetic_flux(values, state), magnetic
            )
        rhs = self.magnetic_space.riesz(magnetic[self.magnetic_indices])
        result = solve(self.magnetic_solve, rhs)
        active = eqx.error_if(
            self.magnetic_space.validate(result.value),
            ~result.successful,
            "FE inverse-permeability solve failed.",
        )
        return jnp.zeros_like(magnetic).at[self.magnetic_indices].set(active)

    def electric_conduction(self, electric: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return jnp.zeros_like(electric)

    def magnetic_conduction(self, magnetic: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return jnp.zeros_like(magnetic)

    def dissipated_power(
        self,
        electric: Array,
        magnetic: Array,
        state: Any,
        electric_star: Array | AbstractVectorSpace,
        magnetic_star: Array | AbstractVectorSpace,
        /,
    ) -> Array:
        del magnetic, electric_star, magnetic_star
        self.validate_state(state)
        return jnp.asarray(0.0, dtype=electric.real.dtype)

    def advance_state(
        self,
        time: Array,
        state: Any,
        displacement: Array,
        magnetic_flux: Array,
        step_size: Array,
        args: Any,
        /,
    ) -> None:
        del time, displacement, magnetic_flux, step_size, args
        self.validate_state(state)

    def energy(
        self,
        displacement: Array,
        magnetic_flux: Array,
        state: Any,
        electric_star: Array | AbstractVectorSpace,
        magnetic_star: Array | AbstractVectorSpace,
        /,
    ) -> Array:
        electric = self.electric_field(displacement, state)
        magnetic = self.magnetic_field(magnetic_flux, state)
        return 0.5 * jnp.real(
            jnp.vdot(electric, _apply_hodge_metric(electric_star, displacement))
            + jnp.vdot(magnetic, _apply_hodge_metric(magnetic_star, magnetic_flux))
        )

    def energy_rate(
        self,
        displacement: Array,
        magnetic_flux: Array,
        displacement_rate: Array,
        magnetic_rate: Array,
        state: Any,
        electric_star: Array | AbstractVectorSpace,
        magnetic_star: Array | AbstractVectorSpace,
        /,
    ) -> Array:
        electric = self.electric_field(displacement, state)
        magnetic = self.magnetic_field(magnetic_flux, state)
        return jnp.real(
            jnp.vdot(electric, _apply_hodge_metric(electric_star, displacement_rate))
            + jnp.vdot(magnetic, _apply_hodge_metric(magnetic_star, magnetic_rate))
        )

    def wave_speed_bound(self, /) -> Array:
        return self.material_speed

    @property
    def auxiliary_degrees(self, /) -> tuple[int, ...]:
        return ()

    def frequency_response(
        self, angular_frequency: ArrayLike, /
    ) -> AbstractMaxwellFrequencyResponse:
        return InstantaneousMaxwellFrequencyResponse(self, angular_frequency)


@final
class UnstructuredMaxwellPlan(StrictModule):
    """Compatible D/B evolution on an arbitrary three-dimensional cochain complex."""

    cochain: AbstractCellDeRhamComplex
    layout: MaxwellCochainLayout
    constitutive: AbstractMaxwellConstitutivePlan
    spectral_upper_bound: float | None = eqx.field(static=True)
    courant_factor: float = eqx.field(static=True)
    boundary: ComplexBoundary = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        complex: AbstractCellDeRhamComplex,
        constitutive: AbstractMaxwellConstitutivePlan,
        /,
        *,
        spectral_upper_bound: float | None = None,
        courant_factor: float,
        boundary: ComplexBoundary = "absolute",
    ) -> None:
        if not isinstance(complex, AbstractCellDeRhamComplex) or complex.dimension != 3:
            raise TypeError(
                "Unstructured Maxwell requires a three-dimensional cell de Rham complex."
            )
        boundary_ = parse(boundary, ComplexBoundary, "boundary")
        bound = None if spectral_upper_bound is None else float(spectral_upper_bound)
        factor = float(courant_factor)
        if bound is not None and (not np.isfinite(bound) or bound <= 0.0):
            raise ValueError("spectral_upper_bound must be finite and positive.")
        if not np.isfinite(factor) or factor <= 0.0 or factor > 1.0:
            raise ValueError("courant_factor must lie in (0, 1].")
        hilbert = complex.hilbert_complex(boundary=boundary_)
        for space in hilbert.spaces:
            if not isinstance(space, ArraySpace):
                raise TypeError(
                    "Unstructured Maxwell requires array-valued cell coordinate spaces."
                )
            pairing = space.pairing
            if isinstance(pairing, OperatorPairing):
                inverse = pairing.prepared_inverse
                if inverse is not None and inverse.plan.backend == "host-sparse":
                    raise ValueError(
                        "Unstructured Maxwell requires native device Hodge solves, not host sparse factors."
                    )
        layout = MaxwellCochainLayout(complex, "full_3d")
        self.cochain = complex
        self.layout = layout
        self.constitutive = constitutive
        self.spectral_upper_bound = bound
        self.courant_factor = factor
        self.boundary = boundary_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "unstructured-maxwell-plan",
                "complex": complex.realization_id,
                "layout": layout.layout_id,
                "constitutive": constitutive.plan_id,
                "spectral_upper_bound": bound,
                "courant_factor": factor,
                "boundary": boundary_,
            }
        )

    def prepare(self, /) -> PreparedUnstructuredMaxwell:
        return PreparedUnstructuredMaxwell(self)


@final
class PreparedUnstructuredMaxwell(StrictModule):
    plan: UnstructuredMaxwellPlan
    constitutive: AbstractPreparedMaxwellConstitutive
    stable_dt: Array
    spectral_evidence: EigenSolveResult | None
    mesh_quality: CellQualityEvaluation | None
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(self, plan: UnstructuredMaxwellPlan, /) -> None:
        if isinstance(plan.constitutive, FiniteElementMaxwellConstitutivePlan):
            constitutive = plan.constitutive.prepare(
                plan.cochain, plan.layout, boundary=plan.boundary
            )
        else:
            constitutive = plan.constitutive.prepare(plan.cochain, plan.layout)
        if (
            not constitutive.capabilities.reversible
            or not constitutive.capabilities.lossless
            or constitutive.capabilities.dispersive
        ):
            raise ValueError(
                "Unstructured Maxwell supports only instantaneous lossless constitutive laws."
            )
        if plan.spectral_upper_bound is None:
            bound, evidence = _spectral_bound(plan.cochain, boundary=plan.boundary)
        else:
            bound = jnp.asarray(plan.spectral_upper_bound, dtype=jnp.float64)
            evidence = None
        speed = constitutive.wave_speed_bound()
        stable_dt = plan.courant_factor * 2.0 / (jnp.sqrt(bound) * speed)
        stable_dt = eqx.error_if(
            stable_dt,
            ~jnp.isfinite(stable_dt) | (stable_dt <= 0.0),
            "Unstructured Maxwell spectral preparation produced an invalid time-step bound.",
        )
        quality = (
            evaluate_cell_quality(plan.cochain.mesh)
            if isinstance(plan.cochain, FiniteElementDeRhamComplex)
            else None
        )
        self.plan = plan
        self.constitutive = constitutive
        self.stable_dt = stable_dt
        self.spectral_evidence = evidence
        self.mesh_quality = quality
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-unstructured-maxwell",
                "plan": plan.plan_id,
                "constitutive": constitutive.prepared_id,
            }
        )

    def initialize(self, /) -> CompatibleMaxwellState:
        counts = self.plan.cochain.cell_counts
        return CompatibleMaxwellState(
            MaxwellPrimaryState(
                jnp.zeros((counts[1],), dtype=jnp.float64),
                jnp.zeros((counts[2],), dtype=jnp.float64),
                jnp.zeros((counts[0],), dtype=jnp.float64),
            ),
            # The unstructured runtime carries no source magnetic currents, so its
            # declared magnetic charge on the tetrahedra stays zero.
            MaxwellAuxiliaryState(
                self.constitutive.initialize_state(),
                None,
                jnp.zeros((counts[3],), dtype=jnp.float64),
                jnp.zeros((counts[3],), dtype=jnp.float64),
            ),
            (),
        )

    def electric_field(self, state: CompatibleMaxwellState, /) -> Array:
        return self.constitutive.electric_field(
            state.primary.electric_displacement,
            state.auxiliary.material,
        )

    def magnetic_field(self, state: CompatibleMaxwellState, /) -> Array:
        return self.constitutive.magnetic_field(
            state.primary.magnetic_flux,
            state.auxiliary.material,
        )

    def step(
        self,
        time: ArrayLike,
        state: CompatibleMaxwellState,
        step_size: ArrayLike,
        /,
        *,
        electric_current: ArrayLike | None = None,
    ) -> CompatibleMaxwellState:
        dt = jnp.asarray(step_size)
        dt = eqx.error_if(
            dt,
            ~jnp.isfinite(dt) | (dt <= 0.0) | (dt > self.stable_dt),
            "Unstructured Maxwell step exceeds its stable bound.",
        )
        half = 0.5 * dt
        electric = self.electric_field(state)
        magnetic_half = (
            state.primary.magnetic_flux
            - half
            * self.plan.cochain.exterior_derivative(
                1, electric, boundary=self.plan.boundary
            )
        )
        magnetic = self.constitutive.magnetic_field(
            magnetic_half, state.auxiliary.material
        )
        current = (
            jnp.zeros_like(state.primary.electric_displacement)
            if electric_current is None
            else jnp.asarray(
                electric_current, dtype=state.primary.electric_displacement.dtype
            )
        )
        if current.shape != state.primary.electric_displacement.shape:
            raise ValueError("Unstructured Maxwell current must be a degree-one cochain.")
        if self.plan.boundary == "relative":
            current = jnp.where(self.plan.cochain.boundary_masks[1], 0.0, current)
        displacement = state.primary.electric_displacement + dt * (
            self.plan.cochain.codifferential(2, magnetic, boundary=self.plan.boundary)
            - current
        )
        electric_new = self.constitutive.electric_field(
            displacement, state.auxiliary.material
        )
        magnetic_new = magnetic_half - half * self.plan.cochain.exterior_derivative(
            1, electric_new, boundary=self.plan.boundary
        )
        del time
        charge = state.primary.charge + dt * self.plan.cochain.codifferential(
            1, current, boundary=self.plan.boundary
        )
        return CompatibleMaxwellState(
            MaxwellPrimaryState(displacement, magnetic_new, charge),
            state.auxiliary,
            state.observations,
        )

    def constraints(self, state: CompatibleMaxwellState, /) -> tuple[Array, Array]:
        return (
            -self.plan.cochain.codifferential(
                1, state.primary.electric_displacement, boundary=self.plan.boundary
            )
            - state.primary.charge,
            self.plan.cochain.exterior_derivative(
                2, state.primary.magnetic_flux, boundary=self.plan.boundary
            ),
        )


__all__ = [
    "FiniteElementMaxwellConstitutivePlan",
    "PreparedFiniteElementMaxwellConstitutive",
    "PreparedUnstructuredMaxwell",
    "UnstructuredMaxwellPlan",
]
