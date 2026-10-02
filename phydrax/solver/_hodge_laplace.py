#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import final, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..exterior._complex import AbstractDeRhamComplex, ComplexBoundary
from ..linalg import (
    HarmonicSubspace,
    HilbertComplex,
    hodge_laplacian,
    mass_form,
    mixed_hodge_laplacian,
)
from ..linalg._certificates import KernelCertificate
from ..linalg._operators import AbstractLinearOperator, FunctionLinearOperator
from ..linalg._policies import (
    LinearSolvePolicy,
    MINRES,
    PCG,
    ProjectedPCG,
    TolerancePolicy,
)
from ..linalg._preconditioning import PreconditioningPolicy
from ..linalg._prepared import PreparedLinearSolve
from ..linalg._problems import LinearSystem
from ..linalg._properties import OperatorProperties
from ..linalg._results import LinearSolveResult
from ..linalg._runtime import prepare, solve
from ..linalg._spaces import BlockSpace
from ..linalg._subspaces import LinearSubspace, NullspacePolicy
from ..linalg.eigen import (
    DenseEigh,
    eigensolve,
    EigenSolveDiagnostics,
    EigenSolvePolicy,
    EigenSolveResult,
    GeneralizedEigenproblem,
    prepare_eigensolve,
)
from ..typing import checked, parse


type HodgeLaplaceFormulation = Literal["mixed", "primal"]


@final
class HodgeLaplaceResult(StrictModule):
    """Solution, mixed variables, and unsuppressed native solve evidence."""

    u: Array
    sigma: Array | None
    p: Array | None
    solve_result: LinearSolveResult
    residual_norm: Array
    harmonic_defect: Array

    @property
    def successful(self) -> Array:
        return self.solve_result.successful


@final
class HodgeLaplacePlan(StrictModule):
    """Prepare a boundary-restricted Hodge solve once for repeated source loads.

    ``harmonic`` is the complete, certified harmonic subspace, including a
    dimension-zero subspace on a contractible relative degree. Sources are
    active-coordinate strong loads. Mixed solves retain their harmonic load in
    ``p``; primal solves reject incompatible loads and fix the minimum-norm gauge.
    """

    complex: HilbertComplex
    harmonic: HarmonicSubspace
    prepared: PreparedLinearSolve
    degree: int = eqx.field(static=True)
    boundary: ComplexBoundary = eqx.field(static=True)
    formulation: HodgeLaplaceFormulation = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        complex: AbstractDeRhamComplex,
        k: int,
        /,
        *,
        boundary: ComplexBoundary,
        formulation: HodgeLaplaceFormulation,
        harmonic: HarmonicSubspace,
        preconditioner: PreconditioningPolicy | None = None,
    ) -> None:
        boundary = parse(boundary, ComplexBoundary, "boundary")
        formulation = parse(formulation, HodgeLaplaceFormulation, "formulation")
        if (
            not isinstance(k, int)
            or isinstance(k, bool)
            or not 0 <= k <= complex.dimension
        ):
            raise ValueError("k must be an integer degree in the complex.")
        hilbert = complex.hilbert_complex(boundary=boundary)
        if harmonic.complex_id != hilbert.complex_id or harmonic.degree != k:
            raise ValueError(
                "Harmonic evidence must belong to this restricted complex and degree."
            )
        if preconditioner is not None and not isinstance(
            preconditioner, PreconditioningPolicy
        ):
            raise TypeError("preconditioner must be a PreconditioningPolicy or None.")
        identifier = canonical_fingerprint(
            {
                "kind": "hodge-laplace-plan",
                "complex": hilbert.complex_id,
                "degree": k,
                "boundary": boundary,
                "formulation": formulation,
            }
        )
        match formulation:
            case "mixed":
                operator = mixed_hodge_laplacian(hilbert, k, harmonic=harmonic)
                problem = LinearSystem(operator, problem_id=identifier)
                method = MINRES()
            case "primal":
                operator = hodge_laplacian(hilbert, k)
                subspace = LinearSubspace(
                    hilbert.space(k),
                    harmonic.basis,
                    orthonormal=True,
                    subspace_id=f"{identifier}:harmonic",
                )
                certificate = KernelCertificate(
                    operator,
                    subspace,
                    complete=True,
                    tolerance=1e-8,
                )
                if harmonic.dimension == 0:
                    if not operator.properties.certifies("positive_semidefinite"):
                        raise ValueError(
                            "An empty Hodge kernel proves positivity only for a certified PSD operator."
                        )
                    base_operator = operator
                    valid = harmonic.valid & certificate.valid
                    operator = FunctionLinearOperator(
                        lambda vector: base_operator.mv(
                            eqx.error_if(
                                vector,
                                ~valid,
                                "Complete empty Hodge kernel evidence is invalid.",
                            )
                        ),
                        source=base_operator.source,
                        target=base_operator.target,
                        properties=OperatorProperties(
                            self_adjoint=True,
                            positive_definite=True,
                            evidence={
                                "self_adjoint": "transformed",
                                "positive_definite": "transformed",
                            },
                        ),
                        operator_id=f"{identifier}:positive-hodge",
                    )
                    problem = LinearSystem(operator, problem_id=identifier)
                    method = PCG()
                else:
                    problem = LinearSystem(
                        operator,
                        nullspace_policy=NullspacePolicy(certificate=certificate),
                        problem_id=identifier,
                    )
                    method = ProjectedPCG()
            case _:
                raise ValueError("Invalid Hodge formulation.")
        prepared = prepare(
            problem,
            LinearSolvePolicy(
                method,
                tolerance=TolerancePolicy(relative=1e-10, absolute=1e-12, max_steps=2000),
                preconditioning=preconditioner,
                require_device_binding=True,
            ),
        )
        self.complex = hilbert
        self.harmonic = harmonic
        self.prepared = prepared
        self.degree = k
        self.boundary = boundary
        self.formulation = formulation
        self.plan_id = identifier

    def solve(self, source: ArrayLike, /) -> HodgeLaplaceResult:
        space = self.complex.space(self.degree)
        load = jnp.asarray(source)
        if load.shape != (space.size,):
            raise ValueError("source must be the active-coordinate strong load.")
        load = eqx.error_if(load, ~self.harmonic.valid, "Harmonic evidence is invalid.")
        source_vector = space.unflatten(load)
        match self.formulation:
            case "primal":
                native = solve(self.prepared, source_vector)
                u = space.flatten(native.value)
                sigma = None
                p = None
                residual = (
                    space.flatten(self.prepared.problem.operator.mv(space.unflatten(u)))
                    - load
                )
            case "mixed":
                block_space = self.prepared.problem.operator.target
                if not isinstance(block_space, BlockSpace):
                    raise TypeError("Mixed Hodge operator must own a BlockSpace.")
                if "u" not in block_space.names:
                    raise ValueError("Mixed Hodge space must declare its solution block.")
                weak_load = space.flatten(space.riesz(source_vector))
                rhs = tuple(
                    member.unflatten(weak_load) if name == "u" else member.zeros()
                    for name, member in zip(
                        block_space.names, block_space.spaces, strict=True
                    )
                )
                native = solve(self.prepared, rhs)
                values = block_space.validate(native.value)
                u_index = block_space.names.index("u")
                u = block_space.spaces[u_index].flatten(values[u_index])
                sigma_index = (
                    block_space.names.index("sigma")
                    if "sigma" in block_space.names
                    else None
                )
                sigma = (
                    block_space.spaces[sigma_index].flatten(values[sigma_index])
                    if sigma_index is not None
                    else None
                )
                p_index = (
                    block_space.names.index("p") if "p" in block_space.names else None
                )
                p = (
                    block_space.spaces[p_index].flatten(values[p_index])
                    if p_index is not None
                    else None
                )
                residual = block_space.flatten(
                    self.prepared.problem.operator.mv(values)
                ) - block_space.flatten(rhs)
            case _:
                raise ValueError("Invalid Hodge formulation.")
        mass_u = space.flatten(space.riesz(space.unflatten(u)))
        defect = jnp.linalg.norm(jnp.conj(self.harmonic.basis.T) @ mass_u)
        return HodgeLaplaceResult(u, sigma, p, native, jnp.linalg.norm(residual), defect)


@final
class MaxwellCavityMaterials(StrictModule):
    """Separate lossless electric and inverse-permeability coordinate Grams.

    Neither constitutive coefficient modifies the metric Hodge. Both operators
    must be certified positive-definite Euclidean-coordinate endomorphisms on
    the relative degree-one and degree-two spaces returned by ``mass_form``.
    """

    electric_mass: AbstractLinearOperator
    magnetic_mass: AbstractLinearOperator

    def __init__(
        self,
        electric_mass: AbstractLinearOperator,
        magnetic_mass: AbstractLinearOperator,
        /,
    ) -> None:
        for operator in (electric_mass, magnetic_mass):
            if not isinstance(operator, AbstractLinearOperator):
                raise TypeError("Material Grams must be linear operators.")
            if not operator.source.compatible(
                operator.target
            ) or not operator.properties.certifies("positive_definite"):
                raise ValueError(
                    "Lossless material Grams must be certified SPD coordinate endomorphisms."
                )
        self.electric_mass = electric_mass
        self.magnetic_mass = magnetic_mass


def _cavity_stiffness(
    hilbert: HilbertComplex, materials: MaxwellCavityMaterials, /
) -> AbstractLinearOperator:
    differential = hilbert.differential(1)
    source = materials.electric_mass.source
    degree_two = hilbert.space(2)

    def action(value: Array) -> Array:
        curl = degree_two.flatten(differential.mv(hilbert.space(1).unflatten(value)))
        weighted = materials.magnetic_mass.target.flatten(
            materials.magnetic_mass.mv(curl)
        )
        transposed = differential.transpose_mv(degree_two.unflatten(jnp.conj(weighted)))
        return jnp.conj(hilbert.space(1).flatten(transposed))

    def transposed(value: Array) -> Array:
        return jnp.conj(action(jnp.conj(value)))

    return FunctionLinearOperator(
        action,
        source=source,
        target=source,
        transpose_action=transposed,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_semidefinite": "construction",
            },
        ),
        operator_id=canonical_fingerprint(
            {
                "kind": "cavity-curl-curl",
                "complex": hilbert.complex_id,
                "electric": materials.electric_mass.operator_id,
                "magnetic": materials.magnetic_mass.operator_id,
            }
        ),
    )


def maxwell_cavity_modes(
    complex: AbstractDeRhamComplex,
    /,
    *,
    count: int,
    materials: MaxwellCavityMaterials | None = None,
) -> EigenSolveResult:
    """Solve a resource-bounded PEC pencil and select its positive Maxwell modes.

    The declared native DenseEigh policy computes the complete generalized
    spectrum, including its exact and topological curl kernel. Positive-mode
    selection preserves the actual dense solve's provenance, status, work counts,
    and per-mode convergence evidence. There is no iterative or dense fallback.
    Native materialization/resource refusals remain visible.
    """
    if not isinstance(complex, AbstractDeRhamComplex):
        raise TypeError("complex must be an AbstractDeRhamComplex.")
    if complex.dimension != 3:
        raise ValueError("Maxwell cavity modes require a three-dimensional complex.")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("count must be a positive integer.")
    hilbert = complex.hilbert_complex(boundary="relative")
    selected = (
        MaxwellCavityMaterials(mass_form(hilbert, 1), mass_form(hilbert, 2))
        if materials is None
        else materials
    )
    if not isinstance(selected, MaxwellCavityMaterials):
        raise TypeError("materials must be MaxwellCavityMaterials or None.")
    for degree, operator in ((1, selected.electric_mass), (2, selected.magnetic_mass)):
        coordinate = mass_form(hilbert, degree).source
        if not operator.source.compatible(coordinate):
            raise ValueError(
                "Material Gram identity must match the relative coordinate space."
            )
    stiffness = _cavity_stiffness(hilbert, selected)
    size = stiffness.source.size
    if size == 0:
        raise ValueError("The relative cavity has no electric degrees of freedom.")
    pencil = GeneralizedEigenproblem(stiffness, selected.electric_mass)
    complete = eigensolve(
        prepare_eigensolve(pencil, EigenSolvePolicy(DenseEigh(), count=size))
    )
    if not bool(np.asarray(complete.successful)):
        raise ValueError(
            "Native cavity kernel admission failed; inspect the spectrum evidence."
        )
    values = np.asarray(complete.eigenvalues)
    tolerance = 1e-9 * max(1.0, float(np.max(np.abs(values))))
    if np.any(values < -tolerance):
        raise ValueError(
            "The admitted lossless cavity pencil is not positive semidefinite."
        )
    indices = np.flatnonzero(np.abs(values) <= tolerance)
    if count > size - indices.size:
        raise ValueError("count exceeds the kernel-deflated cavity dimension.")
    vectors = complete.eigenvectors
    if not isinstance(vectors, Array):
        raise TypeError("Cavity eigenvectors must be array-valued coordinates.")
    positive = np.flatnonzero(values > tolerance)[:count].astype(np.int32)
    mask = complete.mode_mask[positive]
    effective_count = jnp.sum(mask, dtype=jnp.int32)
    original = complete.diagnostics
    diagnostics = EigenSolveDiagnostics(
        original.residual_norms[positive],
        original.relative_residuals[positive],
        original.orthogonality_error,
        original.iterations,
        original.operator_matvec_count,
        original.metric_matvec_count,
        original.preconditioner_apply_count,
        original.converged[positive],
        mask,
        effective_count,
        original.isolation_gaps[positive],
        original.initial_rank,
    )
    return EigenSolveResult(
        complete.eigenvalues[positive],
        vectors[:, positive],
        mask,
        effective_count,
        complete.converged[positive],
        complete.status,
        diagnostics,
        complete.provenance,
    )


__all__ = [
    "HodgeLaplaceFormulation",
    "HodgeLaplacePlan",
    "HodgeLaplaceResult",
    "MaxwellCavityMaterials",
    "maxwell_cavity_modes",
]
