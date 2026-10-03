#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Paired complexes, coordinate weak forms, and reusable Hodge solves."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import final, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, core
from jax.typing import ArrayLike
from jaxtyping import PyTree

from .._strict import StrictModule
from ..typing import parse, PRNGKey
from ._assembly import assemble_diagonal, assemble_sparse, SparseAssemblyPolicy
from ._incomplete_factorizations import SparseFactorizationPreconditionerBuilder
from ._named_blocks import assemble_block_operator
from ._operator_pairing import OperatorPairing
from ._operators import (
    _materialize_by_basis,
    AbstractLinearOperator,
    AdjointLinearOperator,
    BlockLinearOperator,
    DenseLinearOperator,
    DiagonalLinearOperator,
    IdentityLinearOperator,
)
from ._pairings import DiagonalPairing, EuclideanPairing
from ._policies import LinearSolvePolicy, MINRES, TolerancePolicy
from ._preconditioners import DiagonalPreconditioner
from ._preconditioning import PreconditioningPolicy
from ._prepared import PreparedLinearSolve
from ._problems import LinearSystem
from ._properties import LinearCapabilityError, OperatorCapabilities, OperatorProperties
from ._runtime import prepare, solve
from ._spaces import (
    _coordinate_dtype,
    AbstractVectorSpace,
    ArraySpace,
    BlockSpace,
    DualSpace,
    PyTreeSpace,
)
from ._sparse_factorizations import SparseFactorizationPolicy
from .eigen._policies import DenseEigh, EigenSolvePolicy, EigenTolerancePolicy, LOBPCG
from .eigen._problems import GeneralizedEigenproblem
from .eigen._results import EigenSolveResult
from .eigen._runtime import eigensolve


type HodgeLaplacianPart = Literal["lower", "upper", "complete"]


def _explicit_id(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be an explicit nonempty identifier.")
    return value


def _properties(*, positive: bool = False, definite: bool = False) -> OperatorProperties:
    return OperatorProperties(
        self_adjoint=True,
        positive_semidefinite=positive or definite,
        positive_definite=definite,
        evidence={
            "self_adjoint": "construction",
            **({"positive_semidefinite": "construction"} if positive or definite else {}),
            **({"positive_definite": "construction"} if definite else {}),
        },
    )


def coordinate_space(space: AbstractVectorSpace, /) -> ArraySpace:
    """Euclidean coordinate endomorphism space, distinct from the paired primal."""
    return ArraySpace(
        (space.size,),
        dtype=_coordinate_dtype(space),
        space_id=f"{space.space_id}:coordinates",
    )


@final
class _CoordinateOperator(AbstractLinearOperator):
    operator: AbstractLinearOperator
    source: ArraySpace
    target: ArraySpace

    def __init__(self, operator: AbstractLinearOperator, /) -> None:
        self.operator = operator
        self.source = coordinate_space(operator.source)
        self.target = coordinate_space(operator.target)
        self.properties = OperatorProperties()
        self.capabilities = operator.capabilities
        self.batch_shape = operator.batch_shape
        self.operator_id = f"{operator.operator_id}:coordinates"

    def mv(self, vector: PyTree[Array], /) -> Array:
        return self.operator.target.flatten(
            self.operator.mv(self.operator.source.unflatten(self.source.flatten(vector)))
        )

    def transpose_mv(self, vector: PyTree[Array], /) -> Array:
        return self.operator.source.flatten(
            self.operator.transpose_mv(
                self.operator.target.unflatten(self.target.flatten(vector))
            )
        )

    def adjoint_mv(self, vector: PyTree[Array], /) -> Array:
        return jnp.conj(self.transpose_mv(jnp.conj(self.target.flatten(vector))))

    def _materialize(self, /) -> Array:
        return self.operator._materialize().astype(self.target.dtype)

    def _assemble_diagonal(self, /) -> Array:
        return assemble_diagonal(self.operator)


def coordinate_operator(operator: AbstractLinearOperator, /) -> AbstractLinearOperator:
    """Strip Riesz pairing without confusing transpose with Hilbert adjoint."""
    from ..sparse._linear import SparseCoordinateOperator

    source, target = coordinate_space(operator.source), coordinate_space(operator.target)
    identifier = f"{operator.operator_id}:coordinates"
    properties = (
        operator.properties
        if isinstance(operator.source, ArraySpace)
        and isinstance(operator.source.pairing, EuclideanPairing)
        else OperatorProperties()
    )
    if isinstance(operator, SparseCoordinateOperator):
        return SparseCoordinateOperator(
            operator.relation,
            operator.coefficients,
            source=source,
            target=target,
            properties=properties,
            operator_id=identifier,
            accumulation_dtype=np.result_type(
                operator.accumulation_dtype, source.dtype, target.dtype
            ),
            block_shape=operator.block_shape,
            storage_plan=operator._storage_plan,
        )
    if isinstance(operator, DenseLinearOperator):
        return DenseLinearOperator(
            operator.matrix.astype(target.dtype),
            source=source,
            target=target,
            properties=properties,
            operator_id=identifier,
        )
    if isinstance(operator, DiagonalLinearOperator):
        return DiagonalLinearOperator(
            operator.diagonal.astype(source.dtype),
            space=source,
            properties=properties,
            operator_id=identifier,
        )
    if isinstance(operator, IdentityLinearOperator):
        return IdentityLinearOperator(source)
    return _CoordinateOperator(operator)


@final
class _CoordinateFormOperator(AbstractLinearOperator):
    """Certification boundary for a proved Euclidean weak-coordinate form."""

    operator: AbstractLinearOperator

    def __init__(
        self,
        operator: AbstractLinearOperator,
        /,
        *,
        operator_id: str,
        positive: bool = False,
        definite: bool = False,
    ) -> None:
        if not operator.source.compatible(operator.target):
            raise ValueError("A coordinate form must be an endomorphism.")
        identifier = _explicit_id(operator_id, "operator_id")
        self.operator = operator
        self.source, self.target = operator.source, operator.target
        self.properties = _properties(positive=positive, definite=definite)
        self.capabilities = operator.capabilities
        self.batch_shape = operator.batch_shape
        self.operator_id = identifier

    def mv(self, vector: PyTree[Array], /) -> PyTree[Array]:
        return self.operator.mv(vector)

    def transpose_mv(self, vector: PyTree[Array], /) -> PyTree[Array]:
        return self.operator.transpose_mv(vector)

    def adjoint_mv(self, vector: PyTree[Array], /) -> PyTree[Array]:
        return self.operator.adjoint_mv(vector)

    def _materialize(self, /) -> Array:
        return self.operator._materialize()

    def _assemble_diagonal(self, /) -> Array:
        return assemble_diagonal(self.operator)


def _require_spd_pairing(space: AbstractVectorSpace, /) -> None:
    from ._space_extensions import AxisArraySpace, TensorProductSpace

    if isinstance(space, (AxisArraySpace, TensorProductSpace)):
        _require_spd_pairing(space.delegate)
        return
    if isinstance(space, DualSpace):
        _require_spd_pairing(space.primal)
        return
    if isinstance(space, BlockSpace):
        for block in space.spaces:
            _require_spd_pairing(block)
        return
    if isinstance(space, (ArraySpace, PyTreeSpace)) and isinstance(
        space.pairing, (EuclideanPairing, DiagonalPairing, OperatorPairing)
    ):
        return
    raise ValueError("Hilbert complexes require construction-certified SPD pairings.")


@final
class HilbertComplex(StrictModule):
    spaces: tuple[AbstractVectorSpace, ...]
    differentials: tuple[AbstractLinearOperator, ...]
    complex_id: str = eqx.field(static=True)

    def __init__(
        self,
        spaces: Sequence[AbstractVectorSpace],
        differentials: Sequence[AbstractLinearOperator],
        /,
        *,
        complex_id: str,
    ) -> None:
        spaces_, differentials_ = tuple(spaces), tuple(differentials)
        identifier = _explicit_id(complex_id, "complex_id")
        if not spaces_ or len(differentials_) != len(spaces_) - 1:
            raise ValueError(
                "A Hilbert complex requires one space per degree and one fewer differential."
            )
        if not all(isinstance(space, AbstractVectorSpace) for space in spaces_):
            raise TypeError("spaces must be paired AbstractVectorSpace values.")
        for space in spaces_:
            _require_spd_pairing(space)
        for k, differential in enumerate(differentials_):
            if (
                not isinstance(differential, AbstractLinearOperator)
                or not differential.capabilities.transpose
            ):
                raise ValueError("Differentials must provide a coordinate transpose.")
            if differential.batch_shape:
                raise ValueError(
                    "Complex differentials must be unbatched; vmap the complex action."
                )
            if not differential.source.compatible(
                spaces_[k]
            ) or not differential.target.compatible(spaces_[k + 1]):
                raise ValueError(
                    "Differential source/target identity does not match its degree spaces."
                )
        self.spaces, self.differentials, self.complex_id = (
            spaces_,
            differentials_,
            identifier,
        )

    @property
    def top_degree(self) -> int:
        return len(self.spaces) - 1

    def space(self, degree: int, /) -> AbstractVectorSpace:
        if degree < 0 or degree > self.top_degree:
            raise ValueError("Space degree is outside the complex.")
        return self.spaces[degree]

    def differential(self, degree: int, /) -> AbstractLinearOperator:
        if degree < 0 or degree >= self.top_degree:
            raise ValueError("Differential degree is outside the complex.")
        return self.differentials[degree]


@final
class ComplexNilpotencyEvidence(StrictModule):
    nilpotency_defects: Array
    valid: Array


def _probes(space: AbstractVectorSpace, key: PRNGKey, count: int, /) -> Array:
    parse(key, PRNGKey, "key")
    if count < 1:
        raise ValueError("probes must be positive.")
    return jax.random.normal(key, (count, space.size), dtype=_coordinate_dtype(space))


def complex_nilpotency_evidence(
    complex: HilbertComplex, /, *, key: PRNGKey, probes: int = 2, tolerance: float = 1e-10
) -> ComplexNilpotencyEvidence:
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative.")
    defects = []
    for k in range(complex.top_degree - 1):
        vectors = _probes(complex.space(k), jax.random.fold_in(key, k), probes)
        composition = complex.differential(k + 1) @ complex.differential(k)
        images = composition.mv_block(vectors.T)
        defects.append(
            jnp.max(
                jnp.linalg.norm(images, axis=0)
                / jnp.maximum(jnp.linalg.norm(vectors, axis=1), 1)
            )
        )
    values = jnp.stack(defects) if defects else jnp.zeros((0,), dtype=jnp.float64)
    return ComplexNilpotencyEvidence(
        values, jnp.all(jnp.isfinite(values) & (values <= tolerance))
    )


def codifferential(complex: HilbertComplex, degree: int, /) -> AbstractLinearOperator:
    complex.space(degree)
    if degree == 0:
        raise ValueError("A degree-zero codifferential does not exist.")
    return AdjointLinearOperator(complex.differential(degree - 1))


@final
class HodgeLaplacianOperator(AbstractLinearOperator):
    complex: HilbertComplex
    degree: int = eqx.field(static=True)
    part: HodgeLaplacianPart = eqx.field(static=True)

    def __init__(
        self,
        complex: HilbertComplex,
        degree: int,
        /,
        *,
        part: HodgeLaplacianPart = "complete",
    ) -> None:
        space = complex.space(degree)
        part_ = parse(part, HodgeLaplacianPart, "part")
        if (part_ == "lower" and degree == 0) or (
            part_ == "upper" and degree == complex.top_degree
        ):
            raise ValueError("Requested Laplacian part is absent at this degree.")
        self.complex, self.degree, self.part = complex, degree, part_
        self.source, self.target = space, space
        self.properties = _properties(positive=True)
        self.capabilities = OperatorCapabilities(
            transpose=True, adjoint=True, materialize=True
        )
        self.batch_shape = ()
        self.operator_id = f"{complex.complex_id}:laplacian:{degree}:{part_}"

    def mv(self, vector: PyTree[Array], /) -> PyTree[Array]:
        value = self.source.validate(vector)
        result = self.source.zeros()
        if self.degree > 0 and self.part != "upper":
            lower = self.complex.differential(self.degree - 1)
            result = jax.tree.map(jnp.add, result, lower.mv(lower.adjoint_mv(value)))
        if self.degree < self.complex.top_degree and self.part != "lower":
            upper = self.complex.differential(self.degree)
            result = jax.tree.map(jnp.add, result, upper.adjoint_mv(upper.mv(value)))
        return result

    def adjoint_mv(self, vector: PyTree[Array], /) -> PyTree[Array]:
        return self.mv(vector)

    def transpose_mv(self, vector: PyTree[Array], /) -> PyTree[Array]:
        # L*=L in M pairing, hence L^T=conj(M L M^{-1}).
        conjugate = jax.tree.map(jnp.conj, self.target.validate(vector))
        result = self.source.riesz(self.mv(self.target.inverse_riesz(conjugate)))
        return jax.tree.map(jnp.conj, result)

    def _materialize(self, /) -> Array:
        return _materialize_by_basis(self)


def hodge_laplacian(
    complex: HilbertComplex, degree: int, /, *, part: HodgeLaplacianPart = "complete"
) -> HodgeLaplacianOperator:
    return HodgeLaplacianOperator(complex, degree, part=part)


@final
class _CoordinateMassOperator(AbstractLinearOperator):
    paired_space: AbstractVectorSpace
    source: ArraySpace
    target: ArraySpace

    def __init__(self, space: AbstractVectorSpace, /) -> None:
        self.paired_space = space
        self.source = coordinate_space(space)
        self.target = self.source
        self.properties = _properties(definite=True)
        self.capabilities = OperatorCapabilities(
            transpose=True,
            adjoint=True,
            materialize=True,
            diagonal_assembly=True,
        )
        self.batch_shape = ()
        self.operator_id = f"{space.space_id}:mass"

    def mv(self, vector: PyTree[Array], /) -> Array:
        space = self.paired_space
        return space.flatten(space.riesz(space.unflatten(self.source.flatten(vector))))

    def transpose_mv(self, vector: PyTree[Array], /) -> Array:
        return jnp.conj(self.mv(jnp.conj(self.source.flatten(vector))))

    def adjoint_mv(self, vector: PyTree[Array], /) -> Array:
        return self.mv(vector)

    def _materialize(self, /) -> Array:
        return _materialize_by_basis(self)

    def _assemble_diagonal(self, /) -> Array:
        # Sequential probing retains O(n) scratch, never an n-by-n basis/mass.
        def entry(index: Array, diagonal: Array) -> Array:
            basis = (
                jnp.zeros((self.source.size,), dtype=self.source.dtype).at[index].set(1)
            )
            return diagonal.at[index].set(self.mv(basis)[index])

        return jax.lax.fori_loop(
            0,
            self.source.size,
            entry,
            jnp.zeros((self.source.size,), dtype=self.source.dtype),
        )


def _coordinate_mass(space: AbstractVectorSpace, /) -> AbstractLinearOperator:
    coordinates = coordinate_space(space)
    if isinstance(space, ArraySpace):
        if isinstance(space.pairing, EuclideanPairing):
            return IdentityLinearOperator(coordinates)
        if isinstance(space.pairing, DiagonalPairing):
            weights = space.flatten(
                space.riesz(
                    space.unflatten(jnp.ones((space.size,), dtype=coordinates.dtype))
                )
            )
            return DiagonalLinearOperator(
                weights,
                space=coordinates,
                properties=_properties(definite=True),
                operator_id=f"{space.space_id}:mass",
            )
        if isinstance(space.pairing, OperatorPairing):
            original = coordinate_operator(space.pairing.operator)
            if original.source.size != coordinates.size:
                raise ValueError("Pairing coordinates do not match the paired space.")
            # Rebind native leaves to retain exact sparse assembly.
            from ..sparse._linear import SparseCoordinateOperator

            if isinstance(original, SparseCoordinateOperator):
                return SparseCoordinateOperator(
                    original.relation,
                    original.coefficients,
                    source=coordinates,
                    target=coordinates,
                    properties=_properties(definite=True),
                    operator_id=f"{space.space_id}:mass",
                    accumulation_dtype=np.result_type(
                        original.accumulation_dtype, coordinates.dtype
                    ),
                    block_shape=original.block_shape,
                    storage_plan=original._storage_plan,
                )
            if isinstance(original, DenseLinearOperator):
                return DenseLinearOperator(
                    original.matrix.astype(coordinates.dtype),
                    source=coordinates,
                    target=coordinates,
                    properties=_properties(definite=True),
                    operator_id=f"{space.space_id}:mass",
                )
            return _CoordinateMassOperator(space)
    return _CoordinateMassOperator(space)


def mass_form(complex: HilbertComplex, degree: int, /) -> AbstractLinearOperator:
    """Certified SPD Euclidean-coordinate Gram, not a semantic V-to-dual map."""
    return _coordinate_mass(complex.space(degree))


def stiffness_form(complex: HilbertComplex, degree: int, /) -> AbstractLinearOperator:
    """Weak upper Laplacian d* M d, including the zero top-degree form."""
    space = coordinate_space(complex.space(degree))
    operator: AbstractLinearOperator
    if degree == complex.top_degree:
        operator = DiagonalLinearOperator(
            jnp.zeros((space.size,), dtype=space.dtype), space=space
        )
    else:
        differential = coordinate_operator(complex.differential(degree))
        operator = (
            AdjointLinearOperator(differential)
            @ mass_form(complex, degree + 1)
            @ differential
        )
    return _CoordinateFormOperator(
        operator, operator_id=f"{complex.complex_id}:stiffness:{degree}", positive=True
    )


def hodge_laplacian_form(
    complex: HilbertComplex, degree: int, /, *, part: HodgeLaplacianPart = "complete"
) -> AbstractLinearOperator:
    laplacian = hodge_laplacian(complex, degree, part=part)
    operator = mass_form(complex, degree) @ coordinate_operator(laplacian)
    return _CoordinateFormOperator(
        operator, operator_id=f"{laplacian.operator_id}:weak", positive=True
    )


@final
class HarmonicSubspacePolicy(StrictModule):
    dense_dimension: int = eqx.field(static=True)
    oversampling: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    eigen_policy: EigenSolvePolicy | None

    def __init__(
        self,
        *,
        dense_dimension: int = 256,
        oversampling: int = 2,
        tolerance: float = 1e-8,
        eigen_policy: EigenSolvePolicy | None = None,
    ) -> None:
        if (
            dense_dimension < 0
            or oversampling < 1
            or not isfinite(tolerance)
            or tolerance < 0
        ):
            raise ValueError("Harmonic policy dimensions and tolerances are invalid.")
        if eigen_policy is not None and not isinstance(eigen_policy, EigenSolvePolicy):
            raise TypeError("eigen_policy must be an EigenSolvePolicy.")
        self.dense_dimension, self.oversampling, self.tolerance, self.eigen_policy = (
            dense_dimension,
            oversampling,
            tolerance,
            eigen_policy,
        )


@final
class HarmonicSubspace(StrictModule):
    complex_id: str = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    basis: Array
    eigenvalues: Array
    gap: Array
    residual_defect: Array
    orthonormality_defect: Array
    dimension_match: Array
    valid: Array

    def __init__(
        self,
        complex_id: str,
        degree: int,
        basis: ArrayLike,
        eigenvalues: ArrayLike,
        gap: ArrayLike,
        residual_defect: ArrayLike,
        orthonormality_defect: ArrayLike,
        dimension_match: ArrayLike,
        valid: ArrayLike,
        /,
    ) -> None:
        identifier = _explicit_id(complex_id, "complex_id")
        basis_, values_ = jnp.asarray(basis), jnp.asarray(eigenvalues)
        evidence = tuple(
            jnp.asarray(value) for value in (gap, residual_defect, orthonormality_defect)
        )
        match_, valid_ = (
            jnp.asarray(dimension_match, dtype=jnp.bool_),
            jnp.asarray(valid, dtype=jnp.bool_),
        )
        if degree < 0 or basis_.ndim != 2 or values_.shape != (basis_.shape[1],):
            raise ValueError("Harmonic basis and eigenvalue coordinates are invalid.")
        if any(value.shape != () for value in (*evidence, match_, valid_)):
            raise ValueError("Harmonic evidence must consist of scalar arrays.")
        self.complex_id, self.degree = identifier, degree
        self.basis, self.eigenvalues = basis_, values_
        self.gap, self.residual_defect, self.orthonormality_defect = evidence
        self.dimension_match, self.valid = match_, valid_

    @property
    def dimension(self) -> int:
        return self.basis.shape[1]

    def project(
        self, space: AbstractVectorSpace, values: PyTree[Array], /
    ) -> PyTree[Array]:
        if space.size != self.basis.shape[0]:
            raise ValueError("Harmonic basis coordinates do not match the space.")
        if _needs_real_extension(space, values):
            real, imaginary = _real_parts(space, values)
            return jax.tree.map(
                jax.lax.complex, self.project(space, real), self.project(space, imaginary)
            )
        coefficients = self.basis.conj().T @ space.flatten(space.riesz(values))
        return space.unflatten(self.basis @ coefficients)


def _harmonic_pencil(complex: HilbertComplex, degree: int, /) -> AbstractLinearOperator:
    upper = stiffness_form(complex, degree)
    if degree == 0:
        return upper
    mass = mass_form(complex, degree)
    differential = coordinate_operator(complex.differential(degree - 1))
    diagonal = assemble_diagonal(mass_form(complex, degree - 1))
    weights = DiagonalLinearOperator(jnp.reciprocal(diagonal), space=differential.source)
    lower = mass @ differential @ weights @ AdjointLinearOperator(differential) @ mass
    return _CoordinateFormOperator(
        upper + lower,
        operator_id=f"{complex.complex_id}:harmonic-pencil:{degree}",
        positive=True,
    )


def _admit_harmonic_dimension(
    complex: HilbertComplex,
    size: int,
    expected: int | None,
    policy: HarmonicSubspacePolicy,
    /,
) -> None:
    if expected is not None:
        if expected < 0 or expected > size:
            raise ValueError(
                "Expected harmonic dimension is outside the coordinate space."
            )
        return
    if size > policy.dense_dimension:
        raise ValueError("Large harmonic solves require an explicit expected_dimension.")
    if any(isinstance(leaf, core.Tracer) for leaf in jax.tree.leaves(complex)):
        raise ValueError("Traced harmonic solves require an explicit expected_dimension.")


# The default large-space route preconditions LOBPCG with an exact sparse
# factor of ``A + s M``. ``s`` is this fraction of the mean diagonal ratio
# ``diag(A) / diag(M)``: the factor then acts like shift-invert about ``-s``, so
# kernel (near-zero) modes contract by about ``s / (gap + s)`` per iteration,
# while ``A + s M`` stays definite and conditioned near ``1 / fraction``.
_KERNEL_PRECONDITIONER_SHIFT = 1e-6


def _kernel_preconditioning(
    operator: AbstractLinearOperator, mass: AbstractLinearOperator, /
) -> PreconditioningPolicy | None:
    """Exact sparse-factor preconditioner of the definite shifted pencil.

    ``A`` is a certified positive-semidefinite form and ``M`` a certified SPD
    mass, so ``A + s M`` is positive definite for ``s > 0``; its certification
    is that proof. Returns None when the pencil has no canonical sparse
    assembly; the unpreconditioned LOBPCG route still reports its own residual
    and convergence status.
    """
    try:
        sparse_operator = assemble_sparse(operator)
        sparse_mass = assemble_sparse(mass)
    except LinearCapabilityError:
        return None
    shift = (
        _KERNEL_PRECONDITIONER_SHIFT
        * jnp.mean(assemble_diagonal(sparse_operator))
        / jnp.mean(assemble_diagonal(sparse_mass))
    )
    setup = assemble_sparse(
        _CoordinateFormOperator(
            sparse_operator + shift * sparse_mass,
            operator_id=f"{operator.operator_id}:kernel-shift",
            definite=True,
        )
    )
    return PreconditioningPolicy(
        SparseFactorizationPreconditionerBuilder(
            SparseFactorizationPolicy("cholesky", ordering="approximate-minimum-degree")
        ),
        setup_operator=setup,
    )


_ITERATIVE_TOLERANCE_MARGIN = 1e-2


def _harmonic_eigen_policy(
    size: int,
    expected: int | None,
    policy: HarmonicSubspacePolicy,
    operator: AbstractLinearOperator,
    mass: AbstractLinearOperator,
    /,
) -> EigenSolvePolicy:
    if policy.eigen_policy is not None:
        selected = policy.eigen_policy
        required = size if expected is None else min(size, expected + 1)
        if selected.count < required:
            raise ValueError("Harmonic eigensolve must include a gap-checking mode.")
        if isinstance(selected.method, DenseEigh) and size > policy.dense_dimension:
            raise ValueError(
                "Dense harmonic solve exceeds dense_dimension resource policy."
            )
        return selected
    count = size if expected is None else min(size, expected + policy.oversampling)
    dense = size <= policy.dense_dimension
    # The iterative route stops on its own pencil residual; downstream
    # certificates (HarmonicSubspace.valid, validate_harmonic_cohomology) measure
    # different norms of the same kernel at ``policy.tolerance``. Solving to a
    # strictly tighter tolerance keeps a margin between the stop and the
    # certificate (measured: a stop at 1e-7 certified at 1.27e-7). The admission
    # tolerance itself is unchanged.
    solver_tolerance = (
        policy.tolerance if dense else policy.tolerance * _ITERATIVE_TOLERANCE_MARGIN
    )
    return EigenSolvePolicy(
        DenseEigh() if dense else LOBPCG(block_dimension=count),
        count=count,
        key=jax.random.key(0),
        tolerance=EigenTolerancePolicy(
            relative=solver_tolerance,
            absolute=solver_tolerance,
            orthogonality=solver_tolerance,
        ),
        preconditioning=None if dense else _kernel_preconditioning(operator, mass),
    )


def _harmonic_result(
    complex_id: str,
    degree: int,
    result: EigenSolveResult,
    operator: AbstractLinearOperator,
    mass: AbstractLinearOperator,
    expected: int,
    tolerance: float,
    /,
) -> HarmonicSubspace:
    values = result.eigenvalues
    vector_leaves = jax.tree.leaves(result.eigenvectors)
    if len(vector_leaves) != 1:
        raise TypeError("Coordinate harmonic eigensolve must return one array basis.")
    basis = vector_leaves[0][:, :expected]
    selected = values[:expected]
    gap = (
        values[expected]
        if expected < values.shape[0]
        else jnp.asarray(jnp.inf, dtype=values.dtype)
    )
    mass_basis = mass.mv_block(basis)
    residual = jnp.linalg.norm(operator.mv_block(basis), ord="fro")
    orthogonality = jnp.linalg.norm(
        basis.conj().T @ mass_basis - jnp.eye(expected, dtype=basis.dtype),
        ord="fro",
    )
    count_match = jnp.all(jnp.abs(selected) <= tolerance) & (gap > tolerance)
    finite = (
        jnp.all(jnp.isfinite(basis))
        & jnp.isfinite(residual)
        & jnp.isfinite(orthogonality)
    )
    valid = (
        count_match
        & finite
        & (residual <= tolerance)
        & (orthogonality <= tolerance)
        & (result.status == 0)
    )
    return HarmonicSubspace(
        complex_id,
        degree,
        basis,
        selected,
        gap,
        residual,
        orthogonality,
        count_match,
        valid,
    )


def harmonic_subspace(
    complex: HilbertComplex,
    degree: int,
    /,
    *,
    expected_dimension: int | None = None,
    policy: HarmonicSubspacePolicy | None = None,
) -> HarmonicSubspace:
    """Find the harmonic kernel without any inner mass-inverse solve.

    ``expected_dimension`` is a caller-supplied count, optionally exact Betti
    evidence. A missing or extra zero eigenvalue fails dimension_match.
    """
    space = complex.space(degree)
    policy_ = HarmonicSubspacePolicy() if policy is None else policy
    _admit_harmonic_dimension(complex, space.size, expected_dimension, policy_)
    if space.size == 0:
        zero = jnp.asarray(0.0, dtype=jnp.float64)
        return HarmonicSubspace(
            complex.complex_id,
            degree,
            jnp.zeros((0, 0), dtype=_coordinate_dtype(space)),
            jnp.zeros((0,), dtype=jnp.float64),
            jnp.asarray(jnp.inf),
            zero,
            zero,
            jnp.asarray(True),
            jnp.asarray(True),
        )
    operator, mass = _harmonic_pencil(complex, degree), mass_form(complex, degree)
    eigen_policy = _harmonic_eigen_policy(
        space.size, expected_dimension, policy_, operator, mass
    )
    result = eigensolve(
        GeneralizedEigenproblem(
            operator, mass, problem_id=f"{complex.complex_id}:harmonic:{degree}"
        ),
        policy=eigen_policy,
    )
    values = result.eigenvalues
    if expected_dimension is None:
        if isinstance(values, core.Tracer):
            raise ValueError(
                "Traced harmonic solves require an explicit expected_dimension."
            )
        # Explicit eager numerical admission: shape discovery cannot occur in jit.
        expected_dimension = int(
            np.count_nonzero(
                np.abs(np.asarray(jax.device_get(values))) <= policy_.tolerance
            )
        )
    return _harmonic_result(
        complex.complex_id,
        degree,
        result,
        operator,
        mass,
        expected_dimension,
        policy_.tolerance,
    )


def _check_harmonic(
    complex: HilbertComplex, degree: int, harmonic: HarmonicSubspace, /
) -> None:
    if (
        harmonic.complex_id != complex.complex_id
        or harmonic.degree != degree
        or harmonic.basis.shape[0] != complex.space(degree).size
    ):
        raise ValueError("Harmonic basis provenance/degree does not match the complex.")
    if jnp.iscomplexobj(harmonic.basis) and not np.issubdtype(
        _coordinate_dtype(complex.space(degree)), np.complexfloating
    ):
        raise ValueError(
            "A complex harmonic basis requires complex Hilbert-coordinate spaces."
        )


def mixed_hodge_laplacian(
    complex: HilbertComplex, degree: int, /, *, harmonic: HarmonicSubspace | None = None
) -> BlockLinearOperator:
    """Symmetric mixed Hodge saddle, with harmonic constraints when supplied."""
    space = coordinate_space(complex.space(degree))
    entries: list[tuple[tuple[str, ...], tuple[str, ...], AbstractLinearOperator]] = []
    spaces: list[AbstractVectorSpace] = []
    names: list[str] = []
    if degree > 0:
        lower_space = coordinate_space(complex.space(degree - 1))
        spaces.append(lower_space)
        names.append("sigma")
        coupling = mass_form(complex, degree) @ coordinate_operator(
            complex.differential(degree - 1)
        )
        entries.extend(
            [
                (("sigma",), ("sigma",), -mass_form(complex, degree - 1)),
                (("u",), ("sigma",), coupling),
                (("sigma",), ("u",), AdjointLinearOperator(coupling)),
            ]
        )
    spaces.append(space)
    names.append("u")
    entries.append((("u",), ("u",), stiffness_form(complex, degree)))
    if harmonic is not None:
        _check_harmonic(complex, degree, harmonic)
        if harmonic.dimension:
            harmonic_space = ArraySpace(
                (harmonic.dimension,),
                dtype=space.dtype,
                space_id=f"{complex.complex_id}:harmonic-coordinates:{degree}",
            )
            spaces.append(harmonic_space)
            names.append("p")
            basis = DenseLinearOperator(
                harmonic.basis.astype(space.dtype),
                source=harmonic_space,
                target=space,
                operator_id=f"{complex.complex_id}:harmonic-inclusion:{degree}",
            )
            coupling = mass_form(complex, degree) @ basis
            entries.extend(
                [
                    (("u",), ("p",), coupling),
                    (("p",), ("u",), AdjointLinearOperator(coupling)),
                ]
            )
    blocks = BlockSpace(
        spaces,
        names=names,
        space_id=f"{complex.complex_id}:mixed-space:{degree}:{0 if harmonic is None else harmonic.dimension}",
    )
    assembled = assemble_block_operator(entries, source=blocks, target=blocks)
    return BlockLinearOperator(
        assembled.blocks,
        source=blocks,
        target=blocks,
        properties=_properties(),
        operator_id=f"{complex.complex_id}:mixed:{degree}:{0 if harmonic is None else harmonic.dimension}",
    )


@final
class HodgeDecompositionPolicy(StrictModule):
    solve_policy: LinearSolvePolicy
    sparse_assembly: SparseAssemblyPolicy | None
    tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        solve_policy: LinearSolvePolicy | None = None,
        sparse_assembly: SparseAssemblyPolicy | None = None,
        tolerance: float = 1e-8,
    ) -> None:
        if not isfinite(tolerance) or tolerance < 0:
            raise ValueError("Decomposition tolerance must be finite and nonnegative.")
        if solve_policy is not None and not isinstance(solve_policy, LinearSolvePolicy):
            raise TypeError("solve_policy must be a LinearSolvePolicy.")
        if sparse_assembly is not None and not isinstance(
            sparse_assembly, SparseAssemblyPolicy
        ):
            raise TypeError("sparse_assembly must be a SparseAssemblyPolicy.")
        self.solve_policy = (
            LinearSolvePolicy(
                MINRES(),
                tolerance=TolerancePolicy(
                    relative=tolerance * 0.1, absolute=tolerance * 0.01
                ),
            )
            if solve_policy is None
            else solve_policy
        )
        self.sparse_assembly, self.tolerance = sparse_assembly, tolerance


@final
class HodgeDecomposition(StrictModule):
    exact_potential: PyTree[Array] | None
    exact: PyTree[Array]
    coexact: PyTree[Array]
    harmonic: PyTree[Array]
    orthogonality_defect: Array
    reconstruction_defect: Array
    solve_status: Array
    valid: Array


type _HodgeParts = tuple[
    PyTree[Array] | None,
    PyTree[Array],
    PyTree[Array],
    PyTree[Array],
    Array,
]


def _needs_real_extension(space: AbstractVectorSpace, values: PyTree[Array], /) -> bool:
    return not np.issubdtype(_coordinate_dtype(space), np.complexfloating) and any(
        jnp.iscomplexobj(leaf) for leaf in jax.tree.leaves(values)
    )


def _real_parts(
    space: AbstractVectorSpace,
    values: PyTree[Array],
    /,
) -> tuple[PyTree[Array], PyTree[Array]]:
    structure = space.structure()
    real = jax.tree.map(
        lambda value, spec: jnp.real(value).astype(spec.dtype), values, structure
    )
    imaginary = jax.tree.map(
        lambda value, spec: jnp.imag(value).astype(spec.dtype), values, structure
    )
    return space.validate(real), space.validate(imaginary)


def _extended_metric_coordinates(
    space: AbstractVectorSpace,
    values: PyTree[Array],
    /,
) -> tuple[Array, Array]:
    if not _needs_real_extension(space, values):
        return space.flatten(values), space.flatten(space.riesz(values))
    real, imaginary = _real_parts(space, values)
    coordinates = jax.lax.complex(space.flatten(real), space.flatten(imaginary))
    covector = jax.lax.complex(
        space.flatten(space.riesz(real)),
        space.flatten(space.riesz(imaginary)),
    )
    return coordinates, covector


@final
class PreparedHodgeDecomposition(StrictModule):
    complex: HilbertComplex
    harmonic: HarmonicSubspace
    lower_harmonic: HarmonicSubspace | None
    policy: HodgeDecompositionPolicy
    prepared_solve: PreparedLinearSolve | None
    mixed_space: BlockSpace | None
    degree: int = eqx.field(static=True)

    def _components(self, values: PyTree[Array], /) -> _HodgeParts:
        space = self.complex.space(self.degree)
        value = space.validate(values)
        harmonic = self.harmonic.project(space, value)
        potential = None
        exact = space.zeros()
        status = jnp.asarray(0, dtype=jnp.int32)
        if self.degree > 0 and self.complex.space(self.degree - 1).size == 0:
            potential = self.complex.space(self.degree - 1).zeros()
        if self.degree > 0 and self.complex.space(self.degree - 1).size > 0:
            if self.prepared_solve is None or self.mixed_space is None:
                raise RuntimeError(
                    "Prepared positive-degree decomposition has no mixed solve."
                )
            differential = self.complex.differential(self.degree - 1)
            rhs_u = coordinate_operator(differential).adjoint_mv(
                space.flatten(space.riesz(value))
            )
            rhs = tuple(
                rhs_u if name == "u" else block.zeros()
                for name, block in zip(
                    self.mixed_space.names, self.mixed_space.spaces, strict=True
                )
            )
            solved = solve(self.prepared_solve, rhs)
            solved_coordinates = self.prepared_solve.problem.operator.source.flatten(
                solved.value
            )
            solved_blocks = self.mixed_space.unflatten(solved_coordinates)
            potential_coordinates = solved_blocks[self.mixed_space.names.index("u")]
            potential = differential.source.unflatten(potential_coordinates)
            exact = differential.mv(potential)
            status = solved.status
        coexact = jax.tree.map(lambda x, e, h: x - e - h, value, exact, harmonic)
        return potential, exact, coexact, harmonic, status

    def apply(self, values: PyTree[Array], /) -> HodgeDecomposition:
        space = self.complex.space(self.degree)
        if _needs_real_extension(space, values):
            real, imaginary = _real_parts(space, values)
            values = jax.tree.map(jax.lax.complex, real, imaginary)
            first, second = self._components(real), self._components(imaginary)
            potential = None
            if first[0] is not None and second[0] is not None:
                potential = jax.tree.map(jax.lax.complex, first[0], second[0])
            exact = jax.tree.map(jax.lax.complex, first[1], second[1])
            coexact = jax.tree.map(jax.lax.complex, first[2], second[2])
            harmonic = jax.tree.map(jax.lax.complex, first[3], second[3])
            status = jnp.maximum(first[4], second[4])
        else:
            values = space.validate(values)
            potential, exact, coexact, harmonic, status = self._components(values)
        coordinates = tuple(
            _extended_metric_coordinates(space, component)
            for component in (exact, coexact, harmonic)
        )
        orthogonality = jnp.max(
            jnp.stack(
                [
                    jnp.abs(jnp.vdot(coordinates[i][0], coordinates[j][1]))
                    for i, j in ((0, 1), (0, 2), (1, 2))
                ]
            )
        )
        reconstruction = jax.tree.map(
            lambda x, e, c, h: x - e - c - h, values, exact, coexact, harmonic
        )
        reconstructed, metric_reconstructed = _extended_metric_coordinates(
            space, reconstruction
        )
        defect = jnp.sqrt(
            jnp.maximum(jnp.real(jnp.vdot(reconstructed, metric_reconstructed)), 0)
        )
        vector, metric_vector = _extended_metric_coordinates(space, values)
        scale = jnp.maximum(jnp.real(jnp.vdot(vector, metric_vector)), 1)
        valid = (
            self.harmonic.valid
            & (status == 0)
            & jnp.isfinite(orthogonality)
            & (orthogonality <= self.policy.tolerance * scale)
            & (defect <= self.policy.tolerance * jnp.sqrt(scale))
        )
        if self.lower_harmonic is not None:
            valid = valid & self.lower_harmonic.valid
        return HodgeDecomposition(
            potential, exact, coexact, harmonic, orthogonality, defect, status, valid
        )


def prepare_hodge_decomposition(
    complex: HilbertComplex,
    degree: int,
    /,
    *,
    harmonic: HarmonicSubspace,
    lower_harmonic: HarmonicSubspace | None = None,
    policy: HodgeDecompositionPolicy | None = None,
) -> PreparedHodgeDecomposition:
    _check_harmonic(complex, degree, harmonic)
    policy_ = HodgeDecompositionPolicy() if policy is None else policy
    prepared_solve, mixed_space = None, None
    if degree > 0:
        if lower_harmonic is None:
            raise ValueError(
                "Positive-degree decomposition requires a harmonic basis at degree k-1."
            )
        _check_harmonic(complex, degree - 1, lower_harmonic)
        if complex.space(degree - 1).size == 0:
            return PreparedHodgeDecomposition(
                complex, harmonic, lower_harmonic, policy_, None, None, degree
            )
        mixed = mixed_hodge_laplacian(complex, degree - 1, harmonic=lower_harmonic)
        mixed_space = mixed.source
        operator: AbstractLinearOperator = mixed
        if policy_.sparse_assembly is not None:
            operator = assemble_sparse(mixed, policy_.sparse_assembly)
        solve_policy = policy_.solve_policy
        if (
            isinstance(solve_policy.method, MINRES)
            and solve_policy.preconditioning is None
        ):
            diagonal_parts = []
            for name, block in zip(mixed_space.names, mixed_space.spaces, strict=True):
                if name == "p":
                    diagonal = jnp.ones((block.size,), dtype=_coordinate_dtype(block))
                else:
                    k = degree - 2 if name == "sigma" else degree - 1
                    diagonal = assemble_diagonal(mass_form(complex, k))
                diagonal_parts.append(jnp.real(diagonal))
            preconditioner = DiagonalPreconditioner(
                jnp.concatenate(diagonal_parts).astype(_coordinate_dtype(mixed_space)),
                space=mixed_space,
                positive_definite=True,
                preconditioner_id=f"{complex.complex_id}:decomposition-preconditioner:{degree}",
            )
            solve_policy = eqx.tree_at(
                lambda p: p.preconditioning,
                solve_policy,
                PreconditioningPolicy(preconditioner),
                is_leaf=lambda value: value is None,
            )
        prepared_solve = prepare(
            LinearSystem(
                operator, problem_id=f"{complex.complex_id}:decomposition:{degree}"
            ),
            solve_policy,
        )
    return PreparedHodgeDecomposition(
        complex, harmonic, lower_harmonic, policy_, prepared_solve, mixed_space, degree
    )


def hodge_decomposition(
    complex: HilbertComplex,
    degree: int,
    values: PyTree[Array],
    /,
    *,
    harmonic: HarmonicSubspace,
    lower_harmonic: HarmonicSubspace | None = None,
    policy: HodgeDecompositionPolicy | None = None,
) -> HodgeDecomposition:
    return prepare_hodge_decomposition(
        complex, degree, harmonic=harmonic, lower_harmonic=lower_harmonic, policy=policy
    ).apply(values)


@final
class ComplexMap(StrictModule):
    source: HilbertComplex
    target: HilbertComplex
    maps: tuple[AbstractLinearOperator, ...]
    map_id: str = eqx.field(static=True)
    degree_offset: int = eqx.field(static=True)
    first_degree: int = eqx.field(static=True)

    def __init__(
        self,
        source: HilbertComplex,
        target: HilbertComplex,
        maps: Sequence[AbstractLinearOperator],
        /,
        *,
        map_id: str,
        degree_offset: int = 0,
    ) -> None:
        identifier = _explicit_id(map_id, "map_id")
        first, last = (
            max(0, -degree_offset),
            min(source.top_degree, target.top_degree - degree_offset),
        )
        maps_ = tuple(maps)
        if last < first or len(maps_) != last - first + 1:
            raise ValueError(
                "Complex maps must cover exactly the overlapping degree range."
            )
        for degree, operator in zip(range(first, last + 1), maps_, strict=True):
            if (
                not isinstance(operator, AbstractLinearOperator)
                or not operator.capabilities.transpose
                or operator.batch_shape
            ):
                raise ValueError(
                    "Degree maps require an unbatched transposable linear operator."
                )
            if not operator.source.compatible(
                source.space(degree)
            ) or not operator.target.compatible(target.space(degree + degree_offset)):
                raise ValueError("Degree-map scientific space identity mismatch.")
        self.source, self.target, self.maps = source, target, maps_
        self.map_id, self.degree_offset, self.first_degree = (
            identifier,
            degree_offset,
            first,
        )

    def map(self, degree: int, /) -> AbstractLinearOperator:
        index = degree - self.first_degree
        if index < 0 or index >= len(self.maps):
            raise ValueError("Requested degree is outside the map overlap.")
        return self.maps[index]

    def adjoint(self, /) -> ComplexMap:
        """Degreewise Hilbert adjoints; these are not generally chain maps."""
        return ComplexMap(
            self.target,
            self.source,
            tuple(AdjointLinearOperator(operator) for operator in self.maps),
            map_id=f"{self.map_id}:adjoint",
            degree_offset=-self.degree_offset,
        )


@final
class ComplexMapEvidence(StrictModule):
    commuting_defects: Array
    valid: Array


def complex_map_evidence(
    map: ComplexMap, /, *, key: PRNGKey, probes: int = 2, tolerance: float = 1e-10
) -> ComplexMapEvidence:
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative.")
    defects = []
    last = map.first_degree + len(map.maps) - 1
    for degree in range(max(0, map.first_degree - 1), last + 1):
        target_degree = degree + map.degree_offset
        has_left = degree >= map.first_degree and target_degree < map.target.top_degree
        has_right = (
            degree < map.source.top_degree and map.first_degree <= degree + 1 <= last
        )
        if not has_left and not has_right:
            continue
        vectors = _probes(
            map.source.space(degree), jax.random.fold_in(key, degree), probes
        )
        if has_left:
            left = (map.target.differential(target_degree) @ map.map(degree)).mv_block(
                vectors.T
            )
        else:
            left = None
        right = (
            (map.map(degree + 1) @ map.source.differential(degree)).mv_block(vectors.T)
            if has_right
            else None
        )
        if left is None:
            if right is None:
                raise RuntimeError("Commutation endpoint has no action.")
            difference = -right
        else:
            difference = left if right is None else left - right
        defects.append(
            jnp.max(
                jnp.linalg.norm(difference, axis=0)
                / jnp.maximum(jnp.linalg.norm(vectors, axis=1), 1)
            )
        )
    values = jnp.stack(defects) if defects else jnp.zeros((0,), dtype=jnp.float64)
    return ComplexMapEvidence(
        values, jnp.all(jnp.isfinite(values) & (values <= tolerance))
    )


__all__ = [
    "ComplexMap",
    "ComplexMapEvidence",
    "ComplexNilpotencyEvidence",
    "HarmonicSubspace",
    "HarmonicSubspacePolicy",
    "HilbertComplex",
    "HodgeDecomposition",
    "HodgeDecompositionPolicy",
    "HodgeLaplacianOperator",
    "HodgeLaplacianPart",
    "PreparedHodgeDecomposition",
    "codifferential",
    "complex_map_evidence",
    "complex_nilpotency_evidence",
    "coordinate_operator",
    "coordinate_space",
    "harmonic_subspace",
    "hodge_decomposition",
    "hodge_laplacian",
    "hodge_laplacian_form",
    "mass_form",
    "mixed_hodge_laplacian",
    "prepare_hodge_decomposition",
    "stiffness_form",
]
