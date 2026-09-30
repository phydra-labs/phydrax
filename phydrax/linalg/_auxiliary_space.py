#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from ._complexes import coordinate_operator, coordinate_space, HilbertComplex
from ._operators import AbstractLinearOperator, adjoint, IdentityLinearOperator
from ._preconditioner_properties import PreconditionerProperties
from ._preconditioners import AbstractPreconditioner
from ._preconditioning import AbstractPreconditionerBuilder, PreconditionerSource
from ._subspace_correction import (
    AdditiveSubspaceCorrectionBuilder,
    SubspaceCorrectionTerm,
)


def hiptmair_xu_preconditioner_builder(
    complex: HilbertComplex,
    k: int,
    /,
    *,
    vector_interpolation: AbstractLinearOperator | tuple[AbstractLinearOperator, ...],
    vector_builder: PreconditionerSource,
    potential_builder: PreconditionerSource,
    smoother: PreconditionerSource,
) -> AdditiveSubspaceCorrectionBuilder:
    """Build the native additive HX (degree one) or ADS (degree two) correction.

    All actions use Euclidean weak-form coordinates, not a metric-dependent
    Hilbert adjoint. Degree two accepts interpolations in descending degree
    order, ``(Pi_2, Pi_1)``; its potential correction is itself an HX recipe.
    Prepared auxiliary actions may encode independent vector elliptic solves;
    builders instead receive the corresponding Galerkin setup operator.
    """
    if not isinstance(complex, HilbertComplex):
        raise TypeError("complex must be a HilbertComplex.")
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer form degree.")
    if k not in (1, 2) or k > complex.top_degree:
        raise ValueError("Hiptmair–Xu requires an existing degree one or two space.")
    interpolations = (
        vector_interpolation
        if isinstance(vector_interpolation, tuple)
        else (vector_interpolation,)
    )
    if len(interpolations) != k:
        raise ValueError("Provide one vector interpolation for every recursive degree.")
    space = coordinate_space(complex.space(k))
    interpolation = coordinate_operator(interpolations[0])
    if not interpolation.target.compatible(space):
        raise ValueError(
            "Vector interpolation must target the degree-k coordinate space."
        )
    identity = IdentityLinearOperator(space)
    terms = [SubspaceCorrectionTerm(identity, identity, smoother)]
    if interpolation.source.size:
        terms.append(
            SubspaceCorrectionTerm(adjoint(interpolation), interpolation, vector_builder)
        )
    differential = coordinate_operator(complex.differential(k - 1))
    potential: PreconditionerSource = potential_builder
    if k == 2 and isinstance(potential_builder, AbstractPreconditionerBuilder):
        recursive = hiptmair_xu_preconditioner_builder(
            complex,
            1,
            vector_interpolation=interpolations[1],
            vector_builder=vector_builder,
            potential_builder=potential_builder,
            smoother=potential_builder,
        )
        # The recursive gradient correction is annihilated by d1 d0 = 0.
        # Preparing its zero Galerkin operator would invent an invalid solve.
        count = 1 + (interpolations[1].source.size > 0)
        potential = AdditiveSubspaceCorrectionBuilder(recursive.terms[:count])
    if differential.source.size:
        terms.append(
            SubspaceCorrectionTerm(adjoint(differential), differential, potential)
        )
    sources = tuple(term.local_solver for term in terms)
    certified = all(
        isinstance(source, AbstractPreconditioner)
        and source.properties.certifies("positive_definite")
        for source in sources
    )
    properties = (
        PreconditionerProperties(
            linear=True,
            stationary=True,
            self_adjoint=True,
            positive_definite=True,
            evidence={"positive_definite": "construction"},
        )
        if certified
        else None
    )
    return AdditiveSubspaceCorrectionBuilder(tuple(terms), properties=properties)


__all__ = ["hiptmair_xu_preconditioner_builder"]
