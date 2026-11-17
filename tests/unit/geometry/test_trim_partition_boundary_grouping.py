from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.geometry._domain_boundary import MeshingDomainBoundarySource
from phydrax.geometry._meshing_domain import (
    MeshingDomain,
    MeshingDomainCurve,
    MeshingSurfacePatch,
    PatchCurveUse,
)
from phydrax.geometry.brep._patches import CircleCurve, LineCurve, PlanePatch


type Point = tuple[Fraction, Fraction]
type Triangle = tuple[Point, Point, Point]


@pytest.fixture
def sector() -> tuple[
    MeshingDomainBoundarySource, tuple[Point, ...], tuple[Triangle, ...]
]:
    arc = CircleCurve((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), 1.0)
    angle = 1.0
    end = np.asarray(arc.evaluate(jnp.asarray((0.0, angle))), dtype=np.float64)[1]
    uses = (
        PatchCurveUse(0, LineCurve((0.0, 0.0), (1.0, 0.0)), 0.0, 1.0),
        PatchCurveUse(1, arc, 0.0, angle),
        PatchCurveUse(2, LineCurve(end, -end), 0.0, 1.0),
    )
    domain = MeshingDomain(
        (
            MeshingSurfacePatch(
                PlanePatch((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                (uses,),
            ),
        ),
        (MeshingDomainCurve(0, 1), MeshingDomainCurve(1, 2), MeshingDomainCurve(2, 0)),
        3,
        source_id="boundary-grouping-sector",
        source_revision="source-revision-1",
        tolerance=1.0e-12,
    )
    points = np.asarray(arc.evaluate(jnp.linspace(0.0, angle, 5)), dtype=np.float64)
    chain = tuple((Fraction(float(u)), Fraction(float(v))) for u, v in points)
    origin = (Fraction(0), Fraction(0))
    fan = tuple((origin, chain[row], chain[row + 1]) for row in range(4))
    return MeshingDomainBoundarySource(domain, (0,)), chain, fan


def test_shared_edges_and_mixed_winding_preserve_curved_partition(
    sector: tuple[MeshingDomainBoundarySource, tuple[Point, ...], tuple[Triangle, ...]],
) -> None:
    source, chain, fan = sector
    declared, lenses = source.trim_partition_reference(0, fan)
    mixed = (fan[2][::-1], fan[0], fan[3][::-1], fan[1])
    assert source.trim_partition_reference(0, mixed) == (declared, lenses)
    assert tuple(lens.chord for lens in lenses) == tuple(zip(chain, chain[1:]))


def test_exact_distinct_vertices_do_not_join_boundary_chains(
    sector: tuple[MeshingDomainBoundarySource, tuple[Point, ...], tuple[Triangle, ...]],
) -> None:
    source, chain, fan = sector
    near = (chain[0][0] + Fraction(1, 2**100), chain[0][1])
    extra = (near, (Fraction(2), Fraction(0)), (Fraction(2), Fraction(1)))
    assert float(near[0]) == float(chain[0][0])
    assert source.trim_partition_reference(0, fan + (extra,)) == (
        source.trim_partition_reference(0, fan)
    )


def test_nonmanifold_shared_vertex_and_edge_refuse_ambiguous_chain(
    sector: tuple[MeshingDomainBoundarySource, tuple[Point, ...], tuple[Triangle, ...]],
) -> None:
    source, chain, fan = sector
    branch = (chain[0], (Fraction(2), Fraction(0)), (Fraction(2), Fraction(1)))
    with pytest.raises(ValueError, match="no unique boundary chain"):
        source.trim_partition_reference(0, fan + (branch,))
    with pytest.raises(ValueError, match="no unique boundary chain"):
        source.trim_partition_reference(0, fan + (fan[0], fan[0]))


def test_degenerate_triangle_error_precedes_boundary_chain_error(
    sector: tuple[MeshingDomainBoundarySource, tuple[Point, ...], tuple[Triangle, ...]],
) -> None:
    source, chain, fan = sector
    degenerate = (chain[0], chain[0], chain[1])
    with pytest.raises(
        ValueError, match="Target chart triangles must have nonzero chart area\\."
    ):
        source.trim_partition_reference(0, fan[1:] + (degenerate,))
