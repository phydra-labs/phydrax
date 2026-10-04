from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from phydrax.nn.atomistic._radial import (
    AgnesiTransform,
    BesselRadialBasis,
    PolynomialCutoff,
    RadialEmbedding,
    RadialMLP,
    RadialPostprocess,
)
from phydrax.nn.atomistic._radial_projection import (
    prepare_radial_tables,
    PreparedRadialTables,
    qualify_radial_tables,
    radial_source_revision,
    RadialSpeciesBinding,
    RadialTableDeclaration,
    RadialTableLayout,
    RadialTableQualificationError,
    RadialTableQualificationPolicy,
    select_radial_table_layout,
    StaleRadialTableBinding,
)


pytestmark = pytest.mark.strict_jax

CUTOFF = 5.0
GRID_MIN = 1.0e-12
NODES = 256
DOMAIN = (1, 6, 8)


def _embedding(*, agnesi: bool) -> RadialEmbedding:
    transform = (
        AgnesiTransform(
            np.asarray([0.31, 0.76, 0.66]),
            atomic_numbers=DOMAIN,
            a=1.0805,
            q=0.9183,
            p=4.5791,
        )
        if agnesi
        else None
    )
    return RadialEmbedding(
        BesselRadialBasis.native(CUTOFF, 8),
        PolynomialCutoff(CUTOFF, 5),
        transform=transform,
    )


def _mlp(postprocess: RadialPostprocess = "none") -> RadialMLP:
    return RadialMLP.initialize(
        (8, 16, 16, 4),
        key=jr.key(5),
        postprocess=postprocess,
    )


def _tables(
    *,
    agnesi: bool = True,
    layout: RadialTableLayout = "projected-width",
    active: tuple[int, ...] = (0, 2),
) -> tuple[RadialEmbedding, RadialMLP, PreparedRadialTables]:
    embedding = _embedding(agnesi=agnesi)
    mlp = _mlp()
    tables = prepare_radial_tables(
        embedding,
        mlp,
        RadialTableDeclaration(GRID_MIN, NODES, layout=layout),
        species_domain=DOMAIN,
        active_species=active,
    )
    return embedding, mlp, tables


def _source_compact_radial_slopes(values: np.ndarray, h: float) -> np.ndarray:
    """Independent transcription of the pinned source slope algorithm
    (compact_radial.cpp:326-365): left not-a-knot, right slope zero."""
    n = values.shape[0]
    unknowns = n - 2
    lower = np.zeros(unknowns)
    diagonal = np.full(unknowns, 4.0)
    upper = np.zeros(unknowns)
    rhs = np.zeros(unknowns)
    left = 2.0 * (-values[0] + 2.0 * values[1] - values[2]) / h
    upper[0] = 2.0
    rhs[0] = 3.0 * (values[2] - values[0]) / h - left
    for node in range(2, n - 1):
        row = node - 1
        lower[row] = 1.0
        if node + 1 <= n - 2:
            upper[row] = 1.0
        rhs[row] = 3.0 * (values[node + 1] - values[node - 1]) / h
    for row in range(1, unknowns):
        factor = lower[row] / diagonal[row - 1]
        diagonal[row] -= factor * upper[row - 1]
        rhs[row] -= factor * rhs[row - 1]
    solution = np.zeros(unknowns)
    solution[-1] = rhs[-1] / diagonal[-1]
    for row in range(unknowns - 2, -1, -1):
        solution[row] = (rhs[row] - upper[row] * solution[row + 1]) / diagonal[row]
    slopes = np.zeros(n)
    slopes[1:-1] = solution
    slopes[0] = slopes[2] + left
    return slopes


def test_table_slopes_follow_source_not_a_knot_clamped_construction() -> None:
    _, _, tables = _tables()
    values = np.asarray(tables.values)
    slopes = np.asarray(tables.slopes)
    h = float(tables.grid.spacing)
    nodes = np.asarray(tables.grid.nodes)

    for row in range(values.shape[0]):
        for channel in range(values.shape[2]):
            np.testing.assert_allclose(
                slopes[row, :, channel],
                _source_compact_radial_slopes(values[row, :, channel], h),
                rtol=1e-9,
                atol=1e-9,
            )
    oracle = CubicSpline(
        nodes, values[0], bc_type=("not-a-knot", (1, np.zeros(values.shape[2])))
    )
    np.testing.assert_allclose(
        slopes[0], oracle.derivative()(nodes), rtol=1e-9, atol=1e-9
    )
    # Fault adequacy: local secant slopes are not the source construction.
    secant = np.gradient(values[0, :, 0], h)
    assert np.max(np.abs(secant - slopes[0, :, 0])) > 1e-6


def test_pair_dependent_rows_cover_every_ordered_active_pair() -> None:
    embedding, mlp, tables = _tables(agnesi=True, active=(0, 2))
    radius = jnp.asarray([1.3, 1.3, 2.2, 2.2], dtype=jnp.float64)
    sender = jnp.asarray([0, 2, 0, 2], dtype=jnp.int32)
    receiver = jnp.asarray([2, 0, 0, 2], dtype=jnp.int32)
    exact = mlp(embedding(radius, sender, receiver))

    assert tables.binding.row_count == 4
    assert tables.binding.species_domain == DOMAIN
    np.testing.assert_allclose(
        tables.evaluate(radius, sender, receiver), exact, atol=1e-5
    )


def test_pair_independent_realization_tabulates_one_row() -> None:
    _, _, tables = _tables(agnesi=False, active=(0, 1, 2))

    assert tables.binding.row_count == 1
    assert tables.values.shape[0] == 1


def test_unbound_species_and_below_support_radii_are_refused() -> None:
    _, _, tables = _tables(active=(0, 2))
    one = jnp.ones((1,), dtype=jnp.int32)
    zero = jnp.zeros((1,), dtype=jnp.int32)

    with pytest.raises(eqx.EquinoxRuntimeError, match="declared support"):
        tables.evaluate(jnp.asarray([1.0]), one, zero)
    with pytest.raises(eqx.EquinoxRuntimeError, match="declared support"):
        tables.evaluate(jnp.asarray([0.5 * GRID_MIN]), zero, zero)
    assert not bool(tables.support(jnp.asarray([1.0]), one, zero)[0])


@pytest.mark.parametrize(
    ("sender", "receiver"),
    [(-1, 0), (0, -3), (len(DOMAIN), 0), (0, 999)],
    ids=["negative-sender", "negative-receiver", "sender-past-domain", "huge-receiver"],
)
def test_out_of_domain_species_indices_are_unsupported_not_another_pair(
    sender: int, receiver: int
) -> None:
    # Every domain species is active, so a wrapped or clipped index would
    # silently evaluate a real species pair's row.
    _, _, tables = _tables(active=(0, 1, 2))
    radius = jnp.asarray([1.0])
    senders = jnp.asarray([sender], dtype=jnp.int32)
    receivers = jnp.asarray([receiver], dtype=jnp.int32)

    assert not bool(tables.support(radius, senders, receivers)[0])
    with pytest.raises(eqx.EquinoxRuntimeError, match="declared support"):
        tables.evaluate(radius, senders, receivers)
    with pytest.raises(eqx.EquinoxRuntimeError, match="declared support"):
        jax.block_until_ready(
            eqx.filter_jit(lambda t, r, s, q: t.evaluate(r, s, q))(
                tables, radius, senders, receivers
            )
        )
    padded = tables.evaluate(
        radius, senders, receivers, valid=jnp.zeros((1,), dtype=jnp.bool_)
    )
    np.testing.assert_array_equal(np.asarray(padded), np.zeros((1, 4)))


def test_atom_type_id_zero_is_a_species_domain_member() -> None:
    embedding = _embedding(agnesi=False)
    mlp = _mlp()
    tables = prepare_radial_tables(
        embedding,
        mlp,
        RadialTableDeclaration(0.3, NODES, layout="projected-width"),
        species_domain=(0, 2),
        active_species=(0, 1),
    )
    tables.validate(embedding, mlp)
    assert tables.binding.species_domain == (0, 2)
    with pytest.raises(ValueError, match="nonnegative"):
        RadialSpeciesBinding((-1, 2), (0,), pair_dependent=False)


@pytest.mark.parametrize(
    "where",
    [
        lambda e: e.basis.prefactor,
        lambda e: e.basis.frequencies,
        lambda e: e.transform.covalent_radii,
    ],
    ids=["bessel-prefactor", "bessel-frequencies", "agnesi-radii"],
)
def test_table_freshness_rebuilds_the_embedding_instead_of_trusting_its_label(
    where: Any,
) -> None:
    embedding = _embedding(agnesi=True)
    mlp = _mlp()
    tables = prepare_radial_tables(
        embedding,
        mlp,
        RadialTableDeclaration(0.3, 8, layout="projected-width"),
        species_domain=DOMAIN,
        active_species=(0,),
    )
    mutated = eqx.tree_at(where, embedding, where(embedding) * 2.0)
    assert mutated.embedding_id == tables.embedding_id
    with pytest.raises(StaleRadialTableBinding, match="not admitted"):
        tables.require_current(mutated, mlp)
    with pytest.raises(StaleRadialTableBinding, match="not admitted"):
        tables.validate(mutated, mlp)


def test_table_is_exact_zero_at_and_beyond_cutoff_and_padded_lanes() -> None:
    _, _, tables = _tables()
    zero = jnp.zeros((4,), dtype=jnp.int32)
    radius = jnp.asarray([CUTOFF, 1.01 * CUTOFF, 3.0 * CUTOFF, 0.0], dtype=jnp.float64)
    valid = jnp.asarray([True, True, True, False])

    np.testing.assert_array_equal(
        np.asarray(tables.evaluate(radius, zero, zero, valid=valid)), np.zeros((4, 4))
    )


def test_table_forces_are_conservative_and_c1_at_cutoff() -> None:
    _, _, tables = _tables()
    zero = jnp.zeros((), dtype=jnp.int32)

    def energy(r: jax.Array) -> jax.Array:
        return jnp.sum(tables.evaluate(r, zero, zero))

    force = jax.grad(energy)
    for radius in (0.77, 2.31, 4.42):
        step = 1.0e-6
        central = (
            energy(jnp.asarray(radius + step)) - energy(jnp.asarray(radius - step))
        ) / (2.0 * step)
        np.testing.assert_allclose(
            force(jnp.asarray(radius)), central, rtol=1e-6, atol=1e-8
        )
    below = force(jnp.asarray(CUTOFF * (1.0 - 1.0e-9)))
    assert abs(float(below)) < 1e-6
    slope, curvature, interior_jump = tables.cutoff_jets()
    np.testing.assert_allclose(np.asarray(slope), 0.0, atol=1e-12)
    assert float(interior_jump) < 1e-6 * max(1.0, float(jnp.max(jnp.abs(curvature))))
    assert tables.admitted_derivative_order == 1


def test_table_derivative_at_support_lower_bound_matches_exact_network() -> None:
    # A positive grid_min makes r == grid_min an admitted query with a large
    # radial slope; its one-sided table derivative must equal the exact one.
    embedding = _embedding(agnesi=False)
    mlp = _mlp()
    grid_min = 0.3
    tables = prepare_radial_tables(
        embedding,
        mlp,
        RadialTableDeclaration(grid_min, 2048, layout="projected-width"),
        species_domain=DOMAIN,
        active_species=(0,),
    )
    zero = jnp.zeros((), dtype=jnp.int32)
    start = jnp.asarray(grid_min, dtype=jnp.float64)

    table_slope = jax.jacfwd(lambda r: tables.evaluate(r, zero, zero))(start)
    exact_slope = jax.jacfwd(lambda r: mlp(embedding(r, zero, zero)))(start)

    scale = float(jnp.max(jnp.abs(exact_slope)))
    assert scale > 1.0
    # Cubic approximation error at the boundary is about 1e-5 of the slope
    # scale; a clamped (tie-gradient) lookup would be off by a factor of two.
    assert float(jnp.max(jnp.abs(table_slope - exact_slope))) < 1e-3 * scale


def test_embedding_width_table_matches_projected_width_table() -> None:
    embedding, mlp, projected = _tables(layout="projected-width")
    _, _, embedded = _tables(layout="embedding-width")
    radius = jnp.asarray(np.linspace(0.2, 4.95, 37))
    species = jnp.zeros(radius.shape, dtype=jnp.int32)
    exact = mlp(embedding(radius, species, species))

    assert embedded.projection is not None
    np.testing.assert_allclose(
        embedded.evaluate(radius, species, species), exact, atol=1e-5
    )
    np.testing.assert_allclose(
        projected.evaluate(radius, species, species), exact, atol=1e-5
    )


def test_layout_choice_follows_objective_and_refuses_nonlinear_embedding() -> None:
    mlp = _mlp()
    bytes_choice = select_radial_table_layout(
        mlp, 4, NODES, 8, objective="minimum-table-bytes"
    )
    work_choice = select_radial_table_layout(
        mlp, 4, NODES, 8, objective="minimum-edge-work"
    )
    density = _mlp("tanh-square")
    density_choice = select_radial_table_layout(
        density, 4, NODES, 8, objective="minimum-table-bytes"
    )

    # Output width 4 < embedding width 16: projected tables are smaller and cheaper.
    assert bytes_choice.selected == "projected-width"
    assert bytes_choice.projected.table_bytes == 2 * 4 * NODES * 4 * 8
    assert work_choice.selected == "projected-width"
    assert density_choice.selected == "projected-width"
    assert density_choice.embedding is None
    with pytest.raises(ValueError, match="postprocessing"):
        prepare_radial_tables(
            _embedding(agnesi=False),
            density,
            RadialTableDeclaration(GRID_MIN, NODES, layout="embedding-width"),
            species_domain=DOMAIN,
            active_species=(0,),
        )


def test_tables_are_bound_to_source_revision_and_refuse_stale_weights() -> None:
    embedding, mlp, tables = _tables()

    tables.require_current(embedding, mlp)
    assert tables.source_revision_id == radial_source_revision(embedding, mlp).revision_id
    updated = eqx.tree_at(lambda m: m.weights[0], mlp, mlp.weights[0] * 1.001)
    with pytest.raises(StaleRadialTableBinding, match="stale"):
        tables.require_current(embedding, updated)
    with pytest.raises(StaleRadialTableBinding):
        qualify_radial_tables(
            tables,
            embedding,
            updated,
            RadialTableQualificationPolicy(
                value_tolerance=1e-5, first_derivative_tolerance=1e-3
            ),
        )


@pytest.mark.parametrize(
    "field",
    ["node_count", "active_species"],
    ids=["huge-node-count", "extra-active-species"],
)
def test_validate_refuses_corrupted_table_metadata_before_retabulation(
    field: str,
) -> None:
    embedding = _embedding(agnesi=True)
    mlp = _mlp()
    tables = prepare_radial_tables(
        embedding,
        mlp,
        RadialTableDeclaration(0.3, 8, layout="projected-width"),
        species_domain=DOMAIN,
        active_species=(0,),
    )
    tables.validate(embedding, mlp)
    # Tiny stored arrays with metadata implying a huge re-tabulation: a
    # 10**12-node grid would exhaust memory if it were rebuilt before refusal.
    if field == "node_count":
        corrupted = eqx.tree_at(
            lambda t: t.declaration,
            tables,
            RadialTableDeclaration(0.3, 10**12, layout="projected-width"),
        )
    else:
        corrupted = eqx.tree_at(
            lambda t: t.binding,
            tables,
            RadialSpeciesBinding(DOMAIN, (0, 1, 2), pair_dependent=True),
        )

    with pytest.raises(StaleRadialTableBinding, match="extents"):
        corrupted.validate(embedding, mlp)


def test_qualification_admits_first_order_and_records_held_out_errors() -> None:
    # The Agnesi transform is not smooth at r -> 0 (power q < 1), so a passing
    # gate on the full source support uses the pair-independent realization.
    embedding, mlp, tables = _tables(agnesi=False)
    policy = RadialTableQualificationPolicy(
        value_tolerance=1e-5, first_derivative_tolerance=1e-3, probe_count=33
    )
    qualification = qualify_radial_tables(tables, embedding, mlp, policy)

    assert qualification.passed, qualification.failures
    assert qualification.admitted_derivative_order == 1
    assert qualification.below_support_refused
    assert qualification.beyond_cutoff_exact_zero
    categories = {row[0] for row in qualification.maximum_errors}
    assert {"nodes", "mid-spans", "linear", "logarithmic", "near-cutoff"} <= categories
    node_values = [
        row[2] for row in qualification.maximum_errors if row[:2] == ("nodes", 0)
    ]
    assert node_values[0] < 1e-12


def test_second_derivative_request_fails_explicitly_at_cutoff_join() -> None:
    embedding, mlp, tables = _tables(agnesi=False)
    policy = RadialTableQualificationPolicy(
        value_tolerance=1e-5,
        first_derivative_tolerance=1e-3,
        second_derivative_tolerance=1e-2,
        probe_count=33,
    )
    qualification = qualify_radial_tables(tables, embedding, mlp, policy)

    assert not qualification.passed
    assert any("cutoff join is C1" in failure for failure in qualification.failures)
    assert qualification.admitted_derivative_order == 1
    assert qualification.cutoff_second_derivative_jump > 0.0
    with pytest.raises(RadialTableQualificationError, match="C1"):
        qualification.require_passed()


def test_coarse_table_fails_its_value_gate_and_retains_evidence() -> None:
    embedding = _embedding(agnesi=True)
    mlp = _mlp()
    coarse = prepare_radial_tables(
        embedding,
        mlp,
        RadialTableDeclaration(GRID_MIN, 6, layout="projected-width"),
        species_domain=DOMAIN,
        active_species=(1,),
    )
    qualification = qualify_radial_tables(
        coarse,
        embedding,
        mlp,
        RadialTableQualificationPolicy(
            value_tolerance=1e-8, first_derivative_tolerance=1e-6, probe_count=17
        ),
    )

    assert qualification.admitted_derivative_order == -1
    assert any("derivative-0" in failure for failure in qualification.failures)
    assert len(qualification.maximum_errors) > 0
