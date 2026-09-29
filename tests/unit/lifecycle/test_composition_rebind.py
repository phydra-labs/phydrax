#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Cross-owner composition rebind: dispositions, bindings, transports, commit."""

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


lc = phx.lifecycle


def _entry(
    value: object,
    entry_id: str,
    role: lc.CompositionRole,
    structure: str,
    /,
    *,
    revision: str | None = None,
    semantics: str | None = None,
    dependencies: Sequence[lc.CompositionDependency] = (),
) -> lc.CompositionEntry:
    return lc.CompositionEntry(
        value,
        entry_id=entry_id,
        role=role,
        owner_id=entry_id.split("/")[0],
        structure_id=structure,
        revision_id=structure if revision is None else revision,
        semantics_id=f"{entry_id}:meaning" if semantics is None else semantics,
        dependencies=dependencies,
    )


def _source() -> lc.Composition:
    """A solid on mesh m0 and an unchanged fluid, with a probe and a factor."""
    mesh = _entry("coarse-mesh", "solid/mesh", "topology", "m0")
    fluid = _entry("fluid-grid", "fluid/grid", "topology", "g0")
    on_mesh = (mesh.binding("structure"),)
    return lc.Composition(
        (
            mesh,
            fluid,
            _entry(
                jnp.asarray((1.0, 3.0)),
                "solid/T",
                "physical-state",
                "m0",
                revision="r0",
                dependencies=on_mesh,
            ),
            _entry("probe-m0", "solid/probe", "observation", "p0", dependencies=on_mesh),
            _entry("lu-m0", "solid/factor", "preconditioner", "f0", dependencies=on_mesh),
            _entry(
                jnp.asarray((0.5, 0.25, 0.125)),
                "fluid/T",
                "physical-state",
                "g0",
                revision="s0",
                dependencies=(fluid.binding("structure"),),
            ),
        ),
        boundary_id="accepted-window-4",
    )


def _refined(source: lc.Composition, /) -> dict[str, lc.CompositionEntry]:
    mesh = _entry("fine-mesh", "solid/mesh", "topology", "m1")
    on_mesh = (mesh.binding("structure"),)
    return {
        "mesh": mesh,
        "T": _entry(
            jnp.asarray((1.0, 2.0, 3.0)),
            "solid/T",
            "physical-state",
            "m1",
            revision="r1",
            semantics=source.entry("solid/T").semantics_id,
            dependencies=on_mesh,
        ),
        "probe": _entry(
            "probe-m1",
            "solid/probe",
            "observation",
            "p1",
            semantics=source.entry("solid/probe").semantics_id,
            dependencies=on_mesh,
        ),
        "factor": _entry(
            "lu-m1",
            "solid/factor",
            "preconditioner",
            "f1",
            semantics=source.entry("solid/factor").semantics_id,
            dependencies=on_mesh,
        ),
    }


def _remap(
    target: lc.CompositionEntry,
    /,
    *,
    successful: bool = True,
    target_content: float = 4.0,
    structure: str = "m0",
) -> lc.CompositionTransport:
    # Coarse content 1 + 3 = 4 against unit measures; the staged fine content too.
    return lc.CompositionTransport(
        "physical-remap",
        ("solid/T",),
        (target,),
        source_structure_ids=(structure,),
        route_id="p1-prolongation:m0->m1",
        successful=jnp.asarray(successful),
        source_content=jnp.asarray((4.0,)),
        target_content=jnp.asarray((target_content,)),
        content_tolerance=jnp.asarray((1e-12,)),
    )


def _rebind(
    source: lc.Composition,
    /,
    *,
    retain: Sequence[str] = ("fluid/grid", "fluid/T"),
    reprepare: Sequence[lc.CompositionEntry] | None = None,
    transports: Sequence[lc.CompositionTransport] | None = None,
    invalidate: Sequence[str] = (),
) -> lc.CompositionRebind:
    staged = _refined(source)
    return lc.CompositionRebind(
        source,
        retain=retain,
        reprepare=(staged["mesh"], staged["probe"], staged["factor"])
        if reprepare is None
        else reprepare,
        transports=(_remap(staged["T"]),) if transports is None else transports,
        invalidate=invalidate,
    )


def test_accepted_rebind_publishes_one_consistent_composition() -> None:
    source = _source()
    receipt = lc.commit_composition_rebind(_rebind(source), accepted_boundary=True)
    published = receipt.composition

    assert receipt.published and receipt.transport_accepted == (True,)
    assert receipt.retained == ("fluid/T", "fluid/grid")
    assert receipt.reprepared == ("solid/factor", "solid/mesh", "solid/probe")
    assert receipt.remapped == ("solid/T",)
    assert receipt.consumed == () and receipt.invalidated == ()
    # The unchanged owner is the very same object; the changed owner is new.
    assert published.value("fluid/T") is source.value("fluid/T")
    np.testing.assert_array_equal(published.value("solid/T"), (1.0, 2.0, 3.0))
    assert published.boundary_id == source.boundary_id
    assert receipt.source_structure_id == source.structure_id
    assert published.structure_id != source.structure_id
    assert receipt.candidate_composition_id == published.composition_id


@pytest.mark.parametrize(
    ("accepted_boundary", "successful", "target_content", "transport_accepted"),
    (
        pytest.param(False, True, 4.0, (True,), id="rejected-boundary"),
        pytest.param(True, False, 4.0, (False,), id="failed-owner-route"),
        pytest.param(True, True, 4.0 + 1e-6, (False,), id="content-not-conserved"),
    ),
)
def test_refused_commit_returns_the_original_composition(
    accepted_boundary: bool,
    successful: bool,
    target_content: float,
    transport_accepted: tuple[bool, ...],
) -> None:
    source = _source()
    staged = _refined(source)
    rebind = _rebind(
        source,
        transports=(
            _remap(staged["T"], successful=successful, target_content=target_content),
        ),
    )
    receipt = lc.commit_composition_rebind(rebind, accepted_boundary=accepted_boundary)

    assert not receipt.published
    assert receipt.boundary_accepted is accepted_boundary
    assert receipt.transport_accepted == transport_accepted
    assert receipt.composition is source


def test_retaining_a_derived_artifact_on_a_changed_structure_is_refused() -> None:
    source = _source()
    staged = _refined(source)
    with pytest.raises(
        ValueError, match="'solid/probe' is stale against the structure of 'solid/mesh'"
    ):
        _rebind(
            source,
            retain=("fluid/grid", "fluid/T", "solid/probe"),
            reprepare=(staged["mesh"], staged["factor"]),
        )


def test_invalidating_a_derived_artifact_drops_it() -> None:
    source = _source()
    staged = _refined(source)
    receipt = lc.commit_composition_rebind(
        _rebind(
            source,
            reprepare=(staged["mesh"], staged["factor"]),
            invalidate=("solid/probe",),
        ),
        accepted_boundary=True,
    )
    assert receipt.invalidated == ("solid/probe",)
    assert "solid/probe" not in receipt.composition.entry_ids


def test_state_cannot_be_invalidated() -> None:
    with pytest.raises(ValueError, match="'solid/T' .* cannot be invalidated"):
        _rebind(_source(), transports=(), invalidate=("solid/T",))


def test_state_cannot_be_reprepared_from_nothing() -> None:
    source = _source()
    staged = _refined(source)
    derived = (staged["mesh"], staged["probe"], staged["factor"])
    with pytest.raises(ValueError, match="'solid/T' .* cannot be reprepared"):
        _rebind(source, transports=(), reprepare=(*derived, staged["T"]))


@pytest.mark.parametrize(
    ("retain", "message"),
    (
        pytest.param(
            ("fluid/grid",), "lack an explicit rebind disposition: fluid/T", id="missing"
        ),
        pytest.param(
            ("fluid/grid", "fluid/T", "solid/T"),
            "more than one rebind disposition: solid/T",
            id="repeated",
        ),
        pytest.param(
            ("fluid/grid", "fluid/T", "solid/absent"),
            "unknown source entry 'solid/absent'",
            id="unknown",
        ),
    ),
)
def test_every_source_entry_takes_exactly_one_disposition(
    retain: tuple[str, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _rebind(_source(), retain=retain)


def test_numeric_refresh_commits_a_new_revision_with_identical_layout() -> None:
    source = _source()
    refreshed = _entry(
        jnp.asarray((0.75, 0.5, 0.25)),
        "fluid/T",
        "physical-state",
        "g0",
        revision="s1",
        semantics=source.entry("fluid/T").semantics_id,
        dependencies=source.entry("fluid/T").dependencies,
    )
    solid = ("solid/mesh", "solid/T", "solid/probe", "solid/factor")
    receipt = lc.commit_composition_rebind(
        lc.CompositionRebind(source, retain=("fluid/grid", *solid), refresh=(refreshed,)),
        accepted_boundary=True,
    )

    assert receipt.refreshed == ("fluid/T",)
    np.testing.assert_array_equal(receipt.composition.value("fluid/T"), (0.75, 0.5, 0.25))
    # Structure is unchanged; only the boundary revision moved.
    assert receipt.composition.structure_id == source.structure_id
    assert receipt.composition.composition_id != source.composition_id


@pytest.mark.parametrize(
    ("value", "revision", "message"),
    (
        pytest.param(
            jnp.asarray((1.0, 2.0)), "s1", "changed its PyTree layout", id="layout"
        ),
        pytest.param(
            jnp.asarray((1.0, 2.0, 3.0)), "s0", "must name a new revision", id="revision"
        ),
    ),
)
def test_numeric_refresh_refuses_structural_change(
    value: object, revision: str, message: str
) -> None:
    source = _source()
    staged = _entry(
        value,
        "fluid/T",
        "physical-state",
        "g0",
        revision=revision,
        semantics=source.entry("fluid/T").semantics_id,
        dependencies=source.entry("fluid/T").dependencies,
    )
    solid = ("solid/mesh", "solid/T", "solid/probe", "solid/factor")
    with pytest.raises(ValueError, match=message):
        lc.CompositionRebind(source, retain=("fluid/grid", *solid), refresh=(staged,))


@pytest.mark.parametrize(
    "bindings",
    (
        pytest.param("rebound", id="structure-rebound-to-new-mesh"),
        pytest.param("dropped", id="dependency-dropped"),
    ),
)
def test_numeric_refresh_cannot_rebind_state_onto_another_structure(
    bindings: str,
) -> None:
    # A renumbered mesh with the same DOF count: the state layout is unchanged,
    # so only the dependency identity distinguishes a silent re-binding of
    # coarse-mesh values onto the new mesh from a legitimate transport.
    source = _source()
    staged = _refined(source)
    renumbered = _entry("renumbered-mesh", "solid/mesh", "topology", "m1")
    dependencies = (renumbered.binding("structure"),) if bindings == "rebound" else ()
    refreshed = _entry(
        jnp.asarray((1.0, 3.0)),
        "solid/T",
        "physical-state",
        "m0",
        revision="r1",
        semantics=source.entry("solid/T").semantics_id,
        dependencies=dependencies,
    )
    reprepared = (
        (renumbered, staged["probe"], staged["factor"]) if bindings == "rebound" else ()
    )
    retained = ("fluid/grid", "fluid/T") + (
        () if bindings == "rebound" else ("solid/mesh", "solid/probe", "solid/factor")
    )
    with pytest.raises(ValueError, match="must keep its dependencies"):
        lc.CompositionRebind(
            source, retain=retained, reprepare=reprepared, refresh=(refreshed,)
        )


def test_numeric_refresh_advances_bound_revisions_together() -> None:
    temperature = _entry(
        jnp.asarray((0.5, 0.25)), "fluid/T", "physical-state", "g0", revision="s0"
    )
    probe = _entry(
        jnp.asarray((0.375,)),
        "fluid/probe",
        "observation",
        "p0",
        revision="o0",
        dependencies=(temperature.binding("revision"),),
    )
    source = lc.Composition((temperature, probe), boundary_id="accepted-window-1")
    advanced = _entry(
        jnp.asarray((0.75, 0.5)), "fluid/T", "physical-state", "g0", revision="s1"
    )
    observed = _entry(
        jnp.asarray((0.625,)),
        "fluid/probe",
        "observation",
        "p0",
        revision="o1",
        dependencies=(advanced.binding("revision"),),
    )
    receipt = lc.commit_composition_rebind(
        lc.CompositionRebind(source, refresh=(advanced, observed)),
        accepted_boundary=True,
    )
    assert receipt.published
    assert receipt.composition.entry("fluid/probe").dependencies == (
        lc.CompositionDependency("fluid/T", "revision", "s1"),
    )


def test_transport_must_consume_the_structure_it_was_prepared_from() -> None:
    source = _source()
    staged = _refined(source)
    with pytest.raises(ValueError, match="prepared from another structure"):
        _rebind(source, transports=(_remap(staged["T"], structure="m-other"),))
    renamed = _entry(
        jnp.asarray((1.0, 2.0, 3.0)),
        "solid/T",
        "physical-state",
        "m1",
        semantics="pressure",
        dependencies=(staged["mesh"].binding("structure"),),
    )
    with pytest.raises(ValueError, match="one role and scientific meaning"):
        _rebind(source, transports=(_remap(renamed),))


def test_ownership_migration_must_report_moved_content() -> None:
    target = _entry(jnp.asarray((1.0,)), "fluid/T", "physical-state", "g1")
    with pytest.raises(ValueError, match="created rows are never physical content"):
        lc.CompositionTransport(
            "ownership-migration",
            ("fluid/T",),
            (target,),
            source_structure_ids=("g0",),
            route_id="repartition",
            successful=jnp.asarray(True),
        )


def test_budget_transport_needs_conserved_content_evidence() -> None:
    ledger = _entry(jnp.asarray((2.0, -2.0)), "coupling/budget", "budget", "rows-0")
    source = lc.Composition((ledger,), boundary_id="accepted-window-3")
    replaced = _entry(
        jnp.asarray((7.0, 5.0)),
        "coupling/budget",
        "budget",
        "rows-1",
        semantics=ledger.semantics_id,
    )
    route = lc.CompositionTransport(
        "physical-remap",
        ("coupling/budget",),
        (replaced,),
        source_structure_ids=("rows-0",),
        route_id="ledger-rows",
        successful=jnp.asarray(True),
    )
    with pytest.raises(ValueError, match="budget crosses a rebind only with"):
        lc.CompositionRebind(source, transports=(route,))
    # The same ledger move with content evidence stages and is judged at commit.
    evidenced = lc.CompositionTransport(
        "physical-remap",
        ("coupling/budget",),
        (replaced,),
        source_structure_ids=("rows-0",),
        route_id="ledger-rows",
        successful=jnp.asarray(True),
        source_content=jnp.asarray((0.0,)),
        target_content=jnp.asarray((12.0,)),
        content_tolerance=jnp.asarray((1e-12,)),
    )
    receipt = lc.commit_composition_rebind(
        lc.CompositionRebind(source, transports=(evidenced,)), accepted_boundary=True
    )
    assert not receipt.published and receipt.transport_accepted == (False,)


def _parameterized(binding_facet: lc.CompositionFacet) -> lc.Composition:
    weights = _entry(
        jnp.asarray((0.1, 0.2)),
        "model/weights",
        "model-parameter",
        "w-layout",
        semantics="closure-weights",
    )
    grid = _entry("grid", "fluid/grid", "topology", "g0")
    closure = _entry(
        "closure-on-g0",
        "fluid/closure",
        "prepared-graph",
        "c0",
        semantics="closure",
        dependencies=(weights.binding(binding_facet), grid.binding("structure")),
    )
    return lc.Composition((weights, grid, closure), boundary_id="accepted-window-2")


def test_parameters_bind_their_consumers_only_by_semantics() -> None:
    with pytest.raises(ValueError, match="parameters bind only by semantics"):
        _parameterized("structure")


def test_parameters_survive_only_through_a_same_semantics_rebinding() -> None:
    source = _parameterized("semantics")
    grid = _entry("fine-grid", "fluid/grid", "topology", "g1")
    weights = source.entry("model/weights")
    unbound = _entry(
        "closure-on-g1",
        "fluid/closure",
        "prepared-graph",
        "c1",
        semantics="closure",
        dependencies=(grid.binding("structure"),),
    )
    with pytest.raises(
        ValueError, match="explicit same-semantics binding of a consumer: model/weights"
    ):
        lc.CompositionRebind(source, retain=("model/weights",), reprepare=(grid, unbound))
    bound = _entry(
        "closure-on-g1",
        "fluid/closure",
        "prepared-graph",
        "c1",
        semantics="closure",
        dependencies=(grid.binding("structure"), weights.binding("semantics")),
    )
    receipt = lc.commit_composition_rebind(
        lc.CompositionRebind(source, retain=("model/weights",), reprepare=(grid, bound)),
        accepted_boundary=True,
    )
    assert receipt.published
    assert receipt.composition.value("model/weights") is weights.value


@pytest.mark.parametrize("disposition", ("refresh", "transport"))
def test_refreshed_or_transported_parameters_still_need_a_consumer(
    disposition: str,
) -> None:
    source = _parameterized("semantics")
    grid = _entry("fine-grid", "fluid/grid", "topology", "g1")
    weights = source.entry("model/weights")
    unbound = _entry(
        "closure-on-g1",
        "fluid/closure",
        "prepared-graph",
        "c1",
        semantics="closure",
        dependencies=(grid.binding("structure"),),
    )
    advanced = _entry(
        jnp.asarray((0.3, 0.4)),
        "model/weights",
        "model-parameter",
        "w-layout",
        revision="w1",
        semantics=weights.semantics_id,
    )
    route = lc.CompositionTransport(
        "physical-remap",
        ("model/weights",),
        (advanced,),
        source_structure_ids=("w-layout",),
        route_id="weights-remap",
        successful=jnp.asarray(True),
    )
    refreshed = disposition == "refresh"
    with pytest.raises(
        ValueError, match="explicit same-semantics binding of a consumer: model/weights"
    ):
        lc.CompositionRebind(
            source,
            reprepare=(grid, unbound),
            refresh=(advanced,) if refreshed else (),
            transports=() if refreshed else (route,),
        )


def test_commit_requires_an_explicit_host_boundary_decision() -> None:
    rebind = _rebind(_source())
    with pytest.raises(TypeError, match="explicit host bool"):
        lc.commit_composition_rebind(
            rebind,
            accepted_boundary=jnp.asarray(True),  # ty: ignore[invalid-argument-type]
        )
