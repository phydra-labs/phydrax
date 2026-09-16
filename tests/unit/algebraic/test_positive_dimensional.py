#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import numpy as np
import pytest

from phydrax.algebraic._grading import PolynomialVariableGroup
from phydrax.algebraic._positive_dimensional import (
    DecompositionStatus,
    MonodromyEvidence,
    MultigradedWitnessCollection,
    NumericalComponent,
    NumericalDecompositionResult,
    PathInventory,
    PathRecord,
    PathStatus,
    PseudoWitnessSet,
    RegenerationEdge,
    RegenerationPlan,
    RegenerationStage,
    TraceTestEvidence,
    WitnessSet,
)


def _parabola_witness():
    return WitnessSet(
        "parabola-and-origin",
        1,
        np.asarray([[0.0, 1.0]]),
        np.asarray([-1.0]),
        np.asarray([[-1.0, 1.0], [1.0, 1.0]], dtype=complex),
        np.asarray([0.0, 0.0]),
    )


def _success_inventory(batch, targets, *, prefix="path"):
    records = tuple(
        PathRecord(
            f"{prefix}:{source}",
            batch,
            source,
            PathStatus.SUCCESS,
            target_index=target,
            residual_norm=0.0,
        )
        for source, target in enumerate(targets)
    )
    return PathInventory(
        tuple(record.path_id for record in records),
        records,
        path_capacity=len(records),
    )


def _swap_monodromy(witness):
    inventory = _success_inventory("loop-0", (1, 0), prefix="loop-0")
    return MonodromyEvidence(
        witness.witness_id,
        witness.degree,
        ("loop-0",),
        (("loop-0", (1, 0)),),
        inventory,
    )


def _trace(witness, residual):
    records = tuple(
        PathRecord(
            f"trace:{sample}:{source}",
            f"trace:{sample}",
            source,
            PathStatus.SUCCESS,
            target_index=source,
            residual_norm=0.0,
        )
        for sample in range(3)
        for source in range(2)
    )
    inventory = PathInventory(
        tuple(record.path_id for record in records),
        records,
        path_capacity=6,
    )
    return TraceTestEvidence(
        witness.witness_id,
        (0, 1),
        np.asarray([-1.0, 0.0, 1.0]),
        np.asarray([[-1.0, 2.0], [0.0, 2.0], [1.0, 2.0]]),
        residual,
        1.0e-8,
        inventory,
    )


def _parabola_collection(witness):
    group = PolynomialVariableGroup("plane", (0, 1))
    return MultigradedWitnessCollection(
        witness.system_id,
        2,
        (group,),
        (((1,), witness),),
    )


def test_isolated_point_and_parabola_form_dimension_indexed_witness_collection():
    group = PolynomialVariableGroup("plane", (0, 1))
    isolated = WitnessSet(
        "parabola-and-origin",
        0,
        np.zeros((0, 2)),
        np.zeros((0,)),
        np.asarray([[0.0, 0.0]], dtype=complex),
        np.asarray([0.0]),
    )
    parabola = _parabola_witness()

    collection = MultigradedWitnessCollection(
        "parabola-and-origin",
        2,
        (group,),
        (((1,), parabola), ((0,), isolated)),
    )

    assert collection.multidegrees == ((0,), (1,))
    assert collection.witness((0,)).dimension == 0
    assert collection.witness((0,)).degree == 1
    assert collection.witness((1,)).dimension == 1
    assert collection.witness((1,)).degree == 2
    np.testing.assert_allclose(
        np.asarray(collection.witness((1,)).points),
        np.asarray([[-1.0, 1.0], [1.0, 1.0]]),
    )


def test_witness_and_multigraded_slice_validation_rejects_malformed_data():
    with pytest.raises(ValueError, match="exactly d"):
        WitnessSet(
            "system",
            1,
            np.zeros((0, 2)),
            np.zeros((0,)),
            np.asarray([[0.0, 0.0]]),
            np.asarray([0.0]),
        )
    with pytest.raises(ValueError, match="do not lie"):
        WitnessSet(
            "system",
            1,
            np.asarray([[1.0, 0.0]]),
            np.asarray([0.0]),
            np.asarray([[1.0, 0.0]]),
            np.asarray([0.0]),
        )

    witness = WitnessSet(
        "system",
        1,
        np.asarray([[1.0, 1.0]]),
        np.asarray([0.0]),
        np.asarray([[1.0, -1.0]]),
        np.asarray([0.0]),
    )
    groups = (
        PolynomialVariableGroup("x", (0,)),
        PolynomialVariableGroup("y", (1,)),
    )
    with pytest.raises(ValueError, match="supported"):
        MultigradedWitnessCollection(
            "system",
            2,
            groups,
            (((1, 0), witness),),
        )


def test_monodromy_validates_each_permutation_against_endpoint_inventory():
    witness = _parabola_witness()
    inventory = _success_inventory("loop-0", (1, 0), prefix="loop-0")

    evidence = MonodromyEvidence(
        witness.witness_id,
        2,
        ("loop-0",),
        (("loop-0", (1, 0)),),
        inventory,
    )
    assert evidence.transitive
    assert evidence.orbits == ((0, 1),)

    with pytest.raises(ValueError, match="valid permutations"):
        MonodromyEvidence(
            witness.witness_id,
            2,
            ("loop-0",),
            (("loop-0", (1, 1)),),
            inventory,
        )
    with pytest.raises(ValueError, match="disagree"):
        MonodromyEvidence(
            witness.witness_id,
            2,
            ("loop-0",),
            (("loop-0", (0, 1)),),
            inventory,
        )


def test_path_inventories_preserve_partial_failure_and_budget_exhaustion():
    partial = PathInventory(
        ("p0", "p1"),
        (
            PathRecord(
                "p0",
                "batch",
                0,
                PathStatus.SUCCESS,
                target_index=0,
                residual_norm=1.0e-12,
            ),
            PathRecord(
                "p1", "batch", 1, PathStatus.TRACKING_FAILED, diagnostic="step limit"
            ),
        ),
        path_capacity=2,
    )
    exhausted = PathInventory(
        ("p0", "p1"),
        (
            PathRecord(
                "p0",
                "batch",
                0,
                PathStatus.SUCCESS,
                target_index=0,
                residual_norm=1.0e-12,
            ),
            PathRecord("p1", "batch", 1, PathStatus.NOT_ATTEMPTED),
        ),
        path_capacity=2,
        budget_exhausted=True,
    )

    assert not partial.successful
    assert partial.successful_count == 1
    assert not exhausted.successful
    assert exhausted.budget_exhausted
    with pytest.raises(ValueError, match="budget exhaustion"):
        PathInventory(
            ("p0",),
            (PathRecord("p0", "batch", 0, PathStatus.NOT_ATTEMPTED),),
            path_capacity=1,
        )


def test_trace_failure_prevents_complete_decomposition_evidence_claim():
    witness = _parabola_witness()
    monodromy = _swap_monodromy(witness)
    trace = _trace(witness, 1.0e-3)
    component = NumericalComponent(witness, (0, 1), monodromy, trace)

    result = NumericalDecompositionResult(
        _parabola_collection(witness),
        (component,),
        (monodromy,),
        (trace,),
    )

    assert not trace.passed
    assert result.status is DecompositionStatus.TRACE_TEST_FAILED
    assert not result.evidence_complete
    assert "not-exact-irreducibility-or-completeness" in result.claim
    with pytest.raises(AttributeError):
        _ = result.complete
    with pytest.raises(AttributeError):
        _ = component.irreducible


def test_passing_monodromy_and_trace_are_qualified_numerical_evidence_only():
    witness = _parabola_witness()
    monodromy = _swap_monodromy(witness)
    trace = _trace(witness, 0.0)
    component = NumericalComponent(witness, (0, 1), monodromy, trace)

    result = NumericalDecompositionResult(
        _parabola_collection(witness),
        (component,),
        (monodromy,),
        (trace,),
    )

    assert result.status is DecompositionStatus.EVIDENCE_COMPLETE
    assert result.evidence_complete
    assert result.claim == (
        "numerical-decomposition-evidence-not-exact-irreducibility-or-completeness"
    )


def test_pseudo_witness_and_regeneration_contracts_validate_dimensions_and_ids():
    pseudo = PseudoWitnessSet(
        "source-system",
        "parabola-map",
        1,
        1,
        np.zeros((0, 1)),
        np.zeros((0,)),
        np.asarray([[1.0, 0.0]]),
        np.asarray([-1.0]),
        np.asarray([[-1.0], [1.0]]),
        np.asarray([[1.0, 1.0], [1.0, -1.0]]),
        np.asarray([0.0, 0.0]),
        image_degree=2,
    )
    assert pseudo.image_degree == 2
    assert pseudo.fiber_degree == 1

    initial = RegenerationStage("initial", (), 2)
    first = RegenerationStage("first", (0,), 1)
    edge = RegenerationEdge(
        initial,
        first,
        0,
        ("regen:0", "regen:1"),
        path_capacity=2,
    )
    plan = RegenerationPlan("source-system", (initial, first), (edge,), path_capacity=2)
    assert plan.edges[0].target_stage_id == first.stage_id
    with pytest.raises(ValueError, match="add exactly"):
        RegenerationEdge(
            first,
            initial,
            1,
            ("invalid",),
            path_capacity=1,
        )
