import hashlib
import json
import math

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax.applications.nucleic_acid_biophysics._construct import NucleicAcidConstruct
from phydrax.applications.nucleic_acid_biophysics.secondary_kinetics import (
    AssociationConvention,
    prepare_secondary_kinetics,
    SecondaryEnergyModel,
    SecondaryRateLaw,
    SecondaryStructureState,
    StrandComplexPartition,
)
from phydrax.qualification import ReferenceArtifactManifest
from phydrax.solver._differential import DifferentialProblem
from phydrax.solver._jump import (
    finite_state_generator,
    JumpDifferentialProblem,
    solve_direct_ssa,
    solve_jump_differential,
    solve_next_reaction,
)
from phydrax.solver._jump_hitting import event_first_hit, finite_generator_hitting
from phydrax.stochastic import JumpProcess, PoissonClockRealization


def _model(*, pair_energy=math.log(1.5), profile="pair_loop", commercial=True, **updates):
    # Independently authored exact analytical fixture, not measured biophysics.
    data = {
        "profile": profile,
        "chemistry": "DNA",
        "pairing_rule": "watson_crick",
        "temperature": 300.0,
        "energy_convention": "dimensionless_molar_G_over_RT",
        "minimum_hairpin_unpaired": 3,
        "pair_energies": {"AT": pair_energy, "TA": pair_energy, "GC": -1.0, "CG": -1.0},
        "stack_energies": {"GC/GC": -2.0} if profile == "nearest_neighbor_loop" else {},
        "hairpin_energies": {"3": 2.0, "4": 2.5, "5": 3.0},
        "bulge_energies": {"1": 1.0},
        "internal_energies": {"1,1": 1.5},
        "multibranch": [1.0, 0.5, 0.1],
        "association_initiation": 0.0,
    }
    data.update(updates)
    content = json.dumps(data, sort_keys=True).encode()
    manifest = ReferenceArtifactManifest(
        "independently-authored-analytical-ctmc-fixture",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        license_id="CC0-1.0",
        commercial_use_permitted=commercial,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"temperature_kelvin": 300.0},
        uncertainty={"analytical_definition": 0.0},
        lineage_ids=("analytical-test-definition",),
    )
    return SecondaryEnergyModel.from_bytes(
        content,
        manifest,
        requested_use={
            "commercial_use": True,
            "redistribution": False,
            "training_use": False,
            "export": False,
        },
    )


def _prepared(
    *, copies=1, alpha=1.0, rate_name="association_metropolis", pair_energy=math.log(1.5)
):
    construct = NucleicAcidConstruct(
        tuple(["a"] + [f"t{i}" for i in range(copies)]),
        tuple(["A"] + ["T"] * copies),
        ("DNA",) * (copies + 1),
        (False,) * (copies + 1),
    )
    convention = AssociationConvention(
        mode="fixed_volume",
        standard_concentration=1000.0,
        volume=alpha / (1000.0 * 6.02214076e23),
    )
    return prepare_secondary_kinetics(
        construct,
        _model(pair_energy=pair_energy),
        convention,
        SecondaryRateLaw(rate_name, 3.0, 3.0),
        temperature=300.0,
    )


def test_partition_and_pair_legality_preserve_labelled_identity():
    first = StrandComplexPartition(("a", "b", "c"), (("c",), ("b", "a")))
    second = StrandComplexPartition(("a", "b", "c"), (("a", "b"), ("c",)))
    assert first == second
    assert first.fingerprint() == second.fingerprint()
    with pytest.raises(ValueError):
        StrandComplexPartition(("a", "b", "c"), (("a", "b"), ("a",)))
    construct = NucleicAcidConstruct(("x",), ("AATT",), ("DNA",), (False,))
    keys = construct.nucleotide_keys
    with pytest.raises(ValueError, match="crossing"):
        SecondaryStructureState(construct, ((keys[0], keys[2]), (keys[1], keys[3])))
    with pytest.raises(ValueError, match="partners"):
        SecondaryStructureState(construct, ((keys[0], keys[2]), (keys[0], keys[3])))
    system = _prepared()
    joined = system.states[1]
    with pytest.raises(ValueError, match="partition"):
        SecondaryStructureState(
            system.construct, joined.pairs, partition=system.states[0].partition
        )
    move = system.moves(system.states[0])[0]
    reverse = system.moves(move.after)[0]
    assert move.kind == "join" and reverse.kind == "split"
    assert reverse.after.fingerprint() == move.before.fingerprint()


@pytest.mark.parametrize(
    "rate_name", ["metropolis", "symmetric_barrier", "association_metropolis"]
)
def test_rate_ratios_and_labelled_combinatorics_match_partition_function(rate_name):
    system = _prepared(copies=2, alpha=7.0, rate_name=rate_name, pair_energy=-1.0)
    generator = system.generator()
    q = generator.matrix
    equilibrium = system.equilibrium_probabilities()
    assert jnp.allclose(q.sum(axis=1), 0.0, atol=1e-12)
    assert jnp.all(jnp.diag(q) <= 0)
    assert jnp.allclose(equilibrium[:, None] * q, equilibrium[None, :] * q.T, atol=1e-12)
    # Two distinct T copies produce two elementary association channels, without
    # duplicating a state by permutation of complex/member ordering.
    bound_ratio = (equilibrium[1] + equilibrium[2]) / equilibrium[0]
    assert jnp.allclose(bound_ratio, 2.0 * math.e / 7.0)
    assert jnp.allclose(q[0, 1], q[0, 2])
    for state in system.states:
        assert all(
            system.decode(system.encode(move.after)) == move.after
            for move in system.moves(state)
        )


def test_legal_toggles_cannot_create_crossings_or_second_partners_under_jit():
    system = _prepared(copies=2)
    state = system.encode(system.states[1])
    rates = eqx.filter_jit(system.process.intensities)(0.0, state)
    assert int(jnp.count_nonzero(rates)) == 1
    forbidden = int(jnp.argmin(rates))
    result = eqx.filter_jit(system.process.jump)(state, forbidden, jnp.asarray(0))
    assert jnp.array_equal(result, state)
    assert jnp.all(jnp.isnan(system.process.intensities(0.0, jnp.asarray([-1]))))


def test_native_generator_refuses_omitted_reachable_states_and_reports_leakage():
    system = _prepared()
    with pytest.raises(ValueError, match="omits reachable"):
        system.generator((system.states[0],))
    leaked = system.generator((system.states[0],), boundary_policy="leak")
    np.testing.assert_allclose(leaked.escaped_rates, [2.0])
    assert jnp.allclose(leaked.matrix.sum(axis=1), -leaked.escaped_rates)
    with pytest.raises(ValueError, match="closed"):
        finite_generator_hitting(leaked, jnp.asarray([False]))
    with pytest.raises(ValueError, match="state capacity"):
        prepare_secondary_kinetics(
            system.construct,
            system.model,
            system.association,
            system.rate_law,
            temperature=300.0,
            max_states=1,
        )


def test_nearest_neighbor_and_loop_terms_are_consumed_and_missing_parameters_refuse():
    construct = NucleicAcidConstruct(("x",), ("GGAAACC",), ("DNA",), (False,))
    keys = construct.nucleotide_keys
    state = SecondaryStructureState(construct, ((keys[0], keys[6]), (keys[1], keys[5])))
    model = _model(profile="nearest_neighbor_loop")
    assert model.standard_free_energy(state) == -2.0
    incomplete = _model(profile="nearest_neighbor_loop", stack_energies={})
    with pytest.raises(ValueError, match="stack"):
        incomplete.standard_free_energy(state)
    system = prepare_secondary_kinetics(
        construct,
        model,
        AssociationConvention(mode="standard_state", standard_concentration=1000),
        SecondaryRateLaw("metropolis", 2.0, 2.0),
        temperature=300,
    )
    assert {move.kind for move in system.moves(state)} == {"removal"}
    assert any(
        move.kind == "formation"
        for move in system.moves(SecondaryStructureState(construct))
    )


def test_parameter_rights_chemistry_and_temperature_are_real_admission_gates():
    with pytest.raises(PermissionError):
        _model(commercial=False)
    system = _prepared()
    with pytest.raises(ValueError, match="temperature"):
        prepare_secondary_kinetics(
            system.construct,
            system.model,
            system.association,
            system.rate_law,
            temperature=310,
        )
    rna = NucleicAcidConstruct(("x", "y"), ("A", "U"), ("RNA", "RNA"), (False, False))
    with pytest.raises(ValueError, match="chemistry"):
        prepare_secondary_kinetics(
            rna, system.model, system.association, system.rate_law, temperature=300
        )


def test_elementary_concentration_scaling_does_not_change_dissociation():
    small, large = _prepared(alpha=1), _prepared(alpha=10)
    q_small, q_large = small.generator().matrix, large.generator().matrix
    assert jnp.allclose(q_small[0, 1], 10 * q_large[0, 1])
    assert jnp.allclose(q_small[1, 0], q_large[1, 0])
    assert small.elementary_association_rate_constant(
        small.moves(small.states[0])[0]
    ) == pytest.approx(0.002)
    unsupported = _prepared(rate_name="symmetric_barrier")
    with pytest.raises(ValueError, match="Bimolecular"):
        unsupported.elementary_association_rate_constant(
            unsupported.moves(unsupported.states[0])[0]
        )


@pytest.mark.parametrize("solver", [solve_next_reaction, solve_direct_ssa])
def test_real_ssa_matches_exact_transients_and_first_passage(solver):
    system = _prepared()
    target = system.joined_target(system.construct.strand_ids)
    initial = system.encode(system.states[0])
    clocks = PoissonClockRealization(
        jr.key(812),
        system.process.num_channels,
        support=(0.0, 1.0),
        max_events_per_channel=32,
        sample_shape=(1024,),
        process_id=system.process.process_id,
    )
    solution = solver(
        system.process,
        clocks,
        initial,
        t0=0.0,
        t1=1.0,
        save_times=jnp.asarray([0.0, 1.0]),
        max_events=32,
    )
    hits = event_first_hit(solution, initial, target, t0=0.0, t1=1.0)
    assert jnp.all(solution.successful)
    expected_bound = float(system.generator().transition_matrix(1.0)[0, 1])
    bound = np.asarray(solution.states[:, -1, 0]) == 1
    assert abs(bound.mean() - expected_bound) < 5 * math.sqrt(
        expected_bound * (1 - expected_bound) / 1024
    )
    first_probability = 1 - math.exp(-2.0)
    assert abs(float(hits.hit.mean()) - first_probability) < 5 * math.sqrt(
        first_probability * (1 - first_probability) / 1024
    )
    # Entering and exiting the target between the only saved nodes is still hit.
    crossed_between_saves = hits.hit & (solution.states[:, -1, 0] == 0)
    assert jnp.any(crossed_between_saves)
    assert jnp.all(hits.time[crossed_between_saves] < 1.0)
    exact = finite_generator_hitting(system.generator(), target.mask)
    assert bool(exact.successful)
    np.testing.assert_allclose(exact.hitting_probability, [1, 1])
    np.testing.assert_allclose(exact.mean_first_passage_time, [0.5, 0.0])


def test_event_first_hit_distinguishes_initial_censoring_and_capacity_failure():
    inert = JumpProcess(
        lambda t, state, args: jnp.zeros(1),
        lambda state, channel, mark, args: state,
        state_shape=(1,),
        num_channels=1,
        process_id="inert-pure-jump-first-passage",
    )
    clocks = PoissonClockRealization(
        jr.key(201),
        1,
        support=(0.0, 1.0),
        max_events_per_channel=1,
        process_id=inert.process_id,
    )
    target = lambda state: state[0] == 1
    for initial_value in (0, 1):
        initial_state = jnp.asarray([initial_value])
        solution = solve_direct_ssa(
            inert,
            clocks,
            initial_state,
            t0=0.0,
            t1=1.0,
            save_times=jnp.asarray([0.25, 0.75]),
        )
        result = eqx.filter_jit(event_first_hit)(
            solution, initial_state, target, t0=0.0, t1=1.0
        )
        assert bool(result.initially_in_target) == (initial_value == 1)
        assert bool(result.hit) == (initial_value == 1)
        assert bool(result.censored) == (initial_value == 0)
        assert not bool(result.incomplete | result.capacity_failure)
        assert result.observation_end == 1.0
        assert result.time == (0.0 if initial_value == 1 else jnp.inf)
    system = _prepared()
    initial_state = system.encode(system.states[0])
    limited = PoissonClockRealization(
        jr.key(202),
        system.process.num_channels,
        support=(0.0, 1000.0),
        max_events_per_channel=1,
        process_id=system.process.process_id,
    )
    solution = solve_direct_ssa(
        system.process,
        limited,
        initial_state,
        t0=0.0,
        t1=1000.0,
        save_times=jnp.asarray([0.0, 1000.0]),
        max_events=1,
    )
    never_reached = event_first_hit(
        solution, initial_state, lambda state: state[0] == 2, t0=0.0, t1=1000.0
    )
    assert bool(never_reached.incomplete & never_reached.capacity_failure)
    assert not bool(never_reached.hit | never_reached.censored)
    assert never_reached.observation_end == solution.events.times[0]


def test_finite_absorption_keeps_non_hitting_classes_and_infinite_mfpt():
    process = JumpProcess(
        lambda t, state, args: jnp.where(
            state[0] == 0, jnp.asarray([1.0, 3.0]), jnp.zeros(2)
        ),
        lambda state, channel, mark, args: jnp.where(
            state[0] == 0, jnp.asarray([channel + 1]), state
        ),
        state_shape=(1,),
        num_channels=2,
        process_id="competing-absorbing-target-and-trap",
    )
    generator = finite_state_generator(process, jnp.arange(4)[:, None])
    result = finite_generator_hitting(generator, jnp.asarray([False, True, False, False]))
    assert bool(result.successful)
    np.testing.assert_allclose(result.hitting_probability, [0.25, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(result.reachable, [True, True, False, False])
    np.testing.assert_array_equal(result.almost_sure, [False, True, False, False])
    np.testing.assert_array_equal(
        result.closed_avoiding_class, [False, False, True, True]
    )
    assert jnp.all(jnp.isinf(result.mean_first_passage_time[jnp.asarray([0, 2, 3])]))
    assert result.mean_first_passage_time[1] == 0.0


def test_actual_ssa_capacity_failure_preserves_an_already_observed_first_hit():
    system = _prepared()
    initial = system.encode(system.states[0])
    clocks = PoissonClockRealization(
        jr.key(199),
        system.process.num_channels,
        support=(0.0, 1000.0),
        max_events_per_channel=1,
        process_id=system.process.process_id,
    )
    solution = solve_direct_ssa(
        system.process,
        clocks,
        initial,
        t0=0.0,
        t1=1000.0,
        save_times=jnp.asarray([0.0, 1000.0]),
        max_events=1,
    )
    result = event_first_hit(
        solution, initial, system.pair_count_target(1), t0=0.0, t1=1000.0
    )
    assert bool(result.capacity_failure)
    assert bool(result.hit)
    assert not bool(result.censored | result.incomplete)
    assert result.time == solution.events.times[0]
    assert not jnp.any(solution.valid)


def test_first_hit_refuses_actual_hybrid_continuous_crossing_and_raw_ledger():
    process_id = "hybrid-continuous-crossing-before-first-event"
    clocks = PoissonClockRealization(
        jr.key(203),
        1,
        support=(0.0, 1.0),
        max_events_per_channel=8,
        process_id=process_id,
    )
    rate = clocks.thresholds[0, 0] / 0.75
    process = JumpProcess(
        lambda t, state, args: jnp.asarray([rate]),
        lambda state, channel, mark, args: state,
        state_shape=(1,),
        num_channels=1,
        process_id=process_id,
    )
    differential = DifferentialProblem(
        lambda t, state, args: jnp.ones_like(state), jnp.asarray([0.0]), t0=0.0, t1=1.0
    )
    hybrid = solve_jump_differential(
        JumpDifferentialProblem(differential, process),
        clocks,
        save_times=jnp.asarray([0.0, 1.0]),
        event_rtol=1e-9,
        event_atol=1e-11,
    )
    assert bool(hybrid.successful)
    assert hybrid.events.times[0] == pytest.approx(0.75, abs=1e-8)
    assert hybrid.events.pre_states[0, 0] == pytest.approx(0.75, abs=1e-8)
    assert hybrid.states[-1, 0] == pytest.approx(1.0, abs=1e-8)
    for unqualified in (hybrid, hybrid.events):
        with pytest.raises(TypeError, match="pure-jump"):
            event_first_hit(
                unqualified,
                jnp.asarray([0.0]),
                lambda state: state[0] >= 0.5,
                t0=0.0,
                t1=1.0,
            )
