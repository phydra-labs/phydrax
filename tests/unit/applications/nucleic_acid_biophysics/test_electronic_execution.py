# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import hashlib
import json
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.nucleic_acid_biophysics._construct import NucleicAcidConstruct
from phydrax.applications.nucleic_acid_biophysics.electronics import (
    electronic_coherences,
    electronic_populations,
    electronic_reduced_density,
    ElectronicChannel,
    ElectronicParameterArtifact,
    ElectronicSiteGraph,
    evolve_electronic_jumps,
    evolve_electronics,
    nucleotide_electronic_populations,
    prepare_electron_hole,
    prepare_electronics,
)
from phydrax.atomistic import AtomisticScaleContract, AtomisticUnitSystem
from phydrax.linalg import HermitianSpectrum
from phydrax.qualification import ReferenceArtifactManifest
from phydrax.units import (
    ANGSTROM,
    conversion_factor,
    DALTON,
    derived_unit,
    ELECTRONVOLT,
    ELEMENTARY_CHARGE,
    FEMTOSECOND,
    JOULE,
    KELVIN,
    KILOJOULE_PER_MOLE,
    PICOSECOND,
)


jax.config.update("jax_enable_x64", True)
USE = dict(commercial_use=False, redistribution=False, training_use=False, export=False)
UNITS = AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
PER_FS = derived_unit("1/fs", ((FEMTOSECOND, -1),))


def _graph(count=2):
    construct = NucleicAcidConstruct(("strand",), ("A" * count,), ("DNA",), (False,))
    sites = tuple(2**40 + index * 13 for index in range(count))
    return ElectronicSiteGraph(
        construct,
        sites,
        construct.nucleotide_keys,
        ("pi",) * count,
        tuple(zip(sites[:-1], sites[1:], strict=True)),
    )


def _parameters(keys, energies=None, couplings=(), channels=(), energy_unit=ELECTRONVOLT):
    energies = (0.0,) * len(keys) if energies is None else tuple(energies)
    record = {
        "basis": keys,
        "energies": energies,
        "couplings": [(a, b, complex(v).real, complex(v).imag) for a, b, v in couplings],
        "channels": [channel.record() for channel in channels],
        "unit": energy_unit.unit_id,
    }
    raw = json.dumps(record, sort_keys=True).encode()
    source = ReferenceArtifactManifest(
        "independently-declared-analytic-electronic-fixture",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
        license_id="CC0-1.0",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"energy_J": float(energy_unit.scale_to_reference)},
        uncertainty=None,
        lineage_ids=("analytic-input-not-a-DNA-calibration",),
    )
    return ElectronicParameterArtifact(
        tuple(keys),
        energies,
        tuple(couplings),
        tuple(channels),
        energy_unit,
        "analytic finite-model fixture; no experimental calibration",
        "real local pi orbitals",
        source,
        raw,
    )


def _model(graph=None, *, coupling=0.0, channels=(), energies=None):
    graph = _graph() if graph is None else graph
    keys = tuple((site,) for site in graph.site_ids)
    edges = ((keys[0], keys[1], coupling),) if coupling else ()
    parameters = _parameters(keys, energies, edges, channels)
    return prepare_electronics(graph, parameters, units=UNITS, requested_use=USE)


def _density(state):
    return state[:, None] * jnp.conj(state[None, :])


@pytest.mark.parametrize("method", ["lindblad", "cptp"])
def test_coherent_two_site_solution_and_density_physicality(method):
    model = _model(coupling=UNITS.reduced_planck_constant)
    initial = model.basis_state(model.basis_keys[0])
    result = evolve_electronics(
        model,
        _density(initial),
        step_size=0.05,
        time_unit=FEMTOSECOND,
        steps=30,
        requested_use=USE,
        method=method,
    )
    times = result.densities.support.coordinates
    populations = electronic_populations(model, result.densities.values)
    np.testing.assert_allclose(populations[:, 1], jnp.sin(times) ** 2, atol=2e-10)
    pair = (model.graphs[0].site_ids,)
    coherence = electronic_coherences(model, result.densities.values, pair)[:, 0]
    np.testing.assert_allclose(
        coherence, 1j * jnp.cos(times) * jnp.sin(times), atol=2e-10
    )
    assert bool(result.native_result.valid)
    assert (
        np.max(np.abs(np.trace(result.densities.values, axis1=-2, axis2=-1) - 1)) < 2e-10
    )
    assert (
        float(jnp.min(HermitianSpectrum(result.densities.values).minimum_eigenvalue))
        > -2e-10
    )
    assert result.artifact.artifact_id != model.artifact.artifact_id
    assert model.artifact.artifact_id in result.artifact.parent_artifact_ids


@pytest.mark.parametrize("method", ["lindblad", "cptp"])
def test_local_dephasing_decay_uses_rate_once(method):
    graph = _graph()
    keys = tuple((site,) for site in graph.site_ids)
    channels = tuple(
        ElectronicChannel(f"dephase-{index}", "dephasing", key, None, 0.4, PER_FS)
        for index, key in enumerate(keys)
    )
    model = _model(graph, channels=channels)
    initial = jnp.ones(2, dtype=complex) / jnp.sqrt(2.0)
    result = evolve_electronics(
        model,
        _density(initial),
        step_size=0.1,
        time_unit=FEMTOSECOND,
        steps=12,
        requested_use=USE,
        method=method,
    )
    times = result.densities.support.coordinates
    np.testing.assert_allclose(
        result.densities.values[:, 0, 1], 0.5 * jnp.exp(-0.4 * times), atol=2e-10
    )
    np.testing.assert_allclose(
        electronic_populations(model, result.densities.values), 0.5, atol=2e-10
    )
    assert bool(result.native_result.valid)


def test_native_unraveling_agrees_with_density_and_retains_event_evidence():
    graph = _graph()
    keys = tuple((site,) for site in graph.site_ids)
    bath = ElectronicChannel(
        "downhill-declared-bath", "bath", keys[1], keys[0], 0.7, PER_FS
    )
    model = _model(graph, channels=(bath,))
    initial = model.basis_state(keys[1])
    dense = evolve_electronics(
        model,
        _density(initial),
        step_size=0.005,
        time_unit=FEMTOSECOND,
        steps=100,
        requested_use=USE,
    )
    jumps = evolve_electronic_jumps(
        model,
        initial,
        jax.random.PRNGKey(712),
        step_size=0.005,
        time_unit=FEMTOSECOND,
        steps=100,
        trajectory_count=1024,
        requested_use=USE,
    )
    expected = float(dense.densities.values[-1, 1, 1].real)
    observed = float(jumps.mean_densities.values[-1, 1, 1].real)
    # Five Bernoulli standard errors plus the declared first-order time-step scale.
    tolerance = 5 * np.sqrt(expected * (1 - expected) / 1024) + 0.005
    assert abs(observed - expected) < tolerance
    assert bool(jumps.native_result.valid)
    assert int(jnp.sum(jumps.native_result.jump_mask)) > 0
    assert not bool(
        jnp.any(jumps.native_result.jump_channels[jumps.native_result.jump_mask] != 0)
    )
    np.testing.assert_allclose(
        jumps.native_result.empirical_density(),
        jumps.mean_densities.values[-1],
        atol=1e-10,
    )
    with pytest.raises(ValueError, match="jump-probability"):
        evolve_electronic_jumps(
            model,
            initial,
            jax.random.PRNGKey(1),
            step_size=1.0,
            time_unit=FEMTOSECOND,
            steps=1,
            trajectory_count=4,
            requested_use=USE,
        )


def test_site_and_parameter_permutations_preserve_mapped_observables():
    graph = _graph(3)
    keys = tuple((site,) for site in graph.site_ids)
    parameters = _parameters(
        keys,
        (0.03, -0.02, 0.04),
        ((keys[0], keys[1], 0.2 + 0.07j), (keys[1], keys[2], -0.11j)),
        (ElectronicChannel("transfer", "bath", keys[2], keys[0], 0.2, PER_FS),),
    )
    first = prepare_electronics(graph, parameters, units=UNITS, requested_use=USE)
    order = (2, 0, 1)
    reordered = replace(
        graph,
        site_ids=tuple(graph.site_ids[i] for i in order),
        nucleotide_keys=tuple(graph.nucleotide_keys[i] for i in order),
        orbital_labels=tuple(graph.orbital_labels[i] for i in order),
        edges=tuple((b, a) for a, b in reversed(graph.edges)),
    )
    permuted_parameters = replace(
        parameters,
        basis_keys=tuple(reversed(keys)),
        site_energies=tuple(reversed(parameters.site_energies)),
        couplings=tuple(
            (b, a, complex(v).conjugate()) for a, b, v in reversed(parameters.couplings)
        ),
    )
    second = prepare_electronics(
        reordered, permuted_parameters, units=UNITS, requested_use=USE
    )
    assert reordered.fingerprint() == graph.fingerprint()
    assert permuted_parameters.fingerprint() == parameters.fingerprint()
    results = [
        evolve_electronics(
            model,
            _density(model.basis_state(keys[0])),
            step_size=0.2,
            time_unit=FEMTOSECOND,
            steps=6,
            requested_use=USE,
        )
        for model in (first, second)
    ]
    np.testing.assert_allclose(
        nucleotide_electronic_populations(first, results[0].densities.values),
        nucleotide_electronic_populations(second, results[1].densities.values),
        atol=1e-10,
    )
    pair = ((graph.site_ids[0], graph.site_ids[2]),)
    np.testing.assert_allclose(
        electronic_coherences(first, results[0].densities.values, pair),
        electronic_coherences(second, results[1].densities.values, pair),
        atol=1e-10,
    )
    for result in results:
        assert bool(result.native_result.valid)


def test_energy_time_and_rate_units_give_equivalent_physical_execution():
    graph = _graph()
    keys = tuple((site,) for site in graph.site_ids)
    channels = (ElectronicChannel("bath", "bath", keys[1], keys[0], 0.3, PER_FS),)
    source = _parameters(keys, (0.05, -0.01), ((keys[0], keys[1], 0.2),), channels)
    first = prepare_electronics(graph, source, units=UNITS, requested_use=USE)
    units = AtomisticUnitSystem(
        AtomisticScaleContract(ANGSTROM, JOULE),
        mass_unit=DALTON,
        time_unit=PICOSECOND,
        charge_unit=ELEMENTARY_CHARGE,
        temperature_unit=KELVIN,
        constant_set_id="codata-2018",
    )
    energy_factor = float(conversion_factor(ELECTRONVOLT, JOULE))
    per_ps = derived_unit("1/ps", ((PICOSECOND, -1),))
    other = _parameters(
        keys,
        tuple(energy * energy_factor for energy in source.site_energies),
        tuple((a, b, v * energy_factor) for a, b, v in source.couplings),
        (replace(channels[0], rate=300.0, rate_unit=per_ps),),
        JOULE,
    )
    second = prepare_electronics(graph, other, units=units, requested_use=USE)
    state = first.basis_state(keys[1])
    a = evolve_electronics(
        first,
        _density(state),
        step_size=0.1,
        time_unit=FEMTOSECOND,
        steps=10,
        requested_use=USE,
    )
    b = evolve_electronics(
        second,
        _density(state),
        step_size=0.0001,
        time_unit=PICOSECOND,
        steps=10,
        requested_use=USE,
    )
    np.testing.assert_allclose(a.densities.values, b.densities.values, atol=1e-10)
    np.testing.assert_allclose(
        a.densities.support.coordinates * 0.001,
        b.densities.support.coordinates,
        atol=1e-15,
    )
    with pytest.raises(ValueError, match="single-system"):
        replace(source, energy_unit=KILOJOULE_PER_MOLE)


def test_electron_hole_tensor_factorization_and_recombination_to_vacuum():
    graph = _graph()
    keys = tuple((site,) for site in graph.site_ids)
    electron = _model(graph, coupling=UNITS.reduced_planck_constant)
    hole = _model(graph, coupling=0.5 * UNITS.reduced_planck_constant)
    pairs = tuple((e[0], h[0]) for e in keys for h in keys)
    interaction = _parameters(pairs)
    joint = prepare_electron_hole(electron, hole, interaction, requested_use=USE)
    result = evolve_electronics(
        joint,
        _density(joint.basis_state(pairs[0])),
        step_size=0.05,
        time_unit=FEMTOSECOND,
        steps=10,
        requested_use=USE,
    )
    times = result.densities.support.coordinates
    np.testing.assert_allclose(
        electronic_populations(joint, result.densities.values, carrier=0)[:, 1],
        jnp.sin(times) ** 2,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        electronic_populations(joint, result.densities.values, carrier=1)[:, 1],
        jnp.sin(0.5 * times) ** 2,
        atol=1e-10,
    )
    assert electronic_reduced_density(
        joint, result.densities.values, carrier=1
    ).shape == (11, 2, 2)
    static = _model(graph)
    recombination = _parameters(
        pairs,
        channels=(
            ElectronicChannel(
                "recombine-pair", "recombination", pairs[0], None, 0.4, PER_FS
            ),
        ),
    )
    with pytest.raises(ValueError, match="vacuum"):
        prepare_electron_hole(static, static, recombination, requested_use=USE)
    sink = prepare_electron_hole(
        static, static, recombination, requested_use=USE, include_vacuum=True
    )
    decayed = evolve_electronics(
        sink,
        _density(sink.basis_state(pairs[0])),
        step_size=0.1,
        time_unit=FEMTOSECOND,
        steps=10,
        requested_use=USE,
        method="cptp",
    )
    times = decayed.densities.support.coordinates
    survival = jnp.exp(-0.4 * times)
    np.testing.assert_allclose(
        decayed.densities.values[:, -1, -1], 1 - survival, atol=1e-10
    )
    for carrier in (0, 1):
        np.testing.assert_allclose(
            jnp.sum(
                electronic_populations(sink, decayed.densities.values, carrier=carrier),
                axis=-1,
            ),
            survival,
            atol=1e-10,
        )
    assert bool(decayed.native_result.valid)


def test_tensor_baths_preserve_each_carriers_independent_rates():
    graph = _graph()
    keys = tuple((site,) for site in graph.site_ids)
    electron = _model(
        graph,
        channels=(ElectronicChannel("bath", "bath", keys[1], keys[0], 0.4, PER_FS),),
    )
    hole = _model(
        graph,
        channels=tuple(
            ElectronicChannel(f"dephase-{index}", "dephasing", key, None, 0.6, PER_FS)
            for index, key in enumerate(keys)
        ),
    )
    pairs = tuple((e[0], h[0]) for e in keys for h in keys)
    joint = prepare_electron_hole(electron, hole, _parameters(pairs), requested_use=USE)
    initial = (joint.basis_state(pairs[2]) + joint.basis_state(pairs[3])) / jnp.sqrt(2.0)
    result = evolve_electronics(
        joint,
        _density(initial),
        step_size=0.1,
        time_unit=FEMTOSECOND,
        steps=8,
        requested_use=USE,
        method="cptp",
    )
    times = result.densities.support.coordinates
    np.testing.assert_allclose(
        electronic_populations(joint, result.densities.values, carrier=0)[:, 1],
        jnp.exp(-0.4 * times),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        electronic_coherences(
            joint, result.densities.values, (graph.site_ids,), carrier=1
        )[:, 0],
        0.5 * jnp.exp(-0.6 * times),
        atol=1e-10,
    )
    assert bool(result.native_result.valid)


def test_host_admission_refuses_missing_support_rights_and_unbounded_models():
    graph = _graph()
    keys = tuple((site,) for site in graph.site_ids)
    parameters = _parameters(keys)
    with pytest.raises(ValueError, match="unique"):
        replace(graph, site_ids=(graph.site_ids[0],) * 2)
    with pytest.raises(ValueError, match="both directions"):
        _parameters(keys, couplings=((keys[0], keys[1], 1.0), (keys[1], keys[0], 1.0)))
    with pytest.raises(ValueError, match="exactly cover"):
        prepare_electronics(graph, _parameters(keys[:1]), units=UNITS, requested_use=USE)
    with pytest.raises(ValueError, match="resource bounds"):
        prepare_electronics(
            graph, parameters, units=UNITS, requested_use=USE, maximum_dimension=1
        )
    with pytest.raises(ValueError, match="Structure-derived"):
        prepare_electronics(
            graph,
            replace(parameters, structure_derived=True),
            units=UNITS,
            requested_use=USE,
        )
    with pytest.raises(ValueError, match="Raw parameter bytes"):
        replace(parameters, raw_content=parameters.raw_content + b"tampered")
    restricted_record = parameters.source.to_record()
    restricted_record.pop("manifest_id")
    restricted_record["export_permitted"] = False
    restricted = replace(
        parameters, source=ReferenceArtifactManifest.from_record(restricted_record)
    )
    model = prepare_electronics(graph, restricted, units=UNITS, requested_use=USE)
    with pytest.raises(PermissionError):
        model.require_rights({**USE, "export": True})
    with pytest.raises(ValueError, match="unit norm"):
        model.jump_problem(jnp.ones(2, dtype=complex))


def test_fixed_support_generator_and_observables_are_jittable_and_differentiable():
    model = _model(coupling=UNITS.reduced_planck_constant)
    problem = model.density_problem(_density(model.basis_state(model.basis_keys[0])))
    evolved_rate = jax.jit(lambda density: problem.generator(density))(
        problem.initial_density
    )
    np.testing.assert_allclose(evolved_rate, jnp.asarray([[0, 1j], [-1j, 0]]), atol=1e-12)

    def population(angle):
        state = jnp.asarray([jnp.cos(angle), -1j * jnp.sin(angle)])
        return electronic_populations(model, _density(state))[1]

    angle = 0.31
    value, gradient = jax.jit(jax.value_and_grad(population))(angle)
    np.testing.assert_allclose(
        (value, gradient), (np.sin(angle) ** 2, np.sin(2 * angle)), atol=1e-12
    )
