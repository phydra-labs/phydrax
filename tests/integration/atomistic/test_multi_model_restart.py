"""Portable restart of dynamics executing several learned models with bound inputs."""

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest

import phydrax as phx
from phydrax._array_archive import read_array_archive, write_array_archive
from phydrax.atomistic._checkpoint import (
    _RESTART_LIMITS,
    AtomisticRestartModelBinding,
    read_atomistic_restart,
    read_atomistic_restart_models,
    write_atomistic_restart,
)
from phydrax.atomistic._model_artifact import AtomisticModelArtifactError
from tests._support.compiler_profile import isolated_in_fixed_compiler_profile
from tests._support.multi_model_restart import (
    alpha_conditioning,
    alpha_term,
    assert_identical,
    checkpoint_plan,
    CONTINUED_STEPS,
    fresh_continuation,
    LICENSES,
    named_term,
    nvt,
    SAVED_STEPS,
    template,
    write_scenario,
)


@pytest.fixture(scope="module")
def scenario(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return write_scenario(tmp_path_factory.mktemp("multi-model") / "two-model.restart")


def _resume(scenario: dict[str, Any], terms: list[Any]) -> Any:
    dynamics, thermodynamic = nvt(terms, scenario["system"])
    return read_atomistic_restart(
        scenario["path"],
        checkpoint_plan(dynamics, thermodynamic),
        template(dynamics, thermodynamic),
    )


def test_fresh_process_two_model_restart_retains_bindings_and_continuation(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    if isolated_in_fixed_compiler_profile(request):
        return
    scenario = write_scenario(tmp_path / "two-model.restart")
    output = tmp_path / "continued.chk"
    summary = fresh_continuation(scenario["path"], output, CONTINUED_STEPS)
    program = scenario["dynamics"].potential
    assert sorted(summary["terms"]) == ["alpha", "beta"]
    for term in program.terms:
        restored = summary["terms"][term.name]
        assert restored["term_id"] == term.term_id
        assert restored["force_group"] == term.force_group
        assert restored["licenses"] == LICENSES[term.name]
    assert summary["terms"]["alpha"]["conditioned"] and summary["terms"]["alpha"]["keyed"]
    assert not summary["terms"]["beta"]["conditioned"]
    assert (
        summary["terms"]["alpha"]["artifact_id"]
        != summary["terms"]["beta"]["artifact_id"]
    )
    # The restored force cache equals the saved one exactly, and the rebuilt
    # models with their restored conditioning recompute it.
    saved = scenario["saved"].force
    np.testing.assert_array_equal(
        summary["potential_energy"], np.asarray(saved.potential_energy)
    )
    np.testing.assert_array_equal(
        summary["term_energies"], np.asarray(saved.term_energies)
    )
    np.testing.assert_array_equal(summary["forces"], np.asarray(saved.forces))
    np.testing.assert_allclose(
        summary["recomputed_potential_energy"],
        np.asarray(saved.potential_energy),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        summary["recomputed_forces"], np.asarray(saved.forces), rtol=1e-9, atol=1e-12
    )
    continued = phx.atomistic.read_atomistic_checkpoint(
        output,
        scenario["plan"],
        template(scenario["dynamics"], scenario["thermodynamic"]),
    ).state
    reference = scenario["reference"]
    assert int(continued.step_index) == SAVED_STEPS + CONTINUED_STEPS
    assert_identical(continued, reference)


def test_restart_bindings_rebuild_every_learned_term_exactly(
    scenario: dict[str, Any],
) -> None:
    bindings = read_atomistic_restart_models(scenario["path"])
    assert tuple(bindings) == ("alpha", "beta")
    with pytest.raises(TypeError):
        bindings["gamma"] = bindings["alpha"]  # ty: ignore[invalid-assignment]
    originals = {term.name: term.plan for term in scenario["dynamics"].potential.terms}
    for name, binding in bindings.items():
        assert isinstance(binding, AtomisticRestartModelBinding)
        assert binding.term.potential is binding.artifact.model
        assert binding.term.term_id == originals[name].term_id
        assert binding.term.force_group == originals[name].force_group
        assert binding.term.allow_periodic
        assert binding.artifact.manifest.licenses == tuple(LICENSES[name])
    alpha, beta = bindings["alpha"], bindings["beta"]
    assert alpha.artifact.manifest.artifact_id != beta.artifact.manifest.artifact_id
    assert (
        alpha.artifact.manifest.numeric_revision.revision_id
        != beta.artifact.manifest.numeric_revision.revision_id
    )
    assert beta.term.conditioning is None and beta.term.key is None
    restored, original = alpha.term.conditioning, originals["alpha"].conditioning
    assert restored is not None and original is not None
    assert restored.spec.spec_id == original.spec.spec_id
    assert restored.spec.names == ("solvent_fraction", "temperature", "total_charge")
    assert [field.port.port_id for field in restored.spec.fields] == [
        field.port.port_id for field in original.spec.fields
    ]
    assert not restored.present("solvent_fraction")
    np.testing.assert_array_equal(
        np.asarray(restored.defaults_used), [[False, True, False]]
    )
    np.testing.assert_array_equal(np.asarray(restored.value("temperature")), [0.3])
    np.testing.assert_array_equal(np.asarray(restored.value("total_charge")), [1])
    assert np.asarray(restored.value("total_charge")).dtype == np.int64
    assert alpha.term.key is not None
    assert str(jax.random.key_impl(alpha.term.key)) == str(
        jax.random.key_impl(originals["alpha"].key)
    )
    np.testing.assert_array_equal(
        np.asarray(jax.random.key_data(alpha.term.key)),
        np.asarray(jax.random.key_data(originals["alpha"].key)),
    )
    # The restored terms alone rebuild the runtime and resume its exact state.
    resumed = _resume(scenario, [binding.term for binding in bindings.values()])
    assert_identical(resumed.state, scenario["saved"])


def test_restart_refuses_any_other_learned_binding_set(
    scenario: dict[str, Any],
) -> None:
    model_a, model_b = scenario["models"]
    conditioning = alpha_conditioning(model_a)

    # The plan rebuilt over the original terms resumes.
    assert_identical(
        _resume(
            scenario, [alpha_term(model_a, conditioning), named_term(model_b, "beta")]
        ).state,
        scenario["saved"],
    )
    cases = {
        "swapped artifacts": (
            [
                alpha_term(model_b, conditioning),
                named_term(model_a, "beta", conditioning=conditioning),
            ],
            "executes a model that differs from the bundled artifact",
        ),
        "omitted term": (
            [alpha_term(model_a, conditioning)],
            "differ from the bundled bindings",
        ),
        "added term": (
            [
                alpha_term(model_a, conditioning),
                named_term(model_b, "beta"),
                named_term(model_b, "gamma"),
            ],
            "differ from the bundled bindings",
        ),
        "force group": (
            [
                alpha_term(model_a, conditioning),
                named_term(model_b, "beta", force_group=0),
            ],
            "differs from the bundled binding",
        ),
        "condition value": (
            [
                alpha_term(model_a, alpha_conditioning(model_a, charge=0)),
                named_term(model_b, "beta"),
            ],
            "differs from the bundled binding",
        ),
        "default evidence": (
            [
                alpha_term(
                    model_a, alpha_conditioning(model_a, temperature_default=False)
                ),
                named_term(model_b, "beta"),
            ],
            "differs from the bundled binding",
        ),
        "condition metadata": (
            [
                alpha_term(model_a, alpha_conditioning(model_a, solvent_field=False)),
                named_term(model_b, "beta"),
            ],
            "differs from the bundled binding",
        ),
        "random key": (
            [
                alpha_term(model_a, conditioning, key=jax.random.key(12)),
                named_term(model_b, "beta"),
            ],
            "differs from the bundled binding",
        ),
    }
    for terms, message in cases.values():
        with pytest.raises(AtomisticModelArtifactError, match=message):
            _resume(scenario, terms)


def test_restart_writer_derives_terms_and_deduplicates_identical_artifacts(
    scenario: dict[str, Any], tmp_path: Path
) -> None:
    model_a, model_b = scenario["models"]
    terms = [
        alpha_term(model_a, alpha_conditioning(model_a)),
        named_term(model_b, "beta"),
        named_term(model_b, "gamma"),
    ]
    dynamics, thermodynamic = nvt(terms, scenario["system"])
    plan = checkpoint_plan(dynamics, thermodynamic)
    state = template(dynamics, thermodynamic)
    with pytest.raises(ValueError, match="name no learned term"):
        write_atomistic_restart(
            tmp_path / "unknown.restart", plan, state, licenses={"delta": ["MIT"]}
        )
    assert not (tmp_path / "unknown.restart").exists()

    shared = tmp_path / "shared.restart"
    write_atomistic_restart(
        shared, plan, state, licenses={**LICENSES, "gamma": LICENSES["beta"]}
    )
    manifest, _ = read_array_archive(shared, limits=_RESTART_LIMITS)
    assert len(manifest["models"]["artifacts"]) == 2
    assert [record["name"] for record in manifest["models"]["terms"]] == [
        "alpha",
        "beta",
        "gamma",
    ]
    bindings = read_atomistic_restart_models(shared)
    assert bindings["beta"].artifact is bindings["gamma"].artifact
    assert bindings["beta"].term.potential is bindings["gamma"].term.potential
    assert bindings["beta"].binding_id != bindings["gamma"].binding_id

    # Different provenance of the same model never aliases one artifact.
    distinct = tmp_path / "distinct.restart"
    write_atomistic_restart(
        distinct, plan, state, licenses={**LICENSES, "gamma": ["MIT"]}
    )
    manifest, _ = read_array_archive(distinct, limits=_RESTART_LIMITS)
    assert len(manifest["models"]["artifacts"]) == 3
    bindings = read_atomistic_restart_models(distinct)
    assert (
        bindings["beta"].artifact.manifest.artifact_id
        != bindings["gamma"].artifact.manifest.artifact_id
    )

    # A two-term restart refuses this three-term runtime and vice versa.
    with pytest.raises(
        AtomisticModelArtifactError, match="differ from the bundled bindings"
    ):
        read_atomistic_restart(scenario["path"], plan, state)
    with pytest.raises(
        AtomisticModelArtifactError, match="differ from the bundled bindings"
    ):
        read_atomistic_restart(
            shared,
            scenario["plan"],
            template(scenario["dynamics"], scenario["thermodynamic"]),
        )


def _rewritten(
    source: Path,
    target: Path,
    edit: Callable[[dict[str, Any], dict[str, np.ndarray]], None],
) -> Path:
    manifest, arrays = read_array_archive(source, limits=_RESTART_LIMITS)
    manifest = copy.deepcopy(manifest)
    manifest.pop("arrays")
    arrays = {name: np.array(value) for name, value in arrays.items()}
    edit(manifest, arrays)
    write_array_archive(target, manifest=manifest, limits=_RESTART_LIMITS, arrays=arrays)
    return target


def test_restart_refuses_tampered_binding_records_and_inventories(
    scenario: dict[str, Any], tmp_path: Path
) -> None:
    def force_group(manifest: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        manifest["models"]["terms"][1]["force_group"] = 2

    def default_event(manifest: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        # Conditioning fields are sorted: temperature is the first bound field
        # (solvent_fraction is absent).
        arrays["terms/0000/conditioning/000000"] = np.asarray(0.31)

    def key_data(manifest: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        arrays["terms/0000/key/000000"] = arrays["terms/0000/key/000000"] ^ np.uint32(1)

    def extra_array(manifest: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        arrays["terms/0000/conditioning/000009"] = np.asarray(0.0)

    def foreign_namespace(
        manifest: dict[str, Any], arrays: dict[str, np.ndarray]
    ) -> None:
        arrays["models/0000/extra"] = np.asarray(0.0)

    def dropped_term(manifest: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        manifest["models"]["terms"].pop()
        for name in [name for name in arrays if name.startswith("terms/0001/")]:
            arrays.pop(name)

    def binding_identity(manifest: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        manifest["models"]["terms"][0]["binding_id"] = "0" * 64

    def absent_value(manifest: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        manifest["models"]["terms"][0]["conditioning"]["bindings"][0] = "value"

    cases = (
        (force_group, AtomisticModelArtifactError, "does not reproduce its recorded"),
        (default_event, AtomisticModelArtifactError, "does not reproduce its recorded"),
        (key_data, AtomisticModelArtifactError, "does not reproduce its recorded"),
        (extra_array, ValueError, "do not match its record inventory"),
        (foreign_namespace, ValueError, "outside its namespaces"),
        (dropped_term, ValueError, "not each referenced in canonical order"),
        (binding_identity, AtomisticModelArtifactError, "binding identity is corrupt"),
        (absent_value, ValueError, "do not match its record inventory"),
    )
    for index, (edit, error, message) in enumerate(cases):
        target = _rewritten(
            scenario["path"], tmp_path / f"tampered-{index}.restart", edit
        )
        with pytest.raises(error, match=message):
            read_atomistic_restart_models(target)
