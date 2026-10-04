import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.export._atomistic import _contract, _runtime_guard_sites
from tests._support.mace_deployment import electronvolt_units, provider_plan, tiny_mace


_HAS_IREE = (
    importlib.util.find_spec("iree") is not None
    and importlib.util.find_spec("iree.compiler") is not None
    and importlib.util.find_spec("iree.runtime") is not None
)
needs_iree = pytest.mark.skipif(
    not _HAS_IREE, reason="IREE optional packages are not installed"
)

CELL = np.array([[4.2, 0.0, 0.0], [0.9, 4.0, 0.0], [0.5, -0.6, 4.4]])
WATER = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
MOVED = WATER + np.array([[0.02, -0.01, 0.0], [0.0, 0.015, -0.02], [0.0, 0.0, 0.01]])


def _structure(*, periodic: bool) -> Any:
    return phx.atomistic.AtomicStructure(
        jnp.asarray([8, 1, 1]),
        jnp.asarray(WATER),
        jnp.asarray([15.999, 1.008, 1.008]),
        electronvolt_units().scale,
        cell=jnp.asarray(CELL) if periodic else None,
        periodic_axes=jnp.asarray([True, True, True]) if periodic else None,
    )


def _export(directory: Path, *, periodic: bool) -> tuple[Any, Any, Any]:
    plan = provider_plan(tiny_mace(seed=2))
    structure = _structure(periodic=periodic)
    bundle = phx.export.save_atomistic_iree(
        plan,
        structure,
        electronvolt_units(),
        directory / "mace.phxiree",
        # Float64 MACE transcendentals need the host C math library.
        policy=phx.export.IREEExportPolicy(executable_format="system-library"),
    )
    loaded = phx.export.load_atomistic_iree(
        bundle.path,
        trusted_module_sha256=bundle.module_sha256,
        trusted_contract_id=bundle.contract.contract_id,
    )
    system = phx.atomistic.AtomisticSystemPlan.from_structure(
        structure, electronvolt_units()
    ).prepare()
    return plan.prepare(system), bundle, loaded


@pytest.fixture(scope="module")
def periodic_export(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, Any, Any]:
    return _export(tmp_path_factory.mktemp("periodic-export"), periodic=True)


def _with_geometry(request: Any, *, positions: Any = None, cell: Any = None) -> Any:
    return phx.atomistic.NativeAtomisticRequest(
        request.positions if positions is None else positions,
        request.unwrapped_positions,
        request.cell_vectors if cell is None else cell,
        request.image_counts,
        request.neighborhood,
    )


def _assert_native_parity(frozen: Any, native: Any) -> None:
    assert frozen.successful and bool(native.evaluation.successful)
    np.testing.assert_allclose(frozen.energy, native.evaluation.energy, rtol=1e-10)
    np.testing.assert_allclose(
        frozen.forces, native.evaluation.forces, rtol=1e-8, atol=1e-11
    )
    np.testing.assert_allclose(
        frozen.atom_energy, native.atom_energy, rtol=1e-8, atol=1e-11
    )
    if frozen.stress is not None:
        np.testing.assert_allclose(
            frozen.stress, native.evaluation.stress, rtol=1e-8, atol=1e-11
        )


@needs_iree
def test_frozen_periodic_export_matches_native_across_the_frozen_epoch(
    periodic_export: tuple[Any, Any, Any],
) -> None:
    provider, bundle, loaded = periodic_export
    assert bundle.contract.stress and bundle.contract.route == "native-jax"
    assert bundle.contract.failure_policy == "equinox-nan-status"
    assert bundle.contract.input_names == ("positions", "cell_vectors", "image_counts")
    request = provider.prepare_request(jnp.asarray(WATER), None, None)
    moved = provider.prepare_request(jnp.asarray(MOVED), None, request.neighborhood)
    # Atom 0 leaves the cell through a face: same epoch, new runtime image count.
    unwrapped = np.asarray(request.unwrapped_positions).copy()
    unwrapped[0] -= 0.05 * CELL[0] / np.linalg.norm(CELL[0])
    rewrapped = provider.prepare_request(
        jnp.asarray(unwrapped), None, request.neighborhood
    )
    assert int(rewrapped.neighborhood.epoch) == int(request.neighborhood.epoch)
    assert not np.array_equal(rewrapped.image_counts, request.image_counts)
    for current in (request, moved, rewrapped):
        frozen = loaded(current)
        _assert_native_parity(frozen, provider.evaluate_request(current))
        assert frozen.active_routes > 0


@needs_iree
def test_frozen_export_reports_invalid_runtime_geometry_by_status(
    periodic_export: tuple[Any, Any, Any],
) -> None:
    provider, _, loaded = periodic_export
    request = provider.prepare_request(jnp.asarray(WATER), None, None)
    nonfinite = loaded(
        _with_geometry(request, positions=request.positions.at[1, 2].set(jnp.nan))
    )
    singular = loaded(_with_geometry(request, cell=jnp.zeros_like(request.cell_vectors)))
    compressed = loaded(_with_geometry(request, cell=0.55 * request.cell_vectors))
    assert nonfinite.status is phx.atomistic.AtomisticStatus.NONFINITE
    assert singular.status is phx.atomistic.AtomisticStatus.NONFINITE
    # A cell outside the frozen image certificate is an image/capacity failure.
    assert compressed.status is phx.atomistic.AtomisticStatus.NEIGHBOR_OVERFLOW
    for failed in (nonfinite, singular, compressed):
        assert np.isnan(failed.energy) and np.all(np.isnan(failed.forces))
        assert np.all(np.isnan(failed.atom_energy)) and np.all(np.isnan(failed.stress))


@needs_iree
def test_frozen_export_succeeds_exactly_while_the_native_lifecycle_reuses_its_epoch(
    periodic_export: tuple[Any, Any, Any],
) -> None:
    provider, _, loaded = periodic_export
    request = provider.prepare_request(jnp.asarray(WATER), None, None)
    epoch = int(request.neighborhood.epoch)
    for positions, cell in (
        (WATER, CELL),
        (MOVED, CELL),
        (WATER, 1.01 * CELL),
        (WATER, 0.55 * CELL),
        (WATER + np.asarray([[0.6, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), CELL),
    ):
        # Independent native lifecycle probe: does the frozen epoch survive?
        current = provider.prepare_request(
            jnp.asarray(positions), jnp.asarray(cell), request.neighborhood
        )
        reused = int(current.neighborhood.epoch) == epoch
        # The frozen module sees the same runtime geometry within its own epoch.
        frozen = loaded(
            phx.atomistic.NativeAtomisticRequest(
                current.positions,
                current.unwrapped_positions,
                current.cell_vectors,
                current.image_counts,
                request.neighborhood,
            )
        )
        expected = (
            phx.atomistic.AtomisticStatus.SUCCESS
            if reused
            else phx.atomistic.AtomisticStatus.NEIGHBOR_OVERFLOW
        )
        assert frozen.status is expected
        if reused:
            _assert_native_parity(frozen, provider.evaluate_request(current))
        else:
            assert np.isnan(frozen.energy) and np.all(np.isnan(frozen.forces))


@needs_iree
def test_frozen_export_refuses_unpinned_mismatched_or_other_epoch_inputs(
    periodic_export: tuple[Any, Any, Any],
) -> None:
    provider, bundle, loaded = periodic_export
    request = provider.prepare_request(jnp.asarray(WATER), None, None)
    with pytest.raises(PermissionError, match="trusted pin"):
        phx.export.load_atomistic_iree(
            bundle.path,
            trusted_module_sha256="0" * 64,
            trusted_contract_id=bundle.contract.contract_id,
        )
    with pytest.raises(PermissionError, match="caller-supplied pin"):
        phx.export.load_atomistic_iree(
            bundle.path,
            trusted_module_sha256=bundle.module_sha256,
            trusted_contract_id="f" * 64,
        )
    with pytest.raises(TypeError, match="dtype"):
        loaded(_with_geometry(request, positions=request.positions.astype(jnp.float32)))
    stretched = provider.prepare_request(
        jnp.asarray(WATER * 1.6), None, request.neighborhood
    )
    assert int(stretched.neighborhood.epoch) != int(request.neighborhood.epoch)
    with pytest.raises(ValueError, match="candidate graph differs"):
        loaded(stretched)


@needs_iree
def test_frozen_export_refuses_a_pinned_contract_whose_outputs_the_module_lacks(
    periodic_export: tuple[Any, Any, Any], tmp_path: Path
) -> None:
    _, bundle, _ = periodic_export
    contract = bundle.contract
    assert contract.output_names[0] == "energy" and contract.output_dtypes[0] == "<f8"
    # A same-width integer energy, separately pinned, must not reinterpret floats.
    forged = _contract(
        contract.route,
        contract.failure_policy,
        contract.program_id,
        contract.provider_id,
        contract.unit_system_id,
        contract.atomic_numbers,
        contract.periodic,
        contract.stress,
        contract.graph_id,
        contract.input_names,
        contract.input_shapes,
        contract.input_dtypes,
        contract.output_shapes,
        ("<i8", *contract.output_dtypes[1:]),
    )
    copy = tmp_path / "forged.phxiree"
    shutil.copytree(bundle.path, copy)
    manifest = json.loads((copy / "manifest.json").read_text(encoding="utf-8"))
    manifest["domain_contract"] = json.dumps(forged.to_dict())
    (copy / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from the executable ABI"):
        phx.export.load_atomistic_iree(
            copy,
            trusted_module_sha256=bundle.module_sha256,
            trusted_contract_id=forged.contract_id,
        )


@needs_iree
def test_loaded_frozen_export_refuses_differentiation(
    periodic_export: tuple[Any, Any, Any],
) -> None:
    provider, _, loaded = periodic_export
    request = provider.prepare_request(jnp.asarray(WATER), None, None)

    def energy(positions: Any) -> Any:
        return loaded(_with_geometry(request, positions=positions)).energy

    with pytest.raises(TypeError, match="JAX transformations"):
        jax.grad(energy)(request.positions)


@needs_iree
def test_frozen_aperiodic_export_matches_native(tmp_path: Path) -> None:
    provider, bundle, loaded = _export(tmp_path, periodic=False)
    assert not bundle.contract.stress and bundle.contract.input_names == ("positions",)
    request = provider.prepare_request(jnp.asarray(WATER), None, None)
    moved = provider.prepare_request(jnp.asarray(MOVED), None, request.neighborhood)
    for current in (request, moved):
        _assert_native_parity(loaded(current), provider.evaluate_request(current))
        assert loaded(current).stress is None


def test_guard_audit_separates_frozen_constant_guards_from_runtime_guards() -> None:
    frozen = jnp.asarray([1, 2, 3])

    def program(values: Any) -> Any:
        # Integer guard over frozen data only: provably inactive at export.
        indices = eqx.error_if(frozen, jnp.any(frozen < 0), "frozen index")

        def step(carry: Any, value: Any) -> tuple[Any, Any]:
            # The carry depends on the runtime input after the first step.
            guarded = eqx.error_if(carry, carry > 1.0e9, "runtime carry")
            return guarded + value, guarded

        total, _ = jax.lax.scan(step, jnp.zeros(()), values)
        return total + jnp.sum(indices)

    sites = _runtime_guard_sites(program, (jnp.ones(3),))
    assert [site.kind for site in sites] == ["guard", "guard"]
    assert [site.dynamic for site in sites] == [False, True]
