#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Frozen fixed-capacity atomistic E/F/S inference through IREE.

The exported module evaluates one native learned model whose weights, species,
and reference candidate relation (route identities, image shifts, masks, and
the host-prepared route topology of one lifecycle epoch) are immutable
artifact data. Runtime inputs are only the wrapped positions and, for periodic
systems, the cell vectors and lattice image counts. The module evaluates the
frozen routes in their reference frame ``positions + (image_counts -
reference_image_counts) @ cell``, which is exactly the native lifecycle's
whole-lattice re-expression, so rewrapped atoms keep the same physical routes.
Neighbor discovery and cache lifecycles stay on the host: a request from
another lifecycle epoch, provider, or failed lifecycle is refused before
execution and requires a new export.

Runtime geometry failures (non-finite coordinates, singular or deformed cells,
image-certificate or capacity overflow) are reported by the native program's
traced status, never by host callbacks. Before export the parent traces the
frozen program under its ordinary raising guards and refuses any host
callback and any Equinox guard whose predicate depends on a runtime input: an
input-dependent guard under Equinox's ``nan`` policy would poison floats with
NaN and integers with their maximum value without a status, so its failure
would not provably reach ``status``. Every remaining guard depends only on
frozen artifact data and is proven inactive by the parent's raise-mode
evaluation of the same program. Tracing then runs in a pinned isolated worker
interpreter started with ``EQX_ON_ERROR=nan``
(``failure_policy="equinox-nan-status"``), so those constant guards lower
without callbacks; the caller's runtime keeps its raising semantics.

The worker imports the parent's own implementation tree, rebuilds the model,
provider plan, structure, and contract from one pickle-free registered
archive, and returns the compiled module as bounded file artifacts. The parent
publishes nothing until the loaded module reproduces its raise-mode
evaluation's statuses and values at the reference geometry and at declared
failure probes, and that evaluation matches the native provider. Outputs are
energy, forces, per-atom energy, optional tensile stress, the status, and the
frozen route count; failed values cross the wire as zero carriers and the
loader restores NaN. Loaded executables refuse JAX transformations.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
import sysconfig
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.extend import core as jax_core

from .._array_archive import read_array_archive, write_array_archive
from .._external_runtime import (
    _require_execution,
    ExternalExecutionPolicy,
    ExternalRuntimeError,
    pin_executable,
    PinnedFileOutputs,
    PinnedFileRequest,
    run_pinned_command,
)
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._model import model_structure_recipe, register_artifact_value
from .._model._component import ExecutionCapabilities
from .._model._structure import model_from_array_recipe, pack_model_array_tree
from .._publication import publish_resource_set
from .._resource_set import ResourceSetLimits
from ..atomistic._born_oppenheimer import (
    NativeAtomisticProvider,
    NativeAtomisticProviderPlan,
    NativeAtomisticRequest,
)
from ..atomistic._graph import (
    AtomisticGraphExecutionPlan,
    particle_atomistic_graph_topology,
)
from ..atomistic._model_artifact import (
    ATOMISTIC_MODEL_ARTIFACT_LIMITS,
    model_artifact_from_section,
    model_artifact_section,
)
from ..atomistic._system import AtomisticSystemPlan
from ..atomistic._types import AtomicStructure, AtomisticStatus
from ..atomistic._units import AtomisticUnitSystem
from ..discretization._core import DiscretizationKey, DiscretizationRole
from ..discretization._periodic_cell import lattice_right_inverse_with_status
from ..discretization.particle._image_neighborhood import (
    ParticleImageCapacity,
    ParticleImageNeighborhoodState,
)
from ..discretization.particle._neighborhood import (
    DenseParticleNeighborhoodPlan,
    ParticleNeighborhoodState,
)
from ..discretization.particle._verlet import (
    ParticleImageVerletState,
    ParticleVerletState,
    PreparedImageVerletParticleNeighborhood,
    PreparedVerletParticleNeighborhood,
)
from ..sparse._streamed import StreamedRelationPlan
from ..typing import parse
from ._iree import (
    IREEArtifactManifest,
    IREEExecutable,
    IREEExportPolicy,
    load_iree,
)


AtomisticExportRoute: TypeAlias = Literal["native-jax"]
AtomisticExportFailurePolicy: TypeAlias = Literal["equinox-nan-status"]

_CONTRACT_KIND = "atomistic-iree-contract"
_SCALAR_OUTPUTS = ("energy", "forces", "atom_energy")
_STATUS_OUTPUTS = ("status", "active_routes")
_FAILURE_POLICY: AtomisticExportFailurePolicy = "equinox-nan-status"

_REQUEST_FORMAT = "phydrax-atomistic-iree-export-request"
_RESULT_FORMAT = "phydrax-atomistic-iree-export-result"
_REQUEST_FIELDS = frozenset(
    {
        "arrays",
        "contract",
        "format",
        "iree_policy",
        "model",
        "provider",
        "runtime",
        "structure",
        "units",
    }
)
_PROVIDER_FIELDS = frozenset(
    {"deformation_margin", "finite_neighborhood", "graph_execution", "plan_id", "skin"}
)
_STRUCTURE_FIELDS = frozenset(
    {
        "cell",
        "coordinate_dtype",
        "name",
        "numeric_version",
        "periodic_axes",
        "structure_id",
    }
)
_RESULT_FIELDS = frozenset(
    {"contract_id", "format", "implementation_root", "module_sha256", "runtime"}
)
_STRUCTURE_ARRAYS = (
    "atomic_numbers",
    "positions",
    "masses",
    "particle_ids",
    "active_mask",
)
# A model artifact plus the structure arrays and plan recipes of one export.
_REQUEST_LIMITS = dataclasses.replace(
    ATOMISTIC_MODEL_ARTIFACT_LIMITS,
    max_members=ATOMISTIC_MODEL_ARTIFACT_LIMITS.max_members + 16,
)

# Worker process boundary: exact staged/published names and byte bounds.
_REQUEST_NAME = "request.phxarchive"
_MODULE_NAME = "module.vmfb"
_MANIFEST_NAME = "manifest.json"
_RESULT_NAME = "result.json"
_MAX_MODULE_BYTES = 4 * 1024 * 1024 * 1024
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_RESULT_BYTES = 1024 * 1024
_MAX_LOG_BYTES = 16 * 1024 * 1024
_REFUSED_EXIT = 3
_WORKER_MODULE = "phydrax.export._atomistic_export_worker"
# Distributions whose exact releases the worker must import from the declared
# site directories; a different release refuses the export.
_RUNTIME_DISTRIBUTIONS = (
    "equinox",
    "iree-base-compiler",
    "iree-base-runtime",
    "jax",
    "jaxlib",
    "numpy",
)
# Fixed isolated bootstrap (run with ``-I -S``): append the parent's own
# implementation root after the standard library and before the declared site
# directories, so the worker imports exactly the parent's Phydrax tree and
# never another checkout reachable through a site ``.pth`` path.
_WORKER_BOOTSTRAP = (
    "import runpy, site, sys\n"
    "root, count = sys.argv[1], int(sys.argv[2])\n"
    "sys.path.append(root)\n"
    "for path in sys.argv[3:3 + count]:\n"
    "    site.addsitedir(path)\n"
    "sys.argv = [sys.argv[0], *sys.argv[3 + count:]]\n"
    f"runpy.run_module({_WORKER_MODULE!r}, run_name='__main__', alter_sys=True)\n"
)

# Provider and neighborhood plan types persisted by export requests, registered
# here by their first artifact consumer.
for _artifact_id, _artifact_type in (
    ("phydrax.atomistic:AtomisticGraphExecutionPlan", AtomisticGraphExecutionPlan),
    ("phydrax.discretization:ParticleImageCapacity", ParticleImageCapacity),
    (
        "phydrax.discretization:DenseParticleNeighborhoodPlan",
        DenseParticleNeighborhoodPlan,
    ),
    ("phydrax.discretization:DiscretizationKey", DiscretizationKey),
    ("phydrax.discretization:DiscretizationRole", DiscretizationRole),
):
    register_artifact_value(_artifact_id, _artifact_type)
del _artifact_id, _artifact_type


class AtomisticExportRefusal(ValueError):
    """A model, plan, program, or runtime outside the frozen export contract."""


@dataclass(frozen=True, slots=True)
class AtomisticIREEContract:
    """Fixed E/F/S ABI of one frozen native program and candidate graph.

    ``route`` names the executed realization: ``"native-jax"`` is the ordinary
    JAX program lowered by ``jax.export``; accelerated kernels are not
    exportable and are never silently substituted. ``failure_policy`` names
    the declared trace-time guard semantics. Species, units, periodicity,
    capacities, the frozen graph epoch (``graph_id``), and the ordered input
    and output shapes and dtypes are bound to the identity.
    """

    route: AtomisticExportRoute
    failure_policy: AtomisticExportFailurePolicy
    program_id: str
    provider_id: str
    unit_system_id: str
    atomic_numbers: tuple[int, ...]
    periodic: bool
    stress: bool
    graph_id: str
    input_names: tuple[str, ...]
    input_shapes: tuple[tuple[int, ...], ...]
    input_dtypes: tuple[str, ...]
    output_names: tuple[str, ...]
    output_shapes: tuple[tuple[int, ...], ...]
    output_dtypes: tuple[str, ...]
    contract_id: str

    def to_dict(self, /) -> dict[str, Any]:
        return {
            "kind": _CONTRACT_KIND,
            "route": self.route,
            "failure_policy": self.failure_policy,
            "program_id": self.program_id,
            "provider_id": self.provider_id,
            "unit_system_id": self.unit_system_id,
            "atomic_numbers": list(self.atomic_numbers),
            "periodic": self.periodic,
            "stress": self.stress,
            "graph_id": self.graph_id,
            "input_names": list(self.input_names),
            "input_shapes": [list(shape) for shape in self.input_shapes],
            "input_dtypes": list(self.input_dtypes),
            "output_names": list(self.output_names),
            "output_shapes": [list(shape) for shape in self.output_shapes],
            "output_dtypes": list(self.output_dtypes),
            "contract_id": self.contract_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], /) -> AtomisticIREEContract:
        expected = {
            "kind",
            "route",
            "failure_policy",
            "program_id",
            "provider_id",
            "unit_system_id",
            "atomic_numbers",
            "periodic",
            "stress",
            "graph_id",
            "input_names",
            "input_shapes",
            "input_dtypes",
            "output_names",
            "output_shapes",
            "output_dtypes",
            "contract_id",
        }
        if set(value) != expected or value["kind"] != _CONTRACT_KIND:
            raise ValueError("Atomistic IREE contract fields are not canonical.")
        contract = _contract(
            parse(value["route"], AtomisticExportRoute, "route"),
            parse(
                value["failure_policy"], AtomisticExportFailurePolicy, "failure_policy"
            ),
            str(value["program_id"]),
            str(value["provider_id"]),
            str(value["unit_system_id"]),
            tuple(int(number) for number in value["atomic_numbers"]),
            bool(value["periodic"]),
            bool(value["stress"]),
            str(value["graph_id"]),
            tuple(str(name) for name in value["input_names"]),
            tuple(tuple(int(size) for size in shape) for shape in value["input_shapes"]),
            tuple(str(dtype) for dtype in value["input_dtypes"]),
            tuple(tuple(int(size) for size in shape) for shape in value["output_shapes"]),
            tuple(str(dtype) for dtype in value["output_dtypes"]),
        )
        if contract.contract_id != value["contract_id"] or contract.output_names != tuple(
            value["output_names"]
        ):
            raise ValueError("Atomistic IREE contract identity is corrupt.")
        return contract

    def pack_inputs(self, request: NativeAtomisticRequest, /) -> tuple[Array, ...]:
        """Order one host request as inputs, refusing any other frozen graph.

        The request's lifecycle must be the frozen epoch of this export and
        must have succeeded; runtime geometry is checked only by the module.
        """

        frozen = _frozen_graph(request)
        if frozen.graph_id != self.graph_id:
            raise ValueError(
                "Request candidate graph differs from the graph frozen in the export; "
                "a rebuilt or other lifecycle epoch requires a new export."
            )
        if not bool(np.asarray(request.neighborhood.successful)):
            raise ValueError("Request candidate lifecycle failed on the host.")
        values: tuple[Array, ...] = (request.positions,)
        if self.periodic:
            if request.cell_vectors is None:
                raise ValueError("Periodic atomistic export requires cell vectors.")
            values = (*values, request.cell_vectors, request.image_counts)
        elif request.cell_vectors is not None:
            raise ValueError("Aperiodic atomistic export accepts no cell vectors.")
        self.check_inputs(values)
        return values

    def check_inputs(self, values: Sequence[Any], /) -> None:
        """Refuse inputs whose count, shape, or dtype differs from the ABI."""

        if len(values) != len(self.input_names):
            raise ValueError("Atomistic IREE input count differs from the contract.")
        for value, shape, dtype in zip(
            values, self.input_shapes, self.input_dtypes, strict=True
        ):
            if tuple(np.shape(value)) != shape:
                raise ValueError("Atomistic IREE input shape differs from the contract.")
            if np.dtype(value.dtype).str != dtype:
                raise TypeError("Atomistic IREE input dtype differs from the contract.")


def _output_names(stress: bool, /) -> tuple[str, ...]:
    return (*_SCALAR_OUTPUTS, *(("stress",) if stress else ()), *_STATUS_OUTPUTS)


def _contract(
    route: AtomisticExportRoute,
    failure_policy: AtomisticExportFailurePolicy,
    program_id: str,
    provider_id: str,
    unit_system_id: str,
    atomic_numbers: tuple[int, ...],
    periodic: bool,
    stress: bool,
    graph_id: str,
    input_names: tuple[str, ...],
    input_shapes: tuple[tuple[int, ...], ...],
    input_dtypes: tuple[str, ...],
    output_shapes: tuple[tuple[int, ...], ...],
    output_dtypes: tuple[str, ...],
    /,
) -> AtomisticIREEContract:
    outputs = _output_names(stress)
    if len(output_shapes) != len(outputs) or len(output_dtypes) != len(outputs):
        raise ValueError("Atomistic IREE contract must describe every output.")
    identity = canonical_fingerprint(
        {
            "kind": _CONTRACT_KIND,
            "route": route,
            "failure_policy": failure_policy,
            "program_id": program_id,
            "provider_id": provider_id,
            "unit_system_id": unit_system_id,
            "atomic_numbers": list(atomic_numbers),
            "periodic": periodic,
            "stress": stress,
            "graph_id": graph_id,
            "input_names": list(input_names),
            "input_shapes": [list(shape) for shape in input_shapes],
            "input_dtypes": list(input_dtypes),
            "output_names": list(outputs),
            "output_shapes": [list(shape) for shape in output_shapes],
            "output_dtypes": list(output_dtypes),
        }
    )
    return AtomisticIREEContract(
        route,
        failure_policy,
        program_id,
        provider_id,
        unit_system_id,
        atomic_numbers,
        periodic,
        stress,
        graph_id,
        input_names,
        input_shapes,
        input_dtypes,
        outputs,
        output_shapes,
        output_dtypes,
        identity,
    )


@dataclass(frozen=True, slots=True)
class _FrozenGraph:
    """Reference relation of one lifecycle epoch and its content identity."""

    graph: ParticleImageNeighborhoodState | ParticleNeighborhoodState
    reference_image_counts: Array | None
    graph_id: str


def _array_identity(value: Any, /) -> dict[str, Any]:
    arrays, static = eqx.partition(value, eqx.is_array)
    return {
        "structure": str(jax.tree.structure(arrays)),
        "static": str(jax.tree.structure(static)),
        "arrays": array_tree_fingerprint(
            [np.asarray(leaf) for leaf in jax.tree.leaves(arrays)]
        ),
    }


def _frozen_graph(request: NativeAtomisticRequest, /) -> _FrozenGraph:
    """Return the epoch's reference relation that the module freezes."""

    state = request.neighborhood
    if isinstance(state, ParticleImageVerletState):
        graph: ParticleImageNeighborhoodState | ParticleNeighborhoodState = (
            state.reference
        )
        counts: Array | None = state.reference_image_counts
    elif isinstance(state, ParticleVerletState):
        graph, counts = state.neighborhood, None
    else:
        raise TypeError("Frozen export requires a native Verlet lifecycle state.")
    identity = canonical_fingerprint(
        {
            "kind": "atomistic-export-graph",
            "lifecycle": state.prepared_verlet_id,
            "epoch": int(np.asarray(state.epoch)),
            "graph": _array_identity(graph),
            "reference_image_counts": (
                None if counts is None else array_tree_fingerprint(np.asarray(counts))
            ),
        }
    )
    return _FrozenGraph(graph, counts, identity)


def prepare_atomistic_iree_contract(
    provider: NativeAtomisticProvider,
    request: NativeAtomisticRequest,
    /,
    *,
    route: AtomisticExportRoute = "native-jax",
) -> AtomisticIREEContract:
    """Prepare the fixed ABI for one provider and frozen candidate epoch.

    The output shapes and dtypes are those of the frozen program over the
    request's own inputs.
    """

    if not isinstance(provider, NativeAtomisticProvider):
        raise TypeError("provider must be a NativeAtomisticProvider.")
    route_ = parse(route, AtomisticExportRoute, "route")
    system = provider.program.system
    periodic = system.cell is not None
    arrays: list[Array] = [request.positions]
    names = ["positions"]
    if periodic:
        if request.cell_vectors is None:
            raise ValueError("Periodic atomistic export requires cell vectors.")
        arrays += [request.cell_vectors, request.image_counts]
        names += ["cell_vectors", "image_counts"]
    graph_id = _frozen_graph(request).graph_id
    stress = provider.stress_available
    outputs = jax.eval_shape(
        _frozen_evaluation(provider, request, graph_id, periodic, stress), *arrays
    )
    return _contract(
        route_,
        _FAILURE_POLICY,
        provider.program.prepared_id,
        provider.provider_id,
        system.plan.units.unit_system_id,
        tuple(int(number) for number in np.asarray(system.plan.atomic_numbers)),
        periodic,
        stress,
        graph_id,
        tuple(names),
        tuple(tuple(value.shape) for value in arrays),
        tuple(np.dtype(value.dtype).str for value in arrays),
        tuple(tuple(output.shape) for output in outputs),
        tuple(np.dtype(output.dtype).str for output in outputs),
    )


def _exported_evaluation(
    provider: NativeAtomisticProvider,
    request: NativeAtomisticRequest,
    contract: AtomisticIREEContract,
    /,
) -> Callable[..., tuple[Array, ...]]:
    """Frozen-graph E/F/S program of ``contract`` over the runtime geometry inputs."""

    return _frozen_evaluation(
        provider, request, contract.graph_id, contract.periodic, contract.stress
    )


def _frozen_evaluation(
    provider: NativeAtomisticProvider,
    request: NativeAtomisticRequest,
    graph_id: str,
    periodic: bool,
    stress: bool,
    /,
) -> Callable[..., tuple[Array, ...]]:
    """Frozen-graph E/F/S program over the runtime geometry inputs."""

    frozen = _frozen_graph(request)
    if frozen.graph_id != graph_id:
        raise ValueError("Request candidate graph differs from the contract.")
    graph = frozen.graph
    reference_counts = frozen.reference_image_counts
    program = provider.program
    execution = program.graph_execution
    # The route topology and streamed schedule are prepared once on the host
    # from the frozen relation, so the executable never re-sorts its routes.
    topology = (
        {}
        if execution is None or not program.plan.requirements.directed_graph
        else {
            "topology": particle_atomistic_graph_topology(
                program.system,
                graph,
                cell=program.system.cell,
                streamed=execution.streamed,
            )
        }
    )
    if isinstance(graph, ParticleImageNeighborhoodState):
        stored_routes = int(np.sum(np.asarray(graph.evidence.stored_edges)))
    else:
        stored_routes = int(np.sum(np.asarray(graph.pair_count)))

    lifecycle = provider.neighborhood
    frozen_state = request.neighborhood
    image_lifecycle = isinstance(
        lifecycle, PreparedImageVerletParticleNeighborhood
    ) and isinstance(frozen_state, ParticleImageVerletState)
    if not image_lifecycle and not (
        isinstance(lifecycle, PreparedVerletParticleNeighborhood)
        and isinstance(frozen_state, ParticleVerletState)
    ):
        raise TypeError("Frozen export requires a native Verlet lifecycle and its state.")

    def certifies(inputs: Sequence[Array]) -> Array:
        """The lifecycle's own reuse certificate of the frozen epoch."""

        lattice = (
            {"cell_vectors": inputs[1], "image_counts": inputs[2]} if periodic else {}
        )
        if isinstance(lifecycle, PreparedImageVerletParticleNeighborhood) and isinstance(
            frozen_state, ParticleImageVerletState
        ):
            return lifecycle.certifies(inputs[0], frozen_state, **lattice)
        if isinstance(lifecycle, PreparedVerletParticleNeighborhood) and isinstance(
            frozen_state, ParticleVerletState
        ):
            lattice.pop("image_counts", None)
            return lifecycle.certifies(inputs[0], frozen_state, **lattice)
        raise TypeError("Frozen export requires a native Verlet lifecycle and its state.")

    def certified_and_valid(inputs: Sequence[Array]) -> tuple[Array, Array]:
        """Lifecycle reuse certificate of the frozen epoch, and geometry validity."""

        valid = jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(item)) for item in inputs)))
        if periodic:
            _, solved = lattice_right_inverse_with_status(inputs[1])
            valid = valid & solved
        return certifies(inputs), valid

    def evaluate(inputs: Sequence[Array]) -> tuple[tuple[Array, ...], Array, Array]:
        positions = inputs[0]
        context: dict[str, Array] = {}
        if periodic:
            if reference_counts is None:
                raise RuntimeError("Internal invariant failed: periodic counts absent.")
            cell_vectors, image_counts = inputs[1], inputs[2]
            dtype = positions.dtype
            context = {
                "cell_vectors": cell_vectors,
                "unwrapped_positions": positions
                + image_counts.astype(dtype) @ cell_vectors,
            }
            # Whole-lattice re-expression of the current representation in the
            # frozen routes' reference frame.
            positions = (
                positions + (image_counts - reference_counts).astype(dtype) @ cell_vectors
            )
        value = program.evaluate(
            positions, graph, compute_stress=stress, **topology, **context
        )
        carriers = [value.energy, value.forces, value.atom_energy]
        if stress:
            if value.stress is None:
                raise RuntimeError("Internal invariant failed: requested stress absent.")
            carriers.append(value.stress)
        return tuple(carriers), value.successful, value.graph_overflow

    def forward(*inputs: Array, key: Array | None = None) -> tuple[Array, ...]:
        if key is not None:
            raise ValueError("Atomistic IREE export requires key=None.")
        certified, valid = certified_and_valid(inputs)
        shapes = jax.eval_shape(evaluate, inputs)

        def refused(_: Sequence[Array]) -> tuple[tuple[Array, ...], Array, Array]:
            # The frozen relation does not cover this geometry: no graph runs.
            zeros = tuple(jnp.zeros(item.shape, item.dtype) for item in shapes[0])
            return zeros, jnp.asarray(False), jnp.asarray(False)

        carriers, successful, overflow = jax.lax.cond(
            certified & valid, evaluate, refused, inputs
        )
        accepted = (
            certified
            & valid
            & successful
            & jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(item)) for item in carriers)))
        )
        status = jnp.where(
            accepted,
            int(AtomisticStatus.SUCCESS),
            jnp.where(
                valid & (~certified | overflow),
                int(AtomisticStatus.NEIGHBOR_OVERFLOW),
                int(AtomisticStatus.NONFINITE),
            ),
        ).astype(jnp.int32)
        outputs = tuple(jnp.where(accepted, item, 0.0) for item in carriers)
        return (*outputs, status, jnp.asarray(stored_routes, dtype=jnp.int32))

    return forward


@dataclass(frozen=True, slots=True)
class _GuardSite:
    kind: Literal["guard", "host-callback"]
    dynamic: bool
    location: str


def _sub_jaxprs(eqn: Any, /) -> tuple[Any, ...]:
    found = []
    for value in eqn.params.values():
        for item in value if isinstance(value, (tuple, list)) else (value,):
            if isinstance(item, jax_core.ClosedJaxpr):
                found.append(item.jaxpr)
            elif isinstance(item, jax_core.Jaxpr):
                found.append(item)
    return tuple(found)


def _is_callback(eqn: Any, /) -> bool:
    return "callback" in eqn.primitive.name


def _location(eqn: Any, /) -> str:
    return str(eqn.source_info.name_stack) or eqn.primitive.name


def _scan_layout(eqn: Any, /) -> tuple[int, int] | None:
    """Constant and carry counts of one ``scan``, or ``None`` if unknown."""

    params = eqn.params
    if "num_consts" in params and "num_carry" in params:
        return int(params["num_consts"]), int(params["num_carry"])
    try:
        consts, carry, _ = params["ft_in"].unpack()
        return len(list(consts.vals)), len(list(carry.vals))
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def _input_dependence(
    jaxpr: Any, flags: Sequence[bool], sites: list[_GuardSite] | None, /
) -> list[bool]:
    """Propagate runtime-input dependence and record guard and callback sites.

    Equinox raise-mode guards appear as a ``cond`` whose error branch calls a
    host callback directly; the guard is dynamic exactly when its branch index
    depends on a runtime input. Control-flow carries reach a fixed point;
    unknown higher-order primitives are treated conservatively. ``sites`` is
    ``None`` during fixed-point iterations.
    """

    dependent = {
        variable for variable, flag in zip(jaxpr.invars, flags, strict=True) if flag
    }

    def depends(variable: Any, /) -> bool:
        return isinstance(variable, jax_core.Var) and variable in dependent

    for eqn in jaxpr.eqns:
        inputs = [depends(variable) for variable in eqn.invars]
        count = len(eqn.outvars)
        name = eqn.primitive.name
        subs = _sub_jaxprs(eqn)
        if _is_callback(eqn):
            if sites is not None:
                sites.append(_GuardSite("host-callback", any(inputs), _location(eqn)))
            outputs = [any(inputs)] * count
        elif name == "cond":
            branches = [branch.jaxpr for branch in eqn.params["branches"]]
            if any(_is_callback(inner) for branch in branches for inner in branch.eqns):
                if sites is not None:
                    sites.append(_GuardSite("guard", inputs[0], _location(eqn)))
                outputs = [any(inputs)] * count
            else:
                outputs = [inputs[0]] * count
                for branch in branches:
                    result = _input_dependence(branch, inputs[1:], sites)
                    outputs = [a or b for a, b in zip(outputs, result, strict=True)]
        elif name == "scan" and (layout := _scan_layout(eqn)) is not None:
            consts, carried = layout
            body = eqn.params["jaxpr"].jaxpr
            head, carry, tail = (
                inputs[:consts],
                inputs[consts : consts + carried],
                inputs[consts + carried :],
            )
            while True:
                result = _input_dependence(body, head + carry + tail, None)
                updated = [a or b for a, b in zip(carry, result[:carried], strict=True)]
                if updated == carry:
                    break
                carry = updated
            result = _input_dependence(body, head + carry + tail, sites)
            outputs = [
                a or b for a, b in zip(carry, result[:carried], strict=True)
            ] + result[carried:]
        elif name == "scan":
            for sub in subs:
                _input_dependence(sub, [any(inputs)] * len(sub.invars), sites)
            outputs = [any(inputs)] * count
        elif name == "while":
            condition_consts = eqn.params["cond_nconsts"]
            body_consts = eqn.params["body_nconsts"]
            condition = eqn.params["cond_jaxpr"].jaxpr
            body = eqn.params["body_jaxpr"].jaxpr
            condition_head = inputs[:condition_consts]
            body_head = inputs[condition_consts : condition_consts + body_consts]
            carry = inputs[condition_consts + body_consts :]
            while True:
                result = _input_dependence(body, body_head + carry, None)
                updated = [a or b for a, b in zip(carry, result, strict=True)]
                if updated == carry:
                    break
                carry = updated
            trips = _input_dependence(condition, condition_head + carry, sites)[0]
            result = _input_dependence(body, body_head + carry, sites)
            outputs = [a or b or trips for a, b in zip(carry, result, strict=True)]
        elif subs and all(len(sub.invars) == len(eqn.invars) for sub in subs):
            outputs = [False] * count
            for sub in subs:
                result = _input_dependence(sub, inputs, sites)
                outputs = (
                    [a or b for a, b in zip(outputs, result, strict=True)]
                    if len(result) == count
                    else [any(inputs)] * count
                )
        else:
            for sub in subs:
                _input_dependence(sub, [any(inputs)] * len(sub.invars), sites)
            outputs = [any(inputs)] * count
        dependent.update(
            variable for variable, flag in zip(eqn.outvars, outputs, strict=True) if flag
        )
    return [depends(variable) for variable in jaxpr.outvars]


def _runtime_guard_sites(
    forward: Callable[..., Any], inputs: Sequence[Array], /
) -> tuple[_GuardSite, ...]:
    """Guards and host callbacks of the frozen program under raise-mode tracing."""

    closed = jax.make_jaxpr(forward)(*inputs)
    sites: list[_GuardSite] = []
    _input_dependence(closed.jaxpr, [True] * len(closed.jaxpr.invars), sites)
    return tuple(sites)


def _require_status_complete(
    forward: Callable[..., Any], inputs: Sequence[Array], /
) -> None:
    """Refuse a program whose runtime failures could bypass its status output."""

    refused = [
        site
        for site in _runtime_guard_sites(forward, inputs)
        if site.kind == "host-callback" or site.dynamic
    ]
    if refused:
        locations = "; ".join(f"{site.kind} at {site.location}" for site in refused[:8])
        raise AtomisticExportRefusal(
            "Frozen export requires every runtime-input failure to be a traced "
            "status; the program keeps host callbacks or input-dependent guards: "
            f"{locations}."
        )


def _require_runtime_free_lowering(
    program: Callable[..., Any], inputs: Sequence[Array], /
) -> None:
    """Refuse host-runtime custom calls (e.g. LAPACK) IREE cannot compile.

    The raise-mode lowering keeps the audited constant guards as callbacks;
    those lower callback-free in the worker, so only other targets refuse.
    """

    text = jax.jit(program).trace(*inputs).lower(lowering_platforms=("cpu",)).as_text()
    targets = sorted(
        {
            target
            for target in re.findall(r"custom_call @([\w.$-]+)", text)
            if "callback" not in target
        }
    )
    if targets:
        raise AtomisticExportRefusal(
            "The frozen program lowers host-runtime custom calls that IREE cannot "
            f"compile: {', '.join(targets)}."
        )


@dataclass(frozen=True, slots=True)
class AtomisticIREEExportBundle:
    """Published frozen E/F/S executable, its digest, and its contract."""

    path: Path
    module_sha256: str
    contract: AtomisticIREEContract


def _native_provider(
    plan: NativeAtomisticProviderPlan,
    structure: AtomicStructure,
    units: AtomisticUnitSystem,
    /,
) -> tuple[NativeAtomisticProvider, NativeAtomisticRequest]:
    system = AtomisticSystemPlan.from_structure(structure, units).prepare()
    provider = plan.prepare(system)
    return provider, provider.prepare_request(structure.positions, None, None)


def _plan_recipe(value: Any, prefix: str, /) -> dict[str, Any]:
    recipe = model_structure_recipe(value, path=prefix)
    if pack_model_array_tree(value, recipe, prefix=prefix):
        raise AtomisticExportRefusal(f"{prefix} must be a static registered plan.")
    return recipe


def _replayed_execution(
    value: AtomisticGraphExecutionPlan, /
) -> AtomisticGraphExecutionPlan:
    """Reconstruct a graph execution plan through its owning constructors."""

    streamed = value.streamed
    capacity = value.image_capacity
    return AtomisticGraphExecutionPlan(
        value.maximum_neighbors,
        backend=value.backend,
        maximum_dense_atoms=value.maximum_dense_atoms,
        image_capacity=(
            None
            if capacity is None
            else ParticleImageCapacity(
                maximum_particles_per_cell=capacity.maximum_particles_per_cell,
                maximum_edges=capacity.maximum_edges,
                maximum_degree=capacity.maximum_degree,
                maximum_images=capacity.maximum_images,
            )
        ),
        streamed=StreamedRelationPlan(
            receiver_tile=streamed.receiver_tile,
            edge_tile=streamed.edge_tile,
            channel_capacity=streamed.channel_capacity,
            accumulation=streamed.accumulation,
            replay=streamed.replay,
            replay_block_size=streamed.replay_block_size,
            transpose=streamed.transpose,
        ),
        maximum_candidate_slots=value.maximum_candidate_slots,
        plan_id=value.plan_id,
    )


def _replayed_finite(value: Any, /) -> DenseParticleNeighborhoodPlan:
    """Reconstruct the admitted finite neighborhood through its constructor."""

    if type(value) is not DenseParticleNeighborhoodPlan or value.box is not None:
        raise AtomisticExportRefusal(
            "Frozen export persists a box-free DenseParticleNeighborhoodPlan finite "
            "neighborhood; other finite neighborhood plans have no registered recipe."
        )
    return DenseParticleNeighborhoodPlan(
        value.maximum_pairs, name=value.key.name, plan_id=value.plan_id
    )


def _restored_plan_part(
    recipe: Mapping[str, Any],
    prefix: str,
    expected: type,
    replay: Callable[[Any], Any],
    /,
) -> Any:
    try:
        raw = model_from_array_recipe(recipe, {}, prefix=prefix, limits=_REQUEST_LIMITS)
    except (KeyError, TypeError, ValueError) as error:
        raise AtomisticExportRefusal(f"{prefix} recipe is invalid: {error}") from error
    if type(raw) is not expected:
        raise AtomisticExportRefusal(f"{prefix} recipe restores another type.")
    replayed = replay(raw)
    if model_structure_recipe(replayed, path=prefix) != recipe:
        raise AtomisticExportRefusal(
            f"{prefix} recipe is not reproduced by its owning constructors."
        )
    return replayed


def _provider_recipe(plan: NativeAtomisticProviderPlan, /) -> dict[str, Any]:
    """Exact registered recipe of every provider member except the model."""

    _replayed_finite(plan.finite_neighborhood)
    recipe = {
        "graph_execution": _plan_recipe(plan.graph_execution, "graph_execution"),
        "finite_neighborhood": _plan_recipe(
            plan.finite_neighborhood, "finite_neighborhood"
        ),
        "skin": plan.skin,
        "deformation_margin": plan.deformation_margin,
        "plan_id": plan.plan_id,
    }
    _restored_provider_plan(recipe, plan.model)
    return recipe


def _restored_provider_plan(
    recipe: Mapping[str, Any], model: Any, /
) -> NativeAtomisticProviderPlan:
    if not isinstance(recipe, Mapping) or set(recipe) != _PROVIDER_FIELDS:
        raise AtomisticExportRefusal("Provider recipe fields are not canonical.")
    execution = _restored_plan_part(
        recipe["graph_execution"],
        "graph_execution",
        AtomisticGraphExecutionPlan,
        _replayed_execution,
    )
    finite = _restored_plan_part(
        recipe["finite_neighborhood"],
        "finite_neighborhood",
        DenseParticleNeighborhoodPlan,
        _replayed_finite,
    )
    plan = NativeAtomisticProviderPlan(
        model,
        execution,
        finite_neighborhood=finite,
        skin=float(recipe["skin"]),
        deformation_margin=float(recipe["deformation_margin"]),
    )
    if plan.plan_id != recipe["plan_id"]:
        raise AtomisticExportRefusal(
            "Native provider plan is not reproduced by its recipe."
        )
    return plan


def _structure_record(
    structure: AtomicStructure, /
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    arrays = {
        f"structure/{name}": np.asarray(getattr(structure, name))
        for name in _STRUCTURE_ARRAYS
    }
    if structure.cell is not None:
        arrays["structure/cell"] = np.asarray(structure.cell)
    if structure.periodic_axes is not None:
        arrays["structure/periodic_axes"] = np.asarray(structure.periodic_axes)
    record = {
        "name": structure.name,
        "numeric_version": structure.particles.numeric_version,
        "coordinate_dtype": str(np.asarray(structure.positions).dtype),
        "cell": structure.cell is not None,
        "periodic_axes": structure.periodic_axes is not None,
        "structure_id": structure.structure_id,
    }
    return record, arrays


def _restored_structure(
    record: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    units: AtomisticUnitSystem,
    /,
) -> AtomicStructure:
    if not isinstance(record, Mapping) or set(record) != _STRUCTURE_FIELDS:
        raise AtomisticExportRefusal("Structure record fields are not canonical.")
    values = {name: arrays[f"structure/{name}"] for name in _STRUCTURE_ARRAYS}
    structure = AtomicStructure(
        values["atomic_numbers"],
        values["positions"],
        values["masses"],
        units.scale,
        particle_ids=values["particle_ids"],
        active_mask=values["active_mask"],
        cell=arrays["structure/cell"] if record["cell"] else None,
        periodic_axes=(
            arrays["structure/periodic_axes"] if record["periodic_axes"] else None
        ),
        name=str(record["name"]),
        coordinate_dtype=str(record["coordinate_dtype"]),
        numeric_version=str(record["numeric_version"]),
    )
    if structure.structure_id != record["structure_id"]:
        raise AtomisticExportRefusal("Restored structure differs from its record.")
    return structure


def _implementation_root() -> Path:
    """Directory holding the Phydrax package this process executes."""

    return Path(__file__).resolve(strict=True).parents[2]


def _runtime_record() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "distributions": {
            name: importlib.metadata.version(name) for name in _RUNTIME_DISTRIBUTIONS
        },
    }


@dataclass(frozen=True, slots=True)
class _ExportRequest:
    """Everything one worker needs to reproduce the parent's frozen program."""

    plan: NativeAtomisticProviderPlan
    structure: AtomicStructure
    units: AtomisticUnitSystem
    contract: AtomisticIREEContract
    policy: IREEExportPolicy
    runtime: dict[str, Any]


def _write_export_request(
    path: Path,
    plan: NativeAtomisticProviderPlan,
    structure: AtomicStructure,
    units: AtomisticUnitSystem,
    contract: AtomisticIREEContract,
    policy: IREEExportPolicy,
    /,
) -> None:
    section, model_arrays, _ = model_artifact_section(plan.model)
    structure_record, structure_arrays = _structure_record(structure)
    write_array_archive(
        path,
        manifest={
            "format": _REQUEST_FORMAT,
            "model": section,
            "provider": _provider_recipe(plan),
            "structure": structure_record,
            "units": units.to_dict(),
            "contract": contract.to_dict(),
            "iree_policy": {
                "target_backend": policy.target_backend,
                "runtime_driver": policy.runtime_driver,
                "executable_format": policy.executable_format,
                "system_linker": policy.system_linker,
            },
            "runtime": _runtime_record(),
        },
        limits=_REQUEST_LIMITS,
        arrays={**model_arrays, **structure_arrays},
    )


def _read_export_request(path: Path, /) -> _ExportRequest:
    manifest, arrays = read_array_archive(path, limits=_REQUEST_LIMITS)
    if set(manifest) != _REQUEST_FIELDS or manifest["format"] != _REQUEST_FORMAT:
        raise AtomisticExportRefusal("Archive is not an atomistic export request.")
    model_arrays = {
        name: value for name, value in arrays.items() if not name.startswith("structure/")
    }
    model = model_artifact_from_section(manifest["model"], model_arrays).model
    units = AtomisticUnitSystem.from_dict(manifest["units"])
    policy = manifest["iree_policy"]
    if not isinstance(policy, Mapping) or set(policy) != {
        "target_backend",
        "runtime_driver",
        "executable_format",
        "system_linker",
    }:
        raise AtomisticExportRefusal("IREE policy record is not canonical.")
    return _ExportRequest(
        _restored_provider_plan(manifest["provider"], model),
        _restored_structure(manifest["structure"], arrays, units),
        units,
        AtomisticIREEContract.from_dict(manifest["contract"]),
        IREEExportPolicy(
            str(policy["target_backend"]),
            str(policy["runtime_driver"]),
            policy["executable_format"],
            policy["system_linker"],
        ),
        dict(manifest["runtime"]),
    )


def _worker_environment() -> dict[str, str]:
    return {
        "EQX_ON_ERROR": "nan",
        "JAX_ENABLE_X64": "1" if jax.config.jax_enable_x64 else "0",
        "JAX_PLATFORMS": "cpu",
    }


def _run_export_worker(
    request: bytes, destination: Path, timeout: float, /
) -> dict[str, Path]:
    """Compile in a pinned isolated interpreter; return its published files."""

    root = _implementation_root()
    sites = tuple(
        dict.fromkeys(
            str(Path(sysconfig.get_path(name)).resolve(strict=True))
            for name in ("purelib", "platlib")
        )
    )
    interpreter = pin_executable(
        sys.executable, version=platform.python_version(), license_id="PSF-2.0"
    )
    environment = _worker_environment()
    files = (
        PinnedFileRequest(_MODULE_NAME, _MAX_MODULE_BYTES),
        PinnedFileRequest(_MANIFEST_NAME, _MAX_MANIFEST_BYTES),
        PinnedFileRequest(_RESULT_NAME, _MAX_RESULT_BYTES),
    )
    try:
        result = run_pinned_command(
            interpreter,
            [
                "-I",
                "-S",
                "-c",
                _WORKER_BOOTSTRAP,
                str(root),
                str(len(sites)),
                *sites,
                str(root),
                _REQUEST_NAME,
            ],
            inputs={_REQUEST_NAME: request},
            timeout=timeout,
            max_output_bytes=len(request) + _MAX_LOG_BYTES,
            environment=environment,
            execution_policy=ExternalExecutionPolicy(
                inherit_environment=False,
                allowed_environment_variables=tuple(environment),
            ),
            artifacts=PinnedFileOutputs(
                str(destination),
                files,
                sum(file.maximum_bytes for file in files),
            ),
        )
    except ExternalRuntimeError as error:
        failed = error.result
        diagnostic = (
            "" if failed is None else failed.stderr.decode("utf-8", "replace").strip()
        )
        if failed is not None and failed.returncode == _REFUSED_EXIT:
            refusal = next(
                (
                    line
                    for line in reversed(diagnostic.splitlines())
                    if line.startswith("REFUSED: ")
                ),
                "REFUSED: export worker refusal without a message",
            )
            raise AtomisticExportRefusal(refusal.removeprefix("REFUSED: ")) from error
        raise RuntimeError(
            f"Atomistic export worker failed: {error}\n{diagnostic[-8192:]}"
        ) from error
    return {file.path: Path(result.file_artifact(file.path).location) for file in files}


@dataclass(frozen=True, slots=True)
class _StagedModule:
    module: bytes
    manifest_bytes: bytes
    manifest: IREEArtifactManifest


def _staged_module(
    files: Mapping[str, Path], contract: AtomisticIREEContract, /
) -> _StagedModule:
    result = json.loads(files[_RESULT_NAME].read_text(encoding="utf-8"))
    expected_runtime = _runtime_record()
    if (
        not isinstance(result, Mapping)
        or set(result) != _RESULT_FIELDS
        or result["format"] != _RESULT_FORMAT
        or result["contract_id"] != contract.contract_id
        or result["implementation_root"] != str(_implementation_root())
        or result["runtime"] != expected_runtime
    ):
        raise RuntimeError(
            "Export worker did not run the parent's implementation, runtime and contract."
        )
    module = files[_MODULE_NAME].read_bytes()
    manifest_bytes = files[_MANIFEST_NAME].read_bytes()
    manifest = IREEArtifactManifest.from_dict(json.loads(manifest_bytes))
    digest = hashlib.sha256(module).hexdigest()
    if digest != manifest.module_sha256 or digest != result["module_sha256"]:
        raise RuntimeError("Export worker module digest is inconsistent.")
    if (
        manifest.domain_contract is None
        or json.loads(manifest.domain_contract) != contract.to_dict()
    ):
        raise RuntimeError("Export worker manifest carries another contract.")
    return _StagedModule(module, manifest_bytes, manifest)


def _publish_bundle(path: Path, staged: _StagedModule, mode: Any, /) -> None:
    members = {
        staged.manifest.module_file: staged.module,
        "manifest.json": staged.manifest_bytes,
    }
    publish_resource_set(
        path,
        members,
        limits=ResourceSetLimits(
            max_total_bytes=sum(len(data) for data in members.values()),
            max_member_bytes=max(len(data) for data in members.values()),
            max_members=len(members),
            max_depth=1,
        ),
        mode=mode,
    )


def _failure_probes(
    contract: AtomisticIREEContract, inputs: tuple[Array, ...], /
) -> tuple[tuple[str, tuple[Array, ...]], ...]:
    """Declared invalid runtime geometries each module must report by status."""

    positions = inputs[0]
    probes = [("nonfinite-coordinates", (positions.at[0, 0].set(jnp.nan), *inputs[1:]))]
    if contract.periodic:
        probes.append(
            ("singular-cell", (positions, jnp.zeros_like(inputs[1]), inputs[2]))
        )
    return tuple(probes)


def _require_parity(
    expected: AtomisticIREEEvaluation,
    observed: AtomisticIREEEvaluation,
    label: str,
    rtol: float,
    atol: float,
    /,
) -> None:
    if observed.status is not expected.status:
        raise RuntimeError(
            f"Frozen executable status differs at {label}: expected "
            f"{expected.status.name}, got {observed.status.name}."
        )
    if not expected.successful:
        return
    names = ("energy", "forces", "atom_energy", "stress")
    failed = [
        f"{name} max_abs={float(np.max(np.abs(value - result))):.3e}"
        for name, value, result in zip(
            names, _carriers(expected), _carriers(observed), strict=False
        )
        if not np.allclose(value, result, rtol=rtol, atol=atol)
    ]
    if failed:
        raise RuntimeError(
            f"Frozen executable failed E/F/S parity at {label}: {'; '.join(failed)}."
        )


def _carriers(value: AtomisticIREEEvaluation, /) -> tuple[np.ndarray, ...]:
    return (
        value.energy,
        value.forces,
        value.atom_energy,
        *(() if value.stress is None else (value.stress,)),
    )


def save_atomistic_iree(
    plan: NativeAtomisticProviderPlan,
    structure: AtomicStructure,
    units: AtomisticUnitSystem,
    path: str | Path,
    /,
    *,
    route: AtomisticExportRoute = "native-jax",
    policy: IREEExportPolicy | None = None,
    rtol: float = 1.0e-6,
    atol: float = 1.0e-9,
    timeout: float = 3600.0,
) -> AtomisticIREEExportBundle:
    """Compile one frozen E/F/S ABI in an isolated worker and verify parity.

    ``structure`` is the reference geometry whose lifecycle epoch is frozen.
    The parent prepares the provider and contract with ordinary raising
    guards, requires a successful native evaluation, proves every runtime
    failure is a traced status, refuses host-runtime custom calls (such as
    LAPACK) that IREE cannot compile, and evaluates the exact frozen program in
    raise mode. A pinned isolated worker (``timeout`` seconds, bounded logs and
    artifacts, only ``EQX_ON_ERROR``, ``JAX_ENABLE_X64`` and ``JAX_PLATFORMS``
    in its environment) rebuilds model, plan, structure and contract from one
    pickle-free archive and compiles the module. ``path`` is published
    atomically only after the loaded module reproduces the reference values
    and the failure-probe statuses within ``rtol``/``atol``.
    """

    if not isinstance(plan, NativeAtomisticProviderPlan):
        raise TypeError("plan must be a NativeAtomisticProviderPlan.")
    policy_ = IREEExportPolicy() if policy is None else policy
    if not isinstance(policy_, IREEExportPolicy):
        raise TypeError("policy must be IREEExportPolicy or None.")
    provider, request = _native_provider(plan, structure, units)
    contract = prepare_atomistic_iree_contract(provider, request, route=route)
    native = provider.evaluate_request(request)
    if not bool(native.evaluation.successful):
        raise ValueError(
            "The export reference geometry has no successful native evaluation."
        )
    inputs = contract.pack_inputs(request)
    forward = _exported_evaluation(provider, request, contract)
    _require_status_complete(forward, inputs)
    _require_runtime_free_lowering(forward, inputs)
    program = jax.jit(forward)
    reference = _decoded(contract, program(*inputs))
    evaluation = native.evaluation
    _require_parity(
        AtomisticIREEEvaluation(
            np.asarray(evaluation.energy),
            np.asarray(evaluation.forces),
            np.asarray(native.atom_energy),
            None if evaluation.stress is None else np.asarray(evaluation.stress),
            AtomisticStatus.SUCCESS,
            reference.active_routes,
        ),
        reference,
        "the frozen program's reference geometry",
        rtol,
        atol,
    )
    probes = tuple(
        (label, values, _decoded(contract, program(*values)))
        for label, values in _failure_probes(contract, inputs)
    )
    if any(expected.successful for _, _, expected in probes):
        raise RuntimeError("A declared failure probe evaluated successfully.")
    destination = Path(path)
    with tempfile.TemporaryDirectory(prefix="phydrax-atomistic-export-") as directory:
        root = Path(directory)
        request_path = root / _REQUEST_NAME
        _write_export_request(request_path, plan, structure, units, contract, policy_)
        (root / "worker").mkdir()
        files = _run_export_worker(request_path.read_bytes(), root / "worker", timeout)
        staged = _staged_module(files, contract)
        candidate = root / "candidate"
        _publish_bundle(candidate, staged, "exclusive")
        loaded = load_atomistic_iree(
            candidate,
            trusted_module_sha256=staged.manifest.module_sha256,
            trusted_contract_id=contract.contract_id,
        )
        _require_parity(
            reference,
            loaded.evaluate_inputs(inputs),
            "the reference geometry",
            rtol,
            atol,
        )
        for label, values, expected in probes:
            _require_parity(expected, loaded.evaluate_inputs(values), label, rtol, atol)
        _publish_bundle(destination, staged, "atomic_replace")
    return AtomisticIREEExportBundle(destination, staged.manifest.module_sha256, contract)


@dataclass(frozen=True, slots=True)
class AtomisticIREEEvaluation:
    """Host E/F/S result of a frozen executable in the contract's units.

    Values are NaN unless ``status`` is ``AtomisticStatus.SUCCESS``.
    ``active_routes`` counts the stored candidate routes of the frozen graph.
    """

    energy: np.ndarray
    forces: np.ndarray
    atom_energy: np.ndarray
    stress: np.ndarray | None
    status: AtomisticStatus
    active_routes: int

    @property
    def successful(self) -> bool:
        return self.status is AtomisticStatus.SUCCESS


def _decoded(
    contract: AtomisticIREEContract, raw: Sequence[Any], /
) -> AtomisticIREEEvaluation:
    outputs = dict(zip(contract.output_names, raw, strict=True))
    status = AtomisticStatus(int(np.asarray(outputs["status"])))
    failed = status is not AtomisticStatus.SUCCESS

    def restored(name: str, /) -> np.ndarray:
        value = np.asarray(outputs[name])
        return np.full_like(value, np.nan) if failed else value

    return AtomisticIREEEvaluation(
        restored("energy"),
        restored("forces"),
        restored("atom_energy"),
        restored("stress") if contract.stress else None,
        status,
        int(np.asarray(outputs["active_routes"])),
    )


class LoadedAtomisticIREE:
    """Digest-pinned frozen atomistic executable; host-only and nondifferentiable."""

    capabilities = ExecutionCapabilities("compiled-inference", host_only=True)

    def __init__(
        self, executable: IREEExecutable, contract: AtomisticIREEContract, /
    ) -> None:
        self.executable = executable
        self.contract = contract

    def __call__(self, request: NativeAtomisticRequest, /) -> AtomisticIREEEvaluation:
        _require_execution(self.capabilities, request)
        return self.evaluate_inputs(self.contract.pack_inputs(request))

    def evaluate_inputs(self, inputs: Sequence[Any], /) -> AtomisticIREEEvaluation:
        """Evaluate raw ABI inputs of the frozen lifecycle epoch.

        The caller asserts the inputs belong to the frozen epoch; prefer
        calling with a lifecycle request, which checks it.
        """

        _require_execution(self.capabilities, *inputs)
        self.contract.check_inputs(inputs)
        raw = self.executable(*(np.asarray(value) for value in inputs))
        if not isinstance(raw, tuple):
            raise RuntimeError("Atomistic IREE executable returned a single output.")
        return _decoded(self.contract, raw)


def load_atomistic_iree(
    path: str | Path,
    /,
    *,
    trusted_module_sha256: str,
    trusted_contract_id: str,
) -> LoadedAtomisticIREE:
    """Load a frozen E/F/S executable pinned by module digest and contract id."""

    executable = load_iree(path, trusted_module_sha256=trusted_module_sha256)
    payload = executable.manifest.domain_contract
    if payload is None:
        raise ValueError("IREE artifact carries no atomistic contract.")
    value = json.loads(payload)
    if not isinstance(value, Mapping):
        raise TypeError("Atomistic IREE contract must be a JSON object.")
    contract = AtomisticIREEContract.from_dict(value)
    if contract.contract_id != trusted_contract_id:
        raise PermissionError(
            "Atomistic IREE contract is not authorized by the caller-supplied pin."
        )
    manifest = executable.manifest
    if (
        manifest.input_names != contract.input_names
        or manifest.input_shapes != contract.input_shapes
        or manifest.input_dtypes != contract.input_dtypes
        or manifest.output_names != contract.output_names
        or manifest.output_shapes != contract.output_shapes
        or manifest.output_dtypes != contract.output_dtypes
    ):
        raise ValueError("Atomistic IREE contract differs from the executable ABI.")
    return LoadedAtomisticIREE(executable, contract)


__all__ = [
    "AtomisticExportFailurePolicy",
    "AtomisticExportRefusal",
    "AtomisticExportRoute",
    "AtomisticIREEContract",
    "AtomisticIREEEvaluation",
    "AtomisticIREEExportBundle",
    "LoadedAtomisticIREE",
    "load_atomistic_iree",
    "prepare_atomistic_iree_contract",
    "save_atomistic_iree",
]
