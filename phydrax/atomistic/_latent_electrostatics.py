#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native latent Ewald summation (LES) of learned latent charge channels.

Scientific definition
---------------------
The latent Ewald summation of Cheng (npj Comput. Mater. 11, 80 (2025)), as
executed by the pinned LES source (``les/module/ewald.py``,
``ewald_vectorized.py`` and ``bec.py`` at revision
``35b971a650cf30adb84fef7138761e81aa47b3a2``), is an energy of ``C`` learned
latent charge channels ``q[i, c]`` smeared by Gaussians whose *pair* kernel has
standard deviation ``sigma``. In angstrom, electronvolt and elementary-charge
equivalents, with the source constant ``N = 90.4756`` (``1 / (2 epsilon_0)``)
and ``k_e = N / (2 pi)``:

* finite branch: ``E = k_e sum_{i<j} sum_c q_ic q_jc erf(r_ij / (sqrt(2) sigma)) / r_ij``
  over *every* pair of the case (no cutoff, no images);
* periodic branch: ``E = (N / V) sum_{k in K} h(k) exp(-sigma^2 k^2 / 2) / k^2
  sum_c |S_c(k)|^2`` with ``S_c(k) = sum_i q_ic exp(i k.r_i)``, Hermitian
  half-space multiplicity ``h = 2`` and ``K`` the nonzero reciprocal modes with
  ``k^2 <= (2 pi / dl)^2``. There is no real-space part, no ``k = 0`` term and
  no neutralizing background: a nonneutral latent configuration is evaluated
  as is. The support is either the complete sphere (legacy ``Ewald``: per-axis
  bounds ``max(1, floor(|a_i| / dl))`` contain every sphere mode, since
  ``|m_i| = |k.a_i| / 2 pi <= |a_i| / dl``) or the sphere intersected with the
  fixed index box ``|m_i| <= N_max`` (``Ewald_vectorized``), a declared source
  truncation reported per case, not a failure;
* self policy: the on-site Gaussian self energy
  ``N sum_i q_i^2 / (sigma (2 pi)^(3/2))`` (``= k_e q^2 lim_{r->0} erf(r/(sqrt2 sigma))/(2 r)``)
  is *subtracted* from the periodic sum when ``remove_self_interaction`` and
  *added* to the finite sum when it is not, so both branches share one
  convention.

Branch semantics follow the source composition. ``branch="per-case"``
(source ``is_periodic=None``) evaluates the periodic sum over the full bound
cell for every case with any periodic axis (MACELES zeroes the cell only when
no axis is periodic) and the finite sum otherwise; ``"periodic"``/``"finite"``
(``is_periodic=True/False``) force one branch for every case. Deviations from
the source executable are explicit refusals, never substitutes: a periodic
case with a non-positively oriented or singular cell (where the source either
silently switches to the finite sum through ``det < 1e-6`` or produces NaN)
is unsuccessful, as is a forced-periodic case without a periodic axis. The
finite kernel uses its exact analytic value ``2a/sqrt(pi) (1 - (ar)^2/3 + ...)``
for ``a r < 10^-2`` (``a = 1 / (sqrt2 sigma)``); the source legacy branch
yields NaN and the vectorized branch a clamped value for coincident atoms.

Latent channels are learned model outputs, not physical partial charges: they
are published only under :attr:`AtomisticObservable.LATENT_CHARGES` and their
explicit charge unit exists because the energy constant fixes it. The source
``BEC`` response is preserved as a requested projection policy
(:class:`LatentBornProjection`): ``P = sqrt(eps) sum_i q~_i r_i`` (finite) or
``P_a = sqrt(eps) sum_i q~_i exp(2 pi i s_ia) / (2 pi i)`` (periodic, ``s`` row
fractional coordinates), ``q~`` mean-removed per case when requested,
``Z_i[a, b] = d P_a / d r_ib`` including the latent-charge position response,
dephased by ``conj(exp(2 pi i s_ia))`` and projected with ``H^T``.

Numerical execution
-------------------
:class:`PreparedLatentEwald` is the one prepared numerical action used by both
the scalar program term :class:`LatentEwaldPotential` (latent channels from a
typed field dependency on a preceding producer) and model graph evaluation
(MACELES). Finite sums stream ``row_block x A`` blocks in a checkpointed scan;
periodic sums run the term-owned :class:`PreparedReciprocalModeSet` in its
bounded mode chunks. Neither materializes an atom-by-atom or a mode-by-atom
array; workspaces are admitted at preparation. Cases run through ``lax.map``
and a per-case ``lax.cond`` executes exactly one branch. Energies are global
per case: a scoped atomic decomposition is never invented.

Source and legal qualification
------------------------------
The LES source is licensed CC BY-NC 4.0. It is not vendored, imported or
executed here; this module is an independent implementation of the published
equations, with the source's numerical constants and argument semantics
recorded as provenance facts in :data:`LATENT_EWALD_SOURCE`. Rights of any
LES-trained checkpoint (for example MACELES-OFF) are a separate gate owned by
model admission and are not implied by this module.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from phydrax.ein import contract

from .._differentiation import (
    DerivativeContract,
    DerivativeRegularity,
    DerivativeRoute,
    DerivativeSurface,
    GradientLevel,
    SurfaceDerivative,
)
from .._fingerprint import canonical_fingerprint
from .._identity import IdentityWords
from .._model import register_artifact_value
from .._model._fixed_binding import require_identical
from .._model._ports import ValuePort
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import lattice_right_inverse_with_status
from ..typing import parse
from ..units import (
    ANGSTROM,
    CHARGE,
    conversion_factor,
    ELECTRONVOLT,
    ELEMENTARY_CHARGE,
    ENERGY,
    UnitDefinition,
)
from ._conditioning import (
    AtomisticConditionLevel,
    AtomisticFieldReference,
    AtomisticObservable,
    AtomisticObservableValue,
)
from ._potential import AtomisticPotentialCapabilities, AtomisticPotentialRequirements
from ._potential_program import (
    AbstractAtomisticEnergyTerm,
    AbstractPreparedAtomisticEnergyTerm,
    AtomisticPotentialContext,
    AtomisticTermEvaluation,
)
from ._reciprocal_mesh import (
    PreparedReciprocalModeSet,
    ReciprocalBudget,
    ReciprocalInfluence,
    ReciprocalModeSetPlan,
)
from ._system import PreparedAtomisticSystem
from ._types import AtomisticScaleContract


LatentEwaldBranch: TypeAlias = Literal["per-case", "periodic", "finite"]
LatentEwaldModeSupport: TypeAlias = Literal["complete-sphere", "index-box-sphere"]

LATENT_EWALD_SOURCE = {
    "repository": "https://github.com/ChengUCB/les",
    "revision": "35b971a650cf30adb84fef7138761e81aa47b3a2",
    "license": "CC-BY-NC-4.0",
    "files": (
        "src/les/les.py",
        "src/les/module/ewald.py",
        "src/les/module/ewald_vectorized.py",
        "src/les/module/bec.py",
    ),
    "composition": "mace-torch 0.3.15/0.3.16 mace/modules/extensions.py MACELES",
    "execution": "independent native implementation; source never imported",
}
"""Provenance of the implemented definition (facts, not a rights grant)."""

# ``Ewald.__init__`` default ``norm_factor``: 1 / (2 epsilon_0) with
# epsilon_0 = 5.52635e-3 e^2 eV^-1 A^-1. ``Les`` never forwards another value.
SOURCE_NORM_FACTOR = 90.4756
SOURCE_SIGMA = 1.0
SOURCE_DL = 2.0
SOURCE_N_MAX = 10

_TWO_PI = 2.0 * math.pi
_SQRT_PI = math.sqrt(math.pi)
_REAL_ITEMSIZE = 8
# ``(a r)^2`` below which the finite kernel uses its even Taylor series; the
# first omitted term is ``(a r)^8 / 216`` relative, below double roundoff.
_SERIES_THRESHOLD = 1.0e-4
_LENGTH_UNIT = ANGSTROM
_ENERGY_UNIT = ELECTRONVOLT
_LATENT_UNIT = ELEMENTARY_CHARGE

# Source ``les_arguments`` read by ``Les._parse_arguments`` and MACELES.
_SOURCE_ARGUMENTS = frozenset(
    {
        "n_layers",
        "n_hidden",
        "add_linear_nn",
        "output_scaling_factor",
        "sigma",
        "dl",
        "remove_self_interaction",
        "is_periodic",
        "N_max",
        "remove_mean",
        "epsilon_factor",
        "use_atomwise",
        "use_fixed_atomic_charges",
        "fixed_atomic_charges_scaling_factor",
        "use_atomic_alpha",
        "use_epsilon_r_scaling",
        "compute_bec",
        "bec_output_index",
    }
)
# Flags whose modules never act in the MACELES forward: it passes only
# ``latent_charges``, ``positions``, ``cell`` and ``batch`` to ``Les``, so no
# descriptor (atomwise head), atomic numbers (fixed charges, atomic alpha) or
# latent polarizabilities (epsilon_r scaling) reach them.
_INERT_IN_MACELES = (
    "use_atomwise",
    "use_fixed_atomic_charges",
    "use_atomic_alpha",
    "use_epsilon_r_scaling",
)


def _finite_positive(value: object, name: str, /) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _positive_int(value: object, name: str, /) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive.")
    return int(value)


def _boolean(value: object, name: str, /) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean.")
    return bool(value)


def _output_row(value: object, /) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("output_index must be an integer or None.")
    if int(value) not in (0, 1, 2):
        raise ValueError("output_index must select a Cartesian row 0, 1 or 2.")
    return int(value)


def _factor(source: UnitDefinition, target: UnitDefinition, name: str, /) -> float:
    try:
        return float(conversion_factor(source, target))
    except ValueError as error:
        raise ValueError(
            f"The latent Ewald {name} unit {source.symbol!r} does not convert to "
            f"{target.symbol!r}; the source constants are physical."
        ) from error


def _stored_scalar(value: object, name: str, /) -> float:
    """Host value of one restored dynamic real scalar (constructor input)."""
    if not isinstance(value, jax.Array):
        raise TypeError(f"The latent Ewald {name} must be a stored array.")
    array = np.asarray(jax.device_get(value))
    if array.shape != () or not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"The latent Ewald {name} must be a real scalar.")
    return float(array)


def latent_charge_port(channels: int = 1, /) -> ValuePort:
    """Atom-level latent charge channels: event ``()`` for one channel, else ``(C,)``."""
    count = _positive_int(channels, "channels")
    return ValuePort(
        "atomistic.latent-ewald.latent-charge",
        event_shape=() if count == 1 else (count,),
        component_ids=("latent_charge",)
        if count == 1
        else tuple(f"latent_charge_{index}" for index in range(count)),
        representation="les-gaussian-latent-charge-channel-not-a-partial-charge",
        dimensions=(CHARGE,) * count,
        normalization_id="identity",
    )


LATENT_CHARGE_PORT = latent_charge_port(1)
"""The single-channel latent charge port used by the MACELES composition."""

LATENT_EWALD_ENERGY_PORT = ValuePort(
    "atomistic.latent-ewald.global-energy",
    event_shape=(),
    component_ids=("energy",),
    representation="les-global-latent-ewald-energy-not-atomized",
    dimensions=(ENERGY,),
    normalization_id="identity",
)
"""Case-level ledger component of the global latent Ewald energy."""


class LatentBornProjection(StrictModule, NonTrainableState):
    """Requested source ``BEC`` projection policy.

    ``remove_mean`` subtracts each case's mean latent charge before forming
    the polarization; ``epsilon_factor`` multiplies the polarization by
    ``sqrt(epsilon_factor)`` (source ``normalization_factor``);
    ``output_index`` selects one polarization row ``a`` (source
    ``bec_output_index``), returning ``Z_i[a, :]``.
    """

    remove_mean: bool = eqx.field(static=True)
    epsilon_factor: float = eqx.field(static=True)
    output_index: int | None = eqx.field(static=True)
    projection_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        remove_mean: bool = True,
        epsilon_factor: float = 1.0,
        output_index: int | None = None,
    ) -> None:
        remove = _boolean(remove_mean, "remove_mean")
        epsilon = _finite_positive(epsilon_factor, "epsilon_factor")
        row = _output_row(output_index)
        self.remove_mean = remove
        self.epsilon_factor = epsilon
        self.output_index = row
        self.projection_id = canonical_fingerprint(
            {
                "kind": "latent-born-projection",
                "remove_mean": remove,
                "epsilon_factor": float(epsilon).hex(),
                "output_index": row,
                "polarization": "finite-dipole-or-row-fractional-berry-phase",
                "dephasing": "own-atom-row-phase",
                "projection": "cell-transpose",
            }
        )

    def port(self) -> ValuePort:
        """Port of the projected latent Born effective charges ``dP_a/dr_b``."""
        axes = ("x", "y", "z")
        if self.output_index is None:
            shape: tuple[int, ...] = (3, 3)
            components = tuple(f"{row}{column}" for row in axes for column in axes)
            selection = "all-rows"
        else:
            shape = (3,)
            components = tuple(f"{axes[self.output_index]}{column}" for column in axes)
            selection = f"row-{axes[self.output_index]}"
        return ValuePort(
            "atomistic.latent-ewald.born-charges",
            event_shape=shape,
            component_ids=components,
            representation=f"les-latent-born-charge-dP-dr-{selection}",
            dimensions=(CHARGE,) * len(components),
            frame_id="input-cartesian",
            normalization_id=self.projection_id,
        )


class LatentEwaldDefinition(StrictModule, NonTrainableState):
    """Immutable scientific identity of one latent Ewald energy.

    ``sigma`` and ``dl`` are in angstrom, ``norm_factor`` in
    ``eV angstrom e^-2`` (the source's fixed physical constants); prepared
    actions convert a consumer's declared scale at preparation. ``channels``
    is the latent channel count ``C`` (MACELES uses one). The energy is the
    sum over channels; channels never interact with each other.

    PyTree layout: ``channels`` (shape), ``branch``, ``mode_support``,
    ``index_bound`` (mode-support extents), ``remove_self_interaction``,
    ``compute_born`` and the Born projection (whose identity is the static
    normalization of its published port) are static. ``sigma``, ``dl`` and
    ``norm_factor`` are dynamic float64 scalars (so ``coulomb_constant``,
    ``reciprocal_cutoff`` and ``self_coefficient`` are float64 arrays), the
    recorded inert source flags are the dynamic ``bool[4]`` mask
    ``inert_source_mask`` over the MACELES-inert flag vocabulary (decoded by
    the host property ``inert_source_flags``) and ``definition_identity`` is
    the exact former ``definition_id`` digest (decoded by the host property).
    """

    sigma: Array
    dl: Array
    norm_factor: Array
    remove_self_interaction: bool = eqx.field(static=True)
    branch: LatentEwaldBranch = eqx.field(static=True)
    mode_support: LatentEwaldModeSupport = eqx.field(static=True)
    index_bound: int | None = eqx.field(static=True)
    channels: int = eqx.field(static=True)
    born: LatentBornProjection
    compute_born: bool = eqx.field(static=True)
    inert_source_mask: Array
    definition_identity: IdentityWords

    def __init__(
        self,
        *,
        sigma: float = SOURCE_SIGMA,
        dl: float = SOURCE_DL,
        remove_self_interaction: bool = True,
        branch: LatentEwaldBranch = "per-case",
        mode_support: LatentEwaldModeSupport = "complete-sphere",
        index_bound: int | None = None,
        norm_factor: float = SOURCE_NORM_FACTOR,
        channels: int = 1,
        born: LatentBornProjection | None = None,
        compute_born: bool = False,
        inert_source_flags: tuple[str, ...] = (),
    ) -> None:
        width = _finite_positive(sigma, "sigma")
        spacing = _finite_positive(dl, "dl")
        norm = _finite_positive(norm_factor, "norm_factor")
        remove = _boolean(remove_self_interaction, "remove_self_interaction")
        branch = parse(branch, LatentEwaldBranch, "branch")
        mode_support = parse(mode_support, LatentEwaldModeSupport, "mode_support")
        if mode_support == "index-box-sphere":
            if index_bound is None:
                raise TypeError("index_bound must be an integer.")
            bound: int | None = _positive_int(index_bound, "index_bound")
        elif index_bound is not None:
            raise ValueError("Only the index-box sphere support declares index_bound.")
        else:
            bound = None
        count = _positive_int(channels, "channels")
        projection = LatentBornProjection() if born is None else born
        if not isinstance(projection, LatentBornProjection):
            raise TypeError("born must be a LatentBornProjection.")
        requested = _boolean(compute_born, "compute_born")
        if requested and count != 1:
            raise ValueError(
                "The source Born projection is defined for one latent channel."
            )
        flags = tuple(sorted(str(value) for value in inert_source_flags))
        if any(value not in _INERT_IN_MACELES for value in flags):
            raise ValueError("Only MACELES-inert source flags may be recorded as inert.")
        self.sigma = jnp.asarray(width, dtype=jnp.float64)
        self.dl = jnp.asarray(spacing, dtype=jnp.float64)
        self.norm_factor = jnp.asarray(norm, dtype=jnp.float64)
        self.remove_self_interaction = remove
        self.branch = branch
        self.mode_support = mode_support
        self.index_bound = bound
        self.channels = count
        self.born = projection
        self.compute_born = requested
        self.inert_source_mask = jnp.asarray(
            np.asarray([name in flags for name in _INERT_IN_MACELES], dtype=np.bool_)
        )
        self.definition_identity = IdentityWords.of(
            {
                "kind": "latent-ewald-definition",
                "pair_kernel": "erf(r/(sqrt2 sigma))/r",
                "reciprocal_kernel": "exp(-sigma^2 k^2/2)/k^2",
                "zero_mode": "excluded-no-background",
                "sigma_angstrom": float(width).hex(),
                "dl_angstrom": float(spacing).hex(),
                "norm_factor_ev_angstrom_e-2": float(norm).hex(),
                "remove_self_interaction": remove,
                "branch": branch,
                "mode_support": mode_support,
                "index_bound": bound,
                "channels": count,
                "born": projection.projection_id,
                "compute_born": requested,
                "inert_source_flags": flags,
                "source_revision": LATENT_EWALD_SOURCE["revision"],
            }
        )

    @property
    def definition_id(self) -> str:
        """Host decoding of the canonical latent Ewald definition digest."""
        return self.definition_identity.fingerprint()

    @property
    def inert_source_flags(self) -> tuple[str, ...]:
        """Host decoding of the recorded MACELES-inert source flags (sorted)."""
        mask = np.asarray(self.inert_source_mask, dtype=np.bool_)
        return tuple(
            sorted(
                name for name, inert in zip(_INERT_IN_MACELES, mask, strict=True) if inert
            )
        )

    def validate(self) -> None:
        """Replay the constructor on the stored declaration; refuse a stale one.

        A restored definition bypasses the constructor. ``sigma``, ``dl`` and
        ``norm_factor`` must be stored real scalars and ``inert_source_mask``
        one boolean per MACELES-inert flag; the constructor replayed from those
        and the static fields (finite positive scalars, branch, mode support
        and index bound, channel count, Born projection) must reproduce every
        field exactly, including the Born ``projection_id`` and
        ``definition_identity`` (which records the source revision).
        """
        born = self.born
        if not isinstance(born, LatentBornProjection):
            raise TypeError("born must be a LatentBornProjection.")
        mask = self.inert_source_mask
        if (
            not isinstance(mask, jax.Array)
            or mask.dtype != jnp.bool_
            or mask.shape != (len(_INERT_IN_MACELES),)
        ):
            raise ValueError("inert_source_mask must hold one flag per inert flag.")
        replay = LatentEwaldDefinition(
            sigma=_stored_scalar(self.sigma, "sigma"),
            dl=_stored_scalar(self.dl, "dl"),
            remove_self_interaction=self.remove_self_interaction,
            branch=self.branch,
            mode_support=self.mode_support,
            index_bound=self.index_bound,
            norm_factor=_stored_scalar(self.norm_factor, "norm_factor"),
            channels=self.channels,
            born=LatentBornProjection(
                remove_mean=born.remove_mean,
                epsilon_factor=born.epsilon_factor,
                output_index=born.output_index,
            ),
            compute_born=self.compute_born,
            inert_source_flags=self.inert_source_flags,
        )
        require_identical(self, replay, "The latent Ewald definition", ValueError)

    @classmethod
    def from_source_arguments(
        cls, arguments: Mapping[str, object], /
    ) -> "LatentEwaldDefinition":
        """Admit pinned MACELES ``les_arguments`` with the source defaults.

        ``is_periodic`` ``None`` selects the legacy per-case ``Ewald`` with
        complete-sphere support; ``True``/``False`` select the vectorized
        module's forced branch and its ``N_max`` index box. Unknown keys are
        refused. Atomwise, fixed-charge, atomic-alpha and epsilon_r flags are
        recorded as inert: the MACELES forward never feeds those modules.
        """
        if not isinstance(arguments, Mapping):
            raise TypeError("les_arguments must be a mapping.")
        unknown = sorted(set(map(str, arguments)) - _SOURCE_ARGUMENTS)
        if unknown:
            raise ValueError(f"Unrecognized LES arguments {unknown!r}.")
        periodic = arguments.get("is_periodic", None)
        if periodic is None:
            branch: LatentEwaldBranch = "per-case"
            support: LatentEwaldModeSupport = "complete-sphere"
            bound = None
        else:
            forced = _boolean(periodic, "is_periodic")
            branch = "periodic" if forced else "finite"
            support = "index-box-sphere"
            bound = _positive_int(arguments.get("N_max", SOURCE_N_MAX), "N_max")
        flags = tuple(
            name
            for name in _INERT_IN_MACELES
            if _boolean(arguments.get(name, False), name)
        )
        born = LatentBornProjection(
            remove_mean=_boolean(arguments.get("remove_mean", True), "remove_mean"),
            epsilon_factor=_finite_positive(
                arguments.get("epsilon_factor", 1.0), "epsilon_factor"
            ),
            output_index=_output_row(arguments.get("bec_output_index", None)),
        )
        return cls(
            sigma=_finite_positive(arguments.get("sigma", SOURCE_SIGMA), "sigma"),
            dl=_finite_positive(arguments.get("dl", SOURCE_DL), "dl"),
            remove_self_interaction=_boolean(
                arguments.get("remove_self_interaction", True), "remove_self_interaction"
            ),
            branch=branch,
            mode_support=support,
            index_bound=bound,
            born=born,
            compute_born=_boolean(arguments.get("compute_bec", False), "compute_born"),
            inert_source_flags=flags,
        )

    @property
    def coulomb_constant(self) -> Array:
        """``k_e = N / (2 pi)`` in ``eV angstrom e^-2`` (float64 scalar)."""
        return self.norm_factor / _TWO_PI

    @property
    def reciprocal_cutoff(self) -> Array:
        """``k_max = 2 pi / dl`` (inverse angstrom, float64), modes with ``k^2 <= k_max^2``."""
        return _TWO_PI / self.dl

    @property
    def self_coefficient(self) -> Array:
        """``N / (sigma (2 pi)^(3/2))``: on-site self energy per squared latent charge."""
        return self.norm_factor / (self.sigma * _TWO_PI**1.5)

    def latent_port(self) -> ValuePort:
        return latent_charge_port(self.channels)

    def mode_cutoff_locus(self) -> str:
        """Parametric non-smooth locus, with ``|k|`` in inverse angstrom.

        ``dl`` is dynamic numerical state. The regularity contract names the
        family ``|k| * dl = 2 pi``; the definition identity and runtime mode
        mask bind its current value without specializing admission metadata.
        """
        return "latent-ewald-mode-sphere:k-max-times-dl=2*pi"

    def derivative_contract(self, *, periodic: bool) -> DerivativeContract:
        """Value-surface regularity of the latent energy in positions and cell.

        Finite: smooth (the kernel is analytic, including coincident atoms).
        Periodic: smooth in positions and latent charges; in the cell the mode
        set changes discretely where a mode crosses the sphere, so the energy
        jumps there and cell derivatives hold almost everywhere.
        """
        if not periodic:
            return DerivativeContract(
                (SurfaceDerivative(DerivativeSurface.INPUT, GradientLevel.SMOOTH),),
                route=DerivativeRoute.DIRECT,
                regularity=DerivativeRegularity.smooth(),
            )
        locus = self.mode_cutoff_locus()
        return DerivativeContract(
            (
                SurfaceDerivative(
                    DerivativeSurface.INPUT,
                    GradientLevel.ALMOST_EVERYWHERE,
                    conditions=(locus,),
                ),
            ),
            route=DerivativeRoute.DIRECT,
            regularity=DerivativeRegularity.piecewise_smooth(
                continuity=-1, conditions=(locus,)
            ),
        )

    def covering_extents(self, maximum_lattice_length: float, /) -> tuple[int, int, int]:
        """Mode extents covering the support of every cell with ``|a_i| <= L`` (angstrom).

        The complete sphere needs ``max(1, floor(L / dl))`` per axis; the index
        box is the definition's own ``N_max``.
        """
        # The constructor declares ``index_bound`` exactly for the index-box support.
        bound = self.index_bound
        if bound is not None:
            return (bound, bound, bound)
        length = _finite_positive(maximum_lattice_length, "maximum_lattice_length")
        extent = max(1, int(math.floor(length / float(self.dl))))
        return (extent, extent, extent)

    def prepare_graph(
        self,
        scale: AtomisticScaleContract,
        atom_capacity: int,
        /,
        *,
        resources: "LatentEwaldResources",
    ) -> "PreparedLatentEwald":
        """Prepare the case-batched action for model graph evaluation.

        Prepared once (outside every numerical iteration) for one atom
        capacity per case; the case count is free. ``resources.mode_extents``
        binds a lattice-family reciprocal mode set (no reference-cell claim);
        per-case runtime cells are bound at evaluation and covered or refused
        there. Without extents the binding is finite-only and refuses any
        periodic case.
        """
        capacity = _positive_int(atom_capacity, "atom_capacity")
        modes = None
        if self.branch != "finite" and resources.mode_extents is not None:
            modes = _mode_plan(self, resources).prepare_lattice_family(capacity)
        return PreparedLatentEwald(self, scale, capacity, resources, modes)


def _mode_plan(
    definition: LatentEwaldDefinition, resources: "LatentEwaldResources", /
) -> ReciprocalModeSetPlan:
    extents = resources.mode_extents
    if extents is None:
        raise ValueError("Periodic latent Ewald evaluation needs explicit mode extents.")
    if definition.mode_support == "index-box-sphere":
        bound = definition.index_bound
        if extents != (bound, bound, bound):
            raise ValueError(
                "The index-box support is the source N_max box; its mode extents must "
                "equal (N_max, N_max, N_max)."
            )
    return ReciprocalModeSetPlan(
        extents, mode_chunk=resources.mode_chunk, budget=resources.budget
    )


class LatentEwaldResources(StrictModule, NonTrainableState):
    """Explicit execution resources, separate from the scientific definition.

    ``row_block`` rows of the finite sum are resident at once
    (``row_block x A`` per channel-contracted block); ``mode_extents`` and
    ``mode_chunk`` bind the term-owned reciprocal mode set under ``budget``.
    ``maximum_finite_workspace_bytes`` admits the finite block workspace at
    preparation.
    """

    row_block: int = eqx.field(static=True)
    mode_extents: tuple[int, int, int] | None = eqx.field(static=True)
    mode_chunk: int = eqx.field(static=True)
    budget: ReciprocalBudget
    maximum_finite_workspace_bytes: int = eqx.field(static=True)
    resources_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        row_block: int = 64,
        mode_extents: tuple[int, int, int] | None = None,
        mode_chunk: int = 128,
        budget: ReciprocalBudget | None = None,
        maximum_finite_workspace_bytes: int = 1 << 28,
    ) -> None:
        block = _positive_int(row_block, "row_block")
        chunk = _positive_int(mode_chunk, "mode_chunk")
        extents: tuple[int, int, int] | None = None
        if mode_extents is not None:
            axes = tuple(_positive_int(value, "mode_extents") for value in mode_extents)
            if len(axes) != 3:
                raise ValueError("mode_extents must name three lattice axes.")
            extents = (axes[0], axes[1], axes[2])
        budget_ = ReciprocalBudget() if budget is None else budget
        if not isinstance(budget_, ReciprocalBudget):
            raise TypeError("budget must be a ReciprocalBudget.")
        workspace = _positive_int(
            maximum_finite_workspace_bytes, "maximum_finite_workspace_bytes"
        )
        self.row_block = block
        self.mode_extents = extents
        self.mode_chunk = chunk
        self.budget = budget_
        self.maximum_finite_workspace_bytes = workspace
        self.resources_id = canonical_fingerprint(
            {
                "kind": "latent-ewald-resources",
                "row_block": block,
                "mode_extents": None if extents is None else list(extents),
                "mode_chunk": chunk,
                "budget": budget_.budget_id,
                "maximum_finite_workspace_bytes": workspace,
            }
        )

    def validate(self) -> None:
        """Replay the constructor on the stored resources; refuse a stale one.

        The reciprocal budget replays its own constructor; the positive row
        block, mode extents, mode chunk, finite workspace limit and
        ``resources_id`` must reproduce every field exactly.
        """
        budget = self.budget
        if not isinstance(budget, ReciprocalBudget):
            raise TypeError("budget must be a ReciprocalBudget.")
        budget.validate()
        replay = LatentEwaldResources(
            row_block=self.row_block,
            mode_extents=self.mode_extents,
            mode_chunk=self.mode_chunk,
            budget=budget,
            maximum_finite_workspace_bytes=self.maximum_finite_workspace_bytes,
        )
        require_identical(self, replay, "The latent Ewald resources", ValueError)

    def finite_workspace_bytes(self, atom_capacity: int, channels: int, /) -> int:
        """Exact resident bytes of one finite row block.

        Displacements ``(R, A, 3)``, squared distances, kernel, routing mask
        and the channel-contracted charge products ``(R, A)`` plus the row
        charges ``(R, C)``.
        """
        rows = min(self.row_block, atom_capacity)
        return rows * (atom_capacity * 7 + channels) * _REAL_ITEMSIZE


class LatentEwaldEvaluation(StrictModule):
    """Global latent energy ledger of each case and its separate statuses.

    Energies are in the prepared consumer energy unit. ``energy`` is NaN on
    unsuccessful cases. ``self_energy`` is the signed self contribution
    actually included (negative removal in the periodic branch, positive
    addition in the finite branch, zero otherwise). ``periodic`` records the
    executed branch. ``mode_coverage_successful`` certifies, through the mode
    set's ``support_radius > k_max``, that the prepared extents contain every
    complete-sphere mode of the case cell (a sufficient test; an uncertified
    cell is refused rather than truncated). ``sphere_truncated`` flags an
    index-box support whose box may cut the sphere (``support_radius <=
    k_max``), a declared source truncation. ``active_modes`` counts executed
    nonzero half-space modes inside the sphere. ``definition_identity`` and
    ``prepared_identity`` are the exact definition and prepared-action digests
    as dynamic words (``definition_id``/``prepared_id`` decode them on the
    host).
    """

    energy: Array
    finite_energy: Array
    reciprocal_energy: Array
    self_energy: Array
    periodic: Array
    active_modes: Array
    mode_coverage_successful: Array
    sphere_truncated: Array
    cell_successful: Array
    charges_successful: Array
    successful: Array
    energy_unit: UnitDefinition
    definition_identity: IdentityWords
    prepared_identity: IdentityWords

    @property
    def definition_id(self) -> str:
        """Host decoding of the latent Ewald definition digest."""
        return self.definition_identity.fingerprint()

    @property
    def prepared_id(self) -> str:
        """Host decoding of the prepared latent Ewald action digest."""
        return self.prepared_identity.fingerprint()

    def global_energy_value(self, atom_capacity: int, /) -> AtomisticObservableValue:
        """Case-level ledger value of the global latent energy (no atomic shares)."""
        cases = int(self.energy.shape[0])
        return AtomisticObservableValue(
            AtomisticObservable.GLOBAL_ENERGY_COMPONENT,
            LATENT_EWALD_ENERGY_PORT,
            self.energy,
            self.successful,
            level=AtomisticConditionLevel.CASE,
            case_count=cases,
            atom_capacity=atom_capacity,
            unit=self.energy_unit,
        )


class LatentPolarization(StrictModule):
    """Source polarization of each case (``sqrt(eps)``-normalized, mean policy applied).

    ``finite`` is ``sum q~ r`` in ``e x length``; ``periodic_real`` and
    ``periodic_imaginary`` are the row components of
    ``sum q~ exp(2 pi i s) / (2 pi i)`` (in ``e``). Unexecuted branch values
    are zero.
    """

    finite: Array
    periodic_real: Array
    periodic_imaginary: Array
    periodic: Array
    successful: Array


class LatentBornCharges(StrictModule):
    """Projected latent Born effective charges and their per-case status."""

    values: Array
    periodic: Array
    successful: Array
    port: ValuePort
    projection_id: str = eqx.field(static=True)

    def value(self, case_count: int, atom_capacity: int, /) -> AtomisticObservableValue:
        return AtomisticObservableValue(
            AtomisticObservable.BORN_CHARGES,
            self.port,
            self.values,
            self.successful,
            level=AtomisticConditionLevel.ATOM,
            case_count=case_count,
            atom_capacity=atom_capacity,
            unit=_LATENT_UNIT,
        )


def _gaussian_pair_kernel(squared: Array, a: Array, /) -> Array:
    """``erf(a r) / r`` from ``r^2``, analytic (and differentiable) through ``r = 0``."""
    x2 = (a * a) * squared
    small = x2 < _SERIES_THRESHOLD
    safe = jnp.where(small, jnp.ones_like(squared), squared)
    distance = jnp.sqrt(safe)
    direct = jsp.erf(a * distance) / distance
    series = (2.0 * a / _SQRT_PI) * (
        1.0 - x2 / 3.0 + x2 * x2 / 10.0 - x2 * x2 * x2 / 42.0
    )
    return jnp.where(small, series, direct)


class _CaseParts(StrictModule):
    finite: Array
    reciprocal: Array
    self_energy: Array
    active_modes: Array
    coverage: Array
    truncated: Array
    cell: Array


def _orientation(vectors: Array, /) -> Array:
    """Signed triple product ``a . (b x c)`` of the row cell."""
    return jnp.dot(vectors[0], jnp.cross(vectors[1], vectors[2]))


def latent_ewald_influence(
    definition: LatentEwaldDefinition, volume: Array, oriented: Array, /
) -> ReciprocalInfluence:
    """Source-normalized latent influence ``N exp(-sigma^2 k^2/2) / (V k^2)``.

    Nonzero only on ``0 < k^2 <= k_max^2``; a non-positively oriented or
    singular cell divides by one (its status is reported separately). The
    definition's dynamic scalars are evaluated in the volume dtype.
    """
    dtype = volume.dtype
    cutoff = (definition.reciprocal_cutoff**2).astype(dtype)
    half_sigma = (0.5 * definition.sigma**2).astype(dtype)
    safe_volume = jnp.where(oriented & (volume > 0.0), volume, 1.0)
    prefactor = definition.norm_factor.astype(dtype) / safe_volume

    def influence(wave: Array, squared: Array) -> Array:
        del wave
        inside = (squared > 0.0) & (squared <= cutoff)
        safe = jnp.where(inside, squared, 1.0)
        return jnp.where(inside, prefactor * jnp.exp(-half_sigma * safe) / safe, 0.0)

    return influence


class PreparedLatentEwald(StrictModule, NonTrainableState):
    """Prepared numerical latent Ewald action for one per-case atom capacity.

    Bound to a definition, the consumer's scale (length and energy units are
    converted by exact static factors), the explicit resources and, for
    periodic cases, a term-owned reciprocal mode set. Inputs are flat
    case-major ``(B * A, ...)`` arrays; stale bindings (another capacity,
    channel layout or scale) are refused.

    PyTree layout: the capacity, unit factors (functions of the static declared
    units only) and workspace are static; ``prepared_identity`` is the exact
    former ``prepared_id`` digest as dynamic words (decoded by the host
    property).
    """

    definition: LatentEwaldDefinition
    scale: AtomisticScaleContract
    resources: LatentEwaldResources
    modes: PreparedReciprocalModeSet | None
    atom_capacity: int = eqx.field(static=True)
    length_factor: float = eqx.field(static=True)
    energy_factor: float = eqx.field(static=True)
    finite_workspace_bytes: int = eqx.field(static=True)
    prepared_identity: IdentityWords

    def __init__(
        self,
        definition: LatentEwaldDefinition,
        scale: AtomisticScaleContract,
        atom_capacity: int,
        resources: LatentEwaldResources,
        modes: PreparedReciprocalModeSet | None,
        /,
    ) -> None:
        if not isinstance(definition, LatentEwaldDefinition):
            raise TypeError("definition must be a LatentEwaldDefinition.")
        if not isinstance(scale, AtomisticScaleContract):
            raise TypeError("scale must be an AtomisticScaleContract.")
        if not isinstance(resources, LatentEwaldResources):
            raise TypeError("resources must be LatentEwaldResources.")
        capacity = _positive_int(atom_capacity, "atom_capacity")
        if modes is not None:
            if definition.branch == "finite":
                raise ValueError("A finite-only latent Ewald binds no reciprocal modes.")
            if not isinstance(modes, PreparedReciprocalModeSet):
                raise TypeError("modes must be a PreparedReciprocalModeSet or None.")
            if modes.source_capacity != capacity or modes.plan.include_zero_mode:
                raise ValueError(
                    "The mode set must cover the atom capacity and exclude k = 0."
                )
            if modes.maximum_mode != resources.mode_extents:
                raise ValueError("The mode set does not match the declared mode extents.")
            if modes.plan.plan_id != _mode_plan(definition, resources).plan_id:
                raise ValueError(
                    "The mode set must bind the definition's mode plan under these "
                    "resources (support box, mode chunk and budget)."
                )
        workspace = resources.finite_workspace_bytes(capacity, definition.channels)
        if workspace > resources.maximum_finite_workspace_bytes:
            raise ValueError(
                f"Latent Ewald finite row blocks need {workspace} bytes, exceeding "
                f"{resources.maximum_finite_workspace_bytes}."
            )
        length = _factor(scale.length_unit, _LENGTH_UNIT, "length")
        energy = _factor(_ENERGY_UNIT, scale.energy_unit, "energy")
        self.definition = definition
        self.scale = scale
        self.resources = resources
        self.modes = modes
        self.atom_capacity = capacity
        self.length_factor = length
        self.energy_factor = energy
        self.finite_workspace_bytes = workspace
        self.prepared_identity = IdentityWords.of(
            {
                "kind": "prepared-latent-ewald",
                "definition": definition.definition_id,
                "scale": scale.scale_id,
                "atom_capacity": capacity,
                "resources": resources.resources_id,
                "modes": None if modes is None else modes.prepared_id,
            }
        )

    @property
    def prepared_id(self) -> str:
        """Host decoding of the canonical prepared latent Ewald digest."""
        return self.prepared_identity.fingerprint()

    def validate(self) -> None:
        """Replay the preparation on the stored binding; refuse a stale action.

        A restored action bypasses its constructor. The definition and the
        resources replay their constructors and the scale contract is rebuilt
        by its constructor. A stored mode set must bind exactly the
        definition's mode plan under these resources and this atom capacity;
        its owner (`PreparedReciprocalModeSet.validate`) then re-admits its
        layout against the stored arrays before comparing them with the
        canonical modes in bounded blocks, without preparing them again. The
        constructor replayed on the validated parts must reproduce every
        stored field exactly: the atom capacity, unit factors, finite
        workspace and ``prepared_identity``.
        """
        definition = self.definition
        resources = self.resources
        scale = self.scale
        modes = self.modes
        if not isinstance(definition, LatentEwaldDefinition):
            raise TypeError("definition must be a LatentEwaldDefinition.")
        if not isinstance(resources, LatentEwaldResources):
            raise TypeError("resources must be LatentEwaldResources.")
        if not isinstance(scale, AtomisticScaleContract):
            raise TypeError("scale must be an AtomisticScaleContract.")
        definition.validate()
        resources.validate()
        require_identical(
            scale,
            AtomisticScaleContract(scale.length_unit, scale.energy_unit),
            "The latent Ewald scale contract",
            ValueError,
        )
        capacity = _positive_int(self.atom_capacity, "atom_capacity")
        if modes is not None:
            if not isinstance(modes, PreparedReciprocalModeSet):
                raise TypeError("modes must be a PreparedReciprocalModeSet or None.")
            require_identical(
                modes.plan,
                _mode_plan(definition, resources),
                "The latent Ewald mode plan",
                ValueError,
            )
            if modes.source_capacity != capacity:
                raise ValueError("The mode set must cover the atom capacity.")
            modes.validate()
        replay = PreparedLatentEwald(definition, scale, capacity, resources, modes)
        require_identical(self, replay, "The prepared latent Ewald action", ValueError)

    # -- input binding --------------------------------------------------------

    def _layout(self, positions: Array, /) -> int:
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("Latent Ewald positions must be flat (B * A, 3).")
        if positions.shape[0] % self.atom_capacity:
            raise ValueError(
                "Positions do not tile the prepared atom capacity; the latent Ewald "
                "binding is stale for this layout."
            )
        return positions.shape[0] // self.atom_capacity

    def _charges(self, charges: ArrayLike, cases: int, dtype: jnp.dtype, /) -> Array:
        values = jnp.asarray(charges)
        flat = cases * self.atom_capacity
        channels = self.definition.channels
        if values.shape == (flat,) and channels == 1:
            values = values[:, None]
        if values.shape != (flat, channels):
            raise ValueError(
                f"Latent charges must have shape ({flat},) or ({flat}, {channels})."
            )
        return values.astype(dtype).reshape((cases, self.atom_capacity, channels))

    def _periodic_cases(
        self, cell_vectors: ArrayLike, periodic_mask: ArrayLike, cases: int, /
    ) -> tuple[Array | None, Array, Array]:
        """Cell vectors, executed branch, and declared-periodicity admission."""
        vectors = jnp.asarray(cell_vectors)
        mask = jnp.asarray(periodic_mask, dtype=jnp.bool_)
        if vectors.ndim != 3 or vectors.shape[0] != cases or vectors.shape[2] != 3:
            raise ValueError("cell_vectors must have shape (B, rank, 3).")
        rank = vectors.shape[1]
        if rank not in (0, 3) or mask.shape != (cases, rank):
            raise ValueError(
                "Latent Ewald admits lattice rank 0 or 3 with (B, rank) masks."
            )
        if rank == 0:
            if self.definition.branch == "periodic":
                return (
                    None,
                    jnp.ones((cases,), dtype=jnp.bool_),
                    jnp.zeros((cases,), dtype=jnp.bool_),
                )
            return (
                None,
                jnp.zeros((cases,), dtype=jnp.bool_),
                jnp.ones((cases,), dtype=jnp.bool_),
            )
        match self.definition.branch:
            case "per-case":
                periodic = jnp.any(mask, axis=1)
            case "periodic":
                periodic = jnp.ones((cases,), dtype=jnp.bool_)
            case _:
                periodic = jnp.zeros((cases,), dtype=jnp.bool_)
        if self.modes is None and self.definition.branch != "finite":
            raise ValueError(
                "This latent Ewald binding was prepared without reciprocal modes and "
                "cannot evaluate a periodic lattice layout."
            )
        admitted = (
            jnp.any(mask, axis=1)
            if self.definition.branch == "periodic"
            else jnp.ones((cases,), dtype=jnp.bool_)
        )
        return vectors, periodic, admitted

    # -- per-case numerics (angstrom, eV, e) ------------------------------------

    def _self(self, charges: Array, /) -> Array:
        return self.definition.self_coefficient.astype(charges.dtype) * jnp.sum(
            charges * charges
        )

    def _finite_case(self, charges: Array, positions: Array, mask: Array, /) -> Array:
        """``k_e sum_{i<j} q_i.q_j K(r_ij)`` by checkpointed row blocks."""
        count = self.atom_capacity
        block = min(self.resources.row_block, count)
        blocks = -(-count // block)
        rows = jnp.arange(blocks * block, dtype=jnp.int32).reshape((blocks, block))
        columns = jnp.arange(count, dtype=jnp.int32)
        dtype = positions.dtype
        a = (1.0 / (math.sqrt(2.0) * self.definition.sigma)).astype(dtype)

        def body(total: Array, row: Array) -> tuple[Array, None]:
            valid = row < count
            safe = jnp.where(valid, row, 0)
            displacement = positions[safe][:, None, :] - positions[None, :, :]
            squared = jnp.sum(displacement * displacement, axis=-1)
            routed = (
                valid[:, None]
                & mask[safe][:, None]
                & mask[None, :]
                & (safe[:, None] != columns[None, :])
            )
            kernel = _gaussian_pair_kernel(jnp.where(routed, squared, 1.0), a)
            product = contract("rc,nc->rn", charges[safe], charges, backend="jax")
            return total + jnp.sum(jnp.where(routed, product * kernel, 0.0)), None

        total, _ = jax.lax.scan(jax.checkpoint(body), jnp.zeros((), dtype=dtype), rows)
        return 0.5 * self.definition.coulomb_constant.astype(dtype) * total

    def _periodic_case(
        self, charges: Array, positions: Array, vectors: Array, /
    ) -> tuple[Array, Array, Array, Array, Array]:
        """Reciprocal energy, active modes, coverage, truncation and cell status."""
        modes = self.modes
        if modes is None:
            raise ValueError("This latent Ewald binding has no reciprocal mode set.")
        definition = self.definition
        geometry = modes.geometry(vectors)
        squared = jax.lax.stop_gradient(geometry.squared_wavenumber)
        cutoff = definition.reciprocal_cutoff.astype(squared.dtype)
        cutoff_squared = (definition.reciprocal_cutoff**2).astype(squared.dtype)
        volume = geometry.volume
        oriented = _orientation(vectors) > 0.0
        influence = latent_ewald_influence(definition, volume, oriented)

        quadratic = modes.quadratic_energy(
            positions, charges, geometry, influence, zero_mode="exclude"
        )
        stored = modes.hermitian_weights > 0.0
        active = jnp.sum(stored & (squared > 0.0) & (squared <= cutoff_squared))
        # Every sphere mode is represented iff the stored box reaches past k_max.
        support = jax.lax.stop_gradient(geometry.support_radius)
        if definition.mode_support == "complete-sphere":
            coverage = support > cutoff
            truncated = jnp.asarray(False)
        else:
            coverage = jnp.asarray(True)
            truncated = support <= cutoff
        cell = geometry.successful & oriented & quadratic.routes_successful
        return quadratic.value, active, coverage, truncated, cell & quadratic.successful

    def _case(
        self,
        charges: Array,
        positions: Array,
        mask: Array,
        vectors: Array | None,
        periodic: Array,
        /,
    ) -> _CaseParts:
        definition = self.definition
        dtype = positions.dtype
        self_energy = self._self(charges)
        zero = jnp.zeros((), dtype=dtype)
        no = jnp.asarray(False)

        def finite_parts() -> _CaseParts:
            pair = self._finite_case(charges, positions, mask)
            added = zero if definition.remove_self_interaction else self_energy
            return _CaseParts(
                finite=pair,
                reciprocal=zero,
                self_energy=added,
                active_modes=jnp.zeros((), dtype=jnp.int32),
                coverage=jnp.asarray(True),
                truncated=no,
                cell=jnp.asarray(True),
            )

        if vectors is None:
            if definition.branch == "periodic":
                # Forced periodic without a lattice: the source divides by a zero
                # cell. Refuse without evaluating either sum.
                return _CaseParts(
                    finite=zero,
                    reciprocal=zero,
                    self_energy=zero,
                    active_modes=jnp.zeros((), dtype=jnp.int32),
                    coverage=jnp.asarray(True),
                    truncated=no,
                    cell=no,
                )
            return finite_parts()

        def periodic_parts() -> _CaseParts:
            safe = jnp.where(periodic, vectors, jnp.eye(3, dtype=vectors.dtype))
            reciprocal, active, coverage, truncated, cell = self._periodic_case(
                charges, positions, safe
            )
            removed = -self_energy if definition.remove_self_interaction else zero
            return _CaseParts(
                finite=zero,
                reciprocal=reciprocal,
                self_energy=removed,
                active_modes=active.astype(jnp.int32),
                coverage=coverage,
                truncated=truncated,
                cell=cell,
            )

        match definition.branch:
            case "finite":
                return finite_parts()
            case "periodic":
                parts = periodic_parts()
                return _CaseParts(
                    finite=parts.finite,
                    reciprocal=parts.reciprocal,
                    self_energy=parts.self_energy,
                    active_modes=parts.active_modes,
                    coverage=parts.coverage,
                    truncated=parts.truncated,
                    # A forced-periodic case without a periodic axis is refused.
                    cell=parts.cell & periodic,
                )
            case _:
                return jax.lax.cond(periodic, periodic_parts, finite_parts)

    # -- public actions -------------------------------------------------------

    def energy(
        self,
        charges: ArrayLike,
        positions: ArrayLike,
        atom_mask: ArrayLike,
        cell_vectors: ArrayLike,
        periodic_mask: ArrayLike,
        /,
    ) -> LatentEwaldEvaluation:
        """Global latent energy of every case (consumer units).

        ``positions``/``cell_vectors`` are in the prepared length unit,
        ``charges`` latent channels in ``e``; ``cell_vectors`` ``(B, 0, 3)``
        marks a finite layout. Inactive atoms contribute nothing. The result
        is differentiable in charges, positions and cell vectors.
        """
        position = jnp.asarray(positions)
        cases = self._layout(position)
        dtype = position.dtype
        count = self.atom_capacity
        mask = jnp.asarray(atom_mask, dtype=jnp.bool_)
        if mask.shape != (cases * count,):
            raise ValueError("atom_mask must align with the flat positions.")
        latent = self._charges(charges, cases, dtype)
        mask = mask.reshape((cases, count))
        finite_charges = jnp.all(~mask[..., None] | jnp.isfinite(latent), axis=(1, 2))
        finite_positions = jnp.all(
            ~mask[..., None] | jnp.isfinite(position.reshape((cases, count, 3))),
            axis=(1, 2),
        )
        latent = jnp.where(mask[..., None] & jnp.isfinite(latent), latent, 0.0)
        scaled = jnp.where(
            mask[..., None], position.reshape((cases, count, 3)), 0.0
        ) * jnp.asarray(self.length_factor, dtype=dtype)
        vectors, periodic, admitted = self._periodic_cases(
            cell_vectors, periodic_mask, cases
        )
        if vectors is None:
            parts = jax.lax.map(
                lambda item: self._case(item[0], item[1], item[2], None, item[3]),
                (latent, scaled, mask, periodic),
            )
        else:
            lattice = vectors.astype(dtype) * jnp.asarray(self.length_factor, dtype=dtype)
            parts = jax.lax.map(
                lambda item: self._case(item[0], item[1], item[2], item[3], item[4]),
                (latent, scaled, mask, lattice, periodic),
            )
        factor = jnp.asarray(self.energy_factor, dtype=dtype)
        finite = factor * parts.finite
        reciprocal = factor * parts.reciprocal
        self_energy = factor * parts.self_energy
        total = finite + reciprocal + self_energy
        charges_ok = finite_charges
        cell_successful = parts.cell & admitted
        successful = (
            charges_ok
            & finite_positions
            & cell_successful
            & parts.coverage
            & jnp.isfinite(total)
        )
        return LatentEwaldEvaluation(
            energy=jnp.where(successful, total, jnp.nan),
            finite_energy=finite,
            reciprocal_energy=reciprocal,
            self_energy=self_energy,
            periodic=periodic,
            active_modes=parts.active_modes,
            mode_coverage_successful=parts.coverage,
            sphere_truncated=parts.truncated,
            cell_successful=cell_successful,
            charges_successful=charges_ok,
            successful=successful,
            energy_unit=self.scale.energy_unit,
            definition_identity=self.definition.definition_identity,
            prepared_identity=self.prepared_identity,
        )

    def _polarization_parts(
        self,
        charges: Array,
        positions: Array,
        mask: Array,
        vectors: Array | None,
        periodic: Array,
        /,
    ) -> tuple[Array, Array, Array, Array, Array]:
        """``(finite[B,3], real[B,3], imaginary[B,3], phase angle[B,A,3], cell ok[B])``."""
        policy = self.definition.born
        cases = periodic.shape[0]
        count = self.atom_capacity
        dtype = positions.dtype
        latent = jnp.where(mask, charges.reshape((cases, count)), 0.0)
        if policy.remove_mean:
            active = jnp.sum(mask, axis=1, keepdims=True)
            mean = jnp.sum(latent, axis=1, keepdims=True) / jnp.maximum(active, 1)
            latent = jnp.where(mask, latent - mean, 0.0)
        norm = jnp.asarray(math.sqrt(policy.epsilon_factor), dtype=dtype)
        position = jnp.where(mask[..., None], positions.reshape((cases, count, 3)), 0.0)
        finite = norm * contract("ba,bad->bd", latent, position, backend="jax")
        if vectors is None:
            zeros = jnp.zeros_like(finite)
            return (
                finite,
                zeros,
                zeros,
                jnp.zeros((cases, count, 3), dtype=dtype),
                jnp.ones((cases,), dtype=jnp.bool_),
            )
        safe = jnp.where(periodic[:, None, None], vectors, jnp.eye(3, dtype=dtype))
        inverse, solved = lattice_right_inverse_with_status(safe)
        oriented = jax.vmap(_orientation)(safe) > 0.0
        fractional = contract(
            "bad,bdk->bak", position, inverse.astype(dtype), backend="jax"
        )
        angle = _TWO_PI * fractional
        real = (
            norm * contract("ba,bak->bk", latent, jnp.sin(angle), backend="jax") / _TWO_PI
        )
        imaginary = (
            -norm
            * contract("ba,bak->bk", latent, jnp.cos(angle), backend="jax")
            / _TWO_PI
        )
        return finite, real, imaginary, angle, ~periodic | (solved & oriented)

    def polarization(
        self,
        charges: ArrayLike,
        positions: ArrayLike,
        atom_mask: ArrayLike,
        cell_vectors: ArrayLike,
        periodic_mask: ArrayLike,
        /,
    ) -> LatentPolarization:
        """The source polarization under the requested Born projection policy."""
        if self.definition.channels != 1:
            raise ValueError("The source polarization is defined for one latent channel.")
        position = jnp.asarray(positions)
        cases = self._layout(position)
        mask = jnp.asarray(atom_mask, dtype=jnp.bool_).reshape(
            (cases, self.atom_capacity)
        )
        latent = self._charges(charges, cases, position.dtype)[..., 0]
        vectors, periodic, admitted = self._periodic_cases(
            cell_vectors, periodic_mask, cases
        )
        lattice = None if vectors is None else vectors.astype(position.dtype)
        finite, real, imaginary, _, cell = self._polarization_parts(
            latent, position, mask, lattice, periodic
        )
        selected = periodic[:, None]
        charges_ok = jnp.all(~mask | jnp.isfinite(latent), axis=1)
        return LatentPolarization(
            finite=jnp.where(selected, 0.0, finite),
            periodic_real=jnp.where(selected, real, 0.0),
            periodic_imaginary=jnp.where(selected, imaginary, 0.0),
            periodic=periodic,
            successful=admitted & cell & charges_ok,
        )

    def born_charges(
        self,
        charge_function: Callable[[Array], Array],
        positions: ArrayLike,
        atom_mask: ArrayLike,
        cell_vectors: ArrayLike,
        periodic_mask: ArrayLike,
        /,
    ) -> LatentBornCharges:
        """Source ``BEC``: ``Z_i[a, b] = dP_a / dr_ib`` including ``dq/dr``.

        ``charge_function`` maps flat positions to the latent charges of the
        same layout (the model's own differentiable latent head). Periodic
        cases dephase by ``conj(exp(2 pi i s_ia))`` and project with ``H^T``.
        Work is one reverse pass per real polarization component (3 finite,
        plus 6 periodic), never per atom.
        """
        if self.definition.channels != 1:
            raise ValueError(
                "The source Born projection is defined for one latent channel."
            )
        policy = self.definition.born
        position = jnp.asarray(positions)
        cases = self._layout(position)
        count = self.atom_capacity
        dtype = position.dtype
        mask = jnp.asarray(atom_mask, dtype=jnp.bool_).reshape((cases, count))
        vectors, periodic, admitted = self._periodic_cases(
            cell_vectors, periodic_mask, cases
        )
        lattice = (
            None if vectors is None else jax.lax.stop_gradient(vectors.astype(dtype))
        )

        def components(value: Array) -> tuple[Array, tuple[Array, Array, Array]]:
            latent = self._charges(charge_function(value), cases, dtype)[..., 0]
            finite, real, imaginary, angle, cell = self._polarization_parts(
                latent, value, mask, lattice, periodic
            )
            charges_ok = jnp.all(~mask | jnp.isfinite(latent), axis=1)
            stacked = (
                finite
                if lattice is None
                else jnp.concatenate((finite, real, imaginary), axis=1)
            )
            return stacked, (angle, cell, charges_ok)

        stacked, vjp, (angle, cell, charges_ok) = jax.vjp(
            components, position, has_aux=True
        )
        width = stacked.shape[1]
        basis = jnp.broadcast_to(
            jnp.eye(width, dtype=dtype)[:, None, :], (width, cases, width)
        )
        (gradients,) = jax.vmap(vjp)(basis)
        gradients = gradients.reshape((width, cases, count, 3))
        # Z_fin[b, i, a, d] = dP_a / dr_id.
        finite = jnp.moveaxis(gradients[:3], 0, 2)
        if lattice is None:
            values = finite
        else:
            real = jnp.moveaxis(gradients[3:6], 0, 2)
            imaginary = jnp.moveaxis(gradients[6:9], 0, 2)
            basis_values = (
                real * jnp.cos(angle)[..., None] + imaginary * jnp.sin(angle)[..., None]
            )
            safe = jnp.where(periodic[:, None, None], lattice, jnp.eye(3, dtype=dtype))
            projected = contract("bac,biad->bicd", safe, basis_values, backend="jax")
            values = jnp.where(periodic[:, None, None, None], projected, finite)
        values = jnp.where(mask[..., None, None], values, 0.0)
        if policy.output_index is not None:
            values = values[..., policy.output_index, :]
        successful = (
            admitted
            & cell
            & charges_ok
            & jnp.all(jnp.isfinite(values.reshape((cases, -1))), axis=1)
        )
        return LatentBornCharges(
            values=values.reshape((cases * count,) + values.shape[2:]),
            periodic=periodic,
            successful=successful,
            port=policy.port(),
            projection_id=policy.projection_id,
        )


def _checked_latent_field(
    reference: AtomisticFieldReference, channels: int, /
) -> AtomisticFieldReference:
    if not isinstance(reference, AtomisticFieldReference):
        raise TypeError("latent_field must be an AtomisticFieldReference.")
    expected = latent_charge_port(channels)
    if (
        reference.observable is not AtomisticObservable.LATENT_CHARGES
        or reference.level is not AtomisticConditionLevel.ATOM
        or reference.port.port_id != expected.port_id
        or reference.unit is None
    ):
        raise ValueError(
            "latent_field must reference atom-level LATENT_CHARGES on the latent "
            "charge port of the definition's channel count, with an explicit unit."
        )
    return reference


class LatentEwaldPotential(AbstractAtomisticEnergyTerm, NonTrainableState):
    """Latent Ewald energy of a preceding producer's latent charge channels.

    The latent field is resolved inside the same scalar energy, so position,
    cell and producer-parameter derivatives include the latent response. A
    nonperiodic system runs the finite branch; a rank-3 periodic system runs
    the branch selected by the definition. The energy is global
    (``atom_energy_complete=False``).

    PyTree layout: ``name`` and ``force_group`` are static; ``term_identity`` is
    the dynamic exact former ``term_id`` digest (the base class decodes
    ``term_id`` on the host).
    """

    definition: LatentEwaldDefinition
    latent_field: AtomisticFieldReference
    resources: LatentEwaldResources
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_identity: IdentityWords
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self,
        definition: LatentEwaldDefinition,
        latent_field: AtomisticFieldReference,
        /,
        *,
        resources: LatentEwaldResources | None = None,
        name: str = "latent-ewald",
        force_group: int = 0,
    ) -> None:
        if not isinstance(definition, LatentEwaldDefinition):
            raise TypeError("definition must be a LatentEwaldDefinition.")
        reference = _checked_latent_field(latent_field, definition.channels)
        resources_ = LatentEwaldResources() if resources is None else resources
        if not isinstance(resources_, LatentEwaldResources):
            raise TypeError("resources must be LatentEwaldResources.")
        identifier = str(name).strip()
        group = int(force_group)
        if not identifier or group < 0:
            raise ValueError("Term name must be non-empty and force_group non-negative.")
        periodic = definition.branch != "finite"
        self.definition = definition
        self.latent_field = reference
        self.resources = resources_
        self.name = identifier
        self.force_group = group
        self.capabilities = AtomisticPotentialCapabilities(
            conservative_energy=True,
            finite_geometry=definition.branch != "periodic",
            orthorhombic_periodic=periodic,
            triclinic_periodic=periodic,
            cell_derivative=periodic,
            local_energy=False,
        )
        self.requirements = AtomisticPotentialRequirements(
            reciprocal_grid=periodic, field_dependencies=(reference,)
        )
        self.term_identity = IdentityWords.of(
            {
                "kind": "latent-ewald-potential",
                "definition": definition.definition_id,
                "latent_field": reference.reference_id,
                "resources": resources_.resources_id,
                "name": identifier,
                "force_group": group,
            }
        )

    def prepare(
        self, system: PreparedAtomisticSystem, /
    ) -> "PreparedLatentEwaldPotential":
        return PreparedLatentEwaldPotential(self, system)


class PreparedLatentEwaldPotential(AbstractPreparedAtomisticEnergyTerm):
    """System-bound latent Ewald term.

    PyTree layout: the executed branch ``periodic`` (a selector of the static
    definition branch and cell periodicity) and the exact latent unit factor
    are static; ``term_identity``/``prepared_identity`` are the exact former
    ``term_id``/``prepared_id`` digests as dynamic words. ``energy`` folds the
    numerical binding of the context system into the term status.
    """

    plan: LatentEwaldPotential
    system: PreparedAtomisticSystem
    binding: PreparedLatentEwald
    periodic: bool = eqx.field(static=True)
    latent_factor: float = eqx.field(static=True)
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_identity: IdentityWords
    prepared_identity: IdentityWords
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self, plan: LatentEwaldPotential, system: PreparedAtomisticSystem, /
    ) -> None:
        definition = plan.definition
        cell = system.cell
        if cell is None:
            if definition.branch == "periodic":
                raise ValueError("A forced-periodic latent Ewald needs a periodic cell.")
            periodic = False
        else:
            if cell.rank != 3 or cell.ambient_dimension != 3:
                raise ValueError(
                    "Latent Ewald evaluates the full three-dimensional cell; a lower-rank "
                    "lattice has no source definition."
                )
            match definition.branch:
                case "per-case":
                    periodic = any(cell.periodic_axes)
                case "periodic":
                    periodic = True
                case _:
                    periodic = False
        reference = plan.latent_field
        unit = reference.unit
        if unit is None:
            raise ValueError("latent_field must declare an explicit latent charge unit.")
        latent = _factor(unit, _LATENT_UNIT, "latent charge")
        capacity = system.capacity
        scale = system.plan.units.scale
        # A nonperiodic system binds no mode set: under ``per-case`` its only
        # executable branch is the finite sum, and a lattice layout is refused.
        modes = (
            _mode_plan(definition, plan.resources).prepare(cell, capacity)
            if periodic and cell is not None
            else None
        )
        binding = PreparedLatentEwald(definition, scale, capacity, plan.resources, modes)
        self.plan = plan
        self.system = system
        self.binding = binding
        self.periodic = periodic
        self.latent_factor = latent
        self.name = plan.name
        self.force_group = plan.force_group
        self.term_identity = plan.term_identity
        self.capabilities = plan.capabilities
        self.requirements = plan.requirements
        self.prepared_identity = IdentityWords.of(
            {
                "kind": "prepared-latent-ewald-potential",
                "term": plan.term_id,
                "system": system.prepared_id,
                "binding": binding.prepared_id,
                "periodic": periodic,
            }
        )

    def energy(self, context: AtomisticPotentialContext, /) -> AtomisticTermEvaluation:
        # The exact numerical system binding (the latent Ewald binding is
        # prepared for it) is compared as words and folded into the status.
        bound = context.system.prepared_identity.equals(self.system.prepared_identity)
        capacity = self.system.capacity
        channels = self.plan.definition.channels
        field = context.field(self.plan.latent_field)
        if field.available.shape != (1,) or field.values.shape[0] != capacity:
            raise ValueError("Latent charges must be one atom-level case of the system.")
        dtype = context.positions.dtype
        values = field.values.astype(dtype).reshape((capacity, channels))
        usable = field.available[0] & field.successful[0]
        latent = jnp.where(usable, self.latent_factor * values, 0.0)
        cell = self.system.cell
        if self.periodic:
            if cell is None:
                raise ValueError(
                    "The periodic latent Ewald branch needs a periodic cell."
                )
            vectors = jnp.asarray(context.cell_vectors, dtype=dtype).reshape((1, 3, 3))
            mask = jnp.asarray(cell.periodic_mask, dtype=jnp.bool_)[None]
        else:
            vectors = jnp.zeros((1, 0, 3), dtype=dtype)
            mask = jnp.zeros((1, 0), dtype=jnp.bool_)
        evaluation = self.binding.energy(
            latent, context.positions, self.system.active_mask, vectors, mask
        )
        successful = evaluation.successful[0] & usable & bound
        return AtomisticTermEvaluation(
            jnp.where(successful, evaluation.energy[0], jnp.nan),
            jnp.zeros((capacity,), dtype=dtype),
            successful,
            atom_energy_complete=False,
            evidence=(evaluation,),
        )


# Portable data of the prepared latent Ewald action a MACELES model stores;
# `PreparedLatentEwald.validate` replays every one of them on restore.
for _artifact in (
    LatentBornProjection,
    LatentEwaldDefinition,
    LatentEwaldResources,
    PreparedLatentEwald,
):
    register_artifact_value(f"phydrax.atomistic:{_artifact.__name__}", _artifact)
del _artifact


__all__ = [
    "LATENT_CHARGE_PORT",
    "LATENT_EWALD_ENERGY_PORT",
    "LATENT_EWALD_SOURCE",
    "latent_charge_port",
    "LatentBornCharges",
    "LatentBornProjection",
    "LatentEwaldBranch",
    "LatentEwaldDefinition",
    "LatentEwaldEvaluation",
    "LatentEwaldModeSupport",
    "LatentEwaldPotential",
    "LatentEwaldResources",
    "LatentPolarization",
    "PreparedLatentEwald",
    "PreparedLatentEwaldPotential",
    "SOURCE_DL",
    "SOURCE_N_MAX",
    "SOURCE_NORM_FACTOR",
    "SOURCE_SIGMA",
]
