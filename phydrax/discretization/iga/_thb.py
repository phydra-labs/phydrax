#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class THBLevel(StrictModule, NonTrainableState):
    """One nested tensor level and its deterministic active sets."""

    level: int = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)
    cell_active: Array
    function_active: Array
    level_id: str = eqx.field(static=True)

    def __init__(
        self,
        level: int,
        basis_id: str,
        cell_active: ArrayLike,
        function_active: ArrayLike,
        /,
    ):
        level_ = int(level)
        basis = str(basis_id)
        cells = np.asarray(cell_active, dtype=bool)
        functions = np.asarray(function_active, dtype=bool)
        if level_ < 0 or not basis or cells.ndim != 1 or functions.ndim != 1:
            raise ValueError("THB levels require nonnegative identity and rank-1 masks.")
        if cells.size == 0 or functions.size == 0 or not np.any(cells):
            raise ValueError("THB levels require active cells and functions.")
        self.level = level_
        self.basis_id = basis
        self.cell_active = jnp.asarray(cells)
        self.function_active = jnp.asarray(functions)
        self.level_id = canonical_fingerprint(
            {
                "kind": "thb-level",
                "level": level_,
                "basis": basis,
                "cells": array_tree_fingerprint(cells),
                "functions": array_tree_fingerprint(functions),
            }
        )

    @property
    def function_count(self) -> int:
        return int(self.function_active.size)


class THBBasisCertificate(StrictModule, NonTrainableState):
    """Machine-checked nestedness, truncation, and independence evidence."""

    nested: bool = eqx.field(static=True)
    partition_defect: float = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    basis_count: int = eqx.field(static=True)
    prolongation_defect: float = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    diagnostic_codes: tuple[str, ...] = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        nested: bool,
        partition_defect: float,
        rank: int,
        basis_count: int,
        prolongation_defect: float,
        tolerance: float,
        diagnostic_codes: Sequence[str] = (),
    ):
        partition = float(partition_defect)
        prolongation = float(prolongation_defect)
        tolerance_ = float(tolerance)
        rank_ = int(rank)
        count = int(basis_count)
        codes = tuple(str(code) for code in diagnostic_codes)
        if tolerance_ <= 0.0 or min(partition, prolongation) < 0.0:
            raise ValueError("THB certificate tolerances and defects are invalid.")
        passed = (
            bool(nested) and rank_ == count and max(partition, prolongation) <= tolerance_
        )
        self.nested = bool(nested)
        self.partition_defect = partition
        self.rank = rank_
        self.basis_count = count
        self.prolongation_defect = prolongation
        self.passed = passed
        self.diagnostic_codes = codes
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "thb-basis-certificate",
                "nested": bool(nested),
                "partition_defect": partition,
                "rank": rank_,
                "basis_count": count,
                "prolongation_defect": prolongation,
                "tolerance": tolerance_,
                "diagnostics": list(codes),
            }
        )


class THBHierarchy(StrictModule, NonTrainableState):
    """Nested tensor hierarchy represented on the finest coefficient space."""

    levels: tuple[THBLevel, ...]
    prolongations: tuple[Array, ...]
    finest_representations: tuple[Array, ...]
    hierarchy_id: str = eqx.field(static=True)

    def __init__(
        self,
        levels: Sequence[THBLevel],
        prolongations: Sequence[ArrayLike],
        /,
    ):
        levels_ = tuple(levels)
        matrices = tuple(np.asarray(value, dtype=float) for value in prolongations)
        if not levels_ or tuple(level.level for level in levels_) != tuple(
            range(len(levels_))
        ):
            raise ValueError("THB levels must be contiguous and start at zero.")
        if len(matrices) != len(levels_) - 1:
            raise ValueError("One THB prolongation is required between adjacent levels.")
        for coarse, fine, matrix in zip(levels_[:-1], levels_[1:], matrices, strict=True):
            if matrix.shape != (fine.function_count, coarse.function_count):
                raise ValueError(
                    "THB prolongation dimensions do not match adjacent levels."
                )
            if not np.all(np.isfinite(matrix)):
                raise ValueError("THB prolongations must be finite.")
        finest_count = levels_[-1].function_count
        representations: list[np.ndarray] = []
        for level_index, level in enumerate(levels_):
            representation = np.eye(level.function_count)
            for matrix in matrices[level_index:]:
                representation = matrix @ representation
            if representation.shape[0] != finest_count:
                raise ValueError("THB finest-space representation is inconsistent.")
            representations.append(representation)
        self.levels = levels_
        self.prolongations = tuple(jnp.asarray(value) for value in matrices)
        self.finest_representations = tuple(
            jnp.asarray(value) for value in representations
        )
        self.hierarchy_id = canonical_fingerprint(
            {
                "kind": "thb-hierarchy",
                "levels": [level.level_id for level in levels_],
                "prolongations": [array_tree_fingerprint(value) for value in matrices],
            }
        )

    def transformation(self, /) -> Array:
        """Return active hierarchical functions in the finest tensor basis."""
        columns = []
        finest_active_level = len(self.levels) - 1
        for index, (level, representation) in enumerate(
            zip(self.levels, self.finest_representations, strict=True)
        ):
            active = np.flatnonzero(np.asarray(level.function_active, dtype=bool))
            block = np.asarray(representation)[:, active]
            if index < finest_active_level:
                fine_active = np.asarray(
                    self.levels[index + 1].function_active, dtype=bool
                )
                if fine_active.size == block.shape[0]:
                    block = block.copy()
                    block[fine_active, :] = 0.0
            columns.append(block)
        return jnp.asarray(np.concatenate(columns, axis=1))

    def certify(self, /, *, tolerance: float = 1.0e-10) -> THBBasisCertificate:
        matrices = tuple(np.asarray(value) for value in self.prolongations)
        nested = all(np.linalg.matrix_rank(value) == value.shape[1] for value in matrices)
        partition = max(
            (
                float(np.max(np.abs(value @ np.ones(value.shape[1]) - 1.0)))
                for value in matrices
            ),
            default=0.0,
        )
        transform = np.asarray(self.transformation())
        rank = int(np.linalg.matrix_rank(transform, tol=tolerance))
        prolongation_defect = max(
            (float(np.max(np.abs(np.minimum(value, 0.0)))) for value in matrices),
            default=0.0,
        )
        codes = []
        if not nested:
            codes.append("thb.not_nested")
        if partition > tolerance:
            codes.append("thb.partition")
        if rank != transform.shape[1]:
            codes.append("thb.dependent")
        if prolongation_defect > tolerance:
            codes.append("thb.negative_prolongation")
        return THBBasisCertificate(
            nested=nested,
            partition_defect=partition,
            rank=rank,
            basis_count=transform.shape[1],
            prolongation_defect=prolongation_defect,
            tolerance=tolerance,
            diagnostic_codes=codes,
        )


__all__ = ["THBBasisCertificate", "THBHierarchy", "THBLevel"]
