#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


class CalorimeterGeometry(StrictModule, NonTrainableState):
    """Fixed heterogeneous cell support; no image-grid assumption is introduced."""

    cell_ids: Array
    channel_ids: Array
    layer_ids: Array
    subdetector_ids: Array
    material_ids: Array
    readout_ids: Array
    centroids: Array
    volumes: Array
    active: Array
    dead: Array
    senders: Array
    receivers: Array
    geometry_features: Array
    conditions_id: str = eqx.field(static=True)
    cell_count: int = eqx.field(static=True)
    edge_count: int = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        cell_ids: ArrayLike,
        channel_ids: ArrayLike,
        layer_ids: ArrayLike,
        subdetector_ids: ArrayLike,
        material_ids: ArrayLike,
        readout_ids: ArrayLike,
        centroids: ArrayLike,
        volumes: ArrayLike,
        active: ArrayLike,
        dead: ArrayLike,
        senders: ArrayLike,
        receivers: ArrayLike,
        conditions_id: str,
    ):
        cells = np.asarray(cell_ids)
        channels = np.asarray(channel_ids)
        layers = np.asarray(layer_ids)
        subdetectors = np.asarray(subdetector_ids)
        materials = np.asarray(material_ids)
        readouts = np.asarray(readout_ids)
        centroids_ = np.asarray(centroids, dtype=float)
        volumes_ = np.asarray(volumes, dtype=float)
        active_ = np.asarray(active, dtype=bool)
        dead_ = np.asarray(dead, dtype=bool)
        senders_ = np.asarray(senders)
        receivers_ = np.asarray(receivers)
        if (
            cells.ndim != 1
            or cells.size < 1
            or not np.issubdtype(cells.dtype, np.integer)
        ):
            raise ValueError("cell_ids must be a non-empty integer vector.")
        count = int(cells.size)
        scalar_arrays = (
            channels,
            layers,
            subdetectors,
            materials,
            readouts,
            volumes_,
            active_,
            dead_,
        )
        if any(value.shape != (count,) for value in scalar_arrays):
            raise ValueError("Calorimeter scalar metadata must align with cell_ids.")
        if any(
            not np.issubdtype(value.dtype, np.integer)
            for value in (channels, layers, subdetectors, materials, readouts)
        ):
            raise TypeError("Calorimeter identities must contain integers.")
        if centroids_.shape != (count, 3):
            raise ValueError("centroids must have shape (cell_count, 3).")
        if (
            np.any(~np.isfinite(centroids_))
            or np.any(~np.isfinite(volumes_))
            or np.any(volumes_[active_] <= 0.0)
        ):
            raise ValueError("Active cell geometry must be finite with positive volume.")
        if len(set(cells[active_].tolist())) != int(np.sum(active_)) or len(
            set(channels[active_].tolist())
        ) != int(np.sum(active_)):
            raise ValueError("Active cell and channel identities must be unique.")
        if (
            senders_.shape != receivers_.shape
            or senders_.ndim != 1
            or not np.issubdtype(senders_.dtype, np.integer)
            or not np.issubdtype(receivers_.dtype, np.integer)
        ):
            raise ValueError(
                "Adjacency senders and receivers must be aligned integer vectors."
            )
        if (
            np.any(senders_ < 0)
            or np.any(senders_ >= count)
            or np.any(receivers_ < 0)
            or np.any(receivers_ >= count)
        ):
            raise ValueError("Calorimeter adjacency contains out-of-range cell indices.")
        conditions = str(conditions_id).strip()
        if not conditions:
            raise ValueError("conditions_id must be non-empty.")
        scales = np.maximum(np.max(np.abs(centroids_), axis=0), 1.0)
        maximum_layer = max(float(np.max(layers[active_], initial=0)), 1.0)
        features = np.concatenate(
            (
                centroids_ / scales,
                (layers / maximum_layer)[:, None],
                np.log(np.maximum(volumes_, np.finfo(float).tiny))[:, None],
                active_[:, None].astype(float),
                dead_[:, None].astype(float),
            ),
            axis=1,
        )
        self.cell_ids = jnp.asarray(cells, dtype=jnp.int32)
        self.channel_ids = jnp.asarray(channels, dtype=jnp.int32)
        self.layer_ids = jnp.asarray(layers, dtype=jnp.int32)
        self.subdetector_ids = jnp.asarray(subdetectors, dtype=jnp.int32)
        self.material_ids = jnp.asarray(materials, dtype=jnp.int32)
        self.readout_ids = jnp.asarray(readouts, dtype=jnp.int32)
        self.centroids = jnp.asarray(centroids_)
        self.volumes = jnp.asarray(volumes_)
        self.active = jnp.asarray(active_)
        self.dead = jnp.asarray(dead_)
        self.senders = jnp.asarray(senders_, dtype=jnp.int32)
        self.receivers = jnp.asarray(receivers_, dtype=jnp.int32)
        self.geometry_features = jnp.asarray(features)
        self.conditions_id = conditions
        self.cell_count = count
        self.edge_count = int(senders_.size)
        self.geometry_id = canonical_fingerprint(
            {
                "kind": "calorimeter-geometry",
                "conditions": conditions,
                "cells": array_tree_fingerprint(
                    {
                        "cell_ids": cells,
                        "channel_ids": channels,
                        "layers": layers,
                        "subdetectors": subdetectors,
                        "materials": materials,
                        "readouts": readouts,
                        "centroids": centroids_,
                        "volumes": volumes_,
                        "active": active_,
                        "dead": dead_,
                        "senders": senders_,
                        "receivers": receivers_,
                    }
                ),
            }
        )


__all__ = ["CalorimeterGeometry"]
