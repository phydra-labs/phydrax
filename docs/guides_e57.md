# E57 multi-scan admission

`E57Provider` verifies source rights and checksum, enumerates scan headers, enforces file and per-scan point limits, preserves invalid-state masks, intensity, color, row/column acquisition indices, scan GUIDs, and sensor poses, and returns `E57ScanCollection`.

Each scan remains a separate lineage-bearing point product. Scan poses are never flattened without retention. Embedded image assets can be linked into the same `MeasurementCollection`; backends that cannot expose them produce a declared semantic loss.

E57 usually stores derived Cartesian or spherical products, not authoritative emission rays or time-of-flight. The adapter never claims unavailable raw beam semantics.
