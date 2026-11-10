# arborStats/bulk_export.py
"""
Export a compact Parquet file of per-segment stats for bulk (many-segment) reads.

The SQLite database keeps each segment's full stats as JSON (~16 KB per row),
most of which is density maps (xy_dict, z_dict, hull points) that the webapp's
cell-class pages never read. Those pages only need the scalar ``morpho_stats``
and the ``morph_dists`` histograms, so this module writes just those, one
column per value, to a single Parquet file the webapp can load in about a second:

    segment_id                    int64
    stats_method                  string
    has_morpho_stats              bool           (False when morpho_stats is missing)
    morpho_stats.<key>            float64        (null when missing or not numeric)
    morph_dists.<name>.<part>     list<float64>  (e.g. morph_dists.sholl.radii)

Rows are sorted by (segment_id, stats_method). The file metadata carries
``arbor_bulk_format`` so readers can reject layouts they don't understand;
bump BULK_FORMAT_VERSION whenever the layout changes incompatibly.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

BULK_FORMAT_VERSION = "1"
BULK_PARQUET_FILENAME = "arbor_stats_bulk.parquet"
MORPHO_PREFIX = "morpho_stats."
DISTS_PREFIX = "morph_dists."

_BATCH_ROWS = 5000


def _parse_stats_json(value: Any) -> dict:
    if value is None:
        return {}
    if isinstance(value, (bytes, bytearray, memoryview)):
        value = bytes(value).decode("utf-8", errors="replace")
    try:
        payload = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _is_scalar_like(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float))


def _scalar(value: Any) -> float | None:
    if isinstance(value, (bool, int, float)):
        return float(value)
    return None


def _float_list(value: Any) -> list[float] | None:
    if not isinstance(value, list):
        return None
    out = []
    for item in value:
        if isinstance(item, (bool, int, float)):
            out.append(float(item))
        else:
            out.append(math.nan)
    return out


def _iter_rows(conn: sqlite3.Connection) -> Iterator[tuple[int, str, dict]]:
    cursor = conn.execute(
        "SELECT segment_id, stats_method, stats_json FROM arbor_stats "
        "ORDER BY segment_id, stats_method"
    )
    while True:
        batch = cursor.fetchmany(_BATCH_ROWS)
        if not batch:
            return
        for segment_id, stats_method, stats_json in batch:
            yield int(segment_id), str(stats_method), _parse_stats_json(stats_json)


def _collect_columns(conn: sqlite3.Connection) -> tuple[list[str], list[tuple[str, str]]]:
    """First pass: find every morpho_stats key and morph_dists (name, part)."""
    scalar_keys: set[str] = set()
    dist_parts: set[tuple[str, str]] = set()
    for _sid, _method, payload in _iter_rows(conn):
        morpho = payload.get("morpho_stats")
        if isinstance(morpho, dict):
            for key, value in morpho.items():
                if _is_scalar_like(value):
                    scalar_keys.add(str(key))
        dists = payload.get("morph_dists")
        if isinstance(dists, dict):
            for name, parts in dists.items():
                if not isinstance(parts, dict):
                    continue
                for part, value in parts.items():
                    if isinstance(value, list):
                        dist_parts.add((str(name), str(part)))
    return sorted(scalar_keys), sorted(dist_parts)


def _build_schema(scalar_keys: list[str], dist_parts: list[tuple[str, str]], metadata: dict) -> pa.Schema:
    fields = [
        pa.field("segment_id", pa.int64(), nullable=False),
        pa.field("stats_method", pa.string(), nullable=False),
        pa.field("has_morpho_stats", pa.bool_(), nullable=False),
    ]
    fields += [pa.field(f"{MORPHO_PREFIX}{key}", pa.float64()) for key in scalar_keys]
    fields += [
        pa.field(f"{DISTS_PREFIX}{name}.{part}", pa.list_(pa.float64()))
        for name, part in dist_parts
    ]
    return pa.schema(fields, metadata=metadata)


def export_bulk_parquet(sqlite_path: Path, parquet_path: Path) -> dict:
    """
    Write the bulk Parquet file for ``sqlite_path`` to ``parquet_path``.

    The file is written next to the destination and renamed into place, so a
    reader never sees a partial file. Returns a summary dict.
    """
    sqlite_path = Path(sqlite_path)
    parquet_path = Path(parquet_path)
    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite file not found: {sqlite_path}")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    conn = sqlite3.connect(f"{sqlite_path.resolve().as_uri()}?mode=ro", uri=True)
    tmp_path = parquet_path.with_name(parquet_path.name + ".tmp")
    rows = 0
    try:
        scalar_keys, dist_parts = _collect_columns(conn)
        metadata = {
            "arbor_bulk_format": BULK_FORMAT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_sqlite": sqlite_path.name,
        }
        schema = _build_schema(scalar_keys, dist_parts, metadata)

        def _empty_batch() -> dict[str, list]:
            return {name: [] for name in schema.names}

        def _flush(writer: pq.ParquetWriter, batch: dict[str, list]) -> None:
            arrays = [pa.array(batch[f.name], type=f.type) for f in schema]
            writer.write_table(pa.Table.from_arrays(arrays, schema=schema))

        # Format version 2.4 and zstd are readable by pyarrow >= 11 (the webapp's pin).
        with pq.ParquetWriter(tmp_path, schema, compression="zstd", version="2.4") as writer:
            batch = _empty_batch()
            for segment_id, stats_method, payload in _iter_rows(conn):
                morpho = payload.get("morpho_stats")
                if not isinstance(morpho, dict):
                    morpho = None
                dists = payload.get("morph_dists")
                if not isinstance(dists, dict):
                    dists = {}

                batch["segment_id"].append(segment_id)
                batch["stats_method"].append(stats_method)
                batch["has_morpho_stats"].append(morpho is not None)
                for key in scalar_keys:
                    value = morpho.get(key) if morpho is not None else None
                    batch[f"{MORPHO_PREFIX}{key}"].append(_scalar(value))
                for name, part in dist_parts:
                    parts = dists.get(name)
                    value = parts.get(part) if isinstance(parts, dict) else None
                    batch[f"{DISTS_PREFIX}{name}.{part}"].append(_float_list(value))

                rows += 1
                if rows % _BATCH_ROWS == 0:
                    _flush(writer, batch)
                    batch = _empty_batch()
            if batch["segment_id"]:
                _flush(writer, batch)
        os.replace(tmp_path, parquet_path)
    finally:
        conn.close()
        if tmp_path.exists():
            tmp_path.unlink()

    return {
        "path": parquet_path,
        "rows": rows,
        "scalar_columns": len(scalar_keys),
        "dist_columns": len(dist_parts),
        "bytes": parquet_path.stat().st_size,
        "seconds": time.monotonic() - started,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="arborstats-bulk-export",
        description=(
            "Write the compact Parquet file the webapp uses for multi-segment "
            "reads (scalar morpho_stats and morph_dists only)."
        ),
    )
    parser.add_argument("sqlite_path", type=Path, help="Exported arbor_stats SQLite file")
    parser.add_argument(
        "parquet_path",
        type=Path,
        nargs="?",
        default=None,
        help=f"Output path (default: {BULK_PARQUET_FILENAME} next to the SQLite file)",
    )
    args = parser.parse_args(argv)
    parquet_path = args.parquet_path or args.sqlite_path.with_name(BULK_PARQUET_FILENAME)
    summary = export_bulk_parquet(args.sqlite_path, parquet_path)
    print(
        f"Bulk Parquet export: {summary['rows']} rows, "
        f"{summary['scalar_columns']} scalar and {summary['dist_columns']} distribution "
        f"columns, {summary['bytes'] / 1e6:.1f} MB in {summary['seconds']:.1f}s "
        f"-> {summary['path']}"
    )


if __name__ == "__main__":
    main()
