# arborStats/sheet_export.py
"""
Export the Morpho-Typer Google Sheet tabs as Parquet files for the webapp.

Without these files, every new webapp instance downloads all three tabs live from
Google Sheets (~25 s) before it can render Morpho-Typer. Reading the Parquet
files from the bucket mount takes well under a second.

Layout (in the output directory, and in the publish directory once published):

    manifest.json        written last; lists the tab files of one complete export
    all_cells.parquet    one file per tab
    all_bcs.parquet
    all_glia.parquet

Each tab file has one string column per sheet column, in sheet order, holding the
cell text exactly as Google's CSV export returns it (empty cells are ""). Header
names are kept as is; a blank or repeated header is renamed to "__column_<n>"
(Parquet needs unique names) and the original headers are stored in the file
metadata under ``morpho_sheet_headers``.

Every file carries ``morpho_sheet_format`` and ``export_id`` in its metadata, and
the manifest lists the same ``export_id``, so a reader can tell whether the tab
files it opened belong to the manifest it read. Bump SHEET_FORMAT_VERSION when
the layout changes incompatibly.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlencode

import pyarrow as pa
import pyarrow.parquet as pq

SHEET_FORMAT_VERSION = "1"
MANIFEST_FILENAME = "manifest.json"
DEFAULT_SHEET_ID = "1o4i53h92oyzsBc8jEWKmF8ZnfyXKXtFCTaYSecs8tBk"

# Tab name -> gid. The names must match the webapp's MORPHO_TYPER_TAB_NAMES.
DEFAULT_TABS = {
    "All Cells": "1690313708",
    "All BCs": "1528488002",
    "All Glia": "2051269861",
}

# Headers the webapp needs in each tab (any one alias per group), mirroring
# _REQUIRED_HEADERS_BY_TAB in the webapp. An export missing them is refused.
REQUIRED_HEADERS = {
    "all cells": (
        ("Cell Class", "CellClass"),
        ("Cell Type", "CellType", "Cell Type (Human)", "Cell Type (human)"),
    ),
    "all bcs": (("Cell Type", "CellType"),),
    "all glia": (("Cell Type", "CellType"),),
}

# Refuse to publish when a tab loses more than this share of its rows compared
# with the published export (e.g. a truncated download or a cleared tab).
DEFAULT_MAX_ROW_DROP = 0.5

_DOWNLOAD_ATTEMPTS = 3
_DOWNLOAD_TIMEOUT_SECONDS = 60


class SheetExportError(RuntimeError):
    pass


@dataclass
class TabExport:
    name: str
    gid: str
    filename: str
    headers: list[str]
    rows: list[list[str]]
    content_sha256: str


def _increase_csv_field_size_limit() -> None:
    """Allow cells larger than the csv module's 128 KiB default."""
    limit = sys.maxsize
    while limit:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


_increase_csv_field_size_limit()


def _tab_filename(tab_name: str) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in tab_name.strip().lower())
    return f"{slug.strip('_') or 'tab'}.parquet"


def _normalize_header(name: str) -> str:
    return "".join(ch for ch in str(name or "").strip().lower() if ch.isalnum())


def _export_url(sheet_id: str, gid: str) -> str:
    query = urlencode({"format": "csv", "gid": gid})
    return f"https://docs.google.com/spreadsheets/d/{quote(sheet_id, safe='')}/export?{query}"


def _download_csv(url: str) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "arborstats-sheet-export"})
            with urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < _DOWNLOAD_ATTEMPTS:
                time.sleep(2 * attempt)
    raise SheetExportError(f"download failed after {_DOWNLOAD_ATTEMPTS} attempts: {last_error}")


def _parse_csv(payload: bytes) -> list[list[str]]:
    text = payload.decode("utf-8-sig")
    return [list(row) for row in csv.reader(io.StringIO(text, newline=""))]


def _check_required_headers(tab_name: str, headers: list[str]) -> None:
    present = {_normalize_header(h) for h in headers}
    for aliases in REQUIRED_HEADERS.get(tab_name.strip().casefold(), ()):
        if not any(_normalize_header(alias) in present for alias in aliases):
            raise SheetExportError(
                f"tab '{tab_name}' has none of the required headers {list(aliases)}"
            )


def fetch_tab(sheet_id: str, tab_name: str, gid: str) -> TabExport:
    payload = _download_csv(_export_url(sheet_id, gid))
    table = _parse_csv(payload)
    if not table or not any(cell.strip() for cell in table[0]):
        raise SheetExportError(f"tab '{tab_name}' (gid {gid}) returned no header row")
    headers = table[0]
    _check_required_headers(tab_name, headers)
    width = len(headers)
    rows = []
    for row in table[1:]:
        if len(row) < width:
            row = row + [""] * (width - len(row))
        elif len(row) > width:
            row = row[:width]
        rows.append(row)
    return TabExport(
        name=tab_name,
        gid=gid,
        filename=_tab_filename(tab_name),
        headers=headers,
        rows=rows,
        content_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _unique_column_names(headers: list[str]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for index, header in enumerate(headers):
        name = header if header.strip() else ""
        if not name or name in seen:
            name = f"__column_{index}"
        seen.add(name)
        names.append(name)
    return names


def _write_tab_parquet(tab: TabExport, path: Path, export_id: str) -> None:
    column_names = _unique_column_names(tab.headers)
    columns = list(zip(*tab.rows)) if tab.rows else [()] * len(column_names)
    arrays = [pa.array(list(values), type=pa.string()) for values in columns]
    schema = pa.schema(
        [pa.field(name, pa.string()) for name in column_names],
        metadata={
            "morpho_sheet_format": SHEET_FORMAT_VERSION,
            "export_id": export_id,
            "tab_name": tab.name,
            "gid": tab.gid,
            "morpho_sheet_headers": json.dumps(tab.headers),
        },
    )
    table = pa.Table.from_arrays(arrays, schema=schema)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        # Format version 2.4 and zstd are readable by pyarrow >= 11 (the webapp's pin).
        pq.write_table(table, tmp_path, compression="zstd", version="2.4")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _write_json(path: Path, data: dict) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        tmp_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def read_manifest(directory: Path) -> dict | None:
    path = Path(directory) / MANIFEST_FILENAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _content_hashes(manifest: dict | None) -> dict[str, str]:
    if not manifest:
        return {}
    return {
        str(tab.get("name")): str(tab.get("content_sha256"))
        for tab in manifest.get("tabs", [])
        if isinstance(tab, dict)
    }


def _check_row_drop(tabs: list[TabExport], published: dict | None, max_row_drop: float) -> None:
    if not published:
        return
    previous_rows = {
        str(tab.get("name")): int(tab.get("rows") or 0)
        for tab in published.get("tabs", [])
        if isinstance(tab, dict)
    }
    for tab in tabs:
        before = previous_rows.get(tab.name, 0)
        if before and len(tab.rows) < before * (1 - max_row_drop):
            raise SheetExportError(
                f"tab '{tab.name}' dropped from {before} to {len(tab.rows)} rows "
                f"(more than {max_row_drop:.0%}); refusing to publish. Use --force if expected."
            )


def _publish(output_dir: Path, publish_dir: Path, manifest: dict) -> None:
    """Copy the tab files, then the manifest, so readers never see a manifest
    that points at files from a different export."""
    publish_dir.mkdir(parents=True, exist_ok=True)
    for tab in manifest["tabs"]:
        shutil.copyfile(output_dir / tab["file"], publish_dir / tab["file"])
    shutil.copyfile(output_dir / MANIFEST_FILENAME, publish_dir / MANIFEST_FILENAME)


def export_sheet(
    sheet_id: str,
    tabs: dict[str, str],
    output_dir: Path,
    publish_dir: Path | None = None,
    *,
    max_row_drop: float = DEFAULT_MAX_ROW_DROP,
    force: bool = False,
) -> dict:
    """
    Download ``tabs`` ({name: gid}) and write them to ``output_dir``; when
    ``publish_dir`` is given, copy them there too. Nothing is written or
    published unless every tab downloads and validates. Returns a summary.
    """
    started = time.monotonic()
    output_dir = Path(output_dir)
    publish_dir = Path(publish_dir) if publish_dir else None
    output_dir.mkdir(parents=True, exist_ok=True)

    fetched = [fetch_tab(sheet_id, name, gid) for name, gid in tabs.items()]

    published = read_manifest(publish_dir) if publish_dir else read_manifest(output_dir)
    unchanged = bool(published) and published.get("format") == SHEET_FORMAT_VERSION and (
        _content_hashes(published) == {tab.name: tab.content_sha256 for tab in fetched}
    )
    if unchanged and not force:
        return {
            "status": "unchanged",
            "export_id": published.get("export_id"),
            "tabs": {tab.name: len(tab.rows) for tab in fetched},
            "seconds": time.monotonic() - started,
        }
    if not force:
        _check_row_drop(fetched, published, max_row_drop)

    exported_at = datetime.now(timezone.utc)
    export_id = exported_at.strftime("%Y%m%dT%H%M%S%fZ")
    for tab in fetched:
        _write_tab_parquet(tab, output_dir / tab.filename, export_id)
    manifest = {
        "format": SHEET_FORMAT_VERSION,
        "export_id": export_id,
        "exported_at": exported_at.isoformat(timespec="seconds"),
        "sheet_id": sheet_id,
        "tabs": [
            {
                "name": tab.name,
                "gid": tab.gid,
                "file": tab.filename,
                "rows": len(tab.rows),
                "columns": len(tab.headers),
                "content_sha256": tab.content_sha256,
            }
            for tab in fetched
        ],
    }
    _write_json(output_dir / MANIFEST_FILENAME, manifest)
    if publish_dir:
        _publish(output_dir, publish_dir, manifest)

    return {
        "status": "published" if publish_dir else "written",
        "export_id": export_id,
        "tabs": {tab.name: len(tab.rows) for tab in fetched},
        "bytes": sum((output_dir / tab.filename).stat().st_size for tab in fetched),
        "seconds": time.monotonic() - started,
    }


def _parse_tab_option(values: list[str] | None) -> dict[str, str]:
    if not values:
        return dict(DEFAULT_TABS)
    tabs: dict[str, str] = {}
    for value in values:
        name, sep, gid = value.rpartition("=")
        if not sep or not name.strip() or not gid.strip().isdigit():
            raise SystemExit(f"--tab expects 'Tab Name=gid', got {value!r}")
        tabs[name.strip()] = gid.strip()
    return tabs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="arborstats-sheet-export",
        description=(
            "Export the Morpho-Typer Google Sheet tabs as Parquet files that the "
            "webapp reads from the bucket instead of downloading the sheet."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Local directory the files are written to first",
    )
    parser.add_argument(
        "--publish-dir", type=Path, default=None,
        help="Directory the finished files are copied to, e.g. the mounted bucket folder",
    )
    parser.add_argument("--google-sheet-id", default=DEFAULT_SHEET_ID, help="Spreadsheet ID")
    parser.add_argument(
        "--tab", action="append", metavar="NAME=GID",
        help="Tab to export as 'Tab Name=gid' (repeatable; default: All Cells, All BCs, All Glia)",
    )
    parser.add_argument(
        "--max-row-drop", type=float, default=DEFAULT_MAX_ROW_DROP,
        help="Refuse to publish if a tab loses more than this share of its rows (default 0.5)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Publish even if nothing changed or a tab lost many rows",
    )
    args = parser.parse_args(argv)

    try:
        summary = export_sheet(
            args.google_sheet_id,
            _parse_tab_option(args.tab),
            args.output_dir,
            args.publish_dir,
            max_row_drop=args.max_row_drop,
            force=args.force,
        )
    except SheetExportError as exc:
        print(f"Sheet export failed: {exc}", file=sys.stderr)
        raise SystemExit(1)

    rows = ", ".join(f"{name}: {count}" for name, count in summary["tabs"].items())
    if summary["status"] == "unchanged":
        print(f"Sheet unchanged since export {summary['export_id']} ({rows}); nothing published "
              f"({summary['seconds']:.1f}s)")
    else:
        where = args.publish_dir if summary["status"] == "published" else args.output_dir
        print(f"Sheet export {summary['export_id']} {summary['status']} to {where}: {rows}, "
              f"{summary['bytes'] / 1e6:.1f} MB in {summary['seconds']:.1f}s")


if __name__ == "__main__":
    main()
