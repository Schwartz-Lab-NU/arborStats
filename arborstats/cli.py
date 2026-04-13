# arborStats/cli.py
from __future__ import annotations
import argparse
from pathlib import Path
import sys
from urllib.parse import urlencode
import pandas as pd

from .runner import export_stats_to_sqlite, process_many

# Nice help formatting: show defaults and keep line breaks
class _Fmt(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass

def _split_csvish(items):
    """Accept ['A', 'B,C'] → ['A','B','C'] or None."""
    if not items:
        return None
    out = []
    for it in items:
        out.extend([x.strip() for x in str(it).split(",") if x.strip()])
    return out or None

def _normalize_dtype_name(name: str) -> str:
    n = name.strip().lower()
    # Friendly aliases → pandas dtypes
    alias = {
        "int": "Int64", "int64": "Int64", "i64": "Int64",
        "float": "float64", "f64": "float64", "float64": "float64",
        "str": "string", "string": "string",
        "bool": "boolean", "boolean": "boolean",
        "cat": "category", "category": "category",
    }
    return alias.get(n, name)

def _parse_dtypes_option(pairs):
    """
    --dtypes 'A=Int64' 'B=string' or --dtypes 'A=Int64,B=string'
    → {'A':'Int64', 'B':'string'}
    """
    if not pairs:
        return None
    mapping = {}
    for item in pairs:
        for tok in str(item).split(","):
            tok = tok.strip()
            if not tok:
                continue
            if "=" not in tok:
                raise SystemExit(f"--dtypes expects COL=DTYPE (got: {tok})")
            col, typ = tok.split("=", 1)
            mapping[col.strip()] = _normalize_dtype_name(typ.strip())
    return mapping or None

def _coerce_seg_id_value(v):
    if pd.isna(v):
        return None
    s = str(v).strip()
    if not s:
        return None
    if s.endswith(".0"):
        s = s[:-2]
    s = s.replace(",", "")
    try:
        return int(s)
    except Exception:
        return None


def _safe_parse_segids(series, name: str) -> list[int]:
    """
    Robustly parse potentially huge integer IDs without float round-off.
    Accepts strings/numbers; ignores blanks; raises on fully missing col.
    """
    values = []
    for v in series.tolist():
        iv = _coerce_seg_id_value(v)
        if iv is not None:
            values.append(iv)
    if not values:
        raise SystemExit(f"No usable segment IDs found in column '{name}'.")
    return values


def _uniform_cell_classes(segids: list[int], args) -> dict[int, str | None] | None:
    cell_class = getattr(args, "cell_class", None)
    if cell_class is None:
        return None
    cell_class = str(cell_class).strip()
    if not cell_class:
        return None
    return {segid: cell_class for segid in segids}


def _read_segids_from_source(args) -> tuple[list[int], dict[int, str | None] | None]:
    """
    Read segids from explicit --segids, or from CSV/Google Sheet with user-provided
    --read-columns and --dtypes. Also honors filters/column names from CLI.
    """
    # 1) Direct segids wins
    if args.segids:
        segids = [int(s) for s in args.segids]
        return segids, _uniform_cell_classes(segids, args)

    usecols = _split_csvish(args.read_columns)   # None or list[str]
    dtypes = _parse_dtypes_option(args.dtypes)   # None or dict[str,str]

    segid_col = args.segid_col

    # 2) Google Sheet
    if args.google_sheet_id:
        params = {"format": "csv"}
        google_sheet_gid = getattr(args, "google_sheet_gid", None)
        if google_sheet_gid:
            params["gid"] = str(google_sheet_gid)
        url = f"https://docs.google.com/spreadsheets/d/{args.google_sheet_id}/export?{urlencode(params)}"
        df = pd.read_csv(url, usecols=usecols, dtype=dtypes)

    # 3) CSV
    elif args.csv:
        df = pd.read_csv(args.csv, usecols=usecols, dtype=dtypes)

    else:
        raise SystemExit("Provide --segids, or --google-sheet-id, or --csv")

    # Optional filtering (only if the columns exist)
    if args.status_filter and args.status_col in df.columns:
        df = df[df[args.status_col].astype("string").isin(set(args.status_filter))]
    if args.cell_review_filter and args.cell_review_col in df.columns:
        df = df[df[args.cell_review_col].astype("string").isin(set(args.cell_review_filter))]

    if segid_col not in df.columns:
        raise SystemExit(
            f"Column '{segid_col}' not found. Available columns: {list(df.columns)}.\n"
            "Use --segid-col to point at the correct column, and --read-columns/--dtypes if needed."
        )

    cell_classes: dict[int, str | None] | None = None
    segids: list[int]
    class_col = getattr(args, "cell_class_col", None)
    if class_col and class_col in df.columns:
        cell_classes = {}
        segids = []
        for _, row in df.iterrows():
            segid_val = _coerce_seg_id_value(row[segid_col])
            if segid_val is None:
                continue
            segids.append(segid_val)
            raw_cell = row[class_col]
            if pd.isna(raw_cell):
                cell_classes[segid_val] = None
            else:
                cell_str = str(raw_cell).strip()
                cell_classes[segid_val] = cell_str if cell_str else None
        if not segids:
            raise SystemExit(f"No usable segment IDs found in column '{segid_col}'.")
    else:
        segids = _safe_parse_segids(df[segid_col], segid_col)
        cell_classes = _uniform_cell_classes(segids, args)

    print(len(segids))
    print(len(list(set(segids))))
    return segids, cell_classes


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="arborstats",
        description="Run flatone + compute arbor statistics per segment.",
        formatter_class=_Fmt,
        epilog=(
            "Notes:\n"
            "  • Sections marked '(mutually exclusive)' mean you may pick at most one option from that section.\n"
            "  • The 'Usage' line above also shows exclusivity with parentheses and the '|' separator.\n"
        ),
    )

    # ---------- Input source (MUTUALLY EXCLUSIVE & required) ----------
    src = p.add_argument_group(
        "Input source (choose exactly one) — mutually exclusive",
        "Exactly one of these must be provided to supply segment IDs."
    )
    gsrc = src.add_mutually_exclusive_group(required=True)
    gsrc.add_argument("--segids", nargs="+", help="one or more segment IDs")
    gsrc.add_argument("--google-sheet-id", help="Google Sheet ID to read")
    gsrc.add_argument("--csv", type=Path, help="CSV path containing segment IDs")
    p.add_argument(
        "--google-sheet-gid",
        "--gid",
        dest="google_sheet_gid",
        help="Google Sheet tab gid to read when using --google-sheet-id",
    )

    # Extra source-related helpers (not mutually exclusive)
    # ---- schema controls for CSV/Sheets ----
    p.add_argument(
        "--read-columns",
        nargs="+",
        default=["Updated Seg ID (Sept 2)", 
                 "Status", 
                 "Cell Requires Review (DO NOT use Updated IDs for those cells)"],
        help="Columns to read from CSV/Sheet (space or comma separated). "
             "Example: --read-columns 'Status' 'Final SegID'"
    )
    p.add_argument(
        "--dtypes",
        nargs="+",
        default=["Updated Seg ID (Sept 2)=Int64", 
                 "Status=string", 
                 "Cell Requires Review (DO NOT use Updated IDs for those cells)=string"],
        metavar="COL=DTYPE",
        help="Per-column dtypes (space or comma separated). "
             "Use pandas dtypes; aliases: int→Int64, str→string, bool→boolean. "
             "Example: --dtypes 'Final SegID=Int64' 'Status=string'"
    )

    # ---- Which column holds the segids & filter column names (customizable) ----
    p.add_argument("--segid-col", 
                   default="Updated Seg ID (Sept 2)", 
                   help="Column containing segment IDs (applies to CSV/Sheet)")
    p.add_argument("--status-col", 
                   default="Status", 
                   help="Column name used for status filtering")
    p.add_argument("--cell-review-col",
                   default="Cell Requires Review (DO NOT use Updated IDs for those cells)", 
                   help="Column name used for cell-review filtering")
    p.add_argument("--cell-class-col",
                   default="Cell Class",
                   help="Column name containing cell class labels (optional)")
    p.add_argument("--cell-class",
                   default=None,
                   help=(
                       "Cell class label to use for every segment ID when "
                       "--cell-class-col is missing or not present in the input"
                   ))

    
    p.add_argument("--status-filter", 
                   nargs="*", 
                   default=[None, "WIP", "Complete", "Complete (cut off)", "Wrong Type", "611 Assigned"],
                   help="Values in the 'Status' column to include when reading a sheet/csv")
    p.add_argument("--cell-review-filter", 
                   nargs="*", 
                   default=["FALSE"],
                   help="Values in the 'Cell Requires Review' column to include when reading a sheet/csv")
    
    # Common options
    p.add_argument("--output-dir", 
                   type=Path, 
                   required=True,
                   help="Root output directory (flatone writes SEG_ID/ here)")
    p.add_argument("-j", "--jobs", 
                   type=int, 
                   default=1, 
                   help="parallel workers")
    p.add_argument(
        "--stats-method",
        choices=("flatone", "morphopy"),
        default="flatone",
        help=(
            "Pick which backend computes arbor statistics. "
            "'flatone' (default) runs flatone + arborStatsFromSkeleton. "
            "'morphopy' runs the in-repo morphopy pipeline using cached global maps."
        ),
    )
    p.add_argument(
        "--export-sqlite",
        action="store_true",
        help=(
            "If set, export per-segment arbor_stats.pkl into a SQLite database "
            "under --output-dir."
        ),
    )

    # ---------- Overwrite policy (MUTUALLY EXCLUSIVE) ----------
    ow = p.add_argument_group(
        "Overwrite policy — mutually exclusive",
        "Choose at most one. If neither is set, existing outputs may be reused."
    )
    og = ow.add_mutually_exclusive_group()
    og.add_argument(
        "--overwrite-all",
        action="store_true",
        help="Compute flatone results and arbor stats even if output exists"
    )
    og.add_argument(
        "--new-only",
        action="store_true",
        help="Compute flatone results and arbor stats only for new segment IDs"
    )

    # ---------- Mode selector (MUTUALLY EXCLUSIVE) ----------
    mode = p.add_argument_group(
        "Which tasks to run — mutually exclusive",
        "Pick at most one. Default is to run both flatone and arbor stats."
    )
    mg = mode.add_mutually_exclusive_group()
    mg.add_argument(
        "--flatone-arbor-stats-both",
        action="store_true",
        help="Run flatone and compute arbor stats (default)"
    )
    mg.add_argument(
        "--arbor-stats-only",
        action="store_true",
        help="Skip flatone; compute arbor stats only (uses existing SWC if present)"
    )
    mg.add_argument(
        "--flatone-only",
        action="store_true",
        help="Run flatone only; skip arbor stats"
    )
    mg.add_argument(
        "--export-only",
        action="store_true",
        help="Skip processing and export existing arbor stats to SQLite"
    )
    
    return p

def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    segids, segid_cell_classes = _read_segids_from_source(args)
    if not segids:
        print("No segment IDs found.", file=sys.stderr)
        sys.exit(2)

    should_export_sqlite = bool(args.export_sqlite or args.export_only)

    if args.export_only and (args.overwrite_all or args.new_only):
        parser.error("--export-only cannot be combined with --overwrite-all or --new-only")

    if not args.export_only:
        # Derive mode (default is both)
        if getattr(args, "arbor_stats_only", False):
            mode = "arbor-only"
        elif getattr(args, "flatone_only", False):
            mode = "flatone-only"
        else:
            mode = "both"  # default or when --flatone-arbor-stats-both is set

        # Map overwrite policy
        overwrite = bool(getattr(args, "overwrite_all", False))
        new_only = bool(getattr(args, "new_only", False))

        if args.stats_method == "morphopy" and mode == "flatone-only":
            parser.error("--stats-method=morphopy cannot be combined with --flatone-only")

        process_many(
            segids,
            args.output_dir,
            overwrite=overwrite,
            jobs=args.jobs,
            mode=mode,
            new_only=new_only,
            stats_method=args.stats_method,
            cell_classes=segid_cell_classes,
        )

    if should_export_sqlite:
        sqlite_path = args.output_dir / "arbor_stats.sqlite3"
        export_stats_to_sqlite(
            segids,
            args.output_dir,
            stats_method=args.stats_method,
            sqlite_path=sqlite_path,
        )

if __name__ == "__main__":
    main()
