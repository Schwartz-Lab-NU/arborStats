# arborStats/cli.py
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd

from .runner import process_many

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

def _safe_parse_segids(series, name: str) -> list[int]:
    """
    Robustly parse potentially huge integer IDs without float round-off.
    Accepts strings/numbers; ignores blanks; raises on fully missing col.
    """
    def _coerce(v):
        if pd.isna(v):
            return None
        s = str(v).strip()
        if not s:
            return None
        # Strip common formatting artifacts
        if s.endswith(".0"):
            s = s[:-2]
        s = s.replace(",", "")
        try:
            return int(s)
        except Exception:
            return None

    values = []
    for v in series.tolist():
        iv = _coerce(v)
        if iv is not None:
            values.append(iv)
    if not values:
        raise SystemExit(f"No usable segment IDs found in column '{name}'.")
    return values

def _read_segids_from_source(args) -> list[int]:
    """
    Read segids from explicit --segids, or from CSV/Google Sheet with user-provided
    --read-columns and --dtypes. Also honors filters/column names from CLI.
    """
    # 1) Direct segids wins
    if args.segids:
        return [int(s) for s in args.segids]

    usecols = _split_csvish(args.read_columns)   # None or list[str]
    dtypes = _parse_dtypes_option(args.dtypes)   # None or dict[str,str]

    segid_col = args.segid_col

    # 2) Google Sheet
    if args.google_sheet_id:
        # Export the sheet as CSV; you can add a &gid=... if you need a specific tab.
        url = f"https://docs.google.com/spreadsheets/d/{args.google_sheet_id}/export?format=csv"
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

    segids = _safe_parse_segids(df[segid_col], segid_col)
    print(segids[:15])
    return segids[:15]


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

    
    p.add_argument("--status-filter", 
                   nargs="*", 
                   default=["Complete", "Complete (cut off)"],
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
    
    return p

def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    segids = _read_segids_from_source(args)
    if not segids:
        print("No segment IDs found.", file=sys.stderr)
        sys.exit(2)

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

    process_many(
        segids,
        args.output_dir,
        overwrite=overwrite,
        jobs=args.jobs,
        mode=mode,
        new_only=new_only,
    )

if __name__ == "__main__":
    main()
