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

def _read_segids_from_source(args) -> list[int]:
    if args.segids:
        return [int(s) for s in args.segids]

    if args.google_sheet_id:
        url = f"https://docs.google.com/spreadsheets/d/{args.google_sheet_id}/export?format=csv"
        df = pd.read_csv(url, dtype={"Updated Seg ID (Sept 2)": "Int64", "Status": "string", "Cell Requires Review (DO NOT use Updated IDs for those cells)": "string"},
                         usecols=["Updated Seg ID (Sept 2)", "Status", "Cell Requires Review (DO NOT use Updated IDs for those cells)"])
        df = df.dropna(subset=["Updated Seg ID (Sept 2)"])
        df["Updated Seg ID (Sept 2)"] = df["Updated Seg ID (Sept 2)"].astype("int64")
        if args.status_filter:
            df = df[df["Status"].isin(args.status_filter)]
        if args.cell_review_filter:
            df = df[df["Cell Requires Review (DO NOT use Updated IDs for those cells)"].isin(args.cell_review_filter)]
        return df["Updated Seg ID (Sept 2)"].tolist()

    if args.csv:
        df = pd.read_csv(args.csv)
        col = args.csv_col or "Final SegID"
        return [int(x) for x in df[col].dropna().astype("int64").tolist()]

    raise SystemExit("Provide --segids, or --google-sheet-id, or --csv")

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
    p.add_argument("--csv-col", 
                   default=None, 
                   help="Column name in --csv with segment IDs")
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
