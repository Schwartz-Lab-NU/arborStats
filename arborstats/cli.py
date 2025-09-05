# arborStats/cli.py
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd

from .runner import process_many

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
        description="Run flatone + compute arbor statistics per segment."
    )
    gsrc = p.add_mutually_exclusive_group(required=True)
    gsrc.add_argument("--segids", nargs="+", help="one or more segment IDs")
    gsrc.add_argument("--google-sheet-id", help="Google Sheet ID to read")
    gsrc.add_argument("--csv", type=Path, help="CSV path containing segment IDs")
    p.add_argument("--csv-col", default=None, help="Column name in --csv with segment IDs")
    p.add_argument("--status-filter", nargs="*", default=["Complete", "Complete (cut off)"],
                   help="Values in the 'Status' column to include when reading a sheet/csv")
    p.add_argument("--cell-review-filter", nargs="*", default=["FALSE"],
                   help="Values in the 'Cell Requires Review' column to include when reading a sheet/csv")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Root output directory (flatone writes SEG_ID/ here)")
    p.add_argument("--overwrite", action="store_true", help="Pass --overwrite to flatone")
    p.add_argument("-j", "--jobs", type=int, default=1, help="parallel workers")
    return p

def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    segids = _read_segids_from_source(args)
    if not segids:
        print("No segment IDs found.", file=sys.stderr)
        sys.exit(2)

    process_many(segids, args.output_dir, overwrite=args.overwrite, jobs=args.jobs)

if __name__ == "__main__":
    main()
