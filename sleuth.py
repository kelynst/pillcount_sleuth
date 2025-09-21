#!/usr/bin/env python3
"""
sleuth.py — pillcount_sleuth

A tiny medication adherence analyzer for refill CSVs.
- Cleans obvious issues (blank rows, stray whitespace, date parsing)
- Computes per-patient MPR (Medication Possession Ratio)
- Checks optional pill-count spot checks for large discrepancies
- Flags low-adherence or suspicious cases
- Saves cleaned data, summary, and simple charts in ./outputs

CSV expected columns (see README for details):
    patient_id, medication, strength_mg, fill_date, qty_dispensed,
    days_supply, prescribed_daily_dose, count_check_date, observed_pills_remaining

All data is synthetic/demo-only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List

import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Helpers / small utilities
# -------------------------

def _ensure_out(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _coerce_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def _print_section(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


# -------------------------
# Load & clean
# -------------------------

def load_refills(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Drop fully empty rows / cols
    df = df.dropna(how="all").dropna(axis=1, how="all")

    # Trim strings
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]):
            df[c] = df[c].astype("string").str.strip()

    # Coerce types
    numeric_cols = ["strength_mg", "qty_dispensed", "days_supply", "prescribed_daily_dose", "observed_pills_remaining"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Parse dates
    if "fill_date" in df.columns:
        df["fill_date"] = _coerce_date(df["fill_date"])
    if "count_check_date" in df.columns:
        df["count_check_date"] = _coerce_date(df["count_check_date"])

    # Remove duplicates (same patient, med, fill_date)
    before = len(df)
    subset_cols = [c for c in ["patient_id", "medication", "fill_date"] if c in df.columns]
    if subset_cols:
        df = df.drop_duplicates(subset=subset_cols)
    after = len(df)

    if before != after:
        print(f"• Removed {before - after} duplicate rows")

    return df


# ---------------------------------------
# Adherence (MPR) and pill-count checks
# ---------------------------------------

def compute_mpr(refills: pd.DataFrame) -> pd.DataFrame:
    """
    Compute MPR per patient (and per medication). MPR = total days supplied /
    days in observation window. Observation = from first fill to last fill end.

    This is a simple demo MPR; many real-world variants exist.
    """
    df = refills.copy()

    # We’ll group by patient_id + medication (so multiple meds don’t mix)
    group_cols = [c for c in ["patient_id", "medication"] if c in df.columns]
    if not group_cols:
        raise ValueError("Missing required columns: patient_id and/or medication")

    # When any of these are missing, MPR calc will be NaN
    needed = ["fill_date", "days_supply"]
    for need in needed:
        if need not in df.columns:
            raise ValueError(f"Missing required column: {need}")

    # For each group, compute observation window and sum of days_supply
    def _mpr_for_group(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("fill_date")
        first_fill = g["fill_date"].min()
        # Last fill end = last fill_date + last days_supply
        g["fill_end"] = g["fill_date"] + pd.to_timedelta(g["days_supply"].fillna(0), unit="D")
        last_end = g["fill_end"].max()

        # Guard: if any fill_date is NaT, observation could be NaT → handle
        if pd.isna(first_fill) or pd.isna(last_end) or last_end <= first_fill:
            obs_days = pd.NA
        else:
            obs_days = (last_end - first_fill).days

        total_days_supply = g["days_supply"].sum(min_count=1)

        mpr = None
        if pd.notna(total_days_supply) and pd.notna(obs_days) and obs_days > 0:
            mpr = float(total_days_supply) / float(obs_days)
        return pd.Series(
            {
                "first_fill": first_fill,
                "last_fill_end": last_end,
                "obs_days": obs_days,
                "total_days_supply": total_days_supply,
                "mpr": mpr,
                "fills": len(g),
            }
        )

    summary = df.groupby(group_cols, dropna=False, as_index=False).apply(_mpr_for_group).reset_index(drop=True)
    return summary


def check_pillcount_discrepancies(refills: pd.DataFrame, discrepancy_threshold: float = 5.0) -> pd.DataFrame:
    """
    If a count_check_date and observed_pills_remaining exist, estimate what the
    remaining pills *should* be and compute a discrepancy.

    expected_remaining = qty_dispensed - (prescribed_daily_dose * days_since_fill)

    Rows with |discrepancy| > threshold are flagged.
    """
    df = refills.copy()

    needed = ["fill_date", "qty_dispensed", "prescribed_daily_dose", "count_check_date", "observed_pills_remaining"]
    for n in needed:
        if n not in df.columns:
            # If any key column is missing, return empty frame gracefully
            return pd.DataFrame(columns=["patient_id", "medication", "fill_date", "discrepancy", "flag_discrepancy"])

    # Compute days between fill and spot-check
    df["days_since_fill"] = (df["count_check_date"] - df["fill_date"]).dt.days

    # Only consider rows with a valid check date AND observed count
    mask = df["days_since_fill"].notna() & df["observed_pills_remaining"].notna()
    spot = df.loc[mask].copy()

    if spot.empty:
        return pd.DataFrame(columns=["patient_id", "medication", "fill_date", "discrepancy", "flag_discrepancy"])

    # Expected remaining (floored at 0 to avoid negative “should-be”)
    spot["expected_remaining"] = (
        spot["qty_dispensed"].fillna(0) - spot["prescribed_daily_dose"].fillna(0) * spot["days_since_fill"].clip(lower=0)
    ).clip(lower=0)

    spot["discrepancy"] = spot["observed_pills_remaining"] - spot["expected_remaining"]
    spot["flag_discrepancy"] = spot["discrepancy"].abs() > discrepancy_threshold

    return spot[["patient_id", "medication", "fill_date", "count_check_date", "observed_pills_remaining",
                 "expected_remaining", "discrepancy", "flag_discrepancy"]]


# -------------------------
# Plotting (simple & quick)
# -------------------------

def plot_outputs(summary: pd.DataFrame, out_dir: Path) -> List[Path]:
    _ensure_out(out_dir)
    paths: List[Path] = []

    # Histogram of MPR values
    if "mpr" in summary.columns and summary["mpr"].notna().any():
        plt.figure(figsize=(7, 4))
        summary["mpr"].dropna().clip(upper=2.0).hist(bins=20)
        plt.title("MPR Distribution (clipped at 2.0)")
        plt.xlabel("MPR")
        plt.ylabel("Count")
        plt.tight_layout()
        p = out_dir / "mpr_histogram.png"
        plt.savefig(p, dpi=140)
        plt.close()
        paths.append(p)

    # Bar of worst 10 (lowest MPR)
    if "mpr" in summary.columns:
        worst = summary.sort_values("mpr").head(10)
        if not worst.empty:
            label_col = "patient_id"
            if "medication" in worst.columns:
                # combine label for a bit more context
                worst["label"] = worst["patient_id"].astype(str) + " • " + worst["medication"].astype(str)
                label_col = "label"

            plt.figure(figsize=(8, 4))
            plt.barh(worst[label_col], worst["mpr"].fillna(0))
            plt.gca().invert_yaxis()
            plt.xlabel("MPR")
            plt.title("Lowest MPR (Top 10)")
            plt.tight_layout()
            p2 = out_dir / "lowest_mpr_top10.png"
            plt.savefig(p2, dpi=140)
            plt.close()
            paths.append(p2)

    return paths


# -------------------------
# Main “pipeline”
# -------------------------

def run_pipeline(
    input_csv: Path,
    out_dir: Path,
    low_mpr_threshold: float = 0.80,
    discrepancy_threshold: float = 5.0,
    out_clean_name: Optional[str] = None,
    out_summary_name: Optional[str] = None,
    out_flags_name: Optional[str] = None,
) -> None:
    _print_section("Loading data")
    df = load_refills(input_csv)
    print(f"• Rows: {len(df)}  Cols: {len(df.columns)}")
    print(f"• Columns: {list(df.columns)}")

    _print_section("Computing MPR")
    summary = compute_mpr(df)
    print(f"• Summary rows: {len(summary)}")

    # Flag low MPR
    summary["flag_low_mpr"] = summary["mpr"].notna() & (summary["mpr"] < low_mpr_threshold)

    _print_section("Checking pill-count discrepancies")
    discrepancies = check_pillcount_discrepancies(df, discrepancy_threshold=discrepancy_threshold)
    print(f"• Spot-check rows evaluated: {len(discrepancies)}")
    if not discrepancies.empty:
        print(f"• Discrepancy flags (> {discrepancy_threshold} pills): {int(discrepancies['flag_discrepancy'].sum())}")

    # Merge flags together for a single “flags” table
    flags = pd.DataFrame()
    if "flag_low_mpr" in summary.columns and summary["flag_low_mpr"].any():
        low_mpr = summary.loc[summary["flag_low_mpr"], ["patient_id", "medication", "mpr", "obs_days", "fills"]]
        low_mpr = low_mpr.assign(flag="low_mpr")
        flags = pd.concat([flags, low_mpr], ignore_index=True)

    if not discrepancies.empty and discrepancies["flag_discrepancy"].any():
        disc = discrepancies.loc[discrepancies["flag_discrepancy"], [
            "patient_id", "medication", "fill_date", "count_check_date", "discrepancy"
        ]]
        disc = disc.assign(flag="pillcount_discrepancy")
        flags = pd.concat([flags, disc], ignore_index=True)

    # Outputs
    _print_section("Saving outputs")
    _ensure_out(out_dir)

    clean_path = input_csv.with_name(f"cleaned_{input_csv.stem}.csv") if not out_clean_name else Path(out_clean_name)
    summary_path = input_csv.with_name(f"summary_{input_csv.stem}.csv") if not out_summary_name else Path(out_summary_name)
    flags_path = input_csv.with_name(f"flags_{input_csv.stem}.csv") if not out_flags_name else Path(out_flags_name)

    df.to_csv(clean_path, index=False)
    summary.to_csv(summary_path, index=False)
    if not flags.empty:
        flags.to_csv(flags_path, index=False)

    print(f"• Cleaned CSV : {clean_path}")
    print(f"• Summary CSV : {summary_path}")
    if not flags.empty:
        print(f"• Flags CSV   : {flags_path}")
    else:
        print("• Flags CSV   : (none — no rows crossed thresholds)")

    _print_section("Plotting")
    charts = plot_outputs(summary, out_dir)
    if charts:
        for p in charts:
            print(f"• Chart saved : {p}")
    else:
        print("• No charts produced (not enough data)")

    _print_section("Done")
    # friendly footer like your previous projects
    print("✅ Pill Count Analysis Complete")
    print(f"• Input file     : {input_csv.name}")
    n_patients = df["patient_id"].nunique() if "patient_id" in df.columns else "N/A"
    print(f"• Patients       : {n_patients}")
    if "mpr" in summary.columns and summary["mpr"].notna().any():
        print(f"• Avg Adherence  : {summary['mpr'].mean() * 100:,.1f}%")
        print(f"• Non-compliant  : {int((summary['mpr'] < low_mpr_threshold).sum())} rows (MPR < {int(low_mpr_threshold*100)}%)")
    if charts:
        print(f"• Charts dir     : {out_dir}")


# -------------------------
# CLI
# -------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pill count + adherence sleuth on synthetic refill data.")
    p.add_argument("input", help="Path to refill CSV (e.g., demo_refills.csv)")
    p.add_argument("--outputs-dir", default="outputs", help="Where to save charts. Default: outputs")
    p.add_argument("--low-mpr", type=float, default=0.80, help="Flag MPR below this threshold. Default: 0.80")
    p.add_argument("--discrepancy", type=float, default=5.0, help="Flag pill-count discrepancies beyond this many pills.")
    p.add_argument("--out-clean", default="", help="Override cleaned CSV path")
    p.add_argument("--out-summary", default="", help="Override summary CSV path")
    p.add_argument("--out-flags", default="", help="Override flags CSV path")
    return p


def main() -> int:
    ap = build_parser()
    args = ap.parse_args()

    input_csv = Path(args.input)
    out_dir = Path(args.outputs_dir)

    run_pipeline(
        input_csv=input_csv,
        out_dir=out_dir,
        low_mpr_threshold=args.low_mpr,
        discrepancy_threshold=args.discrepancy,
        out_clean_name=(args.out_clean or None),
        out_summary_name=(args.out_summary or None),
        out_flags_name=(args.out_flags or None),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())