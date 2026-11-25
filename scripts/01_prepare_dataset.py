#!/usr/bin/env python3
"""
Step 2: Parse raw SSAH dataset workbook (data/raw/ssah_raw_dataset.xls), construct row-level provenance, align Sheet2 solar
incidence angles & efficiency, deduplicate exact duplicate row, compute physical
derived quantities, and run distribution validation / imbalance audits.
"""

import json
import math
import os
from collections import Counter

import numpy as np
import pandas as pd
import xlrd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_XLS = os.path.join(ROOT, "data", "raw", "ssah_raw_dataset.xls")
XLS_PATH = _DEFAULT_XLS if os.path.exists(_DEFAULT_XLS) else os.path.join(ROOT, "数据合集.xls")
OUT_DATA_DIR = os.path.join(ROOT, "data")
OUT_RES_DIR = os.path.join(ROOT, "results")
os.makedirs(OUT_DATA_DIR, exist_ok=True)
os.makedirs(OUT_RES_DIR, exist_ok=True)

CP_AIR = 1005.0  # J/(kg*K)

SOURCE_META = {
    "Jia2019": {
        "source_id": "Jia2019",
        "citation": "Jia et al. (2019), Chem. Ind. Eng. Prog. 38(2): 819-825 [Ref 27]",
        "data_type": "Numerical (CFD)",
        "collector_area_m2": 1.93 * 0.93,  # 1.7949 m^2 absorber
        "gross_area_m2": 2.0 * 1.0,
        "extraction_method": "Reported tables/curves in Jia et al. (2019)",
        "test_date_or_year": "2019 (CFD)",
    },
    "Jia2021": {
        "source_id": "Jia2021",
        "citation": "Jia et al. (2021), Sol. Energy 225: 969-977 [Ref 30]",
        "data_type": "Numerical (CFD)",
        "collector_area_m2": 1.93 * 0.93,  # 1.7949 m^2 internal channel
        "gross_area_m2": 2.0 * 1.0,
        "extraction_method": "Reported CFD parametric sweep in Jia et al. (2021)",
        "test_date_or_year": "2021 (CFD)",
    },
    "Exp_Authors": {
        "source_id": "Exp_Authors",
        "citation": "Authors' experimental campaign (10 test days: Jan 23 - Mar 12)",
        "data_type": "Experimental",
        "collector_area_m2": 1.93 * 0.93,  # 1.7949 m^2 absorber
        "gross_area_m2": 2.0 * 1.0,
        "extraction_method": "Direct DAQ measurement (Agilent 34972A + TBQ-2 + NK400/fan)",
        "test_date_or_year": "10 winter/spring test days (01.23-03.12)",
    },
    "Amara2023": {
        "source_id": "Amara2023",
        "citation": "Amara & Bouabidi (2023), J. Build. Eng. 65: 105662 [Ref 15]",
        "data_type": "Numerical (3D CFD)",
        "collector_area_m2": 1.50 * 0.75,  # 1.125 m^2 absorber
        "gross_area_m2": 1.50 * 0.75,
        "extraction_method": "Digitized/reconstructed from Amara & Bouabidi (2023) efficiency-flow curves",
        "test_date_or_year": "2023 (CFD)",
    },
    "BenAmara2024": {
        "source_id": "BenAmara2024",
        "citation": "Ben Amara et al. (2024), J. Sol. Energy Eng. 146(3): 031007 [Ref 29]",
        "data_type": "Experimental/CFD",
        "collector_area_m2": 1.50 * 0.75,  # 1.125 m^2 absorber
        "gross_area_m2": 1.50 * 0.75,
        "extraction_method": "Digitized/reconstructed from Ben Amara et al. (2024)",
        "test_date_or_year": "2024 (Exp/CFD)",
    },
}


def map_label_to_source(label: str):
    s = label.strip()
    if "化工进展" in s:
        return "Jia2019", "Jia2019_CFD"
    if "Solar Energy, 2021" in s:
        return "Jia2021", "Jia2021_CFD"
    if "Journal of Building Engineering, 2023" in s:
        return "Amara2023", "Amara2023_CFD"
    if "Gabes Region" in s.replace("\n", " "):
        return "BenAmara2024", "BenAmara2024_ExpCFD"
    # Otherwise author experimental campaign day label
    # Normalize day labels cleanly
    day_map = {
        "实验数据，1.23": "Exp_01_23",
        "实验数据1.25": "Exp_01_25",
        "实验数据1.29": "Exp_01_29",
        "2.5实验数据": "Exp_02_05",
        "2.6实验": "Exp_02_06",
        "2.26实验": "Exp_02_26",
        "3.5数据": "Exp_03_05",
        "3.6数据": "Exp_03_06",
        "3.9数据": "Exp_03_09",
        "3.12数据": "Exp_03_12",
    }
    return "Exp_Authors", day_map.get(s, s)


def main():
    wb = xlrd.open_workbook(XLS_PATH)
    sh1 = wb.sheet_by_index(0)
    sh2 = wb.sheet_by_index(1)

    # Parse Sheet1 provenance blocks
    current_label = str(sh1.cell_value(0, 6)).strip()
    raw_rows = []
    for r in range(1, sh1.nrows):
        lbl = str(sh1.cell_value(r, 6)).strip()
        if lbl:
            current_label = lbl
        tin = float(sh1.cell_value(r, 0))
        h = float(sh1.cell_value(r, 1))
        n_baf = int(round(float(sh1.cell_value(r, 2))))
        m_dot = float(sh1.cell_value(r, 3))
        irrad = float(sh1.cell_value(r, 4))
        tout = float(sh1.cell_value(r, 5))
        src_id, campaign_id = map_label_to_source(current_label)
        eff_sheet2 = float(sh2.cell_value(r + 1, 8))
        raw_rows.append(
            {
                "raw_xls_row": r + 1,  # 1-based Excel row
                "source_id": src_id,
                "campaign_id": campaign_id,
                "raw_label": current_label.replace("\n", " "),
                "data_type": SOURCE_META[src_id]["data_type"],
                "is_experimental": 1 if src_id == "Exp_Authors" else 0,
                "Tin_C": tin,
                "H_m": h,
                "N_baffles": n_baf,
                "m_dot_kg_s": m_dot,
                "I_W_m2": irrad,
                "Tout_C": tout,
                "A_c_m2": SOURCE_META[src_id]["collector_area_m2"],
                "eff_sheet2_raw": eff_sheet2,
            }
        )

    df_raw = pd.DataFrame(raw_rows)
    assert len(df_raw) == 781, f"Expected 781 raw rows, got {len(df_raw)}"

    # Align Sheet2 right-hand table (cols 13..20) by rounded value tuple
    # Note: Sheet2 right table omitted row (Tin=10.244, Tout=44.638, I=520.35), shifting subsequent rows by 1
    right_records = []
    for r in range(2, sh2.nrows):
        v_tin = sh2.cell_value(r, 13)
        if v_tin == "" or v_tin is None:
            continue
        right_records.append(
            {
                "s2_row": r + 1,
                "key": (
                    round(float(sh2.cell_value(r, 13)), 4),
                    round(float(sh2.cell_value(r, 14)), 4),
                    round(float(sh2.cell_value(r, 17)), 6),
                    round(float(sh2.cell_value(r, 19)), 2),
                ),
                "theta_deg": float(sh2.cell_value(r, 18)),
                "used": False,
            }
        )

    thetas = []
    unmatched_exp_rows = []
    for idx, row in df_raw.iterrows():
        k = (
            round(row["Tin_C"], 4),
            round(row["Tout_C"], 4),
            round(row["m_dot_kg_s"], 6),
            round(row["I_W_m2"], 2),
        )
        matched_theta = np.nan
        for rec in right_records:
            if not rec["used"] and rec["key"] == k:
                rec["used"] = True
                matched_theta = rec["theta_deg"]
                break
        if row["is_experimental"] == 1:
            if np.isnan(matched_theta):
                unmatched_exp_rows.append(int(row["raw_xls_row"]))
            thetas.append(matched_theta)
        else:
            # Literature/CFD studies do not report time-varying incidence angle (0 placeholder in Sheet2)
            thetas.append(np.nan)

    df_raw["theta_deg"] = thetas

    # Identify exact duplicates across all 6 core columns
    dup_mask = df_raw.duplicated(
        subset=["Tin_C", "H_m", "N_baffles", "m_dot_kg_s", "I_W_m2", "Tout_C"], keep="first"
    )
    dup_rows = df_raw[dup_mask][["raw_xls_row", "source_id", "Tin_C", "I_W_m2", "Tout_C"]].to_dict(
        orient="records"
    )

    df = df_raw[~dup_mask].copy().reset_index(drop=True)
    df.insert(0, "sample_id", np.arange(1, len(df) + 1))

    # Derived physical features & targets
    df["dT_C"] = df["Tout_C"] - df["Tin_C"]
    df["G_kg_m2_s"] = df["m_dot_kg_s"] / df["A_c_m2"]
    df["eta_abs"] = df["m_dot_kg_s"] * CP_AIR * df["dT_C"] / (df["I_W_m2"] * df["A_c_m2"])
    df["eta_A_m2"] = df["m_dot_kg_s"] * CP_AIR * df["dT_C"] / df["I_W_m2"]

    # Sanity checks
    assert len(df) == 780, f"Expected 780 unique rows, got {len(df)}"
    assert (df["dT_C"] > 0).all(), "Found non-positive dT_C!"
    assert not df[["Tin_C", "H_m", "N_baffles", "m_dot_kg_s", "I_W_m2", "Tout_C", "A_c_m2"]].isna().any().any()

    clean_csv = os.path.join(OUT_DATA_DIR, "ssah_dataset_clean.csv")
    df.to_csv(clean_csv, index=False)

    # Build provenance summary table (by source_id and by campaign_id)
    prov_rows = []
    for src_id in ["Exp_Authors", "Jia2019", "Jia2021", "Amara2023", "BenAmara2024"]:
        sub = df[df["source_id"] == src_id]
        meta = SOURCE_META[src_id]
        prov_rows.append(
            {
                "source_id": src_id,
                "citation": meta["citation"],
                "data_type": meta["data_type"],
                "n_samples": int(len(sub)),
                "pct_total": round(100.0 * len(sub) / len(df), 2),
                "A_c_m2": round(meta["collector_area_m2"], 4),
                "Tin_min": round(float(sub["Tin_C"].min()), 2),
                "Tin_max": round(float(sub["Tin_C"].max()), 2),
                "I_min": round(float(sub["I_W_m2"].min()), 2),
                "I_max": round(float(sub["I_W_m2"].max()), 2),
                "m_dot_min": round(float(sub["m_dot_kg_s"].min()), 5),
                "m_dot_max": round(float(sub["m_dot_kg_s"].max()), 5),
                "H_values_m": ",".join(str(x) for x in sorted(sub["H_m"].unique())),
                "N_values": ",".join(str(int(x)) for x in sorted(sub["N_baffles"].unique())),
                "Tout_min": round(float(sub["Tout_C"].min()), 2),
                "Tout_max": round(float(sub["Tout_C"].max()), 2),
                "dT_min": round(float(sub["dT_C"].min()), 2),
                "dT_max": round(float(sub["dT_C"].max()), 2),
                "eta_abs_min": round(float(sub["eta_abs"].min()), 4),
                "eta_abs_max": round(float(sub["eta_abs"].max()), 4),
                "extraction_method": meta["extraction_method"],
            }
        )
    df_prov = pd.DataFrame(prov_rows)
    df_prov.to_csv(os.path.join(OUT_DATA_DIR, "provenance_summary.csv"), index=False)

    # Per-day experimental breakdown
    exp_days = []
    for cid, sub in df[df["source_id"] == "Exp_Authors"].groupby("campaign_id", sort=False):
        exp_days.append(
            {
                "campaign_id": cid,
                "n_samples": int(len(sub)),
                "Tin_range": [round(float(sub["Tin_C"].min()), 2), round(float(sub["Tin_C"].max()), 2)],
                "I_range": [round(float(sub["I_W_m2"].min()), 1), round(float(sub["I_W_m2"].max()), 1)],
                "m_dot_unique": [round(float(x), 5) for x in sorted(sub["m_dot_kg_s"].unique())],
                "Tout_range": [round(float(sub["Tout_C"].min()), 2), round(float(sub["Tout_C"].max()), 2)],
                "eta_abs_mean": round(float(sub["eta_abs"].mean()), 4),
                "eta_abs_max": round(float(sub["eta_abs"].max()), 4),
                "theta_count_valid": int(sub["theta_deg"].notna().sum()),
                "theta_range": [
                    round(float(sub["theta_deg"].min()), 2),
                    round(float(sub["theta_deg"].max()), 2),
                ],
            }
        )

    # Distribution validation: Experimental (n=694) vs Literature/Numerical (n=86)
    exp_df = df[df["is_experimental"] == 1]
    lit_df = df[df["is_experimental"] == 0]
    dist_tests = {}
    for col in ["Tin_C", "H_m", "N_baffles", "m_dot_kg_s", "I_W_m2", "Tout_C", "dT_C", "eta_abs", "eta_A_m2"]:
        ks_stat, ks_p = stats.ks_2samp(exp_df[col].values, lit_df[col].values)
        mw_stat, mw_p = stats.mannwhitneyu(exp_df[col].values, lit_df[col].values, alternative="two-sided")
        dist_tests[col] = {
            "exp_mean": round(float(exp_df[col].mean()), 5),
            "exp_std": round(float(exp_df[col].std()), 5),
            "exp_median": round(float(exp_df[col].median()), 5),
            "lit_mean": round(float(lit_df[col].mean()), 5),
            "lit_std": round(float(lit_df[col].std()), 5),
            "lit_median": round(float(lit_df[col].median()), 5),
            "ks_stat": round(float(ks_stat), 4),
            "ks_p": float(ks_p),
            "mannwhitney_p": float(mw_p),
        }

    # Joint feature space coverage & imbalance metrics
    geom_counts = Counter(zip(df["H_m"].round(4), df["N_baffles"]))
    geom_summary = [
        {"H_m": float(h), "N_baffles": int(n), "count": int(c), "pct": round(100.0 * c / len(df), 2)}
        for (h, n), c in sorted(geom_counts.items(), key=lambda x: -x[1])
    ]
    probs = np.array([c / len(df) for c in geom_counts.values()])
    shannon_h = float(-np.sum(probs * np.log2(probs)))
    max_h = float(math.log2(len(geom_counts)))

    # Detect transient / high-efficiency points (>0.80)
    high_eff = df[df["eta_abs"] > 0.80][
        ["sample_id", "raw_xls_row", "campaign_id", "Tin_C", "I_W_m2", "m_dot_kg_s", "Tout_C", "eta_abs"]
    ].to_dict(orient="records")

    audit = {
        "raw_rows": int(len(df_raw)),
        "clean_unique_rows": int(len(df)),
        "removed_duplicates": dup_rows,
        "experimental_count": int(len(exp_df)),
        "experimental_pct": round(100.0 * len(exp_df) / len(df), 2),
        "literature_count": int(len(lit_df)),
        "literature_pct": round(100.0 * len(lit_df) / len(df), 2),
        "unmatched_theta_exp_xls_rows": unmatched_exp_rows,
        "valid_theta_count": int(df["theta_deg"].notna().sum()),
        "theta_range_deg": [round(float(df["theta_deg"].min()), 2), round(float(df["theta_deg"].max()), 2)],
        "overall_ranges": {
            col: [round(float(df[col].min()), 5), round(float(df[col].max()), 5), round(float(df[col].mean()), 5), round(float(df[col].std()), 5)]
            for col in ["Tin_C", "H_m", "N_baffles", "m_dot_kg_s", "I_W_m2", "A_c_m2", "Tout_C", "dT_C", "eta_abs"]
        },
        "provenance_by_source": prov_rows,
        "experimental_days": exp_days,
        "distribution_tests_exp_vs_lit": dist_tests,
        "geometry_coverage": geom_summary,
        "unique_geometry_configs": len(geom_counts),
        "geometry_shannon_entropy_bits": round(shannon_h, 4),
        "geometry_max_entropy_bits": round(max_h, 4),
        "geometry_evenness_ratio": round(shannon_h / max_h, 4),
        "high_apparent_eff_gt080": high_eff,
    }

    with open(os.path.join(OUT_RES_DIR, "data_audit.json"), "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

    print(f"Prepared {len(df)} unique samples (removed {len(dup_rows)} exact duplicate).")
    print(f"Experimental: {len(exp_df)} ({100*len(exp_df)/len(df):.2f}%), Literature: {len(lit_df)} ({100*len(lit_df)/len(df):.2f}%).")
    print(f"Unique (H, N) geometries: {len(geom_counts)}; dominant (0.04, 8): {geom_counts[(0.04, 8)]}/{len(df)} ({100*geom_counts[(0.04, 8)]/len(df):.2f}%).")
    print(f"Aligned theta_deg: {df['theta_deg'].notna().sum()} exp rows (missing {len(unmatched_exp_rows)} row: {unmatched_exp_rows}).")
    print(f"Saved: {clean_csv}")


if __name__ == "__main__":
    main()
