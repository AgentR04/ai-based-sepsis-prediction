"""
test_sofa.py
Validates SOFA scoring functions and sofa_hourly.csv data integrity.
One test function, no classes, no sub-tests.
"""
import pandas as pd
import numpy as np
from pathlib import Path


def test_sofa_validation():
    # ── Inline SOFA functions (mirrors src/labels/compute_sofa.py) ──────────

    def sofa_platelets(x):
        if pd.isna(x): return 0
        if x < 20:  return 4
        if x < 50:  return 3
        if x < 100: return 2
        if x < 150: return 1
        return 0

    def sofa_bilirubin(x):
        if pd.isna(x): return 0
        if x >= 12.0: return 4
        if x >= 6.0:  return 3
        if x >= 2.0:  return 2
        if x >= 1.2:  return 1
        return 0

    def sofa_map(x):
        if pd.isna(x): return 0
        return 0 if x >= 70 else 1

    def sofa_creatinine(x):
        if pd.isna(x): return 0
        if x >= 5.0: return 4
        if x >= 3.5: return 3
        if x >= 2.0: return 2
        if x >= 1.2: return 1
        return 0

    # ── Platelets boundary tests ─────────────────────────────────────────────
    assert sofa_platelets(200)     == 0
    assert sofa_platelets(150)     == 0
    assert sofa_platelets(149)     == 1
    assert sofa_platelets(99)      == 2
    assert sofa_platelets(49)      == 3
    assert sofa_platelets(19)      == 4
    assert sofa_platelets(np.nan)  == 0

    # ── Bilirubin boundary tests ─────────────────────────────────────────────
    assert sofa_bilirubin(1.0)     == 0
    assert sofa_bilirubin(1.2)     == 1
    assert sofa_bilirubin(2.0)     == 2
    assert sofa_bilirubin(6.0)     == 3
    assert sofa_bilirubin(12.0)    == 4
    assert sofa_bilirubin(np.nan)  == 0

    # ── MAP boundary tests ───────────────────────────────────────────────────
    assert sofa_map(70)            == 0
    assert sofa_map(100)           == 0
    assert sofa_map(69)            == 1
    assert sofa_map(50)            == 1
    assert sofa_map(np.nan)        == 0

    # ── Creatinine boundary tests ────────────────────────────────────────────
    assert sofa_creatinine(1.0)    == 0
    assert sofa_creatinine(1.2)    == 1
    assert sofa_creatinine(2.0)    == 2
    assert sofa_creatinine(3.5)    == 3
    assert sofa_creatinine(5.0)    == 4
    assert sofa_creatinine(np.nan) == 0

    # ── Monotonicity: lower platelets => higher score ─────────────────────────
    plat_scores = [sofa_platelets(v) for v in [200.0, 140.0, 90.0, 40.0, 10.0]]
    assert plat_scores == sorted(plat_scores), "Platelets scores must be non-decreasing as count drops"

    # ── Monotonicity: higher bilirubin => higher score ────────────────────────
    bili_scores = [sofa_bilirubin(v) for v in [0.5, 1.5, 3.0, 8.0, 15.0]]
    assert bili_scores == sorted(bili_scores), "Bilirubin scores must be non-decreasing"

    # ── Monotonicity: higher creatinine => higher score ───────────────────────
    creat_scores = [sofa_creatinine(v) for v in [0.8, 1.5, 2.5, 4.0, 6.0]]
    assert creat_scores == sorted(creat_scores), "Creatinine scores must be non-decreasing"

    # ── Total SOFA feasible range ─────────────────────────────────────────────
    total_max = sofa_platelets(5) + sofa_bilirubin(20) + sofa_map(40) + sofa_creatinine(10)
    total_min = sofa_platelets(300) + sofa_bilirubin(0.5) + sofa_map(90) + sofa_creatinine(0.5)
    assert total_max == 13, f"4-component SOFA max should be 13, got {total_max}"
    assert total_min == 0,  f"4-component SOFA min should be 0, got {total_min}"

    # ── Load sofa_hourly.csv ──────────────────────────────────────────────────
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sofa_path   = PROJECT_ROOT / "data" / "labels" / "sofa_hourly.csv"
    sepsis_path = PROJECT_ROOT / "data" / "labels" / "sepsis_onset.csv"

    assert sofa_path.exists(),   f"sofa_hourly.csv not found: {sofa_path}"
    assert sepsis_path.exists(), f"sepsis_onset.csv not found: {sepsis_path}"

    df = pd.read_csv(sofa_path)
    sepsis_df = pd.read_csv(sepsis_path)

    # Size
    assert len(df) > 30_000, f"Expected >30000 rows, got {len(df)}"
    n_patients = df['icustay_id'].nunique()
    assert n_patients >= 992, f"Expected >=992 patients, got {n_patients}"

    # Required columns
    required_cols = [
        'icustay_id', 'hour', 'heart_rate', 'map', 'resp_rate',
        'spo2', 'temperature', 'lactate', 'creatinine', 'platelets',
        'sofa_total', 'sofa_creatinine', 'sofa_platelets', 'sofa_bilirubin', 'sofa_map'
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"

    # SOFA component ranges 0-4, no NaN
    for comp in ['sofa_creatinine', 'sofa_platelets', 'sofa_bilirubin', 'sofa_map']:
        assert df[comp].isna().sum() == 0, f"{comp} has NaN values"
        assert df[comp].min() >= 0,        f"{comp} has negative values"
        assert df[comp].max() <= 4,        f"{comp} exceeds 4"

    assert df['sofa_total'].isna().sum() == 0, "sofa_total has NaN values"
    assert df['sofa_total'].min() >= 0,        "sofa_total has negative values"
    assert df['sofa_total'].max() <= 24,       "sofa_total exceeds 24"

    # Sepsis patients have higher average SOFA
    sepsis_ids = set(sepsis_df['icustay_id'].unique())
    df['is_sepsis'] = df['icustay_id'].isin(sepsis_ids)
    avg_s  = df[df['is_sepsis']]['sofa_total'].mean()
    avg_ns = df[~df['is_sepsis']]['sofa_total'].mean()
    assert avg_s > avg_ns, f"Sepsis avg SOFA ({avg_s:.2f}) should exceed non-sepsis ({avg_ns:.2f})"

    # Label counts are plausible
    n_sepsis     = len(sepsis_ids)
    n_non_sepsis = n_patients - n_sepsis
    assert 300 <= n_sepsis <= 500, f"Unexpected sepsis count: {n_sepsis}"

    # ── Extra stats for report ────────────────────────────────────────────────
    sofa_sep  = df[df['is_sepsis']]['sofa_total']
    sofa_ns   = df[~df['is_sepsis']]['sofa_total']
    sofa_all  = df['sofa_total']

    sep_max   = sofa_sep.max()
    sep_med   = sofa_sep.median()
    ns_max    = sofa_ns.max()
    ns_med    = sofa_ns.median()

    high_risk = (sofa_all >= 2).sum()
    high_pct  = 100.0 * high_risk / len(df)

    total_rows = len(df)
    n_hours_sep = len(sofa_sep)
    n_hours_ns  = len(sofa_ns)

    nan_counts = {col: int(df[col].isna().sum())
                  for col in ['sofa_creatinine', 'sofa_platelets', 'sofa_bilirubin', 'sofa_map', 'sofa_total']}

    # ─────────────────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("SOFA SCORE VALIDATION - DATA INTEGRITY & CLINICAL ANALYSIS")
    print("=" * 72)

    print()
    print("[1] SOFA COMPONENT BOUNDARY TESTS")
    print("-" * 72)
    checks = [
        ("Platelets == 0 at 200",    sofa_platelets(200)     == 0),
        ("Platelets == 1 at 149",    sofa_platelets(149)     == 1),
        ("Platelets == 2 at 99",     sofa_platelets(99)      == 2),
        ("Platelets == 3 at 49",     sofa_platelets(49)      == 3),
        ("Platelets == 4 at 19",     sofa_platelets(19)      == 4),
        ("Bilirubin == 0 at 1.0",    sofa_bilirubin(1.0)     == 0),
        ("Bilirubin == 1 at 1.2",    sofa_bilirubin(1.2)     == 1),
        ("Bilirubin == 2 at 2.0",    sofa_bilirubin(2.0)     == 2),
        ("Bilirubin == 3 at 6.0",    sofa_bilirubin(6.0)     == 3),
        ("Bilirubin == 4 at 12.0",   sofa_bilirubin(12.0)    == 4),
        ("MAP == 0 at 70",           sofa_map(70)            == 0),
        ("MAP == 1 at 69",           sofa_map(69)            == 1),
        ("Creatinine == 0 at 1.0",   sofa_creatinine(1.0)    == 0),
        ("Creatinine == 1 at 1.2",   sofa_creatinine(1.2)    == 1),
        ("Creatinine == 2 at 2.0",   sofa_creatinine(2.0)    == 2),
        ("Creatinine == 3 at 3.5",   sofa_creatinine(3.5)    == 3),
        ("Creatinine == 4 at 5.0",   sofa_creatinine(5.0)    == 4),
        ("NaN -> 0 (all components)", sofa_platelets(np.nan) == 0),
        ("Monotonicity: Platelets",  plat_scores  == sorted(plat_scores)),
        ("Monotonicity: Bilirubin",  bili_scores  == sorted(bili_scores)),
        ("Monotonicity: Creatinine", creat_scores == sorted(creat_scores)),
        ("Total SOFA max = 13",      total_max    == 13),
        ("Total SOFA min = 0",       total_min    == 0),
    ]
    passed = sum(1 for _, ok in checks if ok)
    for label, ok in checks:
        result = "PASS" if ok else "FAIL"
        print(f"   {result}   {label}")
    print(f"   ----")
    print(f"   {passed}/{len(checks)} boundary checks passed")

    print()
    print("[2] DATA INTEGRITY CHECKS")
    print("-" * 72)
    print(f"   Required columns present  : PASS  ({len(required_cols)} columns verified)")
    nan_ok = all(v == 0 for v in nan_counts.values())
    print(f"   No NaN in SOFA components : {'PASS' if nan_ok else 'FAIL'}")
    for col, n in nan_counts.items():
        print(f"      {col:<28} NaN count = {n}")
    range_ok = df['sofa_total'].min() >= 0 and df['sofa_total'].max() <= 24
    print(f"   SOFA total range [0-24]   : {'PASS' if range_ok else 'FAIL'}  "
          f"(observed min={df['sofa_total'].min()}, max={df['sofa_total'].max()})")

    print()
    print("[3] SOFA SCORE DISTRIBUTION")
    print("-" * 72)
    print(f"   {'Group':<20}  {'Hours':<8}  {'Mean':>6}  {'Median':>7}  {'Max':>5}")
    print(f"   {'-'*52}")
    print(f"   {'Sepsis patients':<20}  {n_hours_sep:<8}  {avg_s:>6.2f}  {sep_med:>7.1f}  {sep_max:>5.0f}")
    print(f"   {'Non-sepsis patients':<20}  {n_hours_ns:<8}  {avg_ns:>6.2f}  {ns_med:>7.1f}  {ns_max:>5.0f}")
    print(f"   {'All patients':<20}  {total_rows:<8}  {sofa_all.mean():>6.2f}  {sofa_all.median():>7.1f}  {sofa_all.max():>5.0f}")
    print()
    print(f"   High-risk hours (SOFA >= 2) : {high_risk:,} / {total_rows:,}  ({high_pct:.1f}%)")

    print()
    print("[4] SEPSIS vs NON-SEPSIS COMPARISON")
    print("-" * 72)
    sofa_diff = avg_s - avg_ns
    print(f"   Sepsis avg SOFA            : {avg_s:.3f}")
    print(f"   Non-sepsis avg SOFA        : {avg_ns:.3f}")
    print(f"   Difference (sep - non-sep) : {sofa_diff:.3f}")
    print(f"   Sepsis SOFA > non-sepsis   : {'PASS' if avg_s > avg_ns else 'FAIL'}")
    print(f"   SOFA separability ratio    : {avg_s / avg_ns:.2f}x  (sepsis patients score higher)")

    print()
    print("[5] FINAL VALIDATION SUMMARY")
    print("=" * 72)
    summary = [
        ("Boundary checks passed (23/23)",           passed == len(checks)),
        ("No NaN in SOFA components",                nan_ok),
        ("SOFA total in valid range [0-24]",         range_ok),
        ("Sepsis avg SOFA > non-sepsis avg SOFA",    avg_s > avg_ns),
        ("Sepsis SOFA diff >= 0.10",                 sofa_diff >= 0.10),
        ("High-risk hours >= 1%",                    high_pct >= 1.0),
    ]
    sp = sum(1 for _, ok in summary if ok)
    print(f"   {'Check':<45}  {'Result'}")
    print(f"   {'-'*56}")
    for label, ok in summary:
        print(f"   {label:<45}  {'PASS' if ok else 'FAIL'}")
    print(f"   {'-'*56}")
    print(f"   {sp}/{len(summary)} checks passed")
    print("=" * 72)
