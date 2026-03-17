from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
EXISTING_PATH = BASE_DIR / "data" / "heart_unified_clean.csv"
PTB_CORE_PATH = (
    BASE_DIR
    / "raw"
    / "ptb-xl"
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    / "ptbxl_database.csv"
)
PTB_FEATURES_PATH = (
    BASE_DIR
    / "raw"
    / "ptb-xl"
    / "ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1"
    / "features"
    / "ecgdeli_features.csv"
)
CROSSWALK_PATH = (
    BASE_DIR
    / "raw"
    / "ptb-xl"
    / "ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1"
    / "features"
    / "feature_description.csv"
)
OUTPUT_PATH = BASE_DIR / "data" / "heart_unified_v2.csv"


POSITIVE_CODES = {
    "MI",
    "ASMI",
    "ILMI",
    "IMI",
    "IPMI",
    "IPLMI",
    "INJAL",
    "INJAS",
    "INJIL",
    "INJIN",
    "INJLA",
    "LMI",
    "PMI",
    "AMI",
    "ALMI",
    "STTC",
    "LVH",
}


def divider(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_csv_resilient(path: Path, *, force_python: bool = False, **kwargs) -> pd.DataFrame:
    if force_python:
        return pd.read_csv(path, engine="python", **kwargs)

    try:
        return pd.read_csv(path, **kwargs)
    except (pd.errors.ParserError, MemoryError) as exc:
        print(f"Warning: default CSV parser failed for {path.name}: {exc}")
        print("Falling back to engine='python'.")
        return pd.read_csv(path, engine="python", **kwargs)


def print_dataframe_summary(name: str, df: pd.DataFrame) -> None:
    print(f"\n{name}")
    print(f"shape: {df.shape}")
    print(f"columns ({len(df.columns)}):")
    print(list(df.columns))
    print("first 3 rows:")
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        220,
        "display.max_colwidth",
        120,
    ):
        print(df.head(3).to_string(index=False))


def parse_scp_codes(raw_value: str) -> dict[str, float]:
    if pd.isna(raw_value):
        return {}
    parsed = ast.literal_eval(raw_value)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected dict-like scp_codes, got: {type(parsed)!r}")
    return {str(key): float(value) for key, value in parsed.items()}


def label_from_scp_codes(code_map: dict[str, float]) -> tuple[float | None, str, list[str]]:
    high_conf_codes = sorted([code for code, conf in code_map.items() if conf >= 80.0])
    high_conf_set = set(high_conf_codes)
    positive_hits = sorted(high_conf_set & POSITIVE_CODES)

    if positive_hits:
        if high_conf_set == {"NORM"}:
            return 1.0, "positive_label_conflict_only_norm", high_conf_codes
        return 1.0, "positive", high_conf_codes

    if "NORM" in high_conf_set:
        return 0.0, "normal", high_conf_codes

    if not high_conf_codes:
        return None, "no_high_confidence_codes", high_conf_codes

    return None, "non_target_high_confidence_codes", high_conf_codes


def choose_preferred_column(
    feature_columns: list[str],
    candidates: list[str],
) -> str | None:
    for candidate in candidates:
        if candidate in feature_columns:
            return candidate
    return None


def build_feature_mapping(
    crosswalk: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[dict[str, tuple[str | None, Callable[[pd.DataFrame], pd.Series], str]], pd.DataFrame]:
    mapping_rows: list[dict[str, str]] = []
    resolved: dict[str, tuple[str | None, Callable[[pd.DataFrame], pd.Series], str]] = {}

    def query_crosswalk(pattern: str) -> pd.DataFrame:
        return crosswalk[crosswalk["description"].fillna("").str.contains(pattern, case=False, regex=True)]

    heart_rate_hits = query_crosswalk(r"Ventricular rate|Heart rate beat-to-beat|R-R interval \(Mean value during study\)")
    rr_column = "RR_Mean_Global" if "RR_Mean_Global" in feature_columns else None
    if rr_column:
        reason = (
            "Crosswalk has ventricular-rate concepts but no direct ecgdeli rate column in this release; "
            "using RRi_max's current CSV header RR_Mean_Global and converting ms to bpm as 60000 / RR."
        )
        resolved["max_hr"] = (
            rr_column,
            lambda df: np.where(df[rr_column] > 0, 60000.0 / df[rr_column], np.nan),
            reason,
        )
        chosen = rr_column
    else:
        reason = (
            "Warning: no usable RR or ventricular-rate column was found in ecgdeli_features.csv; "
            "max_hr will be NaN for PTB-XL rows."
        )
        resolved["max_hr"] = (None, lambda df: pd.Series(np.nan, index=df.index), reason)
        chosen = "NaN"
    mapping_rows.append(
        {
            "clinical_concept": "Heart rate / ventricular",
            "ecgdeli_column_name_chosen": chosen,
            "reason_for_choice": reason,
        }
    )
    print("\nCrosswalk matches for heart rate / ventricular:")
    print(heart_rate_hits[["unig_feature", "12sl_feature", "ecgdeli_feature", "description"]].to_string(index=False))

    interval_specs = [
        ("qrs_duration", "QRS duration", r"QRS duration$", "QRSd_max", "QRS_Dur_Global"),
        ("pr_interval", "PR interval", r"P-R Interval$", "PRi_max", "PR_Int_Global"),
        ("qt_interval", "QT interval", r"Q-T interval$", "QTi_max", "QT_Int_Global"),
        ("p_duration", "P-wave duration", r"P wave duration$", "PWd_max", "P_Dur_Global"),
    ]
    for target, concept, pattern, short_code, actual_col in interval_specs:
        hits = query_crosswalk(pattern)
        if actual_col in feature_columns:
            reason = (
                f"Crosswalk short code {short_code} maps to the current ecgdeli CSV header {actual_col} "
                "in this release."
            )
            resolved[target] = (
                actual_col,
                lambda df, col=actual_col: df[col],
                reason,
            )
            chosen = actual_col
        else:
            reason = (
                f"Warning: crosswalk identified {short_code} for {concept}, but {actual_col} is missing "
                "from ecgdeli_features.csv; PTB-XL values will be NaN."
            )
            resolved[target] = (None, lambda df: pd.Series(np.nan, index=df.index), reason)
            chosen = "NaN"
        mapping_rows.append(
            {
                "clinical_concept": concept,
                "ecgdeli_column_name_chosen": chosen,
                "reason_for_choice": reason,
            }
        )
        print(f"\nCrosswalk matches for {concept}:")
        print(hits[["unig_feature", "12sl_feature", "ecgdeli_feature", "description"]].to_string(index=False))

    st_hits = query_crosswalk(r"ST elevation/depression")
    st_preference = ["ST_Elev_V5", "ST_Elev_V6", "ST_Elev_V4", "ST_Elev_II", "ST_Elev_aVF", "ST_Elev_I"]
    st_column = choose_preferred_column(feature_columns, st_preference)
    if st_column:
        reason = (
            "Crosswalk points to STc_X, but that short code is absent from this ecgdeli_features.csv. "
            f"Using {st_column} instead, preferring V4-V6 or limb leads as requested, and negating elevation "
            "so positive values represent depression in the unified schema."
        )
        resolved["st_depression"] = (
            st_column,
            lambda df, col=st_column: -df[col],
            reason,
        )
        chosen = st_column
    else:
        reason = (
            "Warning: no ST elevation columns were available for fallback after STc_X was missing; "
            "st_depression will be NaN for PTB-XL rows."
        )
        resolved["st_depression"] = (None, lambda df: pd.Series(np.nan, index=df.index), reason)
        chosen = "NaN"
    mapping_rows.append(
        {
            "clinical_concept": "ST depression or elevation",
            "ecgdeli_column_name_chosen": chosen,
            "reason_for_choice": reason,
        }
    )
    print("\nCrosswalk matches for ST depression or elevation:")
    print(st_hits[["unig_feature", "12sl_feature", "ecgdeli_feature", "description"]].to_string(index=False))

    t_hits = query_crosswalk(r"T wave amplitude")
    t_column = choose_preferred_column(feature_columns, ["T_Amp_II", "T_Amp_V5", "T_Amp_V6", "T_Amp_V4", "T_Amp_I"])
    if t_column:
        reason = (
            "Crosswalk short code TWa_X expands to lead-specific T-wave amplitude columns in this release; "
            f"using {t_column} because it has near-complete coverage and is a standard representative lead."
        )
        resolved["t_amplitude"] = (
            t_column,
            lambda df, col=t_column: df[col],
            reason,
        )
        chosen = t_column
    else:
        reason = (
            "Warning: no T-wave amplitude column was found in ecgdeli_features.csv; "
            "t_amplitude will be NaN for PTB-XL rows."
        )
        resolved["t_amplitude"] = (None, lambda df: pd.Series(np.nan, index=df.index), reason)
        chosen = "NaN"
    mapping_rows.append(
        {
            "clinical_concept": "T-wave amplitude",
            "ecgdeli_column_name_chosen": chosen,
            "reason_for_choice": reason,
        }
    )
    print("\nCrosswalk matches for T-wave amplitude:")
    print(t_hits[["unig_feature", "12sl_feature", "ecgdeli_feature", "description"]].to_string(index=False))

    mapping_table = pd.DataFrame(mapping_rows)
    return resolved, mapping_table


def apply_range_validation(ptb_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int], int]:
    ptb_df = ptb_df.copy()
    under_18_mask = ptb_df["age"] < 18
    under_18_dropped = int(under_18_mask.sum())
    if under_18_dropped:
        print(f"\nDropping {under_18_dropped} PTB-XL rows with age < 18 before final export.")
    ptb_df = ptb_df.loc[~under_18_mask].copy()

    nulled_counts: dict[str, int] = {}

    def null_invalid(column: str, lower: float, upper: float) -> None:
        mask = ptb_df[column].notna() & ((ptb_df[column] < lower) | (ptb_df[column] > upper))
        nulled_counts[column] = int(mask.sum())
        ptb_df.loc[mask, column] = np.nan

    null_invalid("age", 18, 100)
    null_invalid("max_hr", 30, 220)
    null_invalid("bmi", 12, 70)
    null_invalid("qrs_duration", 40, 300)
    null_invalid("pr_interval", 80, 400)
    null_invalid("qt_interval", 200, 700)
    null_invalid("st_depression", -5, 5)

    return ptb_df, nulled_counts, under_18_dropped


def deduplicate_against_existing(ptb_df: pd.DataFrame, existing_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    dedup_keys = ["age", "sex", "max_hr", "st_depression"]
    existing_complete = existing_df.loc[existing_df[dedup_keys].notna().all(axis=1), dedup_keys]
    existing_keys = set(map(tuple, existing_complete.to_numpy()))

    ptb_complete_mask = ptb_df[dedup_keys].notna().all(axis=1)
    ptb_complete_keys = ptb_df.loc[ptb_complete_mask, dedup_keys].apply(lambda row: tuple(row.to_list()), axis=1)
    drop_mask = pd.Series(False, index=ptb_df.index)
    drop_mask.loc[ptb_complete_mask] = ptb_complete_keys.isin(existing_keys).to_numpy()

    rows_dropped = int(drop_mask.sum())
    deduped = ptb_df.loc[~drop_mask].copy()
    return deduped, rows_dropped


def main() -> None:
    divider("PTB-XL INTEGRATION INTO UNIFIED HEART DATASET")

    original_exists = EXISTING_PATH.exists()
    if not original_exists:
        raise FileNotFoundError(f"Missing required file: {EXISTING_PATH}")

    original_sha = file_sha256(EXISTING_PATH)
    original_existing_rows = len(read_csv_resilient(EXISTING_PATH, force_python=True))

    divider("STEP 1 - LOAD AND INSPECT ALL FILES")
    existing = read_csv_resilient(EXISTING_PATH, force_python=True)
    core = read_csv_resilient(PTB_CORE_PATH)
    features = read_csv_resilient(PTB_FEATURES_PATH)
    crosswalk = read_csv_resilient(CROSSWALK_PATH)

    print_dataframe_summary("Existing dataset: data/heart_unified_clean.csv", existing)
    print_dataframe_summary("PTB-XL core: ptbxl_database.csv", core)
    print("\nRaw scp_codes values for rows 0, 1, 2:")
    for idx in [0, 1, 2]:
        print(f"row {idx}: {core.loc[idx, 'scp_codes']}")
    print("\nsex value_counts:")
    print(core["sex"].value_counts(dropna=False).to_string())

    print_dataframe_summary("PTB-XL+ features: ecgdeli_features.csv", features)
    print_dataframe_summary("Feature crosswalk: feature_description.csv", crosswalk)
    print("\nFull feature_description.csv table:")
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        240,
        "display.max_colwidth",
        160,
    ):
        print(crosswalk.to_string(index=False))

    divider("STEP 2 - MAP CLINICAL FEATURES USING CROSSWALK")
    feature_mapping, mapping_table = build_feature_mapping(crosswalk, list(features.columns))
    print("\nMapping table:")
    with pd.option_context("display.max_colwidth", 200, "display.width", 240):
        print(mapping_table.to_string(index=False))

    divider("STEP 3 - PARSE SCP CODES")
    core = core.copy()
    core["parsed_scp_codes"] = core["scp_codes"].apply(parse_scp_codes)
    labels = core["parsed_scp_codes"].apply(label_from_scp_codes)
    core["target"] = [item[0] for item in labels]
    core["label_reason"] = [item[1] for item in labels]
    core["high_conf_codes"] = [item[2] for item in labels]

    kept_mask = core["target"].notna()
    kept_core = core.loc[kept_mask].copy()
    target_counts = kept_core["target"].value_counts().sort_index()
    target_percent = (target_counts / len(kept_core) * 100.0).round(2)
    print("target value_counts:")
    print(target_counts.to_string())
    print("\ntarget percentages:")
    for target_value, count in target_counts.items():
        print(f"target={int(target_value)}: {count} rows ({target_percent.loc[target_value]:.2f}%)")

    dropped_reasons = core.loc[~kept_mask, "label_reason"].value_counts()
    print("\nDropped rows by reason:")
    print(dropped_reasons.to_string())

    positive_rate = float((kept_core["target"] == 1).mean())
    print(f"\nPTB-XL positive rate after target parsing: {positive_rate:.2%}")
    if positive_rate > 0.55 or positive_rate < 0.15:
        raise RuntimeError(
            f"WARNING: positive rate is {positive_rate:.2%}, outside the expected 15%-55% range. "
            "Stopping because the label logic may be wrong."
        )

    divider("STEP 4 - JOIN PTB-XL CORE WITH ECG FEATURES")
    selected_feature_columns = sorted(
        {
            column_name
            for column_name, _, _ in feature_mapping.values()
            if column_name is not None
        }
    )
    features_subset = features[["ecg_id"] + selected_feature_columns].copy()
    ptb_joined = kept_core.merge(features_subset, on="ecg_id", how="left")
    non_null_feature_rows = int(ptb_joined[selected_feature_columns].notna().any(axis=1).sum())
    print(f"Rows with at least one non-null ECG feature after join: {non_null_feature_rows} / {len(ptb_joined)}")

    divider("STEP 5 - HARMONISE COLUMNS TO UNIFIED VOCABULARY")
    original_columns = list(existing.columns)
    new_columns = ["qrs_duration", "pr_interval", "qt_interval", "p_duration", "t_amplitude", "is_ecg_source"]
    harmonised_columns = original_columns + [column for column in new_columns if column not in original_columns]

    ptb_harmonised = pd.DataFrame(index=ptb_joined.index, columns=harmonised_columns, dtype=object)
    for column in harmonised_columns:
        ptb_harmonised[column] = np.nan

    ptb_harmonised["age"] = ptb_joined["age"].astype(float)
    ptb_harmonised["sex"] = ptb_joined["sex"].map({0: 1.0, 1: 0.0}).astype(float)
    ptb_harmonised["source"] = "ptbxl"
    ptb_harmonised["is_ecg_source"] = 1.0
    ptb_harmonised["target"] = ptb_joined["target"].astype(float)

    for target_name, (_, transform, _) in feature_mapping.items():
        ptb_harmonised[target_name] = transform(ptb_joined)

    height_m = ptb_joined["height"] / 100.0
    bmi = np.where(
        ptb_joined["height"].notna() & ptb_joined["weight"].notna() & (height_m > 0),
        ptb_joined["weight"] / np.square(height_m),
        np.nan,
    )
    ptb_harmonised["bmi"] = bmi

    print(f"shape after harmonisation: {ptb_harmonised.shape}")
    print(f"columns after harmonisation ({len(ptb_harmonised.columns)}):")
    print(list(ptb_harmonised.columns))
    print("\nsex value_counts after harmonisation:")
    print(ptb_harmonised["sex"].value_counts(dropna=False).to_string())

    divider("STEP 6 - RANGE VALIDATION")
    ptb_with_aux = pd.concat(
        [
            ptb_harmonised,
            ptb_joined[
                [
                    "ecg_id",
                    "scp_codes",
                    "parsed_scp_codes",
                    "high_conf_codes",
                    "label_reason",
                ]
            ].reset_index(drop=True),
        ],
        axis=1,
    )

    ptb_validated, nulled_counts, under_18_dropped = apply_range_validation(ptb_with_aux)
    print("Count of values nulled per column:")
    for column, count in nulled_counts.items():
        print(f"{column}: {count}")
    print(f"Rows excluded for age < 18: {under_18_dropped}")

    divider("STEP 7 - DEDUPLICATE AGAINST EXISTING DATASET")
    ptb_before_dedup = len(ptb_validated)
    ptb_deduped, rows_dropped = deduplicate_against_existing(ptb_validated, existing)
    print(f"PTB-XL rows before dedup: {ptb_before_dedup}")
    print(f"PTB-XL rows after dedup:  {len(ptb_deduped)}")
    print(f"Rows dropped:             {rows_dropped}")

    divider("STEP 8 - CONCATENATE AND SAVE")
    final_df = pd.concat(
        [existing, ptb_deduped[harmonised_columns]],
        sort=False,
        ignore_index=True,
    )
    final_df.to_csv(OUTPUT_PATH, index=False)

    source_counts = final_df["source"].value_counts(dropna=False)
    target_counts_final = final_df["target"].value_counts().sort_index()
    neg_count = int(target_counts_final.get(0.0, 0))
    pos_count = int(target_counts_final.get(1.0, 0))
    imbalance_ratio = (neg_count / pos_count) if pos_count else np.nan
    missing_over_50 = final_df.columns[final_df.isna().mean() > 0.5].tolist()
    added_columns = sorted(set(final_df.columns) - set(original_columns))

    divider("FINAL DATASET SUMMARY")
    print(f"Total rows: {len(final_df)}")
    print("Rows by source:")
    print(source_counts.to_string())
    print(f"Target class 0: {neg_count} ({neg_count / len(final_df):.2%})")
    print(f"Target class 1: {pos_count} ({pos_count / len(final_df):.2%})")
    print(f"Imbalance ratio (neg/pos): {imbalance_ratio:.4f}")
    print(f"Recommended scale_pos_weight for XGBoost: {imbalance_ratio:.4f}")
    print(f"Total columns: {len(final_df.columns)}")
    print(f"New columns added vs heart_unified_clean.csv: {added_columns}")
    print("Columns with >50% missing:")
    print(missing_over_50)

    divider("STEP 9 - SANITY CHECKS")
    only_norm_high_conf = ptb_deduped["high_conf_codes"].apply(lambda codes: codes == ["NORM"])
    check_1 = not bool(((ptb_deduped["source"] == "ptbxl") & (ptb_deduped["target"] == 1.0) & only_norm_high_conf).any())
    check_2 = bool((final_df.loc[final_df["source"] == "ptbxl", "is_ecg_source"] == 1.0).all())
    sex_values = set(final_df["sex"].dropna().astype(float).unique().tolist())
    check_3 = sex_values.issubset({0.0, 1.0})
    check_4 = bool(((final_df["age"].isna()) | (final_df["age"] >= 18)).all())
    check_5 = (len(existing) + len(ptb_deduped)) == len(final_df)
    current_sha = file_sha256(EXISTING_PATH)
    current_existing_rows = len(read_csv_resilient(EXISTING_PATH, force_python=True))
    check_6 = original_exists and (current_existing_rows == original_existing_rows) and (current_sha == original_sha)

    checks = [
        ("1. No ptbxl target=1 row has NORM as the only high-confidence scp code", check_1),
        ("2. All ptbxl rows have is_ecg_source = 1", check_2),
        ("3. Sex column contains only 0.0 and 1.0", check_3),
        ("4. All rows have age >= 18 or age was explicitly nulled during validation", check_4),
        ("5. Existing rows + deduped ptbxl rows = total rows in v2", check_5),
        ("6. heart_unified_clean.csv still exists and is unchanged", check_6),
    ]
    for label, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'} - {label}")

    print("\n5 PTB-XL rows with the most extreme risk profile:")
    extreme_rows = (
        ptb_deduped.sort_values(["age", "max_hr", "st_depression"], ascending=[False, False, True])
        .loc[:, ["ecg_id", "age", "sex", "max_hr", "st_depression", "qrs_duration", "pr_interval", "qt_interval", "bmi", "target", "scp_codes"]]
        .head(5)
    )
    with pd.option_context("display.max_columns", None, "display.width", 240, "display.max_colwidth", 120):
        print(extreme_rows.to_string(index=False))

    divider("FINAL OUTPUT")
    added_ptb_rows = len(ptb_deduped)
    summary = (
        f"Added {added_ptb_rows} deduplicated PTB-XL rows into data/heart_unified_v2.csv. "
        "The new ECG-driven fields are qrs_duration, pr_interval, qt_interval, p_duration, "
        "t_amplitude, plus PTB-derived max_hr, st_depression, bmi, source, and is_ecg_source "
        "for the imported rows. The combined dataset is now ready for retraining and downstream "
        "calibrated inference, but retraining should account for the new ptbxl source distribution "
        "and for the fact that some PTB-derived ECG features remain missing when the source file "
        "did not provide a valid value or failed range validation."
    )
    print(summary)


if __name__ == "__main__":
    main()
