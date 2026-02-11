#!/usr/bin/env python3
"""Filter and rank repredicted designs (AF2 + MPNN) using project settings.
Reads repredict_stats.csv under the design_path, applies filters.json, and writes final_design_stats.csv.
"""
import os
import sys
import argparse
import json
import ast
import pandas as pd
from functions import *


def main():
    parser = argparse.ArgumentParser(description="Rank repredicted designs by filter pass count")
    parser.add_argument("--settings", "-s", required=True, help="Path to basic settings.json")
    parser.add_argument("--filters", "-f", default="./settings_filters/default_filters.json", help="Path to filters.json")
    parser.add_argument("--advanced", "-a", default="./settings_advanced/default_4stage_multimer.json", help="Path to advanced settings json")
    parser.add_argument("--prefilters", "-p", default=None, help="Unused placeholder for compatibility")
    args = parser.parse_args()

    settings_path, filters_path, advanced_path, prefilters_path = perform_input_check(args)
    target_settings, advanced_settings, filters, prefilters = load_json_settings(settings_path, filters_path, advanced_path, prefilters_path)

    design_path = target_settings["design_path"]
    stats_path = os.path.join(design_path, "repredict_stats.csv")
    if not os.path.exists(stats_path):
        print(f"repredict_stats.csv not found at {stats_path}")
        sys.exit(1)

    df = pd.read_csv(stats_path)
    if df.empty:
        print("repredict_stats.csv is empty")
        sys.exit(1)

    _, design_labels, _ = generate_dataframe_labels()

    # Ensure InterfaceAAs columns are dict-like for filter evaluation
    interface_cols = [c for c in df.columns if "InterfaceAAs" in c]
    def _to_dict(val):
        if isinstance(val, dict):
            return val
        if isinstance(val, str):
            try:
                parsed = ast.literal_eval(val)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}
    for col in interface_cols:
        df[col] = df[col].apply(_to_dict)

    scored_rows = []
    for _, row in df.iterrows():
        values = [row.get(label, None) for label in design_labels]
        result = check_filters(values, design_labels, filters)
        if result is True:
            failed_list = []
            passed = len(filters)
        else:
            failed_list = result
            passed = len(filters) - len(failed_list)

        row_out = row.to_dict()
        row_out["passed_filters"] = passed
        row_out["failed_filters"] = ",".join(failed_list) if failed_list else ""
        scored_rows.append(row_out)

    if not scored_rows:
        print("No rows processed from repredict_stats.csv")
        sys.exit(1)

    out_df = pd.DataFrame(scored_rows)
    sort_cols = ["passed_filters"]
    if "Average_i_pTM" in out_df.columns:
        sort_cols.append("Average_i_pTM")
    elif "Average_pLDDT" in out_df.columns:
        sort_cols.append("Average_pLDDT")
    out_df = out_df.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    out_df.insert(0, "Rank", range(1, len(out_df) + 1))

    output_path = os.path.join(design_path, "final_design_stats.csv")
    out_df.to_csv(output_path, index=False)
    print(f"Ranked designs saved to {output_path}")


if __name__ == "__main__":
    main()
