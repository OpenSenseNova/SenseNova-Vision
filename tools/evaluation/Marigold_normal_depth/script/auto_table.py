#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ============================================================
# Canonical Dataset Definitions (THE ONLY TRUTH)
# ============================================================

DEPTH_DATASET_MAP = {
    "nyu_v2": ["nyu_v2", "nyuv2", "nyu_test", "nyu_depth"],
    "kitti": ["kitti", "kitti_eigen", "kitti_eigen_test"],
    "eth3d": ["eth3d"],
    "scannet": ["scannet"],
    "diode": ["diode"],
}

NORMAL_DATASET_MAP = {
    "scannet": ["scannet"],
    "ibims": ["ibims", "ibims_normals"],
    "nyu": ["nyu"],
}

# ============================================================
# Canonical Metric Definitions
# ============================================================

DEPTH_METRIC_MAP = {
    "abs_relative_difference": "AbsRel",
    "squared_relative_difference": "SqRel",
    "rmse_linear": "RMSE",
    "rmse_log": "RMSE(log)",
    "log10": "Log10",
    "delta1_acc": "δ1",
    "delta2_acc": "δ2",
    "delta3_acc": "δ3",
    "i_rmse": "iRMSE",
    "silog_rmse": "SiLog",
}

NORMAL_METRIC_MAP = {
    "mean_angular_error": "meanErr",
    "median_angular_error": "medianErr",
    "sub5_error": "δ<5",
    "sub7_5_error": "δ<7.5",
    "sub11_25_error": "δ11.25",
    "sub22_5_error": "δ<22.5",
    "sub30_error": "δ<30",
}

# ============================================================
# Markdown Display Spec (what you asked)
# ============================================================

MD_DISPLAY_SPEC = {
    "Depth": [
        # raw_key_in_csv, header_label, arrow
        ("abs_relative_difference", "absrel", "↓"),
        ("delta1_acc", "delta1", "↑"),
    ],
    "Normal": [
        ("mean_angular_error", "meanErr", "↓"),
        ("median_angular_error", "medianErr", "↓"),
        ("sub11_25_error", "delta11.25", "↑"),
        ("sub30_error", "delta30", "↑"),  # Export delta30
    ],
}

# ============================================================
# Utils
# ============================================================

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def make_key(model_name: str, task: str, dataset: str, metric: str) -> str:
    def norm(x: Any) -> str:
        s = str(x).strip()
        s = s.replace("\t", " ").replace("\n", " ")
        s = s.replace("/", "-")
        s = s.replace(" ", "_")
        return s
    return "__".join([norm(model_name), norm(task), norm(dataset), norm(metric)])

def default_out_jsonl(root_dir: str, model_name: str, task: str) -> str:
    metrics_dir = os.path.join(root_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    fname = f"{model_name}_{task}.jsonl"
    return os.path.join(metrics_dir, fname)

def default_out_md(root_dir: str, model_name: str, task: str) -> str:
    metrics_dir = os.path.join(root_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    fname = f"{model_name}_{task}.md"
    return os.path.join(metrics_dir, fname)

def norm_str(s: str) -> str:
    return s.lower().replace("-", "_").replace(" ", "_")

def round4(v: Any) -> Optional[float]:
    try:
        return float(f"{float(v):.4f}")
    except Exception:
        return None

def to_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        s = str(v).strip()
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return None
        return float(s)
    except Exception:
        return None

def read_single_csv_row(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row:
                return row
    return {}

def pick_dataset_raw(row: Dict[str, Any]) -> str:
    return (
        (row.get("dataset_disp_name") or "").strip()
        or (row.get("dataset") or "").strip()
        or "unknown"
    )

def canonicalize_dataset(raw: str, task: str) -> str:
    raw_n = norm_str(raw)
    dataset_map = DEPTH_DATASET_MAP if task == "Depth" else NORMAL_DATASET_MAP

    for canon, aliases in dataset_map.items():
        for a in aliases:
            if a in raw_n:
                return canon

    raise ValueError(f"[DatasetNormError] Unknown dataset: {raw} (task={task})")

def canonicalize_metric(metric_key: str, task: str) -> str:
    if task == "Depth":
        if metric_key not in DEPTH_METRIC_MAP:
            raise ValueError(f"[MetricNormError] Unknown depth metric: {metric_key}")
        return DEPTH_METRIC_MAP[metric_key]
    else:
        if metric_key not in NORMAL_METRIC_MAP:
            raise ValueError(f"[MetricNormError] Unknown normal metric: {metric_key}")
        return NORMAL_METRIC_MAP[metric_key]

def find_auto_table_csvs(root_dir: str) -> List[str]:
    return sorted(
        glob.glob(os.path.join(os.path.abspath(root_dir), "**", "auto_table.csv"), recursive=True)
    )

def postprocess_value(v: float, task: str) -> float:
    """
    Apply task-specific post-processing to metric value.
    - Depth: multiply by 100
    - Normal: keep as-is
    """
    if task == "Depth":
        return v * 100.0
    return v

# ============================================================
# Core Export Logic (JSONL)
# ============================================================

def build_fact_rows_from_csv(
    csv_path: str,
    model_name: str,
    task: str,
) -> List[Dict[str, Any]]:

    row = read_single_csv_row(csv_path)
    if not row:
        return []

    raw_dataset = pick_dataset_raw(row)
    dataset = canonicalize_dataset(raw_dataset, task)

    metric_map = DEPTH_METRIC_MAP if task == "Depth" else NORMAL_METRIC_MAP

    rows: List[Dict[str, Any]] = []
    for raw_metric, canon_metric in metric_map.items():
        if raw_metric not in row:
            continue
        v = round4(row.get(raw_metric))
        if v is None:
            continue
        v = postprocess_value(v, task)
        # Keep four decimal places for precise fact-table storage.
        v = float(f"{v:.4f}")
        key = make_key(model_name, task, dataset, canon_metric)
        rows.append(
            {
                "ModelName": model_name,
                "Task": task,
                "Dataset": dataset,
                "Metric": canon_metric,
                "Value": v,
                "Key": key,
            }
        )

    return rows

def dedup(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    uniq = {}
    for r in rows:
        k = (r["ModelName"], r["Task"], r["Dataset"], r["Metric"])
        uniq[k] = r
    return list(uniq.values())

# ============================================================
# Markdown Export
# ============================================================

@dataclass
class DatasetMDEntry:
    canon_name: str           # Used for sorting
    display_name: str         # Header display text, including parenthesized parameters
    values: Dict[str, Optional[float]]  # raw_metric_key -> value(after postprocess)

def build_md_entries_from_csvs(csv_paths: List[str], task: str) -> List[DatasetMDEntry]:
    # canon_name -> entry (dedup)
    d: Dict[str, DatasetMDEntry] = {}

    display_spec = MD_DISPLAY_SPEC[task]

    for p in csv_paths:
        row = read_single_csv_row(p)
        if not row:
            continue
        raw_dataset = pick_dataset_raw(row)
        canon = canonicalize_dataset(raw_dataset, task)

        vals: Dict[str, Optional[float]] = {}
        for raw_metric_key, _, _ in display_spec:
            fv = to_float(row.get(raw_metric_key))
            if fv is None:
                vals[raw_metric_key] = None
            else:
                vals[raw_metric_key] = postprocess_value(fv, task)

        # If the same canonical dataset appears more than once, keep the latter value.
        d[canon] = DatasetMDEntry(
            canon_name=canon,
            display_name=raw_dataset,
            values=vals,
        )

    # Sort by the canonical key order in the map.
    order = list(DEPTH_DATASET_MAP.keys()) if task == "Depth" else list(NORMAL_DATASET_MAP.keys())
    def sort_key(e: DatasetMDEntry) -> Tuple[int, str]:
        return (order.index(e.canon_name) if e.canon_name in order else 10**9, e.display_name)

    entries = sorted(d.values(), key=sort_key)
    return entries

def fmt_md(v: Optional[float], decimals: int) -> str:
    if v is None:
        return "-"
    return f"{v:.{decimals}f}"

def build_markdown(task: str, model_name: str, entries: List[DatasetMDEntry], md_decimals: int) -> str:
    display_spec = MD_DISPLAY_SPEC[task]
    metric_header = " / ".join([f"{label} {arrow}" for _, label, arrow in display_spec])

    lines: List[str] = []
    lines.append(f"# {model_name} — {task}")
    lines.append("")
    lines.append(f"**{metric_header}**")
    lines.append("")

    # table header
    headers = [e.display_name for e in entries]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    # single values row
    row_cells: List[str] = []
    for e in entries:
        parts: List[str] = []
        for raw_metric_key, _, _ in display_spec:
            parts.append(fmt_md(e.values.get(raw_metric_key), md_decimals))
        row_cells.append("/".join(parts))
    lines.append("| " + " | ".join(row_cells) + " |")
    lines.append("")

    return "\n".join(lines)

# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser("Export depth/normal metrics to Feishu fact-table JSON + Markdown summary")
    ap.add_argument("--root_dir", type=str, required=True)
    ap.add_argument("--task", type=str, choices=["Depth", "Normal"], required=True)
    ap.add_argument("--model_name", type=str, required=True)

    ap.add_argument("--out_jsonl", type=str, default="")
    ap.add_argument("--out_md", type=str, default="")
    ap.add_argument("--md_decimals", type=int, default=2, help="Decimal places in markdown table values (default: 1)")
    ap.add_argument("--no_jsonl", action="store_true", help="Do not write JSONL (only write MD)")

    args = ap.parse_args()

    root_dir = os.path.abspath(args.root_dir)
    metrics_dir = os.path.join(root_dir, "metrics")
    search_dir = metrics_dir if os.path.isdir(metrics_dir) else root_dir

    csv_paths = find_auto_table_csvs(search_dir)
    if not csv_paths:
        raise SystemExit(f"No auto_table.csv found under: {search_dir}")

    # ---- JSONL (original behavior) ----
    if not args.no_jsonl:
        all_rows: List[Dict[str, Any]] = []
        for p in csv_paths:
            all_rows.extend(
                build_fact_rows_from_csv(
                    csv_path=p,
                    model_name=args.model_name,
                    task=args.task,
                )
            )
        final_rows = dedup(all_rows)

        out_jsonl = args.out_jsonl or default_out_jsonl(
            root_dir=root_dir,
            model_name=args.model_name,
            task=args.task,
        )
        write_jsonl(out_jsonl, final_rows)
        print(f"✅ Wrote {len(final_rows)} rows → {out_jsonl}")
        print(f"   task={args.task}, datasets={sorted(set(r['Dataset'] for r in final_rows))}")

    # ---- Markdown (new) ----
    entries = build_md_entries_from_csvs(csv_paths, args.task)
    out_md = args.out_md or default_out_md(
        root_dir=root_dir,
        model_name=args.model_name,
        task=args.task,
    )
    md = build_markdown(
        task=args.task,
        model_name=args.model_name,
        entries=entries,
        md_decimals=args.md_decimals,
    )
    write_text(out_md, md)
    print(f"✅ Wrote markdown → {out_md}")
    print(f"   task={args.task}, md_decimals={args.md_decimals}, columns={[e.display_name for e in entries]}")

if __name__ == "__main__":
    main()
