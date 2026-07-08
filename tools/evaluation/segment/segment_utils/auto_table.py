#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple


def default_out_jsonl(tmp_dir: str, model_name: str, task: str) -> str:
    out_dir = os.path.join(os.path.abspath(tmp_dir), "metrics")
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{model_name}_{task}.jsonl"
    return os.path.join(out_dir, fname)


def norm_for_key(x: Any) -> str:
    s = str(x).strip()
    s = s.replace("\t", " ").replace("\n", " ")
    s = s.replace("/", "-")
    s = s.replace(" ", "_")
    return s


def make_key(model_name: str, task: str, dataset: str, metric: str) -> str:
    return "__".join([
        norm_for_key(model_name),
        norm_for_key(task),
        norm_for_key(dataset),
        norm_for_key(metric),
    ])


def to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def round4(v: float) -> float:
    return float(f"{v:.4f}")


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def md_table_two_header_rows(header1: List[str], header2: List[str], rows: List[List[str]]) -> str:
    assert len(header1) == len(header2), "header1/header2 length mismatch"
    lines = []
    lines.append("| " + " | ".join(header1) + " |")
    lines.append("| " + " | ".join(["---"] * len(header1)) + " |")
    lines.append("| " + " | ".join(header2) + " |")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines) + "\n"


def fmt_score(v: Optional[float], nd: int = 1, pct_if_le_1: bool = True) -> str:
    """Format a metric value for compact markdown summaries."""
    if v is None:
        return ""
    try:
        x = float(v)
        if pct_if_le_1 and x <= 1.0:
            x *= 100.0
        s = f"{x:.{nd}f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s
    except Exception:
        return str(v)


def read_single_csv_row(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row:
                return row
    return {}


def pick_task_dataset(row: Dict[str, Any], default_task: str = "Segmentation") -> Tuple[str, str]:
    task = (row.get("task") or default_task).strip() or default_task
    dataset = (row.get("dataset") or "unknown").strip() or "unknown"
    return task, dataset


def detect_csv_type(row: Dict[str, Any]) -> str:
    """
    Detect the metric layout of an auto_table.csv file.

    - type1: panoptic/ADE-style metrics
    - type2: grounded caption segmentation metrics
    - type3: referring/region segmentation metrics
    """
    keys = set(row.keys())
    if "mIoU_conf_matrix" in keys or "PQ_panoptic_all" in keys or "AP_50_95_instance" in keys:
        return "type1"
    if "CAP_METEOR" in keys or "CAP_CIDEr" in keys or "failed_cnt" in keys:
        return "type2"
    if "gIoU" in keys or "cIoU" in keys or "pACC" in keys:
        return "type3"
    return "type1"


TYPE2_KEEP = {"CAP_METEOR", "CAP_CIDEr", "mIoU", "AP50", "failed_cnt"}
TYPE3_KEEP = {"gIoU", "cIoU"}

TYPE1_METRIC_RENAME_MAP = {
    "mIoU_conf_matrix": "mIoU for semantic",
    "PQ_panoptic_all": "PQ for panoptic",
    "AP_50_95_instance": "AP for instance",
}

TYPE2_METRIC_RENAME_MAP = {
    "CAP_METEOR": "METEOR",
    "CAP_CIDEr": "CIDEr",
    "mIoU": "mIoU",
    "AP50": "AP50",
    "failed_cnt": "failed_cnt",
}

TYPE3_METRIC_RENAME_MAP = {
    "gIoU": "gIoU",
    "cIoU": "cIoU",
}

DATASET_RENAME_MAP = {
    "GCG val": "GCG_val",
    "GCG_test": "GCG_test",
    "ade_val": "ade_val",
    "GenSeg": "GenSeg",
    "refcoco_val": "refcoco_val",
    "refcocop_val": "refcocop_val",
    "refcocog_val": "refcocog_val",
    "rea_test": "rea_test",
    "rea_val": "rea_val",
}


def canonicalize_dataset(dataset: str) -> str:
    ds = dataset.strip()
    return DATASET_RENAME_MAP.get(ds, ds)


def rename_metric(raw_metric: str, csv_type: str) -> str:
    raw_metric = raw_metric.strip()
    if csv_type == "type1":
        return TYPE1_METRIC_RENAME_MAP.get(raw_metric, raw_metric)
    if csv_type == "type2":
        return TYPE2_METRIC_RENAME_MAP.get(raw_metric, raw_metric)
    if csv_type == "type3":
        return TYPE3_METRIC_RENAME_MAP.get(raw_metric, raw_metric)
    return raw_metric


def extract_metrics(row: Dict[str, Any], csv_type: str) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for k, v in row.items():
        if k is None:
            continue
        k = k.strip()
        if k == "" or k in {"task", "dataset"}:
            continue

        if csv_type == "type2" and k not in TYPE2_KEEP:
            continue
        if csv_type == "type3" and k not in TYPE3_KEEP:
            continue

        fv = to_float(v)
        if fv is None:
            continue

        canon_metric = rename_metric(k, csv_type)
        out.append((canon_metric, round4(fv)))

    # Stable ordering
    if csv_type == "type2":
        order = ["failed_cnt", "mIoU", "AP50", "METEOR", "CIDEr"]
        out.sort(key=lambda kv: (order.index(kv[0]) if kv[0] in order else 999, kv[0]))
    elif csv_type == "type3":
        order = ["gIoU", "cIoU"]
        out.sort(key=lambda kv: (order.index(kv[0]) if kv[0] in order else 999, kv[0]))
    else:
        out.sort(key=lambda kv: kv[0].lower())

    return out


def parse_auto_table(csv_path: str) -> Optional[Tuple[str, str, str, Dict[str, float]]]:
    row = read_single_csv_row(csv_path)
    if not row:
        return None
    task, dataset = pick_task_dataset(row, default_task="Segmentation")
    dataset = canonicalize_dataset(dataset)
    csv_type = detect_csv_type(row)
    metrics = extract_metrics(row, csv_type)
    m: Dict[str, float] = {k: v for k, v in metrics}
    return task, dataset, csv_type, m


def build_fact_rows_from_parsed(
    csv_path: str,
    model_name: str,
    task: str,
    dataset: str,
    csv_type: str,
    metric_map: Dict[str, float],
    keep_debug_fields: bool = False,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metric, value in metric_map.items():
        key = make_key(model_name, task, dataset, metric)
        r = {
            "Key": key,
            "ModelName": model_name,
            "Task": task,
            "Dataset": dataset,
            "Metric": metric,
            "Value": value,
        }
        if keep_debug_fields:
            r["_src_csv"] = os.path.abspath(csv_path)
            r["_csv_type"] = csv_type
        rows.append(r)
    return rows


def find_seg_auto_tables(tmp_dir: str) -> List[str]:
    root = os.path.abspath(tmp_dir)
    return sorted(glob.glob(os.path.join(root, "**", "auto_table.csv"), recursive=True))


def dedup_by_key(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    uniq: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        uniq[r["Key"]] = r
    return list(uniq.values())


def build_seg_summary_md(
    dataset2m: Dict[str, Dict[str, float]],
    out_md: str,
) -> None:
    """
    Produce a compact markdown summary for the benchmark segmentation metrics.
    """

    def g(ds: str, metric: str) -> Optional[float]:
        return dataset2m.get(ds, {}).get(metric, None)

    gen_miou = g("GenSeg", "mIoU for semantic")
    gen_pq = g("GenSeg", "PQ for panoptic")
    gen_ap = g("GenSeg", "AP for instance")

    ade_miou = g("ade_val", "mIoU for semantic")
    ade_pq = g("ade_val", "PQ for panoptic")
    ade_ap = g("ade_val", "AP for instance")

    ref_c1 = g("refcoco_val", "cIoU")
    ref_c2 = g("refcocop_val", "cIoU")
    ref_c3 = g("refcocog_val", "cIoU")
    ref_cell = "/".join([fmt_score(ref_c1), fmt_score(ref_c2), fmt_score(ref_c3)])

    rea_val = f"{fmt_score(g('rea_val','gIoU'))}/{fmt_score(g('rea_val','cIoU'))}".strip("/")
    rea_test = f"{fmt_score(g('rea_test','gIoU'))}/{fmt_score(g('rea_test','cIoU'))}".strip("/")

    def gcg_cell(ds: str) -> str:
        meteor = g(ds, "METEOR")
        cider = g(ds, "CIDEr")
        ap50 = g(ds, "AP50")
        miou = g(ds, "mIoU")
        return "/".join([fmt_score(meteor), fmt_score(cider), fmt_score(ap50), fmt_score(miou)])

    gcg_val = gcg_cell("GCG_val")
    gcg_test = gcg_cell("GCG_test")

    h1 = [
        "GenSeg", "", "",
        "OVSeg(ADE20K)", "", "",
        "RefSeg",          # 1 col
        "ReaSegval/test", "",  # 2 cols
        "GCG Segval/test", "",  # 2 cols
    ]
    h2 = [
        "mIoU for semantic", "PQ for panoptic", "AP for instance",
        "mIoU for semantic", "PQ for panoptic", "AP for instance",
        "cIoU(refcoco/refcoco+/refcocog)",
        "gIoU/cIoU", "gIoU/cIoU",
        "METEOR/CIDEr/AP50/mIoU", "METEOR/CIDEr/AP50/mIoU",
    ]
    row = [
        fmt_score(gen_miou), fmt_score(gen_pq), fmt_score(gen_ap),
        fmt_score(ade_miou), fmt_score(ade_pq), fmt_score(ade_ap),
        ref_cell,
        rea_val, rea_test,
        gcg_val, gcg_test,
    ]

    os.makedirs(os.path.dirname(os.path.abspath(out_md)), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_table_two_header_rows(h1, h2, [row]))

    print(f"Wrote markdown summary: {out_md}")


def main():
    ap = argparse.ArgumentParser("Export segmentation auto_table.csv files to JSONL and summary markdown.")
    ap.add_argument("--tmp_dir", type=str, required=True)
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument(
        "--out_jsonl",
        type=str,
        default="",
        help="Output JSONL path. Default: <tmp_dir>/metrics/<model>_Segmentation.jsonl",
    )
    ap.add_argument(
        "--out_md",
        type=str,
        default="",
        help="Output summary markdown path. Default: <dirname(out_jsonl)>/summary.md",
    )
    ap.add_argument("--keep_debug_fields", action="store_true", help="Keep _src_csv/_csv_type fields in output")
    args = ap.parse_args()

    csv_paths = find_seg_auto_tables(args.tmp_dir)
    if not csv_paths:
        raise SystemExit(f"No auto_table.csv found under: {os.path.abspath(args.tmp_dir)}")

    all_rows: List[Dict[str, Any]] = []
    dataset2m: Dict[str, Dict[str, float]] = {}

    for p in csv_paths:
        parsed = parse_auto_table(p)
        if not parsed:
            continue
        task, dataset, csv_type, metric_map = parsed

        dataset2m[dataset] = dict(metric_map)

        all_rows.extend(build_fact_rows_from_parsed(
            csv_path=p,
            model_name=args.model_name,
            task=task,
            dataset=dataset,
            csv_type=csv_type,
            metric_map=metric_map,
            keep_debug_fields=args.keep_debug_fields,
        ))

    rows = dedup_by_key(all_rows)

    out_jsonl = args.out_jsonl or default_out_jsonl(
        tmp_dir=args.tmp_dir,
        model_name=args.model_name,
        task="Segmentation",
    )
    write_jsonl(out_jsonl, rows)
    print(f"Wrote {len(rows)} rows to {out_jsonl}")
    print(f"   datasets={sorted(set(r['Dataset'] for r in rows))}")

    out_md = args.out_md or os.path.join(os.path.dirname(os.path.abspath(out_jsonl)), "summary.md")
    build_seg_summary_md(dataset2m, out_md)


if __name__ == "__main__":
    main()
