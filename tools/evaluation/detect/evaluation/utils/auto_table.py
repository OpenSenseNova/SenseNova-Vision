#!/usr/bin/env python3
import argparse
import glob
import json
import os
from typing import Any, Dict, List, Tuple, Optional


def make_key(model_name: str, task: str, dataset: str, metric: str) -> str:
    def norm(s: str) -> str:
        s = str(s).strip()
        s = s.replace("\t", " ").replace("\n", " ")
        s = s.replace("/", "-")
        s = s.replace(" ", "_")
        return s
    return "__".join([norm(model_name), norm(task), norm(dataset), norm(metric)])


def build_detection_fact_rows(model_name, name2d, cols, task_name: str = "Detection"):
    rows = []
    task = task_name
    for dataset_name, d in name2d.items():
        for metric_key, metric_disp in cols:
            if metric_key not in d:
                continue
            v = d.get(metric_key)
            if v is None:
                continue

            key = make_key(model_name, task, dataset_name, metric_disp)
            rows.append({
                "Key": key,
                "ModelName": model_name,
                "Task": task,
                "Dataset": dataset_name,
                "Metric": metric_disp,
                "Value": float(v),
            })
    return rows


def dataset_name_from_file(path: str) -> str:
    base = os.path.basename(path)
    if base.endswith(".jsonl"):
        base = base[:-5]
    if base.endswith("_metrics"):
        base = base[:-8]
    base = base.rstrip(".")
    return base


def read_single_jsonl_row(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    return {}


def fmt_cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def fmt_pct(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v * 100:.2f}"
    try:
        fv = float(v)
        return f"{fv * 100:.2f}"
    except Exception:
        return str(v)


def fmt_pct1(v: Any) -> str:
    if v is None:
        return ""
    try:
        return f"{float(v) * 100.0:.1f}"
    except Exception:
        return str(v)


def fmt_float(v: Any, nd: int = 1) -> str:
    if v is None:
        return ""
    try:
        fv = float(v)
        s = f"{fv:.{nd}f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s
    except Exception:
        return str(v)


def md_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines) + "\n"


def norm_key(s: str) -> str:
    s = s.strip().rstrip(".")
    s = s.replace("-", "_").replace(" ", "_")
    return s.lower()


def is_variant_key(k: str) -> bool:
    nk = norm_key(k)
    return nk.endswith("_w") or nk.endswith("_nw") or nk.endswith("_n_w") or nk.endswith("_nw")


def find_best_key(
    name2d: Dict[str, Dict[str, Any]],
    must_include: List[str],
    may_include: List[str] = None
) -> Optional[str]:
    may_include = may_include or []
    candidates: List[str] = []
    for k in name2d.keys():
        nk = norm_key(k)
        if any(sub not in nk for sub in must_include):
            continue
        if is_variant_key(k):
            continue
        candidates.append(k)

    if not candidates:
        return None

    def score(k: str) -> Tuple[int, int, int]:
        nk = norm_key(k)
        hit = sum(1 for sub in may_include if sub in nk)
        has_miou = 1 if (isinstance(name2d.get(k, {}), dict) and ("miou_f1" in name2d[k])) else 0
        return (has_miou, hit, -len(nk))

    candidates.sort(key=score, reverse=True)
    return candidates[0]


def md_table_two_header_rows(header1: List[str], header2: List[str], rows: List[List[str]]) -> str:
    assert len(header1) == len(header2), "header1/header2 length mismatch"
    lines = []
    lines.append("| " + " | ".join(header1) + " |")
    lines.append("| " + " | ".join(["---"] * len(header1)) + " |")
    lines.append("| " + " | ".join(header2) + " |")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines) + "\n"


def split_by_prefix(name: str) -> str:
    if name.startswith("point."):
        return "pointing"
    if name.startswith("visual."):
        return "visual_prompting"
    if name.startswith("gui."):
        return "gui"
    if name.startswith("keypoint."):
        return "keypoint"
    return "Detection"


def first_present(d: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in d and d.get(k) is not None:
            return d.get(k)
    return None


def main():
    ap = argparse.ArgumentParser(
        "Collect metrics jsonl into a Markdown file and normalized detection JSONL."
    )
    ap.add_argument("--tmp_dir", type=str, required=True, help="TMP_DIR that contains metrics/ subfolder")
    ap.add_argument("--metrics_subdir", type=str, default="metrics", help="metrics folder name under tmp_dir")
    ap.add_argument("--out_md", type=str, default="", help="output markdown path (default: <tmp_dir>/metrics/summary.md)")
    ap.add_argument("--out_json", type=str, default="", help="output json path for fact-table rows (default: <tmp_dir>/metrics/detection_fact_rows.jsonl)")
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--include_debug", action="store_true", help="include matched-key debug sections in the markdown summary")
    args = ap.parse_args()

    metrics_dir = os.path.join(args.tmp_dir, args.metrics_subdir)
    paths = sorted(glob.glob(os.path.join(metrics_dir, "*_metrics.jsonl")))
    if not paths:
        raise SystemExit(f"No *_metrics.jsonl found in: {metrics_dir}")

    out_md = args.out_md or os.path.join(metrics_dir, "summary.md")
    os.makedirs(os.path.dirname(os.path.abspath(out_md)), exist_ok=True)

    cols: List[Tuple[str, str]] = [
        ("iou05_recall", "R@IoU 0.5"),
        ("iou05_precision", "P@IoU 0.5"),
        ("iou05_f1", "F1@IoU 0.5"),
        ("iou095_recall", "R@IoU 0.95"),
        ("iou095_precision", "P@IoU 0.95"),
        ("iou095_f1", "F1@IoU 0.95"),
        ("miou_recall", "R@ mIoU"),
        ("miou_precision", "P@ mIoU"),
        ("miou_f1", "F1@ mIoU"),
    ]

    # read all
    data_all: List[Tuple[str, Dict[str, Any]]] = []
    for p in paths:
        name = dataset_name_from_file(p)
        d = read_single_jsonl_row(p)
        data_all.append((name, d))

    # split by task prefix
    det_data: List[Tuple[str, Dict[str, Any]]] = []
    point_data: List[Tuple[str, Dict[str, Any]]] = []
    visual_data: List[Tuple[str, Dict[str, Any]]] = []
    gui_data: List[Tuple[str, Dict[str, Any]]] = []
    keypoint_data: List[Tuple[str, Dict[str, Any]]] = []
    for name, d in data_all:
        t = split_by_prefix(name)
        if t == "pointing":
            point_data.append((name, d))
        elif t == "visual_prompting":
            visual_data.append((name, d))
        elif t == "gui":
            gui_data.append((name, d))
        elif t == "keypoint":
            keypoint_data.append((name, d))
        else:
            det_data.append((name, d))

    name2d_det: Dict[str, Dict[str, Any]] = {name: d for name, d in det_data}
    name2d_point: Dict[str, Dict[str, Any]] = {name: d for name, d in point_data}
    name2d_visual: Dict[str, Dict[str, Any]] = {name: d for name, d in visual_data}
    name2d_gui: Dict[str, Dict[str, Any]] = {name: d for name, d in gui_data}
    name2d_keypoint: Dict[str, Dict[str, Any]] = {name: d for name, d in keypoint_data}

    def get_val_by_key(name2d: Dict[str, Dict[str, Any]], k: Optional[str], metric_key: str) -> Any:
        if not k:
            return None
        d = name2d.get(k, {})
        return d.get(metric_key)

    # ---------- 1) Wide table (Detection only, keep original) ----------
    wide_headers = ["name"] + [disp for _, disp in cols]
    wide_rows: List[List[str]] = []
    for name, d in det_data:
        wide_rows.append([name] + [fmt_pct(d.get(key, None)) for key, _ in cols])

    # ---------- 2) Transposed table (Detection only, keep original) ----------
    trans_headers = ["metric"] + [name for name, _ in det_data]
    trans_rows: List[List[str]] = []
    for key, disp in cols:
        row = [disp]
        for name, d in det_data:
            row.append(fmt_pct(d.get(key, None)))
        trans_rows.append(row)

    # ---------- 3) Image-style grouped table (Detection only, keep original) ----------
    k_coco = find_best_key(name2d_det, must_include=["fast", "eval", "coco"], may_include=["coco"])
    k_lvis = find_best_key(name2d_det, must_include=["fast", "eval"], may_include=["lvis", "livs"])
    if k_lvis and not any(x in norm_key(k_lvis) for x in ["lvis", "livs"]):
        k_lvis = find_best_key(name2d_det, must_include=["fast", "eval", "lvis"], may_include=["lvis"])
        if not k_lvis:
            k_lvis = find_best_key(name2d_det, must_include=["fast", "eval", "livs"], may_include=["livs"])

    k_humanref = find_best_key(name2d_det, must_include=["humanref"], may_include=["humanref"])
    k_refcocog_val = find_best_key(name2d_det, must_include=["refcocog", "val"], may_include=["refcocog", "val"])
    k_refcocog_test = find_best_key(name2d_det, must_include=["refcocog", "test"], may_include=["refcocog", "test"])

    k_dense200 = find_best_key(name2d_det, must_include=["dense200"], may_include=["dense200"])
    k_visdrone = find_best_key(name2d_det, must_include=["visdrone"], may_include=["visdrone"])

    k_hiertext = find_best_key(name2d_det, must_include=["hiertext"], may_include=["hiertext"])
    k_sroie = find_best_key(name2d_det, must_include=["sroie"], may_include=["sroie"])

    k_ic15 = find_best_key(name2d_det, must_include=["ic15"], may_include=["ic15", "icdar"])
    k_totaltext = find_best_key(name2d_det, must_include=["totaltext"], may_include=["totaltext"])

    def join_pct(keys: List[Optional[str]]) -> str:
        return " / ".join([fmt_pct(get_val_by_key(name2d_det, k, "miou_f1")) for k in keys])

    img_headers = [
        "Model",
        "COCO_Common",
        "HumanRef / RefCOCOg (val / test)",
        "LVIS",
        "Dense200 / VisDrone",
        "HierText / SROIE",
        "ICDAR2015 / TotalText",
    ]

    img_row = [
        args.model_name,
        fmt_pct(get_val_by_key(name2d_det, k_coco, "miou_f1")),
        join_pct([k_humanref, k_refcocog_val, k_refcocog_test]),
        fmt_pct(get_val_by_key(name2d_det, k_lvis, "miou_f1")),
        join_pct([k_dense200, k_visdrone]),
        join_pct([k_hiertext, k_sroie]),
        join_pct([k_ic15, k_totaltext]),
    ]

    # ---------- 4) Extra: RefExp 3-dataset table (keep original) ----------
    ref_f1_headers = [
        "Model",
        "HumanRef F1@IoU 0.5", "HumanRef F1@IoU 0.95", "HumanRef F1@ mIoU",
        "RefCOCOg val F1@IoU 0.5", "RefCOCOg val F1@IoU 0.95", "RefCOCOg val F1@ mIoU",
        "RefCOCOg test F1@IoU 0.5", "RefCOCOg test F1@IoU 0.95", "RefCOCOg test F1@ mIoU",
    ]
    ref_f1_row = [
        args.model_name,
        fmt_pct(get_val_by_key(name2d_det, k_humanref, "iou05_f1")),
        fmt_pct(get_val_by_key(name2d_det, k_humanref, "iou095_f1")),
        fmt_pct(get_val_by_key(name2d_det, k_humanref, "miou_f1")),
        fmt_pct(get_val_by_key(name2d_det, k_refcocog_val, "iou05_f1")),
        fmt_pct(get_val_by_key(name2d_det, k_refcocog_val, "iou095_f1")),
        fmt_pct(get_val_by_key(name2d_det, k_refcocog_val, "miou_f1")),
        fmt_pct(get_val_by_key(name2d_det, k_refcocog_test, "iou05_f1")),
        fmt_pct(get_val_by_key(name2d_det, k_refcocog_test, "iou095_f1")),
        fmt_pct(get_val_by_key(name2d_det, k_refcocog_test, "miou_f1")),
    ]

    # ---------- 5) Extra: selected multi-dataset table (keep original) ----------
    k_hiertext_textline = find_best_key(
        name2d_det,
        must_include=["hiertext"],
        may_include=["textline", "text_line", "text", "line", "text-line"]
    )
    k_sroie_textline = find_best_key(
        name2d_det,
        must_include=["sroie"],
        may_include=["textline", "text_line", "text", "line", "text-line"]
    )
    k_ic15_word = find_best_key(
        name2d_det,
        must_include=["ic15"],
        may_include=["word", "icdar2015", "icdar", "2015"]
    )
    k_totaltext_word = find_best_key(
        name2d_det,
        must_include=["totaltext"],
        may_include=["word"]
    )

    groups = [
        ("HumanRef", k_humanref),
        ("RefCOCOg val", k_refcocog_val),
        ("RefCOCOg test", k_refcocog_test),
        ("Dense200", k_dense200),
        ("VisDrone", k_visdrone),
        ("HierText-text line", k_hiertext_textline),
        ("ICDAR2015-word", k_ic15_word),
        ("TotalText-word", k_totaltext_word),
        ("SROIE-text line", k_sroie_textline),
    ]

    multi_h1: List[str] = ["Model"]
    multi_h2: List[str] = [""]
    multi_row: List[str] = [args.model_name]
    for ds_name, k in groups:
        multi_h1 += [ds_name, "", ""]
        multi_h2 += ["F1@IoU 0.5", "F1@IoU 0.95", "F1@ mIoU"]
        multi_row += [
            fmt_pct(get_val_by_key(name2d_det, k, "iou05_f1")),
            fmt_pct(get_val_by_key(name2d_det, k, "iou095_f1")),
            fmt_pct(get_val_by_key(name2d_det, k, "miou_f1")),
        ]

    # ---------- 6) NEW: pointing table (requested output) ----------
    # Only F1@Point is populated, equivalent to F1 at IoU=0.5; other columns are empty.
    point_table_md = ""
    point_debug_md = ""
    if point_data:
        def point_f1(k: Optional[str]) -> Any:
            if not k:
                return None
            d = name2d_point.get(k, {})
            return first_present(d, ["iou05_f1", "f1", "F1"])

        pk_coco = find_best_key(name2d_point, must_include=["point", "coco"], may_include=["coco"])
        pk_lvis = find_best_key(name2d_point, must_include=["point", "lvis"], may_include=["lvis", "livs"])
        pk_dense = find_best_key(name2d_point, must_include=["point", "dense200"], may_include=["dense200"])
        pk_vis = find_best_key(name2d_point, must_include=["point", "visdrone"], may_include=["visdrone"])
        pk_human = find_best_key(name2d_point, must_include=["point", "humanref"], may_include=["humanref"])
        pk_ref_val = find_best_key(name2d_point, must_include=["point", "refcocog", "val"], may_include=["refcocog", "val"])
        pk_ref_test = find_best_key(name2d_point, must_include=["point", "refcocog", "test"], may_include=["refcocog", "test"])

        h1 = ["Model", "COCO", "LVIS", "Dense200", "VisDrone", "HumanRef", "RefCOCOg val", "RefCOCOg test"]
        h2 = [""] + ["F1@Point"] * 7
        row = [
            args.model_name,
            fmt_pct(point_f1(pk_coco)),
            fmt_pct(point_f1(pk_lvis)),
            fmt_pct(point_f1(pk_dense)),
            fmt_pct(point_f1(pk_vis)),
            fmt_pct(point_f1(pk_human)),
            fmt_pct(point_f1(pk_ref_val)),
            fmt_pct(point_f1(pk_ref_test)),
        ]
        point_table_md = md_table_two_header_rows(h1, h2, [row])

        point_debug_md = md_table(
            ["group", "picked_dataset_key"],
            [
                ["point.COCO", str(pk_coco)],
                ["point.LVIS", str(pk_lvis)],
                ["point.Dense200", str(pk_dense)],
                ["point.VisDrone", str(pk_vis)],
                ["point.HumanRef", str(pk_human)],
                ["point.RefCOCOg_val", str(pk_ref_val)],
                ["point.RefCOCOg_test", str(pk_ref_test)],
            ]
        )

    # ---------- 7) NEW: visual prompting table (+ MAE) ----------
    visual_table_md = ""
    visual_debug_md = ""
    if visual_data:
        def visual_vals(k: Optional[str]) -> Tuple[Any, Any, Any, Any]:
            if not k:
                return (None, None, None, None)
            d = name2d_visual.get(k, {})
            f05 = first_present(d, ["iou05_f1", "f1_iou05", "f1_05"])
            f095 = first_present(d, ["iou095_f1", "f1_iou095", "f1_095"])
            fmiou = first_present(d, ["miou_f1", "f1_miou"])
            mae = first_present(d, ["mae", "MAE", "mask_mae", "mean_abs_error"])
            return (f05, f095, fmiou, mae)

        vk_fsc = find_best_key(
            name2d_visual,
            must_include=["visual", "fscd"],
            may_include=["fsc147", "fscd", "test"]
        )
        if not vk_fsc:
            vk_fsc = find_best_key(
                name2d_visual,
                must_include=["visual", "fsc147"],
                may_include=["fsc147", "test"]
            )
        vk_dense = find_best_key(name2d_visual, must_include=["visual", "dense200"], may_include=["dense200"])
        vk_coco = find_best_key(name2d_visual, must_include=["visual", "coco"], may_include=["coco"])
        vk_lvis = find_best_key(name2d_visual, must_include=["visual", "lvis"], may_include=["lvis", "livs"])

        h1 = ["Model"]
        h2 = [""]

        def add_group(name: str):
            nonlocal h1, h2
            h1 += [name, "", "", ""]
            h2 += ["F1@0.5", "F1@0.95", "F1@ mIoU", "MAE"]

        add_group("FSC147-test")
        add_group("Dense200")
        add_group("COCO")
        add_group("LVIS")

        row = [args.model_name]
        for k in [vk_fsc, vk_dense, vk_coco, vk_lvis]:
            f05, f095, fmiou, mae = visual_vals(k)
            row += [fmt_pct(f05), fmt_pct(f095), fmt_pct(fmiou), fmt_float(mae, nd=1)]

        visual_table_md = md_table_two_header_rows(h1, h2, [row])
        visual_debug_md = md_table(
            ["group", "picked_dataset_key"],
            [
                ["visual.FSCD_test / visual.FSC147", str(vk_fsc)],
                ["visual.Dense200", str(vk_dense)],
                ["visual.COCO", str(vk_coco)],
                ["visual.LVIS", str(vk_lvis)],
            ]
        )

    # ---------- 8) GUI table ----------
    gui_table_md = ""
    gui_debug_md = ""
    if gui_data:
        def gui_acc(k: Optional[str]) -> Any:
            if not k:
                return None
            d = name2d_gui.get(k, {})
            return first_present(d, ["gui_acc", "miou_f1", "iou05_f1", "accuracy", "acc"])

        def weighted_gui_acc(keys: List[Optional[str]]) -> Any:
            weighted = 0.0
            total = 0.0
            for key in keys:
                if not key:
                    continue
                d = name2d_gui.get(key, {})
                acc = gui_acc(key)
                samples = d.get("samples")
                if acc is None or samples is None:
                    continue
                weighted += float(acc) * float(samples)
                total += float(samples)
            return weighted / total if total > 0 else None

        screenspot_v2_groups = [
            ("Mobile Text", find_best_key(name2d_gui, ["gui", "mobile", "text"], ["screenspot"])),
            ("Mobile Icon", find_best_key(name2d_gui, ["gui", "mobile", "icon"], ["screenspot"])),
            ("Desktop Text", find_best_key(name2d_gui, ["gui", "desktop", "text"], ["screenspot"])),
            ("Desktop Icon", find_best_key(name2d_gui, ["gui", "desktop", "icon"], ["screenspot"])),
            ("Web Text", find_best_key(name2d_gui, ["gui", "web", "text"], ["screenspot"])),
            ("Web Icon", find_best_key(name2d_gui, ["gui", "web", "icon"], ["screenspot"])),
        ]
        screenspotpro_groups = [
            ("Dev. Text", find_best_key(name2d_gui, ["gui", "screenspotpro", "dev", "text"], ["screenspotpro"])),
            ("Dev. Icon", find_best_key(name2d_gui, ["gui", "screenspotpro", "dev", "icon"], ["screenspotpro"])),
            ("Creative Text", find_best_key(name2d_gui, ["gui", "screenspotpro", "creative", "text"], ["screenspotpro"])),
            ("Creative Icon", find_best_key(name2d_gui, ["gui", "screenspotpro", "creative", "icon"], ["screenspotpro"])),
            ("CAD Text", find_best_key(name2d_gui, ["gui", "screenspotpro", "cad", "text"], ["screenspotpro"])),
            ("CAD Icon", find_best_key(name2d_gui, ["gui", "screenspotpro", "cad", "icon"], ["screenspotpro"])),
            ("Sci. Text", find_best_key(name2d_gui, ["gui", "screenspotpro", "sci", "text"], ["screenspotpro"])),
            ("Sci. Icon", find_best_key(name2d_gui, ["gui", "screenspotpro", "sci", "icon"], ["screenspotpro"])),
            ("Office Text", find_best_key(name2d_gui, ["gui", "screenspotpro", "office", "text"], ["screenspotpro"])),
            ("Office Icon", find_best_key(name2d_gui, ["gui", "screenspotpro", "office", "icon"], ["screenspotpro"])),
            ("OS Text", find_best_key(name2d_gui, ["gui", "screenspotpro", "os", "text"], ["screenspotpro"])),
            ("OS Icon", find_best_key(name2d_gui, ["gui", "screenspotpro", "os", "icon"], ["screenspotpro"])),
        ]
        gui_groups = screenspot_v2_groups + [("ScreenSpot-V2 Avg", None)] + screenspotpro_groups + [("ScreenSpotPro Avg", None)]
        gui_h1 = ["Model"] + ["ScreenSpot-V2"] + [""] * 6 + ["ScreenSpotPro"] + [""] * 12
        gui_h2 = [""] + [label for label, _ in screenspot_v2_groups] + ["Avg"] + [label for label, _ in screenspotpro_groups] + ["Avg"]
        gui_row = [args.model_name]
        gui_row += [fmt_pct1(gui_acc(key)) for _, key in screenspot_v2_groups]
        gui_row.append(fmt_pct1(weighted_gui_acc([key for _, key in screenspot_v2_groups])))
        gui_row += [fmt_pct1(gui_acc(key)) for _, key in screenspotpro_groups]
        gui_row.append(fmt_pct1(weighted_gui_acc([key for _, key in screenspotpro_groups])))
        gui_table_md = md_table_two_header_rows(gui_h1, gui_h2, [gui_row])
        gui_debug_md = md_table(
            ["group", "picked_dataset_key"],
            [[label, str(key)] for label, key in gui_groups],
        )

    # ---------- 9) keypoint table ----------
    keypoint_table_md = ""
    keypoint_debug_md = ""
    if keypoint_data:
        def keypoint_vals(k: Optional[str]) -> Tuple[Any, Any, Any, Any]:
            if not k:
                return (None, None, None, None)
            d = name2d_keypoint.get(k, {})
            f05 = first_present(d, ["keypoint_f1_50", "iou05_f1", "F1@0.50"])
            f095 = first_present(d, ["keypoint_f1_95", "iou095_f1", "F1@0.95"])
            fmoks = first_present(d, ["keypoint_f1_moks", "miou_f1", "F1@mOKS"])
            oks = first_present(d, ["avg_oks", "keypoint_avg_oks"])
            return (f05, f095, fmoks, oks)

        kp_coco = find_best_key(name2d_keypoint, ["keypoint", "coco"], ["coco"])
        kp_ap10k = find_best_key(name2d_keypoint, ["keypoint", "ap"], ["10k", "ap-10k"])
        keypoint_h1 = ["Model", "COCO-Keypoints", "", "", "AP10k-Keypoints", "", ""]
        keypoint_h2 = ["", "F1@OKS 0.5", "F1@OKS 0.95", "F1@OKS mOKS", "F1@OKS 0.5", "F1@OKS 0.95", "F1@OKS mOKS"]
        keypoint_row = [args.model_name]
        for key in [kp_coco, kp_ap10k]:
            f05, f095, fmoks, _ = keypoint_vals(key)
            keypoint_row += [fmt_pct(f05), fmt_pct(f095), fmt_pct(fmoks)]
        keypoint_table_md = md_table_two_header_rows(keypoint_h1, keypoint_h2, [keypoint_row])
        keypoint_debug_md = md_table(
            ["group", "picked_dataset_key"],
            [["COCO-Keypoints", str(kp_coco)], ["AP10k-Keypoints", str(kp_ap10k)]],
        )

    # Write a compact public summary. Detailed diagnostic tables are available
    # with --include_debug.
    parts: List[str] = []
    parts.append("# Detection Metrics Summary\n\n")

    parts.append("## Main Results\n\n")
    parts.append(md_table(img_headers, [img_row]))
    parts.append("\n")

    parts.append("## Referring Expression Detection\n\n")
    parts.append(md_table(ref_f1_headers, [ref_f1_row]))
    parts.append("\n")

    parts.append("## Selected Detection F1\n\n")
    parts.append(md_table_two_header_rows(multi_h1, multi_h2, [multi_row]))
    parts.append("\n")

    parts.append("## Pointing\n\n")
    parts.append(point_table_md if point_table_md else "(no pointing metrics found)\n\n")
    parts.append("\n")
    if args.include_debug and point_debug_md:
        parts.append("### Debug (pointing matched keys)\n\n")
        parts.append(point_debug_md)
        parts.append("\n")

    parts.append("## Visual Prompting\n\n")
    parts.append(visual_table_md if visual_table_md else "(no visual_prompting metrics found)\n\n")
    parts.append("\n")
    if args.include_debug and visual_debug_md:
        parts.append("### Debug (visual_prompting matched keys)\n\n")
        parts.append(visual_debug_md)
        parts.append("\n")

    parts.append("## GUI Grounding\n\n")
    parts.append(gui_table_md if gui_table_md else "(no GUI metrics found)\n\n")
    parts.append("\n")
    if args.include_debug and gui_debug_md:
        parts.append("### Debug (GUI matched keys)\n\n")
        parts.append(gui_debug_md)
        parts.append("\n")

    parts.append("## Keypoint Detection\n\n")
    parts.append(keypoint_table_md if keypoint_table_md else "(no keypoint metrics found)\n\n")
    parts.append("\n")
    if args.include_debug and keypoint_debug_md:
        parts.append("### Debug (keypoint matched keys)\n\n")
        parts.append(keypoint_debug_md)
        parts.append("\n")

    if args.include_debug:
        parts.append("## Debug (all detection metrics)\n\n")
        parts.append(md_table(wide_headers, wide_rows))
        parts.append("\n")

        parts.append("## Debug (all detection metrics, transposed)\n\n")
        parts.append(md_table(trans_headers, trans_rows))
        parts.append("\n")

        debug_rows = [
            ["COCO", str(k_coco)],
            ["LVIS", str(k_lvis)],
            ["HumanRef", str(k_humanref)],
            ["RefCOCOg_val", str(k_refcocog_val)],
            ["RefCOCOg_test", str(k_refcocog_test)],
            ["Dense200", str(k_dense200)],
            ["VisDrone", str(k_visdrone)],
            ["HierText", str(k_hiertext)],
            ["HierText-textline", str(k_hiertext_textline)],
            ["SROIE", str(k_sroie)],
            ["SROIE-textline", str(k_sroie_textline)],
            ["IC15", str(k_ic15)],
            ["IC15-word", str(k_ic15_word)],
            ["TotalText", str(k_totaltext)],
            ["TotalText-word", str(k_totaltext_word)],
        ]
        parts.append("## Debug (matched keys)\n\n")
        parts.append(md_table(["group", "picked_dataset_key"], debug_rows))

        parts.append("\n## Debug (available dataset keys)\n\n")
        avail = sorted({name for name, _ in data_all}, key=lambda x: norm_key(x))
        parts.append("```\n" + "\n".join(avail) + "\n```\n")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("".join(parts))

    print(f"✅ Wrote markdown: {out_md}")

    # ---------- fact-table JSON (Keep the original behavior: only write Detection fact rows by default to avoid affecting the existing Feishu pipeline.) ----------
    fact_rows = build_detection_fact_rows(
        model_name=args.model_name,
        name2d=name2d_det,
        cols=cols,
        task_name="Detection",
    )

    out_json = args.out_json or os.path.join(metrics_dir, "detection_fact_rows.jsonl")
    with open(out_json, "w", encoding="utf-8") as f:
        for r in fact_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ Wrote fact-table json: {out_json} ({len(fact_rows)} rows)")


if __name__ == "__main__":
    main()
