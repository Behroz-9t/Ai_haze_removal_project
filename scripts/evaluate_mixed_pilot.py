"""Comprehensive Multi-Domain Evaluation Script for Stage 2C.2 / 2C.3.

Evaluates and compares:
1. Classical Dark Channel Prior (DCP) baseline
2. Existing indoor pilot checkpoint (pilot_best_model.pth)
3. Existing indoor leading checkpoint (medium_best_model.pth)
4. Newly trained unified mixed pilot (mixed_pilot_best_model.pth)

Evaluated Across:
- Held-out RESIDE SOTS-Indoor (500 test pairs across 50 scenes)
- Held-out RESIDE OTS-Outdoor (20 test pairs across 20 distinct scenes)
- Independent Synthetic Outdoor Benchmark (4 scenes: road, mountains, valley, village)
- Zero-Haze Clear Image Robustness Benchmark (40 images: 20 indoor, 20 outdoor)

Outputs:
- outputs/evaluation/stage2c2_multidomain_summary.json
- outputs/evaluation/stage2c2_per_image_metrics.csv
- outputs/evaluation/stage2c2_clear_image_robustness.json
- Visual comparison mosaics in outputs/evaluation/comparisons/stage2c2/
"""

import csv
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch

from app.config import CHECKPOINTS_DIR, OUTPUTS_DIR, RAW_DATA_DIR, get_default_device
from src.dehazing.dcp import dcp_dehaze
from src.dehazing.model import LightweightUNet, load_checkpoint
from src.metrics.psnr import calculate_psnr
from src.metrics.ssim import calculate_ssim
from src.preprocessing.dataset import PairedHazeDataset
from src.utils.image_ops import load_image, numpy_to_tensor, save_image, tensor_to_numpy
from src.utils.logger import get_logger
from src.utils.visualization import create_multi_panel

logger = get_logger("eval_mixed_pilot")


def load_model(chk_path: Path, device: torch.device) -> LightweightUNet:
    """Load a model instance given checkpoint path."""
    raw = torch.load(str(chk_path), map_location=device, weights_only=False)
    cfg = raw.get("config", {})
    base_ch = cfg.get("base_channels", 24)
    res = cfg.get("residual", True)
    model = LightweightUNet(in_channels=3, out_channels=3, base_channels=base_ch, residual=res).to(device)
    load_checkpoint(chk_path, model=model, map_location=str(device))
    model.eval()
    return model


def evaluate_pairs(
    name: str,
    pairs: List[Tuple[Path, Path, str, str]],  # (hazy_path, clear_path, stem, scene_id)
    models_dict: Dict[str, LightweightUNet],
    device: torch.device,
    save_comp_dir: Optional[Path] = None,
    max_visual_saves: int = 6,
) -> Dict[str, Any]:
    """Evaluate multiple models and DCP on a set of pairs."""
    results_by_model: Dict[str, Dict[str, List[float]]] = {
        "Hazy": {"psnr": [], "ssim": [], "latency_ms": []},
        "DCP": {"psnr": [], "ssim": [], "latency_ms": []},
    }
    for m_name in models_dict:
        results_by_model[m_name] = {"psnr": [], "ssim": [], "latency_ms": []}

    per_image_records: List[Dict[str, Any]] = []
    saved_visuals = 0

    print(f"\n--- Evaluating Benchmark: {name} ({len(pairs)} pairs) ---")
    t0_start = time.perf_counter()

    for idx, (h_p, c_p, stem, scene_id) in enumerate(pairs):
        hazy_bgr = cv2.imread(str(h_p), cv2.IMREAD_COLOR)
        clear_bgr = cv2.imread(str(c_p), cv2.IMREAD_COLOR)
        if hazy_bgr is None or clear_bgr is None:
            continue
        hazy_rgb = cv2.cvtColor(hazy_bgr, cv2.COLOR_BGR2RGB)
        clear_rgb = cv2.cvtColor(clear_bgr, cv2.COLOR_BGR2RGB)

        if hazy_rgb.shape[:2] != clear_rgb.shape[:2]:
            clear_rgb = cv2.resize(clear_rgb, (hazy_rgb.shape[1], hazy_rgb.shape[0]), interpolation=cv2.INTER_AREA)

        # Baseline input hazy
        p_in = calculate_psnr(hazy_rgb, clear_rgb, max_val=255.0)
        s_in = calculate_ssim(hazy_rgb, clear_rgb, max_val=255.0)
        results_by_model["Hazy"]["psnr"].append(p_in)
        results_by_model["Hazy"]["ssim"].append(s_in)
        results_by_model["Hazy"]["latency_ms"].append(0.0)

        # DCP
        t_dcp0 = time.perf_counter()
        dcp_rgb, _ = dcp_dehaze(hazy_rgb, use_guided_filter=True)
        lat_dcp = (time.perf_counter() - t_dcp0) * 1000.0
        p_dcp = calculate_psnr(dcp_rgb, clear_rgb, max_val=255.0)
        s_dcp = calculate_ssim(dcp_rgb, clear_rgb, max_val=255.0)
        results_by_model["DCP"]["psnr"].append(p_dcp)
        results_by_model["DCP"]["ssim"].append(s_dcp)
        results_by_model["DCP"]["latency_ms"].append(lat_dcp)

        row_record: Dict[str, Any] = {
            "benchmark": name,
            "stem": stem,
            "scene_id": scene_id,
            "hazy_psnr": round(p_in, 2),
            "hazy_ssim": round(s_in, 4),
            "dcp_psnr": round(p_dcp, 2),
            "dcp_ssim": round(s_dcp, 4),
            "dcp_lat_ms": round(lat_dcp, 1),
        }

        # AI Models
        outputs_for_mosaic = [
            ("Ground Truth", clear_rgb),
            ("Hazy Input", hazy_rgb),
            ("Classical DCP", dcp_rgb),
        ]

        t_in = numpy_to_tensor(hazy_rgb, device=device).unsqueeze(0)
        for m_name, model in models_dict.items():
            t_ai0 = time.perf_counter()
            with torch.no_grad():
                t_out = model(t_in)
            lat_ai = (time.perf_counter() - t_ai0) * 1000.0
            ai_rgb = tensor_to_numpy(t_out.squeeze(0))

            p_ai = calculate_psnr(ai_rgb, clear_rgb, max_val=255.0)
            s_ai = calculate_ssim(ai_rgb, clear_rgb, max_val=255.0)
            results_by_model[m_name]["psnr"].append(p_ai)
            results_by_model[m_name]["ssim"].append(s_ai)
            results_by_model[m_name]["latency_ms"].append(lat_ai)

            row_record[f"{m_name}_psnr"] = round(p_ai, 2)
            row_record[f"{m_name}_ssim"] = round(s_ai, 4)
            row_record[f"{m_name}_lat_ms"] = round(lat_ai, 1)

            outputs_for_mosaic.append((m_name, ai_rgb))

        per_image_records.append(row_record)

        # Save representative visual strip
        if save_comp_dir is not None and saved_visuals < max_visual_saves and idx % max(1, len(pairs) // max_visual_saves) == 0:
            imgs = [img for _, img in outputs_for_mosaic]
            lbls = [lbl for lbl, _ in outputs_for_mosaic]
            mosaic = create_multi_panel(images=imgs, labels=lbls)
            save_image(mosaic, save_comp_dir / f"{name}_{stem}_compare.png")
            saved_visuals += 1

    dur_total = time.perf_counter() - t0_start
    print(f"Finished {name} in {dur_total:.1f}s.")

    # Aggregate metrics
    summary: Dict[str, Any] = {"total_pairs": len(pairs), "methods": {}}
    for m_name, vals in results_by_model.items():
        if vals["psnr"]:
            summary["methods"][m_name] = {
                "mean_psnr": round(float(np.mean(vals["psnr"])), 2),
                "median_psnr": round(float(statistics.median(vals["psnr"])), 2),
                "mean_ssim": round(float(np.mean(vals["ssim"])), 4),
                "median_ssim": round(float(statistics.median(vals["ssim"])), 4),
                "mean_latency_ms": round(float(np.mean(vals["latency_ms"])), 1),
            }

    return {"summary": summary, "per_image": per_image_records}


def evaluate_clear_robustness(
    manifest_path: Path,
    models_dict: Dict[str, LightweightUNet],
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate models on zero-haze clear images to measure non-distortion fidelity."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    images = data.get("images", [])

    results: Dict[str, Dict[str, List[float]]] = {
        "DCP": {"psnr": [], "ssim": [], "latency_ms": []},
    }
    for m_name in models_dict:
        results[m_name] = {"psnr": [], "ssim": [], "latency_ms": []}

    domain_records = []

    print(f"\n--- Evaluating Zero-Haze Clear Image Robustness ({len(images)} images) ---")
    for item in images:
        p = PROJECT_ROOT / item["image_path"]
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        dom = item.get("domain", "unknown")

        # DCP
        t0 = time.perf_counter()
        dcp_rgb, _ = dcp_dehaze(rgb, use_guided_filter=True)
        lat_d = (time.perf_counter() - t0) * 1000.0
        p_d = calculate_psnr(dcp_rgb, rgb, max_val=255.0)
        s_d = calculate_ssim(dcp_rgb, rgb, max_val=255.0)
        results["DCP"]["psnr"].append(p_d)
        results["DCP"]["ssim"].append(s_d)
        results["DCP"]["latency_ms"].append(lat_d)

        rec = {"image": p.name, "domain": dom, "dcp_psnr": round(p_d, 2), "dcp_ssim": round(s_d, 4)}

        # AI Models
        t_in = numpy_to_tensor(rgb, device=device).unsqueeze(0)
        for m_name, model in models_dict.items():
            t0 = time.perf_counter()
            with torch.no_grad():
                t_out = model(t_in)
            lat_m = (time.perf_counter() - t0) * 1000.0
            ai_rgb = tensor_to_numpy(t_out.squeeze(0))

            p_m = calculate_psnr(ai_rgb, rgb, max_val=255.0)
            s_m = calculate_ssim(ai_rgb, rgb, max_val=255.0)
            results[m_name]["psnr"].append(p_m)
            results[m_name]["ssim"].append(s_m)
            results[m_name]["latency_ms"].append(lat_m)

            rec[f"{m_name}_psnr"] = round(p_m, 2)
            rec[f"{m_name}_ssim"] = round(s_m, 4)

        domain_records.append(rec)

    summary: Dict[str, Any] = {"total_clean_images": len(images), "methods": {}}
    for m_name, vals in results.items():
        summary["methods"][m_name] = {
            "mean_psnr_vs_clean": round(float(np.mean(vals["psnr"])), 2),
            "median_psnr_vs_clean": round(float(statistics.median(vals["psnr"])), 2),
            "mean_ssim_vs_clean": round(float(np.mean(vals["ssim"])), 4),
            "median_ssim_vs_clean": round(float(statistics.median(vals["ssim"])), 4),
        }

    return {"summary": summary, "per_image": domain_records}


def main() -> None:
    device = torch.device(get_default_device())
    eval_dir = OUTPUTS_DIR / "evaluation" / "stage2c2"
    eval_dir.mkdir(parents=True, exist_ok=True)
    comp_dir = eval_dir / "comparisons"
    comp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STAGE 2C.2: UNIFIED INDOOR/OUTDOOR MULTI-DOMAIN BENCHMARK EVALUATION")
    print("=" * 70)

    # 1. Load models
    models_to_eval = {}
    chks_to_load = [
        ("pilot_best", CHECKPOINTS_DIR / "pilot_best_model.pth"),
        ("medium_best", CHECKPOINTS_DIR / "medium_best_model.pth"),
        ("mixed_pilot_best", CHECKPOINTS_DIR / "mixed_pilot_best_model.pth"),
    ]
    for label, chk_path in chks_to_load:
        if chk_path.is_file():
            print(f"Loading checkpoint: {chk_path.name}")
            models_to_eval[label] = load_model(chk_path, device)
        else:
            print(f"Warning: Checkpoint {chk_path.name} not found.")

    # 2. Benchmark A: Held-out SOTS-Indoor (500 pairs)
    sots_indoor_dir = RAW_DATA_DIR / "RESIDE" / "SOTS" / "indoor"
    sots_pairs = []
    if sots_indoor_dir.is_dir():
        ds_sots = PairedHazeDataset(root_dir=sots_indoor_dir, strict_pairs=True)
        for h_p, c_p, stem in ds_sots.pairs:
            sots_pairs.append((h_p, c_p, stem, c_p.stem))
    print(f"Discovered SOTS-Indoor test pairs: {len(sots_pairs)}")

    # 3. Benchmark B: Independent Synthetic Outdoor Benchmark (4 pairs)
    syn_dir = PROJECT_ROOT / "data" / "processed" / "synthetic_benchmark"
    syn_pairs = []
    if syn_dir.is_dir():
        c_map = {p.stem: p for p in (syn_dir / "clear").glob("*.png")}
        for h_p in (syn_dir / "hazy").glob("*.png"):
            if h_p.stem in c_map:
                syn_pairs.append((h_p, c_map[h_p.stem], h_p.stem, h_p.stem))
    print(f"Discovered Independent Synthetic Outdoor pairs: {len(syn_pairs)}")

    # 4. Benchmark C: Held-out OTS Outdoor Test Partition (20 pairs across 20 scenes)
    ots_manifest = PROJECT_ROOT / "data" / "processed" / "manifests" / "ots_dataset_manifest.json"
    ots_test_pairs = []
    if ots_manifest.is_file():
        with open(ots_manifest, "r", encoding="utf-8") as f:
            ots_data = json.load(f)
        all_test = [p for p in ots_data.get("pairs", []) if p.get("split") == "test"]
        # Group by scene and select 1 variant per scene for first 20 scenes
        scenes_seen = set()
        for item in all_test:
            sid = item["scene_id"]
            if sid not in scenes_seen:
                scenes_seen.add(sid)
                ots_test_pairs.append((
                    PROJECT_ROOT / item["hazy_path"],
                    PROJECT_ROOT / item["clear_path"],
                    item["variant_stem"],
                    sid,
                ))
            if len(ots_test_pairs) >= 20:
                break
    print(f"Discovered Held-out OTS Outdoor Test pairs: {len(ots_test_pairs)}")

    # 5. Benchmark D: Zero-Haze Clear Image Robustness (40 clean images)
    clear_mf = PROJECT_ROOT / "data" / "processed" / "manifests" / "clear_test_manifest.json"

    # RUN EVALUATIONS
    all_summaries: Dict[str, Any] = {}
    all_csv_records: List[Dict[str, Any]] = []

    # Run SOTS-Indoor
    if sots_pairs:
        res_sots = evaluate_pairs("sots_indoor", sots_pairs, models_to_eval, device, save_comp_dir=comp_dir)
        all_summaries["sots_indoor"] = res_sots["summary"]
        all_csv_records.extend(res_sots["per_image"])

    # Run Independent Synthetic Outdoor
    if syn_pairs:
        res_syn = evaluate_pairs("synthetic_outdoor", syn_pairs, models_to_eval, device, save_comp_dir=comp_dir)
        all_summaries["synthetic_outdoor"] = res_syn["summary"]
        all_csv_records.extend(res_syn["per_image"])

    # Run Held-Out OTS Outdoor
    if ots_test_pairs:
        res_ots = evaluate_pairs("heldout_ots_outdoor", ots_test_pairs, models_to_eval, device, save_comp_dir=comp_dir)
        all_summaries["heldout_ots_outdoor"] = res_ots["summary"]
        all_csv_records.extend(res_ots["per_image"])

    # Run Clear-Image Robustness
    if clear_mf.is_file():
        res_clear = evaluate_clear_robustness(clear_mf, models_to_eval, device)
        all_summaries["clear_image_robustness"] = res_clear["summary"]

    # Save summary JSON
    summary_path = eval_dir / "stage2c2_multidomain_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved multi-domain summary to: {summary_path}")

    # Save per-image CSV
    if all_csv_records:
        csv_path = eval_dir / "stage2c2_per_image_metrics.csv"
        fieldnames = list(all_csv_records[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_csv_records)
        print(f"Saved per-image metrics to: {csv_path}")

    # Print formatted comparison table
    print("\n" + "=" * 80)
    print("STAGE 2C.2 FINAL MULTI-DOMAIN EVALUATION COMPARISON")
    print("=" * 80)
    header = f"{'Benchmark':<24} | {'Method':<18} | {'PSNR (mean)':<12} | {'PSNR (med)':<12} | {'SSIM (mean)':<12} | {'Lat (ms)':<8}"
    print(header)
    print("-" * 80)

    for bench_name, b_data in all_summaries.items():
        if "methods" in b_data:
            for m_name, m_stats in b_data["methods"].items():
                p_mean = m_stats.get("mean_psnr", m_stats.get("mean_psnr_vs_clean", "N/A"))
                p_med = m_stats.get("median_psnr", m_stats.get("median_psnr_vs_clean", "N/A"))
                s_mean = m_stats.get("mean_ssim", m_stats.get("mean_ssim_vs_clean", "N/A"))
                lat = m_stats.get("mean_latency_ms", "N/A")
                print(f"{bench_name:<24} | {m_name:<18} | {str(p_mean):<12} | {str(p_med):<12} | {str(s_mean):<12} | {str(lat):<8}")
            print("-" * 80)


if __name__ == "__main__":
    main()
