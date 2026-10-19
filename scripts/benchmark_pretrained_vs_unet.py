"""Comprehensive Multi-Domain Benchmark: Pretrained DehazeFormer vs Lightweight U-Net vs DCP (Stage 2C.3).

Fairly benchmarks:
1. Classical Dark Channel Prior (DCP) baseline
2. Existing indoor leading model (medium_best_model.pth)
3. Existing unified mixed pilot (mixed_pilot_best_model.pth)
4. Pretrained DehazeFormer-MCT (dehazeformer_mct.pth)

Evaluated Across:
- Held-out RESIDE SOTS-Indoor: 50 distinct scenes (1 per scene: 1400 to 1449)
- Held-out RESIDE OTS-Outdoor: 20 distinct unseen outdoor scenes
- Independent Synthetic Outdoor Benchmark: 4 scenes (mountains, road, valley, village)
- Zero-Haze Clear Image Robustness: 40 images (20 indoor clear, 20 outdoor clear)

Saves:
- outputs/evaluation/stage2c3_pretrained_benchmark_summary.json
- outputs/evaluation/stage2c3_per_image_metrics.csv
- Visual comparison mosaics in outputs/evaluation/comparisons/stage2c3/
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
from src.dehazing.external.dehazeformer_adapter import DehazeFormerAdapter
from src.dehazing.model import LightweightUNet, load_checkpoint
from src.metrics.psnr import calculate_psnr
from src.metrics.ssim import calculate_ssim
from src.utils.image_ops import numpy_to_tensor, save_image, tensor_to_numpy
from src.utils.logger import get_logger
from src.utils.visualization import create_multi_panel

logger = get_logger("benchmark_pretrained")


def load_unet(chk_path: Path, device: torch.device) -> LightweightUNet:
    """Load LightweightUNet model checkpoint."""
    raw = torch.load(str(chk_path), map_location=device, weights_only=False)
    cfg = raw.get("config", {})
    base_ch = cfg.get("base_channels", 24)
    res = cfg.get("residual", True)
    model = LightweightUNet(in_channels=3, out_channels=3, base_channels=base_ch, residual=res).to(device)
    load_checkpoint(chk_path, model=model, map_location=str(device))
    model.eval()
    return model


def run_unet_inference(
    model: LightweightUNet,
    image_rgb: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Run LightweightUNet with fine-grained timing profiling."""
    t0_pre = time.perf_counter()
    t_in = numpy_to_tensor(image_rgb, device=device).unsqueeze(0)
    t_pre = (time.perf_counter() - t0_pre) * 1000.0

    t0_infer = time.perf_counter()
    with torch.no_grad():
        t_out = model(t_in)
    t_infer = (time.perf_counter() - t0_infer) * 1000.0

    t0_post = time.perf_counter()
    out_rgb = tensor_to_numpy(t_out.squeeze(0))
    t_post = (time.perf_counter() - t0_post) * 1000.0

    total_ms = t_pre + t_infer + t_post
    timing = {
        "pre_ms": round(t_pre, 2),
        "infer_ms": round(t_infer, 2),
        "post_ms": round(t_post, 2),
        "total_ms": round(total_ms, 2),
    }
    return out_rgb, timing


def run_dcp_inference(image_rgb: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """Run Classical DCP with timing profiling."""
    t0 = time.perf_counter()
    dcp_rgb, _ = dcp_dehaze(image_rgb, use_guided_filter=True)
    lat_ms = (time.perf_counter() - t0) * 1000.0
    timing = {
        "pre_ms": 0.0,
        "infer_ms": round(lat_ms, 2),
        "post_ms": 0.0,
        "total_ms": round(lat_ms, 2),
    }
    return dcp_rgb, timing


def evaluate_dataset(
    benchmark_name: str,
    pairs: List[Tuple[Path, Path, str, str]],  # (hazy_path, clear_path, stem, scene_id)
    unets: Dict[str, LightweightUNet],
    dehazeformer: DehazeFormerAdapter,
    device: torch.device,
    save_comp_dir: Optional[Path] = None,
    max_visual_saves: int = 5,
) -> Dict[str, Any]:
    """Evaluate all methods on a paired dataset with timing and metrics."""
    method_names = ["Hazy", "DCP", "medium_best", "mixed_pilot_best", "dehazeformer_mct"]
    metrics_by_method: Dict[str, Dict[str, List[float]]] = {
        m: {"psnr": [], "ssim": [], "infer_ms": [], "total_ms": []} for m in method_names
    }

    per_image_rows = []
    saved_visuals = 0

    print(f"\nEvaluating {benchmark_name} ({len(pairs)} pairs)...")
    t0_bench = time.perf_counter()

    for idx, (h_p, c_p, stem, sid) in enumerate(pairs):
        hazy_bgr = cv2.imread(str(h_p), cv2.IMREAD_COLOR)
        clear_bgr = cv2.imread(str(c_p), cv2.IMREAD_COLOR)
        if hazy_bgr is None or clear_bgr is None:
            continue
        hazy_rgb = cv2.cvtColor(hazy_bgr, cv2.COLOR_BGR2RGB)
        clear_rgb = cv2.cvtColor(clear_bgr, cv2.COLOR_BGR2RGB)

        if hazy_rgb.shape[:2] != clear_rgb.shape[:2]:
            clear_rgb = cv2.resize(clear_rgb, (hazy_rgb.shape[1], hazy_rgb.shape[0]), interpolation=cv2.INTER_AREA)

        # 1. Hazy baseline
        p_in = calculate_psnr(hazy_rgb, clear_rgb, max_val=255.0)
        s_in = calculate_ssim(hazy_rgb, clear_rgb, max_val=255.0)
        metrics_by_method["Hazy"]["psnr"].append(p_in)
        metrics_by_method["Hazy"]["ssim"].append(s_in)
        metrics_by_method["Hazy"]["infer_ms"].append(0.0)
        metrics_by_method["Hazy"]["total_ms"].append(0.0)

        # 2. Classical DCP
        dcp_rgb, dcp_time = run_dcp_inference(hazy_rgb)
        p_dcp = calculate_psnr(dcp_rgb, clear_rgb, max_val=255.0)
        s_dcp = calculate_ssim(dcp_rgb, clear_rgb, max_val=255.0)
        metrics_by_method["DCP"]["psnr"].append(p_dcp)
        metrics_by_method["DCP"]["ssim"].append(s_dcp)
        metrics_by_method["DCP"]["infer_ms"].append(dcp_time["infer_ms"])
        metrics_by_method["DCP"]["total_ms"].append(dcp_time["total_ms"])

        row: Dict[str, Any] = {
            "benchmark": benchmark_name,
            "stem": stem,
            "scene_id": sid,
            "hazy_psnr": round(p_in, 2),
            "hazy_ssim": round(s_in, 4),
            "dcp_psnr": round(p_dcp, 2),
            "dcp_ssim": round(s_dcp, 4),
            "dcp_lat_ms": dcp_time["total_ms"],
        }

        mosaic_panels = [
            ("Ground Truth", clear_rgb),
            ("Hazy Input", hazy_rgb),
            (f"DCP ({p_dcp:.1f}dB)", dcp_rgb),
        ]

        # 3. U-Net models
        for u_name, u_model in unets.items():
            u_rgb, u_time = run_unet_inference(u_model, hazy_rgb, device)
            p_u = calculate_psnr(u_rgb, clear_rgb, max_val=255.0)
            s_u = calculate_ssim(u_rgb, clear_rgb, max_val=255.0)
            metrics_by_method[u_name]["psnr"].append(p_u)
            metrics_by_method[u_name]["ssim"].append(s_u)
            metrics_by_method[u_name]["infer_ms"].append(u_time["infer_ms"])
            metrics_by_method[u_name]["total_ms"].append(u_time["total_ms"])

            row[f"{u_name}_psnr"] = round(p_u, 2)
            row[f"{u_name}_ssim"] = round(s_u, 4)
            row[f"{u_name}_lat_ms"] = u_time["total_ms"]
            mosaic_panels.append((f"{u_name} ({p_u:.1f}dB)", u_rgb))

        # 4. DehazeFormer-MCT
        df_rgb, df_time = dehazeformer.dehaze(hazy_rgb)
        p_df = calculate_psnr(df_rgb, clear_rgb, max_val=255.0)
        s_df = calculate_ssim(df_rgb, clear_rgb, max_val=255.0)
        metrics_by_method["dehazeformer_mct"]["psnr"].append(p_df)
        metrics_by_method["dehazeformer_mct"]["ssim"].append(s_df)
        metrics_by_method["dehazeformer_mct"]["infer_ms"].append(df_time["infer_ms"])
        metrics_by_method["dehazeformer_mct"]["total_ms"].append(df_time["total_ms"])

        row["dehazeformer_psnr"] = round(p_df, 2)
        row["dehazeformer_ssim"] = round(s_df, 4)
        row["dehazeformer_lat_ms"] = df_time["total_ms"]
        row["dehazeformer_infer_ms"] = df_time["infer_ms"]
        mosaic_panels.append((f"DehazeFormer ({p_df:.1f}dB)", df_rgb))

        per_image_rows.append(row)

        # Save representative visual comparison
        if save_comp_dir is not None and saved_visuals < max_visual_saves and idx % max(1, len(pairs) // max_visual_saves) == 0:
            imgs = [img for _, img in mosaic_panels]
            lbls = [lbl for lbl, _ in mosaic_panels]
            mosaic = create_multi_panel(images=imgs, labels=lbls)
            save_image(mosaic, save_comp_dir / f"{benchmark_name}_{stem}_multimodel.png")
            saved_visuals += 1

    dur_sec = time.perf_counter() - t0_bench
    print(f"Finished {benchmark_name} in {dur_sec:.1f}s.")

    # Compute aggregate metrics
    agg: Dict[str, Any] = {"total_pairs": len(pairs), "duration_sec": round(dur_sec, 2), "methods": {}}
    for m in method_names:
        p_list = metrics_by_method[m]["psnr"]
        s_list = metrics_by_method[m]["ssim"]
        i_list = metrics_by_method[m]["infer_ms"]
        t_list = metrics_by_method[m]["total_ms"]
        if p_list:
            agg["methods"][m] = {
                "mean_psnr": round(float(np.mean(p_list)), 2),
                "median_psnr": round(float(statistics.median(p_list)), 2),
                "mean_ssim": round(float(np.mean(s_list)), 4),
                "median_ssim": round(float(statistics.median(s_list)), 4),
                "mean_infer_ms": round(float(np.mean(i_list)), 1),
                "mean_total_ms": round(float(np.mean(t_list)), 1),
            }

    # Calculate win counts vs medium_best
    med_psnrs = [r["medium_best_psnr"] for r in per_image_rows]
    mix_psnrs = [r["mixed_pilot_best_psnr"] for r in per_image_rows]
    df_psnrs = [r["dehazeformer_psnr"] for r in per_image_rows]

    df_wins = sum(1 for df_p, m_p in zip(df_psnrs, med_psnrs) if df_p > m_p)
    mix_wins = sum(1 for mx_p, m_p in zip(mix_psnrs, med_psnrs) if mx_p > m_p)
    agg["scene_wins_vs_medium"] = {
        "dehazeformer_wins": df_wins,
        "dehazeformer_win_pct": round(df_wins / max(1, len(per_image_rows)) * 100.0, 1),
        "mixed_pilot_wins": mix_wins,
        "mixed_pilot_win_pct": round(mix_wins / max(1, len(per_image_rows)) * 100.0, 1),
    }

    return {"summary": agg, "per_image": per_image_rows}


def evaluate_clear_set(
    manifest_path: Path,
    unets: Dict[str, LightweightUNet],
    dehazeformer: DehazeFormerAdapter,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate on 40 zero-haze clear images to measure non-distortion fidelity."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    images = data.get("images", [])

    method_names = ["DCP", "medium_best", "mixed_pilot_best", "dehazeformer_mct"]
    metrics_by_method = {m: {"psnr": [], "ssim": [], "total_ms": []} for m in method_names}
    rows = []

    print(f"\nEvaluating Zero-Haze Clear Robustness ({len(images)} images)...")
    for item in images:
        p = PROJECT_ROOT / item["image_path"]
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        dom = item.get("domain", "unknown")

        dcp_rgb, dcp_time = run_dcp_inference(rgb)
        p_d = calculate_psnr(dcp_rgb, rgb, max_val=255.0)
        s_d = calculate_ssim(dcp_rgb, rgb, max_val=255.0)
        metrics_by_method["DCP"]["psnr"].append(p_d)
        metrics_by_method["DCP"]["ssim"].append(s_d)
        metrics_by_method["DCP"]["total_ms"].append(dcp_time["total_ms"])

        row = {"image": p.name, "domain": dom, "dcp_psnr": round(p_d, 2), "dcp_ssim": round(s_d, 4)}

        for u_name, u_model in unets.items():
            u_rgb, u_time = run_unet_inference(u_model, rgb, device)
            p_u = calculate_psnr(u_rgb, rgb, max_val=255.0)
            s_u = calculate_ssim(u_rgb, rgb, max_val=255.0)
            metrics_by_method[u_name]["psnr"].append(p_u)
            metrics_by_method[u_name]["ssim"].append(s_u)
            metrics_by_method[u_name]["total_ms"].append(u_time["total_ms"])
            row[f"{u_name}_psnr"] = round(p_u, 2)
            row[f"{u_name}_ssim"] = round(s_u, 4)

        df_rgb, df_time = dehazeformer.dehaze(rgb)
        p_df = calculate_psnr(df_rgb, rgb, max_val=255.0)
        s_df = calculate_ssim(df_rgb, rgb, max_val=255.0)
        metrics_by_method["dehazeformer_mct"]["psnr"].append(p_df)
        metrics_by_method["dehazeformer_mct"]["ssim"].append(s_df)
        metrics_by_method["dehazeformer_mct"]["total_ms"].append(df_time["total_ms"])
        row["dehazeformer_psnr"] = round(p_df, 2)
        row["dehazeformer_ssim"] = round(s_df, 4)

        rows.append(row)

    agg = {"total_clean_images": len(images), "methods": {}}
    for m in method_names:
        p_list = metrics_by_method[m]["psnr"]
        s_list = metrics_by_method[m]["ssim"]
        t_list = metrics_by_method[m]["total_ms"]
        agg["methods"][m] = {
            "mean_psnr_vs_clean": round(float(np.mean(p_list)), 2),
            "median_psnr_vs_clean": round(float(statistics.median(p_list)), 2),
            "mean_ssim_vs_clean": round(float(np.mean(s_list)), 4),
            "median_ssim_vs_clean": round(float(statistics.median(s_list)), 4),
            "mean_total_ms": round(float(np.mean(t_list)), 1),
        }
    return {"summary": agg, "per_image": rows}


def main() -> None:
    device = torch.device(get_default_device())
    eval_dir = OUTPUTS_DIR / "evaluation" / "stage2c3"
    eval_dir.mkdir(parents=True, exist_ok=True)
    comp_dir = eval_dir / "comparisons"
    comp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("STAGE 2C.3: PRETRAINED DEHAZEFORMER VS LIGHTWEIGHT U-NET VS DCP BENCHMARK")
    print("=" * 75)

    # 1. Load models
    unets = {
        "medium_best": load_unet(CHECKPOINTS_DIR / "medium_best_model.pth", device),
        "mixed_pilot_best": load_unet(CHECKPOINTS_DIR / "mixed_pilot_best_model.pth", device),
    }
    dehazeformer = DehazeFormerAdapter(PROJECT_ROOT / "models" / "pretrained" / "dehazeformer_mct.pth", device=str(device))
    print(f"Loaded U-Net models ({unets['medium_best'].count_parameters():,} params each) and DehazeFormer ({dehazeformer.count_parameters():,} params).")

    # 2. Benchmark A: SOTS-Indoor (50 distinct scenes, 1 representative variant per scene)
    sots_indoor_gt = RAW_DATA_DIR / "RESIDE" / "SOTS" / "indoor" / "gt"
    sots_indoor_hazy = RAW_DATA_DIR / "RESIDE" / "SOTS" / "indoor" / "hazy"
    sots_pairs = []
    for sid in sorted([p.stem for p in sots_indoor_gt.glob("*.png")]):
        c_p = sots_indoor_gt / f"{sid}.png"
        # Find first matching hazy variant
        h_cand = list(sots_indoor_hazy.glob(f"{sid}_*.png"))
        if h_cand:
            sots_pairs.append((h_cand[0], c_p, h_cand[0].stem, sid))
    print(f"Discovered SOTS-Indoor 50-Scene Benchmark pairs: {len(sots_pairs)}")

    # 3. Benchmark B: Synthetic Outdoor Benchmark (4 scenes)
    syn_dir = PROJECT_ROOT / "data" / "processed" / "synthetic_benchmark"
    syn_pairs = []
    c_map = {p.stem: p for p in (syn_dir / "clear").glob("*.png")}
    for h_p in sorted(list((syn_dir / "hazy").glob("*.png"))):
        if h_p.stem in c_map:
            syn_pairs.append((h_p, c_map[h_p.stem], h_p.stem, h_p.stem))
    print(f"Discovered Synthetic Outdoor Benchmark pairs: {len(syn_pairs)}")

    # 4. Benchmark C: Held-out OTS Outdoor Test Partition (20 distinct scenes)
    ots_manifest = PROJECT_ROOT / "data" / "processed" / "manifests" / "ots_dataset_manifest.json"
    ots_test_pairs = []
    with open(ots_manifest, "r", encoding="utf-8") as f:
        ots_data = json.load(f)
    all_test = [p for p in ots_data.get("pairs", []) if p.get("split") == "test"]
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
    print(f"Discovered Held-Out OTS Outdoor Test pairs: {len(ots_test_pairs)}")

    # 5. Benchmark D: Zero-Haze Clear Image Robustness
    clear_mf = PROJECT_ROOT / "data" / "processed" / "manifests" / "clear_test_manifest.json"

    # EXECUTE EVALUATIONS
    all_summaries: Dict[str, Any] = {}
    all_csv_records: List[Dict[str, Any]] = []

    res_sots = evaluate_dataset("sots_indoor_50scenes", sots_pairs, unets, dehazeformer, device, save_comp_dir=comp_dir)
    all_summaries["sots_indoor_50scenes"] = res_sots["summary"]
    all_csv_records.extend(res_sots["per_image"])

    res_syn = evaluate_dataset("synthetic_outdoor_4scenes", syn_pairs, unets, dehazeformer, device, save_comp_dir=comp_dir)
    all_summaries["synthetic_outdoor_4scenes"] = res_syn["summary"]
    all_csv_records.extend(res_syn["per_image"])

    res_ots = evaluate_dataset("heldout_ots_outdoor_20scenes", ots_test_pairs, unets, dehazeformer, device, save_comp_dir=comp_dir)
    all_summaries["heldout_ots_outdoor_20scenes"] = res_ots["summary"]
    all_csv_records.extend(res_ots["per_image"])

    res_clear = evaluate_clear_set(clear_mf, unets, dehazeformer, device)
    all_summaries["clear_image_robustness_40images"] = res_clear["summary"]

    # Save summary JSON
    summary_path = eval_dir / "stage2c3_pretrained_benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved benchmark summary to: {summary_path}")

    # Save per-image CSV
    csv_path = eval_dir / "stage2c3_per_image_metrics.csv"
    if all_csv_records:
        fieldnames = list(all_csv_records[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_csv_records)
        print(f"Saved per-image metrics to: {csv_path}")

    # Print Formatted Multi-Domain Results Table
    print("\n" + "=" * 95)
    print("STAGE 2C.3 HEAD-TO-HEAD BENCHMARK: DEHAZEFORMER-MCT VS U-NET VS DCP")
    print("=" * 95)
    header = f"{'Benchmark':<28} | {'Method':<18} | {'PSNR (mean)':<11} | {'PSNR (med)':<11} | {'SSIM (mean)':<11} | {'Total (ms)':<10}"
    print(header)
    print("-" * 95)

    for b_name, b_data in all_summaries.items():
        if "methods" in b_data:
            for m_name, m_stats in b_data["methods"].items():
                p_mean = m_stats.get("mean_psnr", m_stats.get("mean_psnr_vs_clean", "N/A"))
                p_med = m_stats.get("median_psnr", m_stats.get("median_psnr_vs_clean", "N/A"))
                s_mean = m_stats.get("mean_ssim", m_stats.get("mean_ssim_vs_clean", "N/A"))
                tot_ms = m_stats.get("mean_total_ms", "N/A")
                print(f"{b_name:<28} | {m_name:<18} | {str(p_mean):<11} | {str(p_med):<11} | {str(s_mean):<11} | {str(tot_ms):<10}")
            print("-" * 95)


if __name__ == "__main__":
    main()
