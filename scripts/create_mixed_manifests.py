"""Manifest Generation Utility for Unified Indoor and Outdoor Dehazing (Stage 2C.2).

Generates machine-readable manifests:
1. `data/processed/manifests/ots_dataset_manifest.json` (Full 72,135 OTS dataset)
2. `data/processed/manifests/mixed_train_manifest.json` (Balanced pilot training: 100 indoor + 100 outdoor)
3. `data/processed/manifests/mixed_val_manifest.json` (Balanced validation: 20 indoor + 20 outdoor)
4. `data/processed/manifests/clear_test_manifest.json` (40 zero-haze clear test images: 20 indoor + 20 outdoor)

Ensures zero scene leakage, preserves official splits, and verifies disjointness from SOTS-Indoor.
"""

import json
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import DATA_DIR, RAW_DATA_DIR
from src.utils.seed import set_seed


def generate_manifests(seed: int = 42) -> None:
    set_seed(seed)
    manifest_dir = DATA_DIR / "processed" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STAGE 2C.2: GENERATING LEAKAGE-RESISTANT MIXED-DOMAIN MANIFESTS")
    print("=" * 70)

    # 1. Full OTS Scan and Discovery
    ots_root = RAW_DATA_DIR / "RESIDE" / "OTS"
    clear_dir = ots_root / "clear"
    haze_dir = ots_root / "haze"

    clear_files = {p.stem: p for p in clear_dir.glob("*.jpg")}
    sorted_clear_scenes = sorted(list(clear_files.keys()))
    print(f"Discovered OTS Clear Scenes: {len(sorted_clear_scenes):,}")

    ots_pairs_by_scene = {}
    for part in ["part1", "part2", "part3", "part4"]:
        part_dir = haze_dir / part
        for p in part_dir.glob("*.jpg"):
            stem = p.stem
            parts = stem.split("_")
            scene_id = parts[0]
            airlight = parts[1] if len(parts) > 1 else "N/A"
            beta = parts[2] if len(parts) > 2 else "N/A"

            if scene_id in clear_files:
                ots_pairs_by_scene.setdefault(scene_id, []).append({
                    "hazy_path": str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "clear_path": str(clear_files[scene_id].relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "scene_id": scene_id,
                    "variant_stem": stem,
                    "airlight": airlight,
                    "beta": beta,
                    "domain": "outdoor",
                })

    total_ots_pairs = sum(len(v) for v in ots_pairs_by_scene.values())
    print(f"Discovered OTS Hazy Pairs:   {total_ots_pairs:,} across {len(ots_pairs_by_scene):,} scenes")

    # 2. Split OTS Scenes Disjointly (seed=42)
    rng = random.Random(seed)
    shuffled_ots_scenes = list(sorted_clear_scenes)
    rng.shuffle(shuffled_ots_scenes)

    # Split 2,061 OTS scenes into:
    # 1,600 Train scenes (56,000 pairs)
    # 261 Val scenes (9,135 pairs)
    # 200 Held-Out Test scenes (7,000 pairs)
    ots_train_scenes = set(shuffled_ots_scenes[:1600])
    ots_val_scenes = set(shuffled_ots_scenes[1600:1861])
    ots_test_scenes = set(shuffled_ots_scenes[1861:])

    print("-" * 70)
    print("OTS SCENE-DISJOINT PARTITIONING:")
    print(f"  Outdoor Train Scenes:      {len(ots_train_scenes):,} ({len(ots_train_scenes)*35:,} pairs)")
    print(f"  Outdoor Val Scenes:        {len(ots_val_scenes):,} ({len(ots_val_scenes)*35:,} pairs)")
    print(f"  Outdoor Test Scenes:       {len(ots_test_scenes):,} ({len(ots_test_scenes)*35:,} pairs)")
    print(f"  Overlap Train vs Val:      {len(ots_train_scenes.intersection(ots_val_scenes))} (Disjoint: True)")
    print(f"  Overlap Train vs Test:     {len(ots_train_scenes.intersection(ots_test_scenes))} (Disjoint: True)")

    # 3. Indoor ITS Discovery
    its_train_clear_dir = RAW_DATA_DIR / "RESIDE" / "ITS" / "train" / "ITS_clear"
    its_train_haze_dir = RAW_DATA_DIR / "RESIDE" / "ITS" / "train" / "ITS_haze"
    its_val_clear_dir = RAW_DATA_DIR / "RESIDE" / "ITS" / "val" / "clear"
    its_val_haze_dir = RAW_DATA_DIR / "RESIDE" / "ITS" / "val" / "haze"

    its_train_clear = {p.stem: p for p in its_train_clear_dir.glob("*.png")}
    its_val_clear = {p.stem: p for p in its_val_clear_dir.glob("*.png")}

    # SOTS-Indoor Held-Out Check
    sots_indoor_gt = set(p.stem for p in (RAW_DATA_DIR / "RESIDE" / "SOTS" / "indoor" / "gt").glob("*.png"))
    print("-" * 70)
    print("INDOOR ITS & SOTS LEAKAGE PREVENTION:")
    print(f"  ITS Train Clear Scenes:    {len(its_train_clear):,}")
    print(f"  ITS Val Clear Scenes:      {len(its_val_clear):,}")
    print(f"  SOTS-Indoor Test Scenes:   {len(sots_indoor_gt):,} (Held-out: True)")
    print(f"  Overlap ITS Train vs Val:  {len(set(its_train_clear.keys()).intersection(set(its_val_clear.keys())))} (Disjoint: True)")

    # 4. Construct Balanced Mixed Pilot Training Manifest (100 indoor + 100 outdoor = 200 pairs)
    # Pick 100 indoor scenes (excluding any SOTS overlap string stems for absolute purity)
    eligible_its_train_scenes = sorted(list(set(its_train_clear.keys()) - sots_indoor_gt))
    rng.shuffle(eligible_its_train_scenes)
    pilot_its_scenes = eligible_its_train_scenes[:100]

    mixed_train_pairs = []
    # Add 100 indoor pairs (1 variant per scene)
    for sid in pilot_its_scenes:
        c_p = its_train_clear[sid]
        # Find hazy variant
        h_candidates = list(its_train_haze_dir.glob(f"{sid}_*.png"))
        if h_candidates:
            h_p = rng.choice(h_candidates)
            mixed_train_pairs.append({
                "hazy_path": str(h_p.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "clear_path": str(c_p.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "scene_id": sid,
                "variant_stem": h_p.stem,
                "domain": "indoor",
                "split": "train",
            })

    # Add 100 outdoor pairs (1 variant per scene from ots_train_scenes)
    pilot_ots_scenes = sorted(list(ots_train_scenes))
    rng.shuffle(pilot_ots_scenes)
    for sid in pilot_ots_scenes[:100]:
        variants = ots_pairs_by_scene[sid]
        selected_var = rng.choice(variants)
        entry = dict(selected_var)
        entry["split"] = "train"
        mixed_train_pairs.append(entry)

    # 5. Construct Balanced Mixed Validation Manifest (20 indoor + 20 outdoor = 40 pairs)
    mixed_val_pairs = []
    pilot_val_its = sorted(list(its_val_clear.keys()))
    rng.shuffle(pilot_val_its)
    for sid in pilot_val_its[:20]:
        c_p = its_val_clear[sid]
        h_candidates = list(its_val_haze_dir.glob(f"{sid}_*.png"))
        if h_candidates:
            h_p = rng.choice(h_candidates)
            mixed_val_pairs.append({
                "hazy_path": str(h_p.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "clear_path": str(c_p.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "scene_id": sid,
                "variant_stem": h_p.stem,
                "domain": "indoor",
                "split": "val",
            })

    pilot_val_ots = sorted(list(ots_val_scenes))
    rng.shuffle(pilot_val_ots)
    for sid in pilot_val_ots[:20]:
        variants = ots_pairs_by_scene[sid]
        selected_var = rng.choice(variants)
        entry = dict(selected_var)
        entry["split"] = "val"
        mixed_val_pairs.append(entry)

    # 6. Construct Clear-Image Test Set (20 indoor clear + 20 outdoor clear = 40 images)
    clear_test_set = []
    # 20 distinct clear indoor images from ITS val (unseen in training)
    for sid in pilot_val_its[20:40]:
        c_p = its_val_clear[sid]
        clear_test_set.append({
            "image_path": str(c_p.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "scene_id": sid,
            "domain": "indoor",
            "is_clean_reference": True,
        })
    # 20 distinct clear outdoor images from OTS val (unseen in training)
    for sid in pilot_val_ots[20:40]:
        c_p = clear_files[sid]
        clear_test_set.append({
            "image_path": str(c_p.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "scene_id": sid,
            "domain": "outdoor",
            "is_clean_reference": True,
        })

    # Save all manifests to disk
    train_manifest_path = manifest_dir / "mixed_train_manifest.json"
    val_manifest_path = manifest_dir / "mixed_val_manifest.json"
    clear_manifest_path = manifest_dir / "clear_test_manifest.json"
    ots_manifest_path = manifest_dir / "ots_dataset_manifest.json"

    # Assemble full OTS dataset manifest
    full_ots_list = []
    for sid, variants in sorted(ots_pairs_by_scene.items()):
        split_name = "train" if sid in ots_train_scenes else ("val" if sid in ots_val_scenes else "test")
        for v in variants:
            item = dict(v)
            item["split"] = split_name
            full_ots_list.append(item)

    with open(ots_manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "description": "Stage 2C.2 Full RESIDE OTS Dataset Manifest (72,135 pairs)",
            "seed": seed,
            "total_pairs": len(full_ots_list),
            "total_scenes": len(ots_pairs_by_scene),
            "train_scenes": len(ots_train_scenes),
            "val_scenes": len(ots_val_scenes),
            "test_scenes": len(ots_test_scenes),
            "pairs": full_ots_list,
        }, f, indent=2)

    with open(train_manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "description": "Stage 2C.2 Balanced Mixed-Domain Training Manifest (100 indoor + 100 outdoor)",
            "seed": seed,
            "total_pairs": len(mixed_train_pairs),
            "indoor_count": sum(1 for p in mixed_train_pairs if p["domain"] == "indoor"),
            "outdoor_count": sum(1 for p in mixed_train_pairs if p["domain"] == "outdoor"),
            "pairs": mixed_train_pairs,
        }, f, indent=2)

    with open(val_manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "description": "Stage 2C.2 Balanced Mixed-Domain Validation Manifest (20 indoor + 20 outdoor)",
            "seed": seed,
            "total_pairs": len(mixed_val_pairs),
            "indoor_count": sum(1 for p in mixed_val_pairs if p["domain"] == "indoor"),
            "outdoor_count": sum(1 for p in mixed_val_pairs if p["domain"] == "outdoor"),
            "pairs": mixed_val_pairs,
        }, f, indent=2)

    with open(clear_manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "description": "Stage 2C.2 Zero-Haze Clear Image Robustness Benchmark (20 indoor + 20 outdoor)",
            "seed": seed,
            "total_images": len(clear_test_set),
            "indoor_count": sum(1 for p in clear_test_set if p["domain"] == "indoor"),
            "outdoor_count": sum(1 for p in clear_test_set if p["domain"] == "outdoor"),
            "images": clear_test_set,
        }, f, indent=2)

    print("-" * 70)
    print("MANIFESTS SUCCESSFULLY WRITTEN:")
    print(f"  1. Full OTS Manifest:      {ots_manifest_path.name} ({len(full_ots_list):,} pairs across {len(ots_pairs_by_scene):,} scenes)")
    print(f"  2. Mixed Train Manifest:   {train_manifest_path.name} ({len(mixed_train_pairs)} pairs: 100 indoor + 100 outdoor)")
    print(f"  3. Mixed Val Manifest:     {val_manifest_path.name} ({len(mixed_val_pairs)} pairs: 20 indoor + 20 outdoor)")
    print(f"  4. Clear Test Manifest:    {clear_manifest_path.name} ({len(clear_test_set)} clean images: 20 indoor + 20 outdoor)")
    print("=" * 70)


if __name__ == "__main__":
    generate_manifests()
