"""Streamlit Testing Interface for AI Atmospheric Visibility Enhancement (Stage 2C.3).

Features:
1. Single Image Dehazing with Pretrained DehazeFormer-MCT and Lightweight U-Net.
2. Atmospheric Haze Density Diagnostics (detects clear vs light vs heavy haze).
3. Side-by-side comparative views (Active Model, Classical DCP Baseline, and Lightweight U-Net).
4. Reference-based quantitative evaluation (PSNR/SSIM) when ground truth is provided.
5. Multi-Model Head-to-Head Comparison across arbitrary checkpoints.
6. Memory-safe Offline Video Dehazing (frame-by-frame) with FPS profiling and audio preservation.
7. Graceful fallback to Classical DCP baseline when no neural checkpoints are installed.
"""

from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is available in module path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import streamlit as st
import torch

from app.config import CHECKPOINTS_DIR, OUTPUT_VIDEOS_DIR, PRETRAINED_MODELS_DIR, get_default_device
from src.dehazing.dcp import dcp_dehaze
from src.dehazing.external.dehazeformer_adapter import DehazeFormerAdapter
from src.dehazing.model import LightweightUNet, load_checkpoint
from src.detection.density import estimate_haze_density
from src.metrics.psnr import calculate_psnr
from src.metrics.ssim import calculate_ssim
from src.utils.image_ops import numpy_to_tensor, tensor_to_numpy
from src.video.processor import get_ffmpeg_path, process_video_file

# Streamlit Page Setup
st.set_page_config(
    page_title="AI Atmospheric Visibility Enhancement",
    page_icon="🌫️",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_cached_unet(checkpoint_path_str: str, device_str: str) -> Tuple[LightweightUNet, Dict[str, Any]]:
    """Load and cache LightweightUNet model instance to prevent redundant disk I/O."""
    chk_path = Path(checkpoint_path_str)
    device = torch.device(device_str)
    raw_chk = torch.load(str(chk_path), map_location=device, weights_only=False)
    chk_config = raw_chk.get("config", {})
    base_channels = chk_config.get("base_channels", 24)
    residual = chk_config.get("residual", True)

    model = LightweightUNet(
        in_channels=3,
        out_channels=3,
        base_channels=base_channels,
        residual=residual,
    ).to(device)

    load_checkpoint(chk_path, model=model, map_location=device_str)
    model.eval()
    return model, raw_chk


@st.cache_resource(show_spinner=False)
def load_cached_dehazeformer(checkpoint_path_str: str, device_str: str) -> DehazeFormerAdapter:
    """Load and cache DehazeFormerAdapter instance."""
    return DehazeFormerAdapter(checkpoint_path_str, device=device_str)


def get_available_models() -> Dict[str, Dict[str, Any]]:
    """Discover all selectable models (Pretrained DehazeFormer and Lightweight U-Net checkpoints)."""
    models: Dict[str, Dict[str, Any]] = {}

    # 1. Pretrained models (DehazeFormer-MCT)
    if PRETRAINED_MODELS_DIR.is_dir():
        for p in sorted(PRETRAINED_MODELS_DIR.glob("*.pth")):
            if "dehazeformer" in p.name.lower():
                display_label = "DehazeFormer-MCT (Pretrained SOTA - Unified)"
                models[display_label] = {
                    "id": p.stem,
                    "type": "dehazeformer",
                    "path": p,
                    "variant": "Multi-Scale Context Transformer (MCT)",
                    "source": "Official IDKiro / HuggingFace Space (MIT License)",
                    "params": "1,233,024 (~1.23M)",
                    "file_name": p.name,
                }

    # 2. Local U-Net checkpoints
    if CHECKPOINTS_DIR.is_dir():
        priority_order = [
            "mixed_pilot_best_model.pth",
            "medium_best_model.pth",
            "pilot_best_model.pth",
            "best_model.pth",
        ]
        all_chks = sorted(list(CHECKPOINTS_DIR.glob("*.pth")), key=lambda p: (
            priority_order.index(p.name) if p.name in priority_order else 999,
            p.name
        ))
        for p in all_chks:
            display_label = f"Lightweight U-Net: {p.name}"
            models[display_label] = {
                "id": p.stem,
                "type": "unet",
                "path": p,
                "variant": "Lightweight Residual U-Net (24 base ch)",
                "source": f"Local Training Run (models/checkpoints/{p.name})",
                "params": "1,084,731 (~1.08M)",
                "file_name": p.name,
            }

    return models


def main() -> None:
    st.title("🌫️ AI Atmospheric Visibility Enhancement System")
    st.markdown(
        "**Stage 2C.3 Multi-Model Benchmarking**: Pretrained DehazeFormer-MCT, "
        "Unified Lightweight U-Net, Classical DCP Baseline, and Offline Video Processing."
    )

    device_str = get_default_device()
    device = torch.device(device_str)

    # Sidebar Information & Settings
    st.sidebar.header("⚙️ Environment & Hardware")
    st.sidebar.markdown(f"**Compute Device**: `{device_str.upper()}` (Intel Core Ultra 5 125U)")
    ffmpeg_available = get_ffmpeg_path() is not None
    st.sidebar.markdown(f"**FFmpeg Audio Engine**: `{'Available ✅' if ffmpeg_available else 'Not Found ⚠️'}`")

    model_registry = get_available_models()
    has_models = len(model_registry) > 0
    model_labels = list(model_registry.keys())

    if not has_models:
        st.info(
            "ℹ️ **Baseline Mode Active: No trained checkpoints detected.**\n\n"
            "The **Classical Dark Channel Prior (DCP)** baseline and **Atmospheric Diagnostics** are ready to use. "
            "To enable deep learning models, please obtain or train a model checkpoint."
        )
        with st.expander("📥 How to Obtain & Configure AI Checkpoints", expanded=False):
            st.markdown("""
            **Option 1: Download Official Pretrained DehazeFormer-MCT (Recommended - 5.65 MB)**
            Run this single command in your terminal:
            ```bash
            python scripts/download_pretrained.py
            ```
            This downloads and verifies the official MIT-licensed checkpoint into `models/pretrained/dehazeformer_mct.pth`.

            **Option 2: Train a Lightweight U-Net Locally**
            Place your dataset in `data/raw/RESIDE` and run:
            ```bash
            python scripts/train.py --train-dir data/processed/reside/train --val-dir data/processed/reside/val --epochs 5
            ```
            Checkpoints are automatically stored in `models/checkpoints/`.
            """)

    # Tabs Interface
    tab_single, tab_compare, tab_video = st.tabs([
        "📸 Single Image Dehazing & Inspection",
        "⚖️ Multi-Model Comparison",
        "🎬 Offline Video Dehazing",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Single Image Dehazing & Inspection
    # -------------------------------------------------------------------------
    with tab_single:
        st.subheader("Image Atmospheric Inspection & Enhancement")
        st.caption("Upload any indoor, outdoor, hazy, or clear image to test dehazing and diagnose haze density.")

        if has_models:
            col_s1, col_s2 = st.columns([1.2, 1.0])
            with col_s1:
                active_model_label = st.selectbox(
                    "Active AI Model",
                    model_labels,
                    index=0,
                    key="single_model_sel",
                )
                active_model_meta = model_registry[active_model_label]

                st.caption(
                    f"**Architecture**: `{active_model_meta['variant']}` | "
                    f"**Params**: `{active_model_meta['params']}`\n\n"
                    f"**Source**: `{active_model_meta['source']}`"
                )

            with col_s2:
                include_dcp = st.checkbox(
                    "Run Classical DCP Baseline Alongside AI",
                    value=True,
                    help="Computes Dark Channel Prior with Guided Filter for side-by-side comparison.",
                )

                compare_unet_label = None
                if active_model_meta["type"] == "dehazeformer":
                    include_unet_cmp = st.checkbox(
                        "Include Lightweight U-Net Comparison",
                        value=True,
                        help="Renders local U-Net model side-by-side to visually inspect differences.",
                    )
                    if include_unet_cmp:
                        unet_opts = [lbl for lbl, meta in model_registry.items() if meta["type"] == "unet"]
                        if unet_opts:
                            compare_unet_label = st.selectbox("Compare With U-Net Checkpoint", unet_opts, index=0)
                else:
                    include_unet_cmp = False
        else:
            active_model_label = None
            active_model_meta = None
            include_dcp = True
            compare_unet_label = None
            st.info("🔧 Active Algorithm: **Classical Dark Channel Prior (DCP) with Guided Filter** (Hardware Independent Baseline)")

        col_up1, col_up2 = st.columns(2)
        with col_up1:
            uploaded_hazy = st.file_uploader(
                "Upload Input Image (JPG, PNG, WebP, BMP)",
                type=["png", "jpg", "jpeg", "bmp", "webp"],
                key="single_hazy_uploader",
            )
        with col_up2:
            uploaded_clear = st.file_uploader(
                "Upload Ground-Truth Clear Image (Optional)",
                type=["png", "jpg", "jpeg", "bmp", "webp"],
                key="single_clear_uploader",
                help="If uploaded, objective reference metrics (PSNR & SSIM) will be calculated.",
            )

        if uploaded_hazy is not None:
            file_bytes = np.frombuffer(uploaded_hazy.read(), np.uint8)
            bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if bgr is None:
                st.error("Failed to decode uploaded image file. Please provide a valid, uncorrupted image.")
            else:
                hazy_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                h, w, _ = hazy_rgb.shape

                # 1. Haze Density Diagnostics
                diag = estimate_haze_density(hazy_rgb)
                sev = diag["severity_label"]

                if sev == "Clear / Near-Clear":
                    st.info(f"🌿 **Atmospheric Diagnostic**: `{sev}` — {diag['recommended_action']}")
                elif sev == "Light Haze":
                    st.warning(f"🌤️ **Atmospheric Diagnostic**: `{sev}` — {diag['recommended_action']}")
                else:
                    st.error(f"🌫️ **Atmospheric Diagnostic**: `{sev}` — {diag['recommended_action']}")

                st.caption(
                    f"Mean Dark Channel: `{diag['dark_channel_mean']:.4f}` | "
                    f"P95 Dark Channel: `{diag['dark_channel_p95']:.4f}` | "
                    f"Contrast Std: `{diag['contrast_std']:.4f}`"
                )

                # Ground Truth Clear Image Handling
                clear_rgb = None
                if uploaded_clear is not None:
                    c_bytes = np.frombuffer(uploaded_clear.read(), np.uint8)
                    c_bgr = cv2.imdecode(c_bytes, cv2.IMREAD_COLOR)
                    if c_bgr is not None:
                        clear_rgb = cv2.cvtColor(c_bgr, cv2.COLOR_BGR2RGB)
                        if clear_rgb.shape[:2] != (h, w):
                            clear_rgb = cv2.resize(clear_rgb, (w, h), interpolation=cv2.INTER_AREA)

                if st.button("🚀 Enhance Visibility", type="primary", key="btn_run_single"):
                    with st.spinner("Processing image through dehazing pipeline..."):
                        try:
                            ai_rgb = None
                            ai_time_breakdown = {}

                            # 1. Run Active AI Model if available
                            if active_model_meta is not None:
                                if active_model_meta["type"] == "dehazeformer":
                                    df_adapter = load_cached_dehazeformer(str(active_model_meta["path"]), device_str)
                                    ai_rgb, ai_time_breakdown = df_adapter.dehaze(hazy_rgb)
                                else:
                                    unet_m, _ = load_cached_unet(str(active_model_meta["path"]), device_str)
                                    t0_pre = time.perf_counter()
                                    t_in = numpy_to_tensor(hazy_rgb, device=device).unsqueeze(0)
                                    t_pre = (time.perf_counter() - t0_pre) * 1000.0

                                    t0_infer = time.perf_counter()
                                    with torch.no_grad():
                                        t_out = unet_m(t_in)
                                    t_infer = (time.perf_counter() - t0_infer) * 1000.0

                                    t0_post = time.perf_counter()
                                    ai_rgb = tensor_to_numpy(t_out.squeeze(0))
                                    t_post = (time.perf_counter() - t0_post) * 1000.0

                                    ai_time_breakdown = {
                                        "pre_ms": round(t_pre, 2),
                                        "infer_ms": round(t_infer, 2),
                                        "post_ms": round(t_post, 2),
                                        "total_ms": round(t_pre + t_infer + t_post, 2),
                                    }

                            # 2. Run Classical DCP if enabled
                            dcp_rgb = None
                            dcp_time_ms = 0.0
                            if include_dcp:
                                t0_dcp = time.perf_counter()
                                dcp_rgb, _ = dcp_dehaze(hazy_rgb, use_guided_filter=True)
                                dcp_time_ms = (time.perf_counter() - t0_dcp) * 1000.0

                            # 3. Run Optional U-Net comparison if enabled
                            unet_cmp_rgb = None
                            unet_cmp_time_ms = 0.0
                            if compare_unet_label and model_registry:
                                u_meta = model_registry[compare_unet_label]
                                u_m, _ = load_cached_unet(str(u_meta["path"]), device_str)
                                t0_u = time.perf_counter()
                                t_in_u = numpy_to_tensor(hazy_rgb, device=device).unsqueeze(0)
                                with torch.no_grad():
                                    t_out_u = u_m(t_in_u)
                                unet_cmp_rgb = tensor_to_numpy(t_out_u.squeeze(0))
                                unet_cmp_time_ms = (time.perf_counter() - t0_u) * 1000.0

                        except Exception as err:
                            st.error(f"Inference error encountered: {err}")
                            return

                    # Metadata Metric Cards
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    col_m1.metric("Resolution", f"{w} × {h}")
                    if active_model_meta is not None:
                        col_m2.metric("Selected Model", active_model_meta["id"])
                        col_m3.metric(
                            "AI Total Latency",
                            f"{ai_time_breakdown['total_ms']:.1f} ms",
                            help=f"Forward Infer: {ai_time_breakdown['infer_ms']:.1f} ms | Pre/Post: {ai_time_breakdown['pre_ms'] + ai_time_breakdown['post_ms']:.1f} ms"
                        )
                        if include_dcp:
                            col_m4.metric("DCP Baseline", f"{dcp_time_ms:.1f} ms")
                        elif compare_unet_label:
                            col_m4.metric("U-Net Comparison", f"{unet_cmp_time_ms:.1f} ms")
                        else:
                            col_m4.metric("Parameters", active_model_meta["params"])
                    else:
                        col_m2.metric("Enhancement Engine", "Classical DCP")
                        col_m3.metric("Processing Latency", f"{dcp_time_ms:.1f} ms")
                        col_m4.metric("Engine Type", "Physical Prior (No Weights)")

                    # Visual Displays
                    st.markdown("---")
                    st.subheader("🖼️ Comparative Visualizations")

                    disp_items = [("Original Input Image", hazy_rgb)]
                    if ai_rgb is not None and active_model_meta is not None:
                        disp_items.append((f"{active_model_meta['id']} ({ai_time_breakdown['total_ms']:.1f} ms)", ai_rgb))
                    if include_dcp and dcp_rgb is not None:
                        disp_items.append((f"Classical DCP ({dcp_time_ms:.1f} ms)", dcp_rgb))
                    if compare_unet_label and unet_cmp_rgb is not None:
                        disp_items.append((f"{model_registry[compare_unet_label]['id']} ({unet_cmp_time_ms:.1f} ms)", unet_cmp_rgb))

                    v_cols = st.columns(len(disp_items))
                    for col_v, (lbl, img_disp) in zip(v_cols, disp_items):
                        with col_v:
                            st.image(img_disp, caption=lbl, use_container_width=True)

                    # Provide download option for enhanced image
                    download_img = ai_rgb if ai_rgb is not None else dcp_rgb
                    download_tag = active_model_meta["id"] if active_model_meta is not None else "dcp"
                    if download_img is not None:
                        is_success, buffer = cv2.imencode(".png", cv2.cvtColor(download_img, cv2.COLOR_RGB2BGR))
                        if is_success:
                            st.download_button(
                                label=f"⬇️ Download {download_tag} Enhanced Image (PNG)",
                                data=buffer.tobytes(),
                                file_name=f"dehazed_{Path(uploaded_hazy.name).stem}_{download_tag}.png",
                                mime="image/png",
                            )

                    # Quantitative Metrics vs Visual-Only Assessment Notice
                    st.markdown("---")
                    if clear_rgb is not None:
                        st.subheader("📊 Objective Reference-Based Benchmarks (Ground Truth Available)")
                        p_hazy = calculate_psnr(hazy_rgb, clear_rgb, max_val=255.0)
                        s_hazy = calculate_ssim(hazy_rgb, clear_rgb, max_val=255.0)

                        qb_cols = st.columns(len(disp_items))
                        with qb_cols[0]:
                            st.info(f"**Input Image**\n\nPSNR: `{p_hazy:.2f} dB`\n\nSSIM: `{s_hazy:.4f}`")

                        col_curr = 1
                        if ai_rgb is not None and active_model_meta is not None:
                            p_ai = calculate_psnr(ai_rgb, clear_rgb, max_val=255.0)
                            s_ai = calculate_ssim(ai_rgb, clear_rgb, max_val=255.0)
                            with qb_cols[col_curr]:
                                st.success(
                                    f"**{active_model_meta['id']}**\n\nPSNR: `{p_ai:.2f} dB` ({p_ai - p_hazy:+.2f} dB)\n\n"
                                    f"SSIM: `{s_ai:.4f}` ({s_ai - s_hazy:+.4f})"
                                )
                            col_curr += 1

                        if include_dcp and dcp_rgb is not None:
                            p_dcp = calculate_psnr(dcp_rgb, clear_rgb, max_val=255.0)
                            s_dcp = calculate_ssim(dcp_rgb, clear_rgb, max_val=255.0)
                            with qb_cols[col_curr]:
                                st.info(
                                    f"**Classical DCP**\n\nPSNR: `{p_dcp:.2f} dB` ({p_dcp - p_hazy:+.2f} dB)\n\n"
                                    f"SSIM: `{s_dcp:.4f}` ({s_dcp - s_hazy:+.4f})"
                                )
                            col_curr += 1

                        if compare_unet_label and unet_cmp_rgb is not None:
                            p_u = calculate_psnr(unet_cmp_rgb, clear_rgb, max_val=255.0)
                            s_u = calculate_ssim(unet_cmp_rgb, clear_rgb, max_val=255.0)
                            with qb_cols[col_curr]:
                                st.info(
                                    f"**{model_registry[compare_unet_label]['id']}**\n\n"
                                    f"PSNR: `{p_u:.2f} dB` ({p_u - p_hazy:+.2f} dB)\n\n"
                                    f"SSIM: `{s_u:.4f}` ({s_u - s_hazy:+.4f})"
                                )
                    else:
                        st.subheader("👁️ Visual-Only Assessment")
                        st.info(
                            "ℹ️ **No ground-truth clean image was provided.** "
                            "Quantitative metrics (PSNR and SSIM) require an aligned reference image. "
                            "Inspect structural edges, color saturation, and dark shadows visually."
                        )

    # -------------------------------------------------------------------------
    # TAB 2: Multi-Model Head-to-Head Comparison
    # -------------------------------------------------------------------------
    with tab_compare:
        st.subheader("Multi-Model Head-to-Head Comparison on the Same Image")
        st.caption("Select multiple AI models (pretrained DehazeFormer, trained U-Net variants) to evaluate side-by-side.")

        if not has_models:
            st.info(
                "⚖️ **Multi-Model Comparison requires at least one trained neural network checkpoint.**\n\n"
                "Run `python scripts/download_pretrained.py` in your terminal to obtain the official DehazeFormer-MCT checkpoint, "
                "or train a local model using `python scripts/train.py`."
            )
        else:
            default_cmp = [lbl for lbl in model_labels if "DehazeFormer" in lbl or "mixed_pilot_best" in lbl or "medium_best" in lbl][:3]
            selected_compare_models = st.multiselect(
                "Select Models to Compare",
                model_labels,
                default=default_cmp,
            )

            cmp_upload = st.file_uploader(
                "Upload Image for Multi-Model Testing",
                type=["png", "jpg", "jpeg", "bmp", "webp"],
                key="cmp_uploader",
            )

            if cmp_upload is not None and selected_compare_models:
                c_bytes = np.frombuffer(cmp_upload.read(), np.uint8)
                c_bgr = cv2.imdecode(c_bytes, cv2.IMREAD_COLOR)
                if c_bgr is not None:
                    img_rgb = cv2.cvtColor(c_bgr, cv2.COLOR_BGR2RGB)
                    ch, cw, _ = img_rgb.shape

                    if st.button("⚖️ Run Comparative Inference", type="primary", key="btn_run_cmp"):
                        cols = st.columns(len(selected_compare_models) + 1)
                        with cols[0]:
                            st.image(img_rgb, caption=f"Original Input ({cw}×{ch})", use_container_width=True)

                        for idx, m_lbl in enumerate(selected_compare_models):
                            m_meta = model_registry[m_lbl]
                            with st.spinner(f"Evaluating `{m_meta['id']}`..."):
                                try:
                                    if m_meta["type"] == "dehazeformer":
                                        df_a = load_cached_dehazeformer(str(m_meta["path"]), device_str)
                                        out_rgb, timing = df_a.dehaze(img_rgb)
                                        lat_ms = timing["total_ms"]
                                    else:
                                        u_m, _ = load_cached_unet(str(m_meta["path"]), device_str)
                                        t_in = numpy_to_tensor(img_rgb, device=device).unsqueeze(0)
                                        t0 = time.perf_counter()
                                        with torch.no_grad():
                                            tout = u_m(t_in)
                                        lat_ms = (time.perf_counter() - t0) * 1000.0
                                        out_rgb = tensor_to_numpy(tout.squeeze(0))

                                    with cols[idx + 1]:
                                        st.image(
                                            out_rgb,
                                            caption=f"{m_meta['id']}\n({lat_ms:.1f} ms)",
                                            use_container_width=True,
                                        )
                                except Exception as ex:
                                    with cols[idx + 1]:
                                        st.error(f"Error evaluating {m_meta['id']}: {ex}")

    # -------------------------------------------------------------------------
    # TAB 3: Offline Video Dehazing
    # -------------------------------------------------------------------------
    with tab_video:
        st.subheader("Offline Video Dehazing & Processing")
        st.markdown(
            "> [!NOTE]\n"
            "> **Offline Processing Notice**: Video enhancement executes sequentially frame-by-frame on CPU. "
            "Real-time stream processing will be developed in future stages."
        )

        vcol1, vcol2 = st.columns([1.2, 1.0])
        with vcol1:
            if has_models:
                video_model_lbl = st.selectbox(
                    "Enhancement Model",
                    model_labels,
                    index=0,
                    key="video_model_sel",
                )
                v_model_meta = model_registry[video_model_lbl]
            else:
                video_model_lbl = None
                v_model_meta = None
                st.info("Enhancement Model: Classical Dark Channel Prior (Physical Baseline)")

        with vcol2:
            if has_models:
                v_method = st.radio(
                    "Enhancement Algorithm",
                    ["ai", "dcp"],
                    horizontal=True,
                    format_func=lambda x: f"AI ({v_model_meta['id']})" if x == "ai" else "Classical DCP",
                )
            else:
                v_method = "dcp"
                st.write("**Algorithm**: Classical DCP")

        col_vopt1, col_vopt2 = st.columns(2)
        with col_vopt1:
            frame_limit = st.slider(
                "Max Frames to Process (0 = Entire Video)",
                min_value=0,
                max_value=600,
                value=60,
                step=10,
                help="Process a subset of frames for fast testing without waiting for long renders on CPU.",
            )
        with col_vopt2:
            try_audio = st.checkbox(
                "Preserve Original Audio (Requires FFmpeg)",
                value=ffmpeg_available,
                disabled=not ffmpeg_available,
                help="When enabled, muxes source audio into output MP4 using FFmpeg.",
            )
            if not ffmpeg_available:
                st.caption("⚠️ FFmpeg is not detected in system PATH. Audio preservation is unavailable.")

        video_file = st.file_uploader(
            "Upload Video File (MP4, AVI, MOV, MKV)",
            type=["mp4", "avi", "mov", "mkv"],
            key="video_uploader",
        )

        if video_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(video_file.name).suffix) as tmp_in:
                tmp_in.write(video_file.read())
                temp_video_input = Path(tmp_in.name)

            st.video(str(temp_video_input))

            if st.button("🎬 Start Offline Video Enhancement", type="primary", key="btn_run_video"):
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                def on_video_progress(current_f: int, total_f: int, elapsed_s: float) -> None:
                    pct = min(1.0, current_f / max(1, total_f))
                    fps_val = current_f / max(elapsed_s, 1e-4)
                    progress_bar.progress(pct)
                    status_text.text(
                        f"Processed {current_f} / {total_f} frames ({pct*100:.1f}%) | "
                        f"Speed: {fps_val:.1f} FPS | Elapsed: {elapsed_s:.1f}s"
                    )

                with st.spinner("Processing video frames..."):
                    try:
                        v_model_obj = None
                        if v_method == "ai" and v_model_meta is not None:
                            if v_model_meta["type"] == "dehazeformer":
                                v_model_obj = load_cached_dehazeformer(str(v_model_meta["path"]), device_str)
                            else:
                                v_model_obj, _ = load_cached_unet(str(v_model_meta["path"]), device_str)

                        max_f_arg = frame_limit if frame_limit > 0 else None
                        res_summary = process_video_file(
                            input_path=temp_video_input,
                            model=v_model_obj,
                            method=v_method,
                            device_str=device_str,
                            max_frames=max_f_arg,
                            progress_callback=on_video_progress,
                            preserve_audio=try_audio,
                        )

                        progress_bar.progress(1.0)
                        status_text.text("Enhancement completed successfully!")

                        st.success(
                            f"✅ Processed **{res_summary['frames_processed']} frames** "
                            f"in **{res_summary['total_duration_sec']:.2f} s** "
                            f"({res_summary['processing_fps']:.2f} FPS average)."
                        )

                        if res_summary["audio_preserved"]:
                            st.info(f"🔊 {res_summary['audio_message']}")
                        else:
                            st.caption(f"ℹ️ {res_summary['audio_message']}")

                        out_path = Path(res_summary["output_file"])
                        if out_path.is_file():
                            st.subheader("Enhanced Output Video")
                            st.video(str(out_path))
                            with open(out_path, "rb") as f_out:
                                st.download_button(
                                    label="⬇️ Download Enhanced Video",
                                    data=f_out,
                                    file_name=out_path.name,
                                    mime="video/mp4",
                                )

                    except Exception as ex:
                        st.error(f"Video enhancement error: {ex}")
                    finally:
                        temp_video_input.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
