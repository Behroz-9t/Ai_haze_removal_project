"""DehazeVision — Neural Image Dehazing Platform.

An End-to-End Deep Learning Framework for Atmospheric Haze Removal.
Academic Enterprise Dashboard supporting SOTA Vision Transformers (DehazeFormer-MCT),
Lightweight Residual U-Net, and Classical Dark Channel Prior (DCP) baseline.
"""

from pathlib import Path
import sys
import tempfile
import time
import io
import base64
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is available in module path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components
import torch

from app.config import CHECKPOINTS_DIR, OUTPUT_VIDEOS_DIR, PRETRAINED_MODELS_DIR, get_default_device
from src.dehazing.dcp import dcp_dehaze
from src.dehazing.external.dehazeformer_adapter import DehazeFormerAdapter
from src.dehazing.model import LightweightUNet, load_checkpoint
from src.detection.density import estimate_haze_density
from src.metrics.psnr import calculate_psnr
from src.metrics.ssim import calculate_ssim
from src.utils.image_ops import numpy_to_tensor, tensor_to_numpy
from src.video.processor import process_video_file

# Streamlit Page Setup
st.set_page_config(
    page_title="DehazeVision — Neural Image Dehazing Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Academic Enterprise Dark Navy & Soft Cyan Theme
# -----------------------------------------------------------------------------
def inject_enterprise_styles() -> None:
    """Inject clean, academic enterprise styling with dark navy and soft cyan."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

        :root {
            --bg-base: #0B132B;
            --bg-surface: #111C30;
            --bg-card: #162238;
            --border-color: #24344D;
            --cyan-accent: #38BDF8;
            --cyan-primary: #0284C7;
            --cyan-subtle: rgba(56, 189, 248, 0.12);
            --text-title: #F8FAFC;
            --text-body: #E2E8F0;
            --text-muted: #94A3B8;
        }

        /* Base Page Background */
        html, body, [data-testid="stAppViewContainer"] {
            background-color: var(--bg-base) !important;
            color: var(--text-body) !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }

        /* Clean Header */
        .platform-header {
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 16px;
            margin-bottom: 24px;
        }
        .platform-title {
            font-size: 1.85rem !important;
            font-weight: 700 !important;
            color: var(--text-title) !important;
            letter-spacing: -0.02em !important;
            margin: 0 0 6px 0 !important;
        }
        .platform-title span {
            color: var(--cyan-accent);
        }
        .platform-subtitle {
            font-size: 0.95rem;
            color: var(--text-muted);
            margin: 0;
            font-weight: 400;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #0D1629 !important;
            border-right: 1px solid var(--border-color) !important;
            padding-top: 20px;
        }
        .sidebar-card {
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 14px;
            margin-top: 18px;
            font-size: 0.85rem;
        }
        .sidebar-card-title {
            font-weight: 600;
            color: var(--cyan-accent);
            margin-bottom: 8px;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* Tabs Styling */
        [data-baseweb="tab-list"] {
            background-color: var(--bg-surface) !important;
            border-radius: 8px !important;
            padding: 4px !important;
            border: 1px solid var(--border-color) !important;
            gap: 4px !important;
            margin-bottom: 22px !important;
        }
        [data-baseweb="tab"] {
            color: var(--text-muted) !important;
            border-radius: 6px !important;
            padding: 8px 18px !important;
            font-weight: 500 !important;
            font-size: 0.92rem !important;
            border: none !important;
            background: transparent !important;
            transition: all 0.2s ease !important;
        }
        [data-baseweb="tab"][aria-selected="true"] {
            background: var(--bg-card) !important;
            color: var(--cyan-accent) !important;
            border: 1px solid var(--border-color) !important;
            font-weight: 600 !important;
        }

        /* Metric Cards */
        .metric-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 14px;
            margin: 18px 0;
        }
        .academic-metric-card {
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 14px 18px;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .metric-title {
            font-size: 0.78rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .metric-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.55rem;
            font-weight: 700;
            color: var(--cyan-accent);
        }
        .metric-sub {
            font-size: 0.78rem;
            color: var(--text-muted);
        }
        .metric-sub.positive {
            color: #34D399;
        }

        /* Clean Card Wrappers */
        .academic-card {
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 18px;
            margin-bottom: 18px;
        }
        .panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 12px;
            background: #152033;
            border: 1px solid var(--border-color);
            border-bottom: none;
            border-radius: 6px 6px 0 0;
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-body);
        }
        .panel-badge {
            background: var(--cyan-subtle);
            color: var(--cyan-accent);
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 4px;
            padding: 2px 8px;
            font-size: 0.75rem;
            font-family: 'JetBrains Mono', monospace;
        }

        /* Clean Buttons */
        .stButton > button[kind="primary"], .stButton > button {
            background: var(--cyan-primary) !important;
            color: #FFFFFF !important;
            border: 1px solid rgba(56, 189, 248, 0.4) !important;
            border-radius: 6px !important;
            padding: 9px 20px !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            transition: all 0.2s ease !important;
            box-shadow: none !important;
            width: 100% !important;
        }
        .stButton > button:hover {
            background: #0275B1 !important;
            border-color: var(--cyan-accent) !important;
        }
        .stDownloadButton > button {
            background: var(--bg-surface) !important;
            color: var(--cyan-accent) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 6px !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            transition: all 0.2s ease !important;
            width: 100% !important;
        }
        .stDownloadButton > button:hover {
            background: var(--bg-card) !important;
            border-color: var(--cyan-accent) !important;
        }

        /* File Uploader */
        section[data-testid="stFileUploadDropzone"] {
            background: var(--bg-surface) !important;
            border: 1px dashed var(--border-color) !important;
            border-radius: 8px !important;
            padding: 20px 14px !important;
            transition: border-color 0.2s ease !important;
        }
        section[data-testid="stFileUploadDropzone"]:hover {
            border-color: var(--cyan-accent) !important;
        }

        /* Diagnostic Banner */
        .diagnostic-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 14px;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 500;
            margin: 8px 0 16px 0;
            border: 1px solid var(--border-color);
            background: var(--bg-surface);
        }
        .diagnostic-pill.clear {
            border-color: rgba(52, 211, 153, 0.4);
            color: #34D399;
        }
        .diagnostic-pill.light {
            border-color: rgba(251, 191, 36, 0.4);
            color: #FBBF24;
        }
        .diagnostic-pill.dense {
            border-color: rgba(56, 189, 248, 0.4);
            color: #38BDF8;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Model Caching & Registration
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_cached_unet(checkpoint_path_str: str, device_str: str) -> Tuple[LightweightUNet, Dict[str, Any]]:
    """Load and cache LightweightUNet model instance."""
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
    """Discover all selectable models (Pretrained DehazeFormer, Lightweight U-Net, and Classical DCP)."""
    models: Dict[str, Dict[str, Any]] = {}

    # 1. Pretrained DehazeFormer-MCT
    if PRETRAINED_MODELS_DIR.is_dir():
        for p in sorted(PRETRAINED_MODELS_DIR.glob("*.pth")):
            if "dehazeformer" in p.name.lower():
                display_label = "DehazeFormer-MCT (Pretrained SOTA Transformer)"
                models[display_label] = {
                    "id": "DehazeFormer-MCT",
                    "type": "dehazeformer",
                    "path": p,
                    "variant": "Vision Transformer (MCT)",
                    "params": "1.23M",
                    "file_name": p.name,
                }

    # 2. Local Lightweight U-Net Checkpoints
    if CHECKPOINTS_DIR.is_dir():
        all_chks = sorted(list(CHECKPOINTS_DIR.glob("*.pth")))
        for p in all_chks:
            display_label = f"Lightweight Residual U-Net ({p.stem})"
            models[display_label] = {
                "id": p.stem,
                "type": "unet",
                "path": p,
                "variant": "Residual U-Net (24 base channels)",
                "params": "1.08M",
                "file_name": p.name,
            }

    # 3. Always register Classical DCP baseline
    models["Classical Dark Channel Prior (DCP Baseline)"] = {
        "id": "Classical DCP",
        "type": "dcp",
        "path": None,
        "variant": "Physical Prior (He et al.)",
        "params": "0 (Prior-Based)",
        "file_name": "None",
    }

    return models


# -----------------------------------------------------------------------------
# Helper: Base64 Encoding & Comparison Slider
# -----------------------------------------------------------------------------
def numpy_to_base64_jpeg(rgb_arr: np.ndarray, quality: int = 92) -> str:
    """Encode an RGB numpy image array to base64 JPEG data URL."""
    pil_img = Image.fromarray(rgb_arr)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def render_academic_comparison_slider(
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    label_before: str = "Hazy Input",
    label_after: str = "Dehazed Output",
) -> None:
    """Render a clean, high-contrast before-and-after comparison slider."""
    h, w, _ = before_rgb.shape
    aspect_ratio = max(0.6, min(2.4, w / max(1, h)))
    slider_height = int(min(580, max(360, 800 / aspect_ratio)))
    total_iframe_height = slider_height + 60

    before_b64 = numpy_to_base64_jpeg(before_rgb)
    after_b64 = numpy_to_base64_jpeg(after_rgb)

    slider_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
    * {{
        box-sizing: border-box;
        margin: 0;
        padding: 0;
        user-select: none;
    }}
    body {{
        background: transparent;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        display: flex;
        flex-direction: column;
        align-items: center;
        overflow: hidden;
    }}
    .slider-container {{
        position: relative;
        width: 100%;
        max-width: 900px;
        height: {slider_height}px;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #24344D;
        background: #0B132B;
        touch-action: none;
        cursor: ew-resize;
    }}
    .img-layer {{
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        object-fit: contain;
        pointer-events: none;
    }}
    .img-after {{
        z-index: 1;
    }}
    .img-before {{
        z-index: 2;
        clip-path: polygon(0 0, 50% 0, 50% 100%, 0 100%);
    }}
    .divider-bar {{
        position: absolute;
        top: 0;
        bottom: 0;
        left: 50%;
        width: 2px;
        background: #38BDF8;
        z-index: 3;
        transform: translateX(-50%);
        pointer-events: none;
    }}
    .slider-handle {{
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 38px;
        height: 38px;
        border-radius: 50%;
        background: #111C30;
        border: 2px solid #38BDF8;
        z-index: 4;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #38BDF8;
        pointer-events: none;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
    }}
    .badge {{
        position: absolute;
        top: 12px;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        z-index: 5;
        pointer-events: none;
    }}
    .badge-before {{
        left: 12px;
        background: rgba(13, 22, 41, 0.9);
        color: #CBD5E1;
        border: 1px solid #334155;
    }}
    .badge-after {{
        right: 12px;
        background: rgba(13, 22, 41, 0.9);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.4);
    }}
    .split-pill {{
        position: absolute;
        bottom: 10px;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(13, 22, 41, 0.9);
        color: #94A3B8;
        font-size: 11px;
        font-family: monospace;
        padding: 3px 10px;
        border-radius: 4px;
        border: 1px solid #24344D;
        z-index: 5;
        pointer-events: none;
    }}
    .toolbar {{
        display: flex;
        gap: 6px;
        margin-top: 8px;
        justify-content: center;
    }}
    .tool-btn {{
        background: #111C30;
        color: #94A3B8;
        border: 1px solid #24344D;
        border-radius: 4px;
        padding: 4px 12px;
        font-size: 11px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.15s ease;
    }}
    .tool-btn:hover {{
        color: #38BDF8;
        border-color: #38BDF8;
    }}
    .tool-btn.active {{
        background: #162238;
        color: #38BDF8;
        border-color: #38BDF8;
    }}
    </style>
    </head>
    <body>
    <div class="slider-container" id="slider">
        <img class="img-layer img-after" src="{after_b64}" alt="Dehazed" />
        <img class="img-layer img-before" id="beforeLayer" src="{before_b64}" alt="Hazy" />
        <div class="divider-bar" id="dividerBar"></div>
        <div class="slider-handle" id="sliderHandle">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="15 18 9 12 15 6"></polyline>
                <polyline points="9 18 15 12 9 6" transform="translate(24, 0) scale(-1, 1)"></polyline>
            </svg>
        </div>
        <div class="badge badge-before">{label_before}</div>
        <div class="badge badge-after">{label_after}</div>
        <div class="split-pill" id="splitPill">50% Split</div>
    </div>
    <div class="toolbar">
        <button class="tool-btn" onclick="setPos(0)">100% Hazy</button>
        <button class="tool-btn" onclick="setPos(25)">25%</button>
        <button class="tool-btn active" id="btn50" onclick="setPos(50)">50% Split</button>
        <button class="tool-btn" onclick="setPos(75)">75%</button>
        <button class="tool-btn" onclick="setPos(100)">100% Dehazed</button>
    </div>

    <script>
    const slider = document.getElementById('slider');
    const beforeLayer = document.getElementById('beforeLayer');
    const dividerBar = document.getElementById('dividerBar');
    const sliderHandle = document.getElementById('sliderHandle');
    const splitPill = document.getElementById('splitPill');
    let isDragging = false;

    function updateSlider(pct) {{
        pct = Math.max(0, Math.min(100, pct));
        beforeLayer.style.clipPath = `polygon(0 0, ${{pct}}% 0, ${{pct}}% 100%, 0 100%)`;
        dividerBar.style.left = `${{pct}}%`;
        sliderHandle.style.left = `${{pct}}%`;
        splitPill.textContent = `${{Math.round(pct)}}% Split`;
        document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
        if (Math.round(pct) === 50) document.getElementById('btn50')?.classList.add('active');
    }}

    slider.addEventListener('pointerdown', (e) => {{
        isDragging = true;
        slider.setPointerCapture(e.pointerId);
        const rect = slider.getBoundingClientRect();
        updateSlider(((e.clientX - rect.left) / rect.width) * 100);
    }});

    slider.addEventListener('pointermove', (e) => {{
        if (!isDragging) return;
        const rect = slider.getBoundingClientRect();
        updateSlider(((e.clientX - rect.left) / rect.width) * 100);
    }});

    slider.addEventListener('pointerup', (e) => {{
        isDragging = false;
        try {{ slider.releasePointerCapture(e.pointerId); }} catch(err){{}}
    }});

    function setPos(pct) {{
        updateSlider(pct);
    }}

    updateSlider(50);
    </script>
    </body>
    </html>
    """

    components.html(slider_html, height=total_iframe_height, scrolling=False)


# -----------------------------------------------------------------------------
# Main Application Entry Point
# -----------------------------------------------------------------------------
def main() -> None:
    inject_enterprise_styles()

    device_str = get_default_device()
    device = torch.device(device_str)

    # 1. Clean Academic Header & Title
    st.markdown(
        """
        <div class="platform-header">
            <h1 class="platform-title">DehazeVision <span>— Neural Image Dehazing Platform</span></h1>
            <p class="platform-subtitle">An End-to-End Deep Learning Framework for Atmospheric Haze Removal & Quality Restoration</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Clean Sidebar: Essential Controls Only
    with st.sidebar:
        st.markdown(
            """
            <div style="font-size: 0.95rem; font-weight: 600; color: #F8FAFC; margin-bottom: 14px;">
                Configuration & Parameters
            </div>
            """,
            unsafe_allow_html=True,
        )

        model_registry = get_available_models()
        model_labels = list(model_registry.keys())

        # Model Selection dropdown
        selected_model_label = st.selectbox(
            "Primary Inference Model",
            model_labels,
            index=0,
            help="Select the deep learning model or physical baseline to execute.",
        )
        selected_model_meta = model_registry[selected_model_label]

        st.caption(
            f"**Architecture**: `{selected_model_meta['variant']}`\n\n"
            f"**Parameters**: `{selected_model_meta['params']}`"
        )

        st.markdown("---")
        st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #CBD5E1;'>Inference Thresholds</div>", unsafe_allow_html=True)

        # Threshold Sliders
        omega_factor = st.slider(
            "Haze Removal Factor (ω)",
            min_value=0.50,
            max_value=1.00,
            value=0.95,
            step=0.01,
            help="Controls haze retention. Standard value is 0.95 (He et al.).",
        )

        guided_radius = st.slider(
            "Guided Filter Radius (r)",
            min_value=10,
            max_value=80,
            value=40,
            step=5,
            help="Spatial window radius for edge-preserving transmission refinement.",
        )

        t0_lower = st.slider(
            "Transmission Lower Bound (t₀)",
            min_value=0.05,
            max_value=0.30,
            value=0.10,
            step=0.01,
            help="Restricts minimum transmission to mitigate noise amplification in dense haze regions.",
        )

        # Clean System Context Card
        st.markdown(
            f"""
            <div class="sidebar-card">
                <div class="sidebar-card-title">System Context</div>
                <div style="color: #94A3B8; line-height: 1.6;">
                    • <strong>Compute Device:</strong> <code>{device_str.upper()}</code><br>
                    • <strong>Framework:</strong> <code>PyTorch 2.x</code><br>
                    • <strong>Precision:</strong> <code>FP32</code><br>
                    • <strong>Status:</strong> Ready
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. Main Workspace Tabs
    tab_single, tab_benchmark, tab_video = st.tabs([
        "Single Image Restoration & Evaluation",
        "Multi-Model Benchmark Comparison",
        "Offline Video Dehazing",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Single Image Restoration & Evaluation
    # -------------------------------------------------------------------------
    with tab_single:
        st.markdown("<div style='font-size: 1.05rem; font-weight: 600; color: #F8FAFC; margin-bottom: 8px;'>Single Image Dehazing Pipeline</div>", unsafe_allow_html=True)

        col_up1, col_up2 = st.columns(2)
        with col_up1:
            uploaded_hazy = st.file_uploader(
                "Upload Hazy Input Image (JPG, PNG, WebP, BMP)",
                type=["png", "jpg", "jpeg", "bmp", "webp"],
                key="tab1_hazy_uploader",
            )
        with col_up2:
            uploaded_clear = st.file_uploader(
                "Upload Ground-Truth Clear Image (Optional)",
                type=["png", "jpg", "jpeg", "bmp", "webp"],
                key="tab1_clear_uploader",
                help="If uploaded, objective reference metrics (PSNR & SSIM) will be calculated.",
            )

        if uploaded_hazy is not None:
            # Detect file upload change and invalidate cached inference result
            file_sig = f"{uploaded_hazy.name}_{uploaded_hazy.size}_{selected_model_label}"
            if st.session_state.get("last_sig") != file_sig:
                st.session_state["active_result"] = None
                st.session_state["last_sig"] = file_sig

            file_bytes = np.frombuffer(uploaded_hazy.read(), np.uint8)
            bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if bgr is None:
                st.error("Failed to decode uploaded image. Please provide a valid, uncorrupted image file.")
            else:
                hazy_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                h, w, _ = hazy_rgb.shape

                # Atmospheric Density Diagnostics
                diag = estimate_haze_density(hazy_rgb)
                sev = diag["severity_label"]
                diag_class = "clear" if "Clear" in sev else ("light" if "Light" in sev else "dense")

                st.markdown(
                    f"""
                    <div class="diagnostic-pill {diag_class}">
                        <span><strong>Atmospheric Density:</strong> {sev}</span>
                        <span>|</span>
                        <span>Dark Channel Mean: <code>{diag['dark_channel_mean']:.4f}</code></span>
                        <span>|</span>
                        <span>Contrast: <code>{diag['contrast_std']:.4f}</code></span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Ground Truth Clear Image
                clear_rgb = None
                if uploaded_clear is not None:
                    c_bytes = np.frombuffer(uploaded_clear.read(), np.uint8)
                    c_bgr = cv2.imdecode(c_bytes, cv2.IMREAD_COLOR)
                    if c_bgr is not None:
                        clear_rgb = cv2.cvtColor(c_bgr, cv2.COLOR_BGR2RGB)
                        if clear_rgb.shape[:2] != (h, w):
                            clear_rgb = cv2.resize(clear_rgb, (w, h), interpolation=cv2.INTER_AREA)

                # Run Inference Action
                btn_run = st.button("Run Dehazing Inference", type="primary", key="btn_run_single")

                if btn_run:
                    with st.spinner(f"Executing {selected_model_meta['id']} inference pipeline..."):
                        try:
                            t_start = time.perf_counter()
                            out_rgb = None
                            time_details: Dict[str, float] = {}

                            # Case A: DehazeFormer-MCT
                            if selected_model_meta["type"] == "dehazeformer":
                                df_adapter = load_cached_dehazeformer(str(selected_model_meta["path"]), device_str)
                                out_rgb, time_details = df_adapter.dehaze(hazy_rgb)
                                exec_time_sec = time_details["total_ms"] / 1000.0

                            # Case B: Lightweight Residual U-Net
                            elif selected_model_meta["type"] == "unet":
                                unet_m, _ = load_cached_unet(str(selected_model_meta["path"]), device_str)
                                t0_infer = time.perf_counter()
                                t_in = numpy_to_tensor(hazy_rgb, device=device).unsqueeze(0)
                                with torch.no_grad():
                                    t_out = unet_m(t_in)
                                out_rgb = tensor_to_numpy(t_out.squeeze(0))
                                exec_time_sec = time.perf_counter() - t0_infer
                                time_details = {
                                    "infer_ms": exec_time_sec * 1000.0,
                                    "total_ms": exec_time_sec * 1000.0,
                                }

                            # Case C: Classical DCP Baseline
                            else:
                                t0_dcp = time.perf_counter()
                                out_rgb, _ = dcp_dehaze(
                                    hazy_rgb,
                                    omega=omega_factor,
                                    guided_radius=guided_radius,
                                    t0=t0_lower,
                                    use_guided_filter=True,
                                )
                                exec_time_sec = time.perf_counter() - t0_dcp
                                time_details = {
                                    "infer_ms": exec_time_sec * 1000.0,
                                    "total_ms": exec_time_sec * 1000.0,
                                }

                            st.session_state["active_result"] = {
                                "out_rgb": out_rgb,
                                "exec_time_sec": exec_time_sec,
                                "time_details": time_details,
                                "model_id": selected_model_meta["id"],
                                "h": h,
                                "w": w,
                            }

                        except Exception as err:
                            st.error(f"Inference execution failed: {err}")
                            return

                # Display Results Workspace
                active_res = st.session_state.get("active_result")
                if active_res is not None:
                    out_rgb = active_res["out_rgb"]
                    exec_time_sec = active_res["exec_time_sec"]
                    model_id = active_res["model_id"]

                    st.markdown("---")

                    # Prominent Quantitative Evaluation Metrics Section (Jira Task 9)
                    st.markdown("<div style='font-size: 0.95rem; font-weight: 600; color: #F8FAFC; margin-bottom: 10px;'>Evaluation Metrics & Execution Performance</div>", unsafe_allow_html=True)

                    # Compute Metrics
                    has_gt = clear_rgb is not None
                    if has_gt:
                        psnr_in = calculate_psnr(hazy_rgb, clear_rgb, max_val=255.0)
                        ssim_in = calculate_ssim(hazy_rgb, clear_rgb, max_val=255.0)
                        psnr_out = calculate_psnr(out_rgb, clear_rgb, max_val=255.0)
                        ssim_out = calculate_ssim(out_rgb, clear_rgb, max_val=255.0)
                        delta_psnr = psnr_out - psnr_in
                        delta_ssim = ssim_out - ssim_in
                        psnr_str = f"{psnr_out:.2f} dB"
                        psnr_sub = f"{delta_psnr:+.2f} dB vs Input"
                        ssim_str = f"{ssim_out:.4f}"
                        ssim_sub = f"{delta_ssim:+.4f} vs Input"
                    else:
                        psnr_str = "N/A"
                        psnr_sub = "Reference Image Required"
                        ssim_str = "N/A"
                        ssim_sub = "Reference Image Required"

                    # 4 Prominent Metric Cards
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    with col_m1:
                        st.markdown(
                            f"""
                            <div class="academic-metric-card">
                                <div class="metric-title">PSNR (Peak Signal-to-Noise)</div>
                                <div class="metric-value">{psnr_str}</div>
                                <div class="metric-sub {'positive' if has_gt and delta_psnr > 0 else ''}">{psnr_sub}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with col_m2:
                        st.markdown(
                            f"""
                            <div class="academic-metric-card">
                                <div class="metric-title">SSIM (Structural Similarity)</div>
                                <div class="metric-value">{ssim_str}</div>
                                <div class="metric-sub {'positive' if has_gt and delta_ssim > 0 else ''}">{ssim_sub}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with col_m3:
                        st.markdown(
                            f"""
                            <div class="academic-metric-card">
                                <div class="metric-title">Execution Time</div>
                                <div class="metric-value">{exec_time_sec:.2f}s</div>
                                <div class="metric-sub">{active_res['time_details'].get('infer_ms', 0):.0f} ms latency</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with col_m4:
                        st.markdown(
                            f"""
                            <div class="academic-metric-card">
                                <div class="metric-title">Resolution & Megapixels</div>
                                <div class="metric-value">{w} × {h}</div>
                                <div class="metric-sub">{(w * h) / 1e6:.2f} MP ({model_id})</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

                    # Side-by-Side Dual View & Slider Toggle
                    view_choice = st.radio(
                        "Comparison Display Format",
                        ["Side-by-Side Comparison", "Interactive Comparison Slider"],
                        horizontal=True,
                        label_visibility="collapsed",
                    )

                    if view_choice == "Side-by-Side Comparison":
                        col_side1, col_side2 = st.columns(2)
                        with col_side1:
                            st.markdown(
                                """
                                <div class="panel-header">
                                    <span>Hazy Input Image</span>
                                    <span class="panel-badge">RAW INPUT</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            st.image(hazy_rgb, use_container_width=True)

                        with col_side2:
                            st.markdown(
                                f"""
                                <div class="panel-header">
                                    <span>Dehazed Output Image</span>
                                    <span class="panel-badge">{model_id.upper()}</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            st.image(out_rgb, use_container_width=True)

                    else:
                        render_academic_comparison_slider(
                            before_rgb=hazy_rgb,
                            after_rgb=out_rgb,
                            label_before="Original (Hazy)",
                            label_after=f"Dehazed ({model_id})",
                        )

                    # Download Action
                    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                    is_ok, png_buf = cv2.imencode(".png", cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))
                    if is_ok:
                        col_dl1, col_dl2 = st.columns([1, 2])
                        with col_dl1:
                            st.download_button(
                                label="Download Dehazed Image (PNG)",
                                data=png_buf.tobytes(),
                                file_name=f"dehazed_{Path(uploaded_hazy.name).stem}_{model_id}.png",
                                mime="image/png",
                            )

    # -------------------------------------------------------------------------
    # TAB 2: Multi-Model Benchmark Comparison
    # -------------------------------------------------------------------------
    with tab_benchmark:
        st.markdown("<div style='font-size: 1.05rem; font-weight: 600; color: #F8FAFC; margin-bottom: 8px;'>Multi-Model Comparative Benchmarking</div>", unsafe_allow_html=True)
        st.caption("Evaluate multiple architectures side-by-side on identical input data with objective latency and restoration profiling.")

        benchmark_candidates = list(model_registry.keys())
        selected_benchmarks = st.multiselect(
            "Select Models to Benchmark",
            benchmark_candidates,
            default=benchmark_candidates[:min(3, len(benchmark_candidates))],
        )

        col_bup1, col_bup2 = st.columns(2)
        with col_bup1:
            bench_hazy_file = st.file_uploader(
                "Upload Benchmark Hazy Image",
                type=["png", "jpg", "jpeg", "bmp", "webp"],
                key="bench_hazy_uploader",
            )
        with col_bup2:
            bench_clear_file = st.file_uploader(
                "Upload Reference Clear Image (Optional)",
                type=["png", "jpg", "jpeg", "bmp", "webp"],
                key="bench_clear_uploader",
            )

        if bench_hazy_file is not None and selected_benchmarks:
            b_bytes = np.frombuffer(bench_hazy_file.read(), np.uint8)
            b_bgr = cv2.imdecode(b_bytes, cv2.IMREAD_COLOR)

            if b_bgr is not None:
                b_rgb = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2RGB)
                bh, bw, _ = b_rgb.shape

                # Reference ground truth
                ref_rgb = None
                if bench_clear_file is not None:
                    ref_b = np.frombuffer(bench_clear_file.read(), np.uint8)
                    ref_bgr = cv2.imdecode(ref_b, cv2.IMREAD_COLOR)
                    if ref_bgr is not None:
                        ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
                        if ref_rgb.shape[:2] != (bh, bw):
                            ref_rgb = cv2.resize(ref_rgb, (bw, bh), interpolation=cv2.INTER_AREA)

                if st.button("Run Multi-Model Benchmark", type="primary", key="btn_run_bench"):
                    cols = st.columns(len(selected_benchmarks) + 1)
                    with cols[0]:
                        st.markdown(
                            """
                            <div class="panel-header">
                                <span>Hazy Input</span>
                                <span class="panel-badge">RAW</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.image(b_rgb, caption=f"{bw}×{bh}", use_container_width=True)

                    bench_records = []

                    for idx, m_lbl in enumerate(selected_benchmarks):
                        m_meta = model_registry[m_lbl]
                        with st.spinner(f"Evaluating {m_meta['id']}..."):
                            try:
                                t_s = time.perf_counter()
                                if m_meta["type"] == "dehazeformer":
                                    df_a = load_cached_dehazeformer(str(m_meta["path"]), device_str)
                                    m_out, m_t = df_a.dehaze(b_rgb)
                                    m_sec = m_t["total_ms"] / 1000.0
                                elif m_meta["type"] == "unet":
                                    u_m, _ = load_cached_unet(str(m_meta["path"]), device_str)
                                    t_in = numpy_to_tensor(b_rgb, device=device).unsqueeze(0)
                                    with torch.no_grad():
                                        tout = u_m(t_in)
                                    m_out = tensor_to_numpy(tout.squeeze(0))
                                    m_sec = time.perf_counter() - t_s
                                else:
                                    m_out, _ = dcp_dehaze(
                                        b_rgb,
                                        omega=omega_factor,
                                        guided_radius=guided_radius,
                                        t0=t0_lower,
                                        use_guided_filter=True,
                                    )
                                    m_sec = time.perf_counter() - t_s

                                # Metrics
                                if ref_rgb is not None:
                                    m_psnr = calculate_psnr(m_out, ref_rgb, max_val=255.0)
                                    m_ssim = calculate_ssim(m_out, ref_rgb, max_val=255.0)
                                    stat_sub = f"{m_psnr:.2f} dB | {m_ssim:.4f}"
                                else:
                                    m_psnr = None
                                    m_ssim = None
                                    stat_sub = f"{m_sec:.2f}s latency"

                                bench_records.append({
                                    "Model": m_meta["id"],
                                    "Architecture": m_meta["variant"],
                                    "Parameters": m_meta["params"],
                                    "Latency (s)": f"{m_sec:.3f}s",
                                    "PSNR (dB)": f"{m_psnr:.2f}" if m_psnr is not None else "N/A",
                                    "SSIM": f"{m_ssim:.4f}" if m_ssim is not None else "N/A",
                                })

                                with cols[idx + 1]:
                                    st.markdown(
                                        f"""
                                        <div class="panel-header">
                                            <span>{m_meta['id']}</span>
                                            <span class="panel-badge">{m_sec:.2f}s</span>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )
                                    st.image(m_out, caption=stat_sub, use_container_width=True)

                            except Exception as ex:
                                with cols[idx + 1]:
                                    st.error(f"Error evaluating {m_meta['id']}: {ex}")

                    if bench_records:
                        st.markdown("---")
                        st.markdown("<div style='font-size: 0.95rem; font-weight: 600; color: #F8FAFC; margin-bottom: 8px;'>Quantitative Benchmark Summary</div>", unsafe_allow_html=True)
                        st.dataframe(bench_records, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 3: Offline Video Dehazing
    # -------------------------------------------------------------------------
    with tab_video:
        st.markdown("<div style='font-size: 1.05rem; font-weight: 600; color: #F8FAFC; margin-bottom: 8px;'>Offline Video Dehazing & Sequence Processing</div>", unsafe_allow_html=True)
        st.caption("Frame-by-frame temporal enhancement pipeline with bounded memory consumption and execution profiling.")

        vcol1, vcol2 = st.columns([1.2, 1.0])
        with vcol1:
            vid_model_label = st.selectbox(
                "Enhancement Architecture",
                model_labels,
                index=0,
                key="vid_model_select",
            )
            v_meta = model_registry[vid_model_label]
        with vcol2:
            frame_limit = st.slider(
                "Frame Limit (0 = Full Video)",
                min_value=0,
                max_value=600,
                value=60,
                step=10,
                help="Process a subset of frames for rapid latency verification on CPU.",
            )

        video_upload = st.file_uploader(
            "Upload Source Video (MP4, AVI, MOV, MKV)",
            type=["mp4", "avi", "mov", "mkv"],
            key="video_file_uploader",
        )

        if video_upload is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(video_upload.name).suffix) as tmp_in:
                tmp_in.write(video_upload.read())
                temp_vid_path = Path(tmp_in.name)

            st.video(str(temp_vid_path))

            if st.button("Start Video Dehazing", type="primary", key="btn_run_video_proc"):
                prog_bar = st.progress(0.0)
                status_box = st.empty()

                def on_video_progress(current_f: int, total_f: int, elapsed_s: float) -> None:
                    pct = min(1.0, current_f / max(1, total_f))
                    fps_val = current_f / max(elapsed_s, 1e-4)
                    prog_bar.progress(pct)
                    status_box.text(
                        f"Processed {current_f}/{total_f} frames ({pct*100:.1f}%) | "
                        f"Throughput: {fps_val:.1f} FPS | Elapsed: {elapsed_s:.1f}s"
                    )

                with st.spinner("Executing sequence dehazing..."):
                    try:
                        v_model = None
                        v_method = "dcp"
                        if v_meta["type"] == "dehazeformer":
                            v_model = load_cached_dehazeformer(str(v_meta["path"]), device_str)
                            v_method = "ai"
                        elif v_meta["type"] == "unet":
                            v_model, _ = load_cached_unet(str(v_meta["path"]), device_str)
                            v_method = "ai"

                        max_f = frame_limit if frame_limit > 0 else None
                        res = process_video_file(
                            input_path=temp_vid_path,
                            model=v_model,
                            method=v_method,
                            device_str=device_str,
                            max_frames=max_f,
                            progress_callback=on_video_progress,
                            preserve_audio=False,
                        )

                        prog_bar.progress(1.0)
                        st.success(
                            f"Successfully enhanced {res['frames_processed']} frames in "
                            f"{res['total_duration_sec']:.2f}s (Average Speed: {res['processing_fps']:.2f} FPS)."
                        )

                        out_v = Path(res["output_file"])
                        if out_v.is_file():
                            st.video(str(out_v))
                            with open(out_v, "rb") as f_v:
                                st.download_button(
                                    label="Download Enhanced Video (MP4)",
                                    data=f_v,
                                    file_name=out_v.name,
                                    mime="video/mp4",
                                )

                    except Exception as ex:
                        st.error(f"Video processing failed: {ex}")
                    finally:
                        temp_vid_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
