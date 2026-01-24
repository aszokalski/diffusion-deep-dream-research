from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Dict, List, Optional

import altair as alt
import gdown
import pandas as pd
from PIL import Image
import streamlit as st

# Imports and Error Handling
try:
    from diffusion_deep_dream_research.utils.deep_dream_results_reading_utils import (
        DeepDreamResult,
        DeepDreamStats,
        IntermediateStep,
    )
    from diffusion_deep_dream_research.utils.prior_results_reading_utils import (
        ChannelPriors,
        ImageWithLatent,
    )
except ImportError:
    st.error(
        "Could not import project classes. Please ensure the 'diffusion_deep_dream_research' "
        "folder is in the same directory as this script."
    )
    st.stop()

st.set_page_config(layout="wide", page_title="Deep Dream Explorer")


# --- Configuration ---

# Paste your Google Drive Folder Link here (ensure it is "Anyone with the link")
# This folder must contain 'index_metadata.pkl' and the 'channels' directory.
GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/YOUR_FOLDER_ID_HERE?usp=sharing"

# Local directory where data will be downloaded
LOCAL_DATA_PATH = Path("downloaded_data")


# --- Data Setup (Gdown) ---
@st.cache_resource
def setup_data_environment():
    """
    Checks for local data. If missing, downloads the folder structure
    from Google Drive using gdown.
    """
    if not LOCAL_DATA_PATH.exists():
        LOCAL_DATA_PATH.mkdir(parents=True, exist_ok=True)

    # Check if critical files exist to avoid re-downloading on every run
    if not (LOCAL_DATA_PATH / "index_metadata.pkl").exists():
        with st.spinner("Downloading experiment data from Google Drive... (This happens once)"):
            try:
                gdown.download_folder(
                    url=GDRIVE_FOLDER_URL,
                    output=str(LOCAL_DATA_PATH),
                    quiet=False,
                    use_cookies=False,
                )
                st.success("Data downloaded successfully.")
            except Exception as e:
                st.error(f"Failed to download data using gdown: {e}")
                st.warning("Ensure the Google Drive link is set to 'Anyone with the link'.")
                st.stop()
    return LOCAL_DATA_PATH


# Initialize data download
BASE_DIR = setup_data_environment()


# --- Data Structures ---
@dataclass
class ChannelMeta:
    activity_profile: object
    max_activation: object
    peaks: List[int]


@dataclass
class TimestepData:
    dataset_examples: List[Dict[str, str]]
    deep_dream: Dict[str, List[DeepDreamResult]]


@dataclass
class ChannelData:
    id: int
    meta: ChannelMeta
    priors: ChannelPriors
    timesteps: Dict[int, TimestepData]


# --- Utility Functions ---
def get_available_modes():
    """
    Detects available sub-experiments (SAE vs non-SAE) in the downloaded folder.
    """
    modes = {}
    std_path = BASE_DIR / "non-sae"
    sae_path = BASE_DIR / "sae"

    if (std_path / "index_metadata.pkl").exists():
        modes["non-SAE"] = std_path
    if (sae_path / "index_metadata.pkl").exists():
        modes["SAE"] = sae_path

    # Fallback to root if no subfolders exist
    if not modes and (BASE_DIR / "index_metadata.pkl").exists():
        modes["Default"] = BASE_DIR

    return modes


@st.cache_data
def load_metadata(base_path: Path):
    path = base_path / "index_metadata.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(max_entries=20, show_spinner=False)
def load_channel_shard(base_path: Path, channel_id: int) -> Optional[ChannelData]:
    path = base_path / "channels" / f"channel_{channel_id:05d}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def load_image_safe(path_str: str) -> Optional[Image.Image]:
    """
    Loads an image. Handles path differences between local dev and cloud.
    """
    try:
        # Check if the path exists exactly as stored in pickle
        p = Path(path_str)
        if p.exists():
            return Image.open(p).convert("RGB")

        # If not, check relative to the downloaded data root
        # This handles cases where pickle stores absolute paths like /net/pr2/...
        # We assume the file name is unique or the relative structure matches.

        # Strategy 1: Check if it's inside the current Base Dir
        rel_p = BASE_DIR / p.name
        if rel_p.exists():
            return Image.open(rel_p).convert("RGB")

        return None
    except Exception:
        return None


# --- Session State Initialization ---
if "selected_timestep" not in st.session_state:
    st.session_state.selected_timestep = 0


# --- Main Application Logic ---
def main():
    st.title("Deep Dream Explorer")

    available_modes = get_available_modes()
    if not available_modes:
        st.error(f"No valid metadata found in {BASE_DIR}. Check your Google Drive structure.")
        st.stop()

    # Sidebar Controls
    with st.sidebar:
        st.header("Settings")
        mode_names = list(available_modes.keys())
        selected_mode_name = st.radio("Analysis Mode", mode_names, horizontal=True)
        current_data_path = available_modes[selected_mode_name]

        with st.spinner("Loading index metadata..."):
            meta = load_metadata(current_data_path)

        if not meta:
            st.error("Failed to load metadata.")
            st.stop()

        active_channels = meta.active_channels
        if not active_channels:
            st.warning("No active channels found.")
            st.stop()

        selected_id = st.selectbox(
            f"Select Channel ({len(active_channels)})",
            active_channels,
            format_func=lambda x: f"Ch {x:04d}",
        )

        with st.spinner(f"Loading data for Channel {selected_id}..."):
            ch_data = load_channel_shard(current_data_path, selected_id)

        if not ch_data:
            st.error(f"Could not load data for Channel {selected_id}")
            st.stop()

        # Timestep Navigation
        st.divider()
        st.subheader("Go To")

        # Activity Peaks
        peaks = ch_data.meta.peaks
        if peaks:
            st.caption("Activity Peaks")
            sorted_peaks = sorted(peaks, reverse=True)
            for p in sorted_peaks:
                if st.button(f"Timestep {p}", key=f"peak_{p}", use_container_width=True):
                    st.session_state.selected_timestep = p

        # Existing Data Navigation
        st.caption("Available Results")
        available_ts = sorted(list(ch_data.timesteps.keys()))
        relevant_ts = [
            t
            for t in available_ts
            if ch_data.timesteps[t].deep_dream or ch_data.timesteps[t].dataset_examples
        ]

        current_ts_idx = 0
        if st.session_state.selected_timestep in relevant_ts:
            current_ts_idx = relevant_ts.index(st.session_state.selected_timestep) + 1

        target_ts = st.selectbox(
            "Jump to timestep with data:",
            options=[st.session_state.selected_timestep] + relevant_ts,
            index=0,
        )
        if target_ts != st.session_state.selected_timestep:
            st.session_state.selected_timestep = target_ts

        # Manual Selection
        st.divider()
        new_t = st.number_input(
            "Manual Timestep",
            min_value=0,
            max_value=1000,
            value=st.session_state.selected_timestep,
        )
        if new_t != st.session_state.selected_timestep:
            st.session_state.selected_timestep = new_t
            st.rerun()

    # Main Content Layout
    current_t = st.session_state.selected_timestep

    col_graphs, col_results = st.columns([1, 1])

    # Left Column: Activity and Activation Charts
    with col_graphs:
        render_charts(ch_data, current_t)

    # Right Column: Detailed Analysis Tabs
    with col_results:
        t_data = ch_data.timesteps.get(current_t)

        tab_dd, tab_ex, tab_priors = st.tabs(["Deep Dreams", "Dataset Examples", "Channel Priors"])

        with tab_dd:
            if not t_data or not t_data.deep_dream:
                st.info(f"No Deep Dreams generated for Timestep {current_t}")
            else:
                render_deep_dream_view(t_data.deep_dream)

        with tab_ex:
            if not t_data or not t_data.dataset_examples:
                st.info(f"No Dataset Examples found for Timestep {current_t}")
            else:
                render_dataset_examples(t_data.dataset_examples)

        with tab_priors:
            render_priors(ch_data.priors)


def render_charts(ch_data: ChannelData, current_t: int):
    """
    Renders the Altair charts for channel activity and navigation.
    """
    act = ch_data.meta.activity_profile
    max_act = ch_data.meta.max_activation
    peaks = set(ch_data.meta.peaks)

    df_context = pd.DataFrame(
        {"Timestep": range(len(act)), "Activity": act, "Activation": max_act}
    )

    marker_data = []

    for p in peaks:
        marker_data.append(
            {
                "Timestep": p,
                "Activity": act[p],
                "Type": "Peak",
                "Color": "#e74c3c",  # Red
                "Shape": "cross",
                "Size": 300,
            }
        )

    for t, data in ch_data.timesteps.items():
        if t in peaks:
            continue
        if data.deep_dream:
            marker_data.append(
                {
                    "Timestep": t,
                    "Activity": act[t],
                    "Type": "Deep Dream",
                    "Color": "#2ecc71",  # Green
                    "Shape": "circle",
                    "Size": 100,
                }
            )

    df_markers = pd.DataFrame(marker_data)
    x_scale = alt.X("Timestep", scale=alt.Scale(domain=[1000, 0]), axis=alt.Axis(title=""))

    # Chart: Global Activity Profile
    base_ctx = alt.Chart(df_context).encode(x=x_scale)
    area = base_ctx.mark_area(color="lightgreen", opacity=0.3).encode(y="Activity")
    line = base_ctx.mark_line(color="green", opacity=0.5).encode(y="Activity")

    rule = (
        alt.Chart(pd.DataFrame({"Timestep": [current_t]}))
        .mark_rule(color="blue", strokeWidth=2)
        .encode(x="Timestep")
    )

    chart_context = (area + line + rule).properties(height=250, title="Feature Activity Context")
    st.altair_chart(chart_context, use_container_width=True)

    # Chart: Maximum Activation Trends
    chart_max = (
        base_ctx.mark_line(color="#e67e22")
        .encode(
            y=alt.Y("Activation", title="Max Activation"),
            tooltip=["Timestep", "Activation"],
        )
        .properties(height=150)
    )
    chart_max = (chart_max + rule).properties(title="Max Activation over Timesteps")
    st.altair_chart(chart_max, use_container_width=True)

    # Chart: Interactive Navigation Markers
    if not df_markers.empty:
        click_selector = alt.selection_point(fields=["Timestep"], on="click", name="ClickSelector")

        base_sel = alt.Chart(df_markers).encode(
            x=alt.X(
                "Timestep",
                scale=alt.Scale(domain=[1000, 0]),
                axis=alt.Axis(title="Timestep (Click to Select)"),
            )
        )

        points = (
            base_sel.mark_point(filled=True, opacity=1.0)
            .encode(
                y=alt.Y("Activity", axis=alt.Axis(title="Marker Activity")),
                color=alt.Color("Color", scale=None),
                shape=alt.Shape("Shape", scale=None),
                size=alt.Size("Size", scale=None),
                tooltip=["Timestep", "Type", "Activity"],
                opacity=alt.condition(click_selector, alt.value(1.0), alt.value(0.4)),
            )
            .add_params(click_selector)
        )

        selection_event = st.altair_chart(
            points.properties(height=200, title="Navigation (Click Markers)"),
            use_container_width=True,
            on_select="rerun",
            key="nav_chart",
        )

        if selection_event.selection:
            if "ClickSelector" in selection_event.selection:
                sel_data = selection_event.selection["ClickSelector"]
                if sel_data:
                    new_t = sel_data[0]["Timestep"]
                    if new_t != st.session_state.selected_timestep:
                        st.session_state.selected_timestep = new_t
                        st.rerun()
            elif "rows" in selection_event.selection:
                rows = selection_event.selection["rows"]
                if rows:
                    new_t = rows[0]["Timestep"]
                    if new_t != st.session_state.selected_timestep:
                        st.session_state.selected_timestep = new_t
                        st.rerun()
    else:
        st.info("No clickable markers (peaks or deep dreams) for this channel.")


def render_deep_dream_view(dd_map: Dict[str, List[DeepDreamResult]]):
    """
    Renders the Deep Dream inspector with intermediate optimization steps.
    """
    # Selection Controls
    c_mode, c_var = st.columns([1, 1])
    with c_mode:
        modes = list(dd_map.keys())
        selected_mode = st.radio("Noise Mode", modes, horizontal=True)

    results_list = dd_map[selected_mode]
    with c_var:
        variant_opts = range(len(results_list))
        selected_idx = st.selectbox(
            "Variant (Seed)", variant_opts, format_func=lambda x: f"Variant #{x + 1}"
        )

    result: DeepDreamResult = results_list[selected_idx]

    # Split view: Stats (Left) | Image (Right)
    col_stats, col_img = st.columns([1, 2])

    intermediates = result.intermediate_steps
    is_browsing_history = False
    current_img = None
    current_stats = None

    # Image Column: Configure slider and display image
    with col_img:
        if intermediates:
            max_step = len(intermediates)
            step_idx = st.slider(
                "Optimization Progress",
                min_value=0,
                max_value=max_step,
                value=max_step,
                help="Position 0 is start, rightmost is final result.",
            )

            if step_idx < max_step:
                is_browsing_history = True
                step_obj: IntermediateStep = intermediates[step_idx]
                current_img = step_obj.get_image()
                current_stats = step_obj.stats
                current_label = f"Step {step_obj.step_idx}"
            else:
                current_img = result.get_final_image()
                try:
                    current_stats = result.stats
                except FileNotFoundError:
                    current_stats = None
                current_label = "Final Result"
        else:
            current_img = result.get_final_image()
            current_stats = result.stats
            current_label = "Final Result"

        st.subheader(current_label)
        if current_img:
            st.image(current_img, use_container_width=True)
        else:
            st.error("Image file missing")

    # Statistics Column: Display metrics
    with col_stats:
        st.subheader("Stats")
        if current_stats:
            st.metric("Activation", f"{current_stats.activation:.4f}")

            with st.expander("Penalty Breakdown", expanded=True):
                for k, v in current_stats.penalties.items():
                    st.write(f"**{k}:** {v:.4f}")

            st.markdown("---")
            st.metric("Total Loss", f"{current_stats.total_loss:.4f}")

        if is_browsing_history:
            st.info("Viewing intermediate step.")


def render_dataset_examples(examples: List[Dict]):
    st.markdown(f"**Found {len(examples)} examples triggering this feature.**")
    cols = st.columns(3)
    for i, ex in enumerate(examples):
        img = load_image_safe(ex["path"])
        with cols[i % 3]:
            if img:
                st.image(img, use_container_width=True)
            else:
                st.warning("Image missing")
            with st.expander("Prompt"):
                st.caption(ex["prompt"])


def render_priors(priors: ChannelPriors):
    if not priors or not priors.images_with_latents:
        st.info("No priors saved for this channel.")
        return

    pairs = priors.images_with_latents
    st.markdown(f"**{len(pairs)} Priors Available**")

    cols = st.columns(4)
    for i, pair in enumerate(pairs):
        with cols[i % 4]:
            img = load_image_safe(str(pair.image_path))
            if img:
                st.image(img, use_container_width=True, caption=f"Prior {i}")


if __name__ == "__main__":
    main()
