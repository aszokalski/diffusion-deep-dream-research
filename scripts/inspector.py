import argparse
from dataclasses import dataclass
from pathlib import Path
import pickle
import sys
from typing import Dict, List, Optional

import altair as alt
from dotenv import load_dotenv
import hydra

load_dotenv()
from hydra.core.global_hydra import GlobalHydra
import pandas as pd
import streamlit as st

from diffusion_deep_dream_research.config.config_schema import ExperimentConfig, register_configs
from diffusion_deep_dream_research.utils.deep_dream_results_reading_utils import (
    DeepDreamResult,
    IntermediateStep,
)
from diffusion_deep_dream_research.utils.prior_results_reading_utils import (
    ChannelPriors,
)


# cli arguments
def get_cli_args():
    parser = argparse.ArgumentParser(description="Deep Dream Explorer")
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Path to the base representation directory",
    )

    try:
        return parser.parse_args()
    except SystemExit:
        sys.exit(0)


args = get_cli_args()
BASE_REPRESENTATION_DIR = Path(args.base_dir)


# hydra setup
@st.cache_resource
def load_config() -> ExperimentConfig:
    GlobalHydra.instance().clear()

    register_configs()
    with hydra.initialize(version_base=None, config_path="../conf"):
        cfg = hydra.compose(config_name="config", overrides=["stage=prior"])
    return cfg


config = load_config()
PROJECT_ROOT = Path(config.outputs_dir)


st.set_page_config(layout="wide", page_title="Deep Dream Explorer")


# data structures
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


# helper functions
def get_available_modes():
    modes = {}
    std_path = BASE_REPRESENTATION_DIR / "non-sae" / "index_metadata.pkl"
    sae_path = BASE_REPRESENTATION_DIR / "sae" / "index_metadata.pkl"

    if std_path.exists():
        modes["non-SAE"] = BASE_REPRESENTATION_DIR / "non-sae"
    if sae_path.exists():
        modes["SAE"] = BASE_REPRESENTATION_DIR / "sae"

    if not modes and (BASE_REPRESENTATION_DIR / "index_metadata.pkl").exists():
        modes["Default"] = BASE_REPRESENTATION_DIR

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


def get_image_path(path_str: str) -> Optional[str]:
    try:
        p = Path(path_str)
        if p.exists():
            return str(p)

        p_alt = PROJECT_ROOT / path_str
        if p_alt.exists():
            return str(p_alt)

        return None
    except Exception:
        return None


# session state
if "selected_timestep" not in st.session_state:
    st.session_state.selected_timestep = 0


# main app
def main():
    st.title("Results Inspector")

    available_modes = get_available_modes()
    if not available_modes:
        st.error(f"No Representation Data Found at {BASE_REPRESENTATION_DIR}")
        st.info("Check your --base_dir argument.")
        st.stop()

    # sidebar settings
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

        if "selected_channel_id" not in st.session_state:
            st.session_state.selected_channel_id = active_channels[0]

        if st.session_state.selected_channel_id not in active_channels:
            st.session_state.selected_channel_id = active_channels[0]

        # navigation controls
        def cycle_channel(offset):
            try:
                current_idx = active_channels.index(st.session_state.selected_channel_id)
                new_idx = (current_idx + offset) % len(active_channels)
                st.session_state.selected_channel_id = active_channels[new_idx]
            except ValueError:
                st.session_state.selected_channel_id = active_channels[0]

        col_prev, col_next = st.columns(2)
        with col_prev:
            st.button("Previous", on_click=cycle_channel, args=(-1,), use_container_width=True)
        with col_next:
            st.button("Next", on_click=cycle_channel, args=(1,), use_container_width=True)

        selected_id = st.selectbox(
            f"Select Channel ({len(active_channels)})",
            active_channels,
            format_func=lambda x: f"Ch {x:04d}",
            key="selected_channel_id",
        )

        with st.spinner(f"Loading data for Channel {selected_id}..."):
            ch_data = load_channel_shard(current_data_path, selected_id)

        if not ch_data:
            st.error(f"Could not load data for Channel {selected_id}")
            st.stop()

        # timestep controls
        st.divider()
        st.subheader("Go To")

        peaks = ch_data.meta.peaks
        if peaks:
            sorted_peaks = sorted(peaks, reverse=True)
            for p in sorted_peaks:
                if st.button(f"Timestep {p}", key=f"peak_{p}"):
                    st.session_state.selected_timestep = p

        st.caption("Available Results")
        available_ts = sorted(list(ch_data.timesteps.keys()))
        relevant_ts = [
            t
            for t in available_ts
            if ch_data.timesteps[t].deep_dream or ch_data.timesteps[t].dataset_examples
        ]

        idx_selection = 0
        if st.session_state.selected_timestep in relevant_ts:
            idx_selection = relevant_ts.index(st.session_state.selected_timestep)

        target_ts = st.selectbox(
            "Jump to timestep with data:",
            options=relevant_ts,
            index=idx_selection,
        )

        if target_ts != st.session_state.selected_timestep:
            st.session_state.selected_timestep = target_ts

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

        st.divider()

        # display options
        with st.expander("View Options"):
            chart_height_main = st.slider(
                "Main Chart Height (px)", min_value=200, max_value=1000, value=400, step=50
            )

            chart_height_secondary = st.slider(
                "Secondary Chart Height (px)", min_value=100, max_value=600, value=250, step=50
            )

    # main layout
    current_t = st.session_state.selected_timestep

    col_graphs, col_results = st.columns([1, 1])

    with col_graphs:
        render_charts(ch_data, current_t, chart_height_main, chart_height_secondary)

    with col_results:
        t_data = ch_data.timesteps.get(current_t)

        tab_dd, tab_ex, tab_priors = st.tabs(
            ["Visualization", "Dataset Examples", "Channel Priors"]
        )

        with tab_dd:
            if not t_data or not t_data.deep_dream:
                st.info(f"No visualizations generated for Timestep {current_t}")
            else:
                render_deep_dream_view(t_data.deep_dream)

        with tab_ex:
            if not t_data or not t_data.dataset_examples:
                st.info(f"No Dataset Examples found for Timestep {current_t}")
            else:
                render_dataset_examples(t_data.dataset_examples)

        with tab_priors:
            render_priors(ch_data.priors)


# content rendering functions
def render_charts(ch_data: ChannelData, current_t: int, h_main: int, h_sec: int):
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
                "Size": 200,
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
                    "Type": "Visualization",
                    "Color": "#2ecc71",  # Green
                    "Shape": "circle",
                    "Size": 100,
                }
            )

    df_markers = pd.DataFrame(marker_data)

    x_scale = alt.X("Timestep", scale=alt.Scale(domain=[1000, 0]), axis=alt.Axis(title="Timestep"))

    base_ctx = alt.Chart(df_context).encode(x=x_scale)
    area = base_ctx.mark_area(color="lightgreen", opacity=0.3).encode(y="Activity")
    line = base_ctx.mark_line(color="green", opacity=0.5).encode(y="Activity")

    rule = (
        alt.Chart(pd.DataFrame({"Timestep": [current_t]}))
        .mark_rule(color="blue", strokeWidth=2)
        .encode(x="Timestep")
    )

    chart_context_base = area + line + rule

    if not df_markers.empty:
        markers_layer = (
            alt.Chart(df_markers)
            .mark_point(filled=True, opacity=1.0)
            .encode(
                x=x_scale,
                y="Activity",
                color=alt.Color("Color", scale=None),
                shape=alt.Shape("Shape", scale=None),
                size=alt.Size("Size", scale=None),
                tooltip=["Timestep", "Type", "Activity"],
            )
        )
        chart_context_base = chart_context_base + markers_layer

    chart_context = chart_context_base.properties(
        title="Time-step Activity Profile", height=h_main
    )

    chart_max = base_ctx.mark_line(color="#e67e22").encode(
        y=alt.Y("Activation", title="Max Activation"), tooltip=["Timestep", "Activation"]
    )
    chart_max = (chart_max + rule).properties(title="Max Activation over Timesteps", height=h_sec)

    st.altair_chart(chart_context, width="stretch")
    st.altair_chart(chart_max, width="stretch")


def render_deep_dream_view(dd_map: Dict[str, List[DeepDreamResult]]):
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

    col_stats, col_img = st.columns([1, 2])

    intermediates = result.intermediate_steps
    is_browsing_history = False

    current_img_path = None
    current_stats = None

    with col_img:
        img_container = st.container()
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
                current_img_path = step_obj.image_path
                current_stats = step_obj.stats
                current_label = f"Step {step_obj.step_idx}"
            else:
                current_img_path = result.final_image_path
                try:
                    current_stats = result.stats
                except FileNotFoundError:
                    current_stats = None
                current_label = "Final Result"
        else:
            current_img_path = result.final_image_path
            current_stats = result.stats
            current_label = "Final Result"

        with img_container:
            st.subheader(current_label)
            if current_img_path and Path(current_img_path).exists():
                st.image(str(current_img_path), width="stretch")
            else:
                st.error(f"Image file missing: {current_img_path}")

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
        img_path = get_image_path(ex["path"])
        with cols[i % 3]:
            if img_path:
                st.image(img_path, width="stretch")
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
            if pair.image_path.exists():
                st.image(str(pair.image_path), width="stretch", caption=f"Prior {i}")
            else:
                st.warning(f"Missing: {pair.image_path.name}")


if __name__ == "__main__":
    main()
