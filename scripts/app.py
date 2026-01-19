import json
import pickle
import random
import textwrap
from pathlib import Path
from typing import Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy.interpolate import interp1d

# --- CONFIGURATION ---
BASE_PATH = Path(
    "/net/pr2/projects/plgrid/plggailpwmm/aszokalski/diffusion-deep-dream-research/outputs/e0/Stage.timestep_analysis/multirun/2026-01-05/20-34-31/0"
)
PROJECT_ROOT = Path(
    "/net/pr2/projects/plgrid/plggailpwmm/aszokalski/diffusion-deep-dream-research"
)
TOTAL_TIMESTEPS = 1000

st.set_page_config(layout="wide", page_title="Deep Dream Research Explorer")


# --- 1. Fast Data Loading ---
@st.cache_data
def load_analysis_data(base_path: Path, sae: bool):
    suffix = "_sae" if sae else ""
    try:
        # Load heavy pickle files
        with open(
            base_path / f"frequency_in_top_k_sorted_timesteps_max_activation{suffix}.pkl", "rb"
        ) as f:
            frequency, sorted_ts, max_act = pickle.load(f)

        # Load JSONs
        with open(base_path / f"activity_peaks{suffix}.json", "r") as f:
            activity_peaks = json.load(f)

        # Load dataset examples
        # Optimization: dataset_examples can be huge. If loading is still slow,
        # consider splitting this file or using a database/parquet.
        with open(base_path / f"dataset_examples{suffix}.json", "r") as f:
            dataset_examples = json.load(f)

        return frequency, np.array(sorted_ts), max_act, activity_peaks, dataset_examples
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}")
        return None


# --- 2. Image Logic ---
def get_random_example(
    dataset_examples: dict, channel_idx: int, timestep: int, project_root: Path
) -> Tuple[Optional[Image.Image], str]:
    c_key = str(channel_idx)
    t_key = str(timestep)

    if c_key not in dataset_examples:
        return None, "Channel not found in examples"
    if t_key not in dataset_examples[c_key]:
        return None, "No image generated for this specific timestep"

    examples_list = dataset_examples[c_key][t_key]
    if not examples_list:
        return None, "Empty example list"

    # Pick random example
    prompt, rel_path_str = random.choice(examples_list)
    image_path = project_root / rel_path_str

    if not image_path.exists():
        return None, f"File not found: {image_path.name}"

    try:
        return Image.open(image_path).convert("RGB"), prompt
    except Exception:
        return None, "Error loading image file"


# --- 3. State Helper ---
def set_timestep(val):
    st.session_state.timestep_input = val


# --- 4. Main App ---
def main():
    st.title("Deep Dream Explorer: Peaks & Channels")

    # -- Sidebar --
    st.sidebar.header("Configuration")
    use_sae = st.sidebar.checkbox("Use SAE (Sparse Autoencoder)", value=False)

    data = load_analysis_data(BASE_PATH, use_sae)
    if not data:
        st.stop()

    frequency, sorted_timesteps, max_activation, activity_peaks, dataset_examples = data

    # Filter Active Channels
    n_channels = frequency.shape[0]
    active_indices = [i for i in range(n_channels) if np.sum(frequency[i]) > 0]

    # Sidebar: Channel Selector
    st.sidebar.subheader("1. Channel")
    selected_channel = st.sidebar.selectbox(
        "Search Channel Index", active_indices, format_func=lambda x: f"Channel {x:04d}"
    )

    peaks = activity_peaks[selected_channel]

    # State Management: Reset timestep if channel changes
    if "last_channel" not in st.session_state or st.session_state.last_channel != selected_channel:
        st.session_state.last_channel = selected_channel
        default_ts = peaks[0] if peaks else int(np.argmax(frequency[selected_channel]))
        st.session_state.timestep_input = default_ts

    # Sidebar: Peaks
    st.sidebar.subheader("2. Jump to Peak")
    if peaks:
        cols = st.sidebar.columns(3)
        for i, peak in enumerate(peaks):
            cols[i % 3].button(
                f"T={peak}",
                key=f"btn_{peak}",
                on_click=set_timestep,
                args=(peak,),
                use_container_width=True,
            )
    else:
        st.sidebar.caption("No distinct peaks.")

    # Sidebar: Manual Input
    st.sidebar.subheader("3. Exact Timestep")
    current_ts = st.sidebar.number_input(
        "Enter Timestep", min_value=0, max_value=TOTAL_TIMESTEPS, key="timestep_input", step=1
    )

    # --- MAIN UI ---
    col_left, col_right = st.columns([2, 1])

    with col_left:
        # --- DATA PREPARATION FOR ALTAIR ---
        # 1. Interpolation Data (Full Range)
        y_observed = frequency[selected_channel]
        x_observed = sorted_timesteps

        # Perform interpolation
        f_interp = interp1d(
            x_observed, y_observed, kind="linear", bounds_error=False, fill_value=0
        )
        x_full = np.arange(0, TOTAL_TIMESTEPS + 1)
        y_full = f_interp(x_full)

        # Create DataFrames
        df_interp = pd.DataFrame({"Timestep": x_full, "Frequency": y_full})
        df_observed = pd.DataFrame({"Timestep": x_observed, "Frequency": y_observed})
        df_peaks = pd.DataFrame(
            {
                "Timestep": peaks,
                "Frequency": [y_full[p] for p in peaks],
                "Label": [str(p) for p in peaks],
            }
        )

        # Max Activation Data
        y_max = max_activation[selected_channel]
        df_max = pd.DataFrame({"Timestep": sorted_timesteps, "Activation": y_max})

        # --- CHART 1: ACTIVITY PROFILE ---
        # Base Chart
        base = alt.Chart(df_interp).encode(
            x=alt.X("Timestep", scale=alt.Scale(domain=[TOTAL_TIMESTEPS, 0]))
        )

        # A. Interpolated Line + Area
        line = base.mark_line(color="#2ecc71").encode(y="Frequency")
        area = base.mark_area(color="#2ecc71", opacity=0.2).encode(y="Frequency")

        # B. Observed Points
        points = (
            alt.Chart(df_observed)
            .mark_circle(color="gray", opacity=0.4)
            .encode(
                x=alt.X("Timestep", scale=alt.Scale(domain=[TOTAL_TIMESTEPS, 0])),
                y="Frequency",
                tooltip=["Timestep", "Frequency"],
            )
        )

        # C. Peaks Markers
        peak_points = (
            alt.Chart(df_peaks)
            .mark_point(shape="cross", color="black", size=100)
            .encode(x="Timestep", y="Frequency")
        )
        peak_labels = (
            alt.Chart(df_peaks)
            .mark_text(dy=-10, fontWeight="bold")
            .encode(x="Timestep", y="Frequency", text="Label")
        )

        # D. Current Selection Rule
        rule = (
            alt.Chart(pd.DataFrame({"Timestep": [current_ts]}))
            .mark_rule(color="#e74c3c", strokeDash=[5, 5])
            .encode(x="Timestep")
        )

        # Combine Chart 1
        chart_activity = (area + line + points + peak_points + peak_labels + rule).properties(
            title=f"Channel {selected_channel} Activity Profile", height=300
        )

        # --- CHART 2: MAX ACTIVATION ---
        chart_max = (
            alt.Chart(df_max)
            .mark_line(color="#3498db")
            .encode(
                x=alt.X("Timestep", scale=alt.Scale(domain=[TOTAL_TIMESTEPS, 0])),
                y="Activation",
                tooltip=["Timestep", "Activation"],
            )
        )

        # Combine Chart 2 (Add the same rule)
        chart_max = (chart_max + rule).properties(title="Max Activation", height=250)

        # Render Charts
        st.altair_chart(chart_activity, use_container_width=True)
        st.altair_chart(chart_max, use_container_width=True)

    with col_right:
        st.markdown(f"### Result @ T={current_ts}")

        img, prompt = get_random_example(
            dataset_examples, selected_channel, current_ts, PROJECT_ROOT
        )

        if img:
            st.image(img, use_container_width=True)
            # Wrap prompt nicely
            st.success(prompt)
        else:
            st.warning("No image found for this timestep.")
            if peaks:
                st.info("💡 Tip: Try clicking a Peak button on the left.")


if __name__ == "__main__":
    main()
