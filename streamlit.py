# mask the pytorch_grad_cam package import, we don't need it and importing causes an error
import sys
from unittest.mock import MagicMock

sys.modules["pytorch_grad_cam"] = MagicMock()


streamlit = True
if streamlit:
    import streamlit as st
    from streamlit_extras.add_vertical_space import add_vertical_space

else:
    streamlit = None

from PIL import Image
from datetime import date
import numpy as np

import pydeck as pdk

# st.write(f"## Ovenbird song map")

import pandas as pd
import numpy as np
import utm
import seaborn as sns

from opensoundscape import Audio, Spectrogram, CNN, BoxedAnnotations

import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path

from matplotlib import pyplot as plt


def figsize(w, h):
    plt.rcParams["figure.figsize"] = [w, h]


figsize(15, 5)  # for big visuals
from opensoundscape.localization import PositionEstimate
from opensoundscape.localization.position_estimate import (
    positions_to_df,
    df_to_positions,
)
import ast
import re
import random

from names import nature_adjectives, bird_nouns
from names import nature_adjectives, bird_nouns

random.seed(2025)
adj = random.choices(nature_adjectives, k=100)
noun = random.choices(bird_nouns, k=100)
names = [f"{a} {n}" for a, n in zip(adj, noun)]


def show_audio(file, start, duration):
    a = Audio.from_file(file, offset=start, duration=duration)

    # Create two columns
    col1, col2 = st.columns(2)

    # Add audio players in each column
    with col1:
        st.write("full speed")
        st.audio(
            a.samples,
            sample_rate=a.sample_rate,
            format="audio/wav",
            start_time=0,
        )

    with col2:

        st.write("1/4 speed:")
        st.audio(
            a.samples,
            sample_rate=a.sample_rate // 4,
            format="audio/wav",
            start_time=0,
        )

    spec = Spectrogram.from_audio(a).bandpass(1500, 8000)
    img = spec.to_image(range=[-80, -20], invert=True)
    st.image(img)


def parse_list_of_lists(cell):
    # Replace newline characters and convert the cell string to a list of lists
    cell = (
        re.sub(r"\s+", ",", cell.replace("\n", ""))
        .replace("[,", "[")
        .replace(",,", ",")
    )
    # cell = cell.strip(', ')
    return ast.literal_eval(cell)


palette = sns.color_palette(palette="Dark2") + sns.color_palette("pastel")


# splits_dir = "/home/sml161/loca26_bird_volume/outputs/7_contrastive_iid/splits"
dataset_path = Path("./data")
# "/media/sml161/BULocaData/SBT/2016/V1"
# val_df = pd.read_csv(f"{splits_dir}/val_labels_2_annotators.csv")
# test_df = pd.read_csv(f"{splits_dir}/test_labels_2_annotators.csv")

import pandas as pd

labels = pd.read_csv(dataset_path / "all_labels_with_clusters.csv")
locations_df = pd.read_csv(dataset_path / "SBT_Stations_2016_rtk_points.csv")

# only use the nearest clips as examples
# (from sets of clips at different distances, same localized event)
labels = labels[labels["nearest_recorder"] == True]

# array_metadata = pd.read_csv(f"{dataset_path}/{array}/{array}_rtk_points.csv")
# array_metadata["x"] = array_metadata["EASTING"] - array_metadata["EASTING"].min()
# array_metadata["y"] = array_metadata["NORTHING"] - array_metadata["NORTHING"].min()

st.write(f"## Ovenbird song map")
# instructions
with st.expander("Info and Instructions"):
    st.write(
        """
    This web page displays Ovenbird songs with individual labels and spatial locations on
    13 acoustic localization grids from Alberta, Canada. See the preprint for details:

    https://www.biorxiv.org/content/10.1101/2025.06.26.661638v1.full.pdf 

    Colors in the map represent individual labels, as assigned by human annotators or
    automated individual ID models. The "custom Ovenbird feature extractor" is our 
    model described in the preprint, and creates very similar labels to the human annotators.

    Colors recycle for each array (no individuals occur across multiple arrays)

    Individual bird names are randomly generated from adjective-noun pairs.

    #### Using the controls: 
    
    Use the first dropdown menu to select human or automated individual labels.

    Use the second dropdown to select a localization grid (array).

    Click and drag on the map to pan around, and use the scroll wheel to zoom in and out.

    Click on points in the map to listen to the audio clip corresponding to that point.

    Scroll down to see more audio examples from each individual on that array,
    with spectrograms and real-time or slowed-down playback.
            
    """
    )

cols = st.columns(2)
label_names = {
    "final_label": "Human-annotated individual",
    "song_based_label": "Human-annotated (w/o spatial info)",
    "ovenbird_cluster_labels": "custom Ovenbird feature extractor",
    "baseline_resnet18_cluster_labels": "ResNet18 baseline",
    "hawkears_cluster_labels": "HawkEars",
    "birdnet_cluster_labels": "BirdNET",
}
with cols[0]:

    label_column = st.selectbox(
        "Select annotated label or automated ID label:",
        options=[
            "final_label",
            "song_based_label",
            "ovenbird_cluster_labels",
            "baseline_resnet18_cluster_labels",
            "hawkears_cluster_labels",
            "birdnet_cluster_labels",
        ],
        format_func=lambda x: label_names[x],
        index=0,
        key="label_column",
        help="Display human annotated labels or automated ID labels from different models.",
    )

labels = labels[labels[label_column].apply(lambda x: x not in ("n", "u"))]
labels[label_column] = labels[label_column].astype(int)
labels["name"] = labels[label_column].apply(lambda x: names[x])

all_arrays = labels["array"].unique()
with cols[1]:
    array = st.selectbox(
        "Select localization grid:",
        all_arrays,
    )

array_metadata = locations_df[locations_df["array_folder_name"] == array].copy()
utm_zone = array_metadata["UTM_ZONE"].values[0]
array_metadata["Latitude"], array_metadata["Longitude"] = utm.to_latlon(
    array_metadata["EASTING"], array_metadata["NORTHING"], utm_zone, "N"
)
array_metadata["name"] = "recorder"

array_origin = array_metadata["EASTING"].min(), array_metadata["NORTHING"].min()
array_dets = labels[labels["array"] == array].copy()
array_dets["EASTING"] = array_dets["bird_position_x"] + array_origin[0]
array_dets["NORTHING"] = array_dets["bird_position_y"] + array_origin[1]

array_dets["Latitude"], array_dets["Longitude"] = utm.to_latlon(
    array_dets["EASTING"], array_dets["NORTHING"], utm_zone, "N"
)

array_dets = array_dets[array_dets[label_column].apply(lambda x: x not in ("n", "u"))]

array_metadata["color"] = [(0, 0, 0, 0.6)] * len(array_metadata)
array_metadata["size"] = 1
# color by aiid label
array_dets[label_column] = array_dets[label_column].astype(int)
aiid_counter_map = {
    label: i for i, label in enumerate(array_dets[label_column].unique())
}
array_dets["aiid_index"] = array_dets[label_column].map(aiid_counter_map)
array_dets["color"] = array_dets["aiid_index"].apply(
    lambda x: tuple(list(palette[x % len(palette)]) + [0.6])
)
array_dets["size"] = 3
all_points = pd.concat([array_metadata, array_dets])

from streamlit.elements.lib.color_util import ColorTuple
from typing import cast

all_points["rgb255"] = all_points["color"].apply(
    lambda x: cast(ColorTuple, (np.array(x) * 255).astype("uint8"))
)
all_points["r"] = all_points["rgb255"].apply(lambda x: x[0])
all_points["g"] = all_points["rgb255"].apply(lambda x: x[1])
all_points["b"] = all_points["rgb255"].apply(lambda x: x[2])
all_points["a"] = all_points["rgb255"].apply(lambda x: x[3])

all_points["rgb255"] = None

clip_dur = 3


def rgb_to_hex(color_tuple):
    c = (np.array(color_tuple) * 255).astype("uint8")
    return "#{:02x}{:02x}{:02x}".format(*c)


# Create Pydeck Layer for Map
layer = pdk.Layer(
    "ScatterplotLayer",
    all_points,
    get_position=["Longitude", "Latitude"],
    get_fill_color=["r", "g", "b", "a"],
    get_radius="size",
    pickable=True,
    id="scatterplot-layer",
)

# Use Pydeck with light background for visibility
terrain_map = pdk.Deck(
    layers=[layer],
    initial_view_state=pdk.ViewState(
        latitude=array_metadata["Latitude"].mean(),
        longitude=array_metadata["Longitude"].mean(),
        zoom=16,
        pitch=0,
    ),
    map_style="light",
    tooltip={"text": "{name}"},
)


# Display the map
event = st.pydeck_chart(
    terrain_map, on_select="rerun", selection_mode="single-object", height=400
)
st.write("Click on points in the map to listen to the corresponding audio clip.")

# if user clicked on a point, show that song:
if "scatterplot-layer" in event.selection["objects"]:
    # event.selection # show everything as dictionary
    click_data = event.selection["objects"]["scatterplot-layer"][0]

    # color bar
    color = palette[click_data["aiid_index"] % len(palette)]
    img = Image.new("RGB", (400, 5), tuple((np.array(color) * 255).astype("uint8")))
    st.image(img)

    st.header(f"selected song: {click_data['name']}")
    show_audio(
        file=dataset_path / click_data["rel_path"],
        start=click_data["song_center_time"] - clip_dur / 2,
        duration=clip_dur,
    )
    add_vertical_space(2)

st.header(f"Examples from each individual")

for indv, dets in array_dets.groupby(label_column):
    name = dets["name"].values[0]
    color = palette[dets["aiid_index"].values[0] % 10]
    if len(dets) > 5:
        dets = dets.sample(5)

    # color bar
    img = Image.new("RGB", (400, 5), tuple((np.array(color) * 255).astype("uint8")))
    # break
    st.image(
        img,
        caption=None,
        width=None,
        use_column_width=None,
        clamp=False,
        channels="RGB",
        output_format="auto",
        use_container_width=False,
    )

    st.markdown(
        f'<span style="color:{rgb_to_hex(color)}; font-weight: bold;">{name}</span>',
        unsafe_allow_html=True,
    )
    with st.expander(f"audio examples"):

        for i, row in dets.iterrows():
            show_audio(
                file=dataset_path / row.rel_path,
                start=10 / 2 - clip_dur / 2,
                duration=clip_dur,
            )

            add_vertical_space(2)
