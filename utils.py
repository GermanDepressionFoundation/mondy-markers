import json
import os

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

PLOT_STYLES = {
    "font": "Arial",
    "colors": {
        "Rohdaten_train": "#5761AD",
        "Rohdaten_test": "#000000",
        "Elasticnet": "#F2846B",
        "RF": "#9E2040",
    },
}


def add_logo_to_figure(
    fig, logo_path="SDD_Logo_rgb_pos_600_reduced2.png", position="top_right", zoom=0.15
):
    if not os.path.exists(logo_path):
        print(f"[INFO] Logo file not found: {logo_path}, skipping logo.")
        return

    """Adds a logo image to a matplotlib figure."""
    logo_img = plt.imread(logo_path)
    ax_logo = fig.add_axes([0, 0, 1, 1], zorder=-1)
    ax_logo.axis("off")

    imagebox = OffsetImage(logo_img, zoom=zoom)

    positions = {
        "top_left": (0.03, 0.97),
        "top_right": (0.97, 0.97),
        "bottom_left": (0.03, 0.03),
        "bottom_right": (0.97, 0.03),
    }
    xy = positions.get(position, (0.97, 0.97))

    ab = AnnotationBbox(imagebox, xy, xycoords="figure fraction", frameon=False)
    ax_logo.add_artist(ab)


pseudonym_to_letter_mapping_path = os.path.join("config", "pseudonym_to_letter.json")
if os.path.exists(pseudonym_to_letter_mapping_path):
    with open(pseudonym_to_letter_mapping_path, "r", encoding="utf-8") as f:
        PSEUDONYM_TO_LETTER = json.load(f)
else:
    print(f"[WARN] Mapping file not found at {pseudonym_to_letter_mapping_path}")
    PSEUDONYM_TO_LETTER = {}
