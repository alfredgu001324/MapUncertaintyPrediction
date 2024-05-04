import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict


def main():
    with open("/Users/YOUR_USERNAME_HERE/Downloads/distance_data.pkl", "rb") as f:
        data = pickle.load(f)

    data_dict = defaultdict(list)
    for map_method in data.keys():
        for subkey in data[map_method].keys():
            if subkey == "labels":
                data_dict["Map Element"].extend(data[map_method][subkey].tolist())
            else:
                data_dict[subkey].extend(data[map_method][subkey].tolist())
        data_dict["method"].extend([map_method if map_method != "MapTRv2_cent" else "MapTRv2-Centerline"] * data[map_method]["distance"].shape[0])

    rename_dict = {
        "Boundary": "Boundary",
        "Centerline": "Centerline",
        "Divider": "Divider",
        "Pedestrian Crossing": "Ped. Crossing"
    }

    data_df = pd.DataFrame(data_dict)
    data_df["Map Element"] = data_df["Map Element"].map(rename_dict)
    data_df["binned_x"] = pd.cut(data_df["distance"], bins=np.arange(0, 31, 5))

    g = sns.catplot(
        data=data_df, x="binned_x", y="uncertainties", 
        col="method", hue="Map Element", hue_order=["Boundary", "Centerline", "Divider", "Ped. Crossing"],
        kind="point", col_order=["MapTR", "MapTRv2", "MapTRv2-Centerline", "StreamMapNet"],
        height=2.7, aspect=0.9, dodge=True,
        palette=["#008000", "#000000", "#FFA500", "#0000FF"],
    )
    g.set_axis_labels("Distance (m)", "Uncertainty (m)")
    # g.set_xticklabels(["Men", "Women", "Children"])
    g.set_titles("{col_name}")
    g.set(ylim=(0, 4))
    for axes in g.axes.flat:
        axes.set_xticklabels(axes.get_xticklabels(), rotation=30)

    g.fig.savefig("./unc_vs_distance.pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()