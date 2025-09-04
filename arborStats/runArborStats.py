import os
import sys
import pandas as pd


if __name__ == "__main__":
    output_dir = "/Volumes/fsmresfiles/Ophthalmology/Research/SchwartzLab/Flatone-Output"
    sheet_id = "1o4i53h92oyzsBc8jEWKmF8ZnfyXKXtFCTaYSecs8tBk"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"

    df = pd.read_csv(url)
    target_values = ["Complete", "Complete (cut off)"]
    indices = df[df["Status"].isin(target_values)].index.tolist()
    #print(indices)
    #print(df.loc[indices, "Status"])
    seg_ids = df.loc[indices, "Final SegID"]
    not_nan_mask = seg_ids.notna()
    print(not_nan_mask)
    not_nan_indices = seg_ids[not_nan_mask].index
    not_nan_seg_ids = seg_ids[not_nan_mask].astype(int).tolist()
    