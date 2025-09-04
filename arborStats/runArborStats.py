import os
import sys
import subprocess
import pandas as pd
from arborStats import load_swc, arborStatsFromSkeleton
import pickle


if __name__ == "__main__":
    output_dir = "/Volumes/fsmresfiles/Ophthalmology/Research/SchwartzLab/flatone_output"
    sheet_id = "1o4i53h92oyzsBc8jEWKmF8ZnfyXKXtFCTaYSecs8tBk"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"

    dtype_map = {
        "Final SegID": "Int64",
        "Status": "string"
    }
    df = pd.read_csv(url, dtype=dtype_map, usecols=["Final SegID", "Status"])
    df = df.dropna(subset=["Final SegID"])
    df["Final SegID"] = df["Final SegID"].astype("int64")

    target_values = ["Complete", "Complete (cut off)"]
    indices = df[df["Status"].isin(target_values)].index.tolist()
    seg_ids = df.loc[indices, "Final SegID"]
    not_nan_mask = seg_ids.isna().all()
    print(seg_ids[:10])
    seg_ids_not_processed = []
    seg_ids_with_arbor_stats_error = []
    delete_files = ["mesh.obj", "skeleton.swc", "skeleton.png", "error_msg.txt"]

    #not_nan_seg_ids = [720575940547803008, 720575940552928256, 720575940564637184]
    for seg_id in seg_ids[:10]:
        print(seg_id)
        result = subprocess.run(
            ["flatone", str(seg_id), "--output-dir", str(output_dir), "--overwrite"],
            capture_output=True,
            text=True
        )
        #print(result.stderr)
        if "No meshes found." in result.stderr:
            print(f"No meshes found for segment ID {seg_id}, skipping.")
            filepath = os.path.join(output_dir, str(seg_id), "error_msg.txt")
            with open(filepath, "w") as f:
                f.write("No meshes found.\n")
            seg_ids_not_processed.append(seg_id)
            
            
        else:
            output_seg_dir = os.path.join(output_dir, str(seg_id))
            with open(os.path.join(output_seg_dir, "processed_seg_ids.txt"), "w") as f:
                f.write(f"{seg_id}\n")

            skeleton_warped_path = os.path.join(output_seg_dir, "skeleton_warped.swc")
            if os.path.exists(skeleton_warped_path):
                print(f"Loading SWC for segment {seg_id} from {skeleton_warped_path}")
                coords, radii, edges = load_swc(skeleton_warped_path)
                print(f"Computing arbor stats for segment {seg_id}")
                try:
                    stats, units = arborStatsFromSkeleton(coords, edges, radii=radii)
                    results = {
                    "segment_id": seg_id,
                    "stats": stats,
                    "units": units
                    }
                    with open(os.path.join(output_seg_dir, "arbor_stats.pkl"), "wb") as f:
                        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    print(f"Error computing arbor stats for segment {seg_id}: {e}")
                    seg_ids_with_arbor_stats_error.append(seg_id)
                    with open(os.path.join(output_seg_dir, "arbor_stats_error.txt"), "w") as f:
                        f.write(f"Error computing arbor stats for segment {seg_id}: {e}")
                    continue
        


    with open(os.path.join(output_dir, "not_processed_seg_ids.txt"), "w") as f:
        for seg_id in seg_ids_not_processed:
            f.write(f"{seg_id}\n")


    with open(os.path.join(output_dir, "arbor_stats_error_seg_ids.txt"), "w") as f:
        for seg_id in seg_ids_with_arbor_stats_error:
            f.write(f"{seg_id}\n")