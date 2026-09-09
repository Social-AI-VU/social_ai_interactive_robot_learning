"""
Merge several robomimic-format demo HDF5 files (each produced by
robomimic's dataset_states_to_obs + split_train_val scripts) into one
dataset, renumbering demos so they don't collide, and writing a fresh
80/20 train/valid mask over the merged set.

Usage:
    python scripts/merge_demos.py --inputs a/obs.hdf5 b/obs.hdf5 --out merged/merged.hdf5
"""

import argparse
import os

import h5py
import numpy as np


def merge_demos(input_paths, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    demo_names = []

    with h5py.File(out_path, "w") as f_out:
        grp_out = f_out.create_group("data")

        next_idx = 0
        env_args_written = False

        for in_path in input_paths:
            with h5py.File(in_path, "r") as f_in:
                if not env_args_written:
                    for key, value in f_in["data"].attrs.items():
                        grp_out.attrs[key] = value
                    env_args_written = True

                for demo_key in f_in["data"].keys():
                    new_name = f"demo_{next_idx}"
                    f_in.copy(f"data/{demo_key}", grp_out, name=new_name)
                    demo_names.append(new_name)
                    next_idx += 1

        grp_out.attrs["total"] = sum(
            int(grp_out[name].attrs.get("num_samples", 0)) for name in demo_names
        )

        rng = np.random.default_rng(0)
        shuffled = list(demo_names)
        rng.shuffle(shuffled)
        split = max(1, int(len(shuffled) * 0.8)) if len(shuffled) > 1 else len(shuffled)
        train_names, valid_names = shuffled[:split], shuffled[split:]

        mask_grp = f_out.create_group("mask")
        mask_grp.create_dataset("train", data=np.array(train_names, dtype="S"))
        mask_grp.create_dataset("valid", data=np.array(valid_names, dtype="S"))

    print(f"Merged {len(demo_names)} demos from {len(input_paths)} file(s) into {out_path}")
    print(f"  train: {len(train_names)}, valid: {len(valid_names)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="Source .hdf5 files to merge")
    parser.add_argument("--out", required=True, help="Path to write the merged .hdf5 file")
    args = parser.parse_args()

    merge_demos(args.inputs, args.out)
