"""Generate an evaluation index JSON for the MVImgNet2 dataset.

Reads a list of scene paths, samples context and target view indices for each
scene using angle-based heuristics, and writes the resulting index to a JSON file.
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

from nvs.idx_sampling import sample_and_split_trajectory
from nvs.mvin2_dataset import sample_nvs_indices


def sample_max_angle_indices(scene_meta, supervise_views=3):
    """Sample the two frames with the largest angular separation and random targets.

    Uses dot-product computation over all frame viewing directions (Z-axis of
    the transform matrix) to find the pair with the maximum angular distance.
    Remaining target views are sampled randomly from the other frames.

    Args:
        scene_meta: Dictionary containing a ``frames`` list, where each frame
            has a ``transform_matrix`` key.
        supervise_views: Number of target (supervision) views to sample.

    Returns:
        A list of frame indices ``[ctx1, ctx2, tgt1, tgt2, ...]``, or ``None``
        if there are fewer than 3 frames.
    """
    frames = scene_meta["frames"]
    num_frames = len(frames)
    if num_frames < 3:
        return None

    # 1. Extract the viewing direction (Z-axis) of every frame into an Nx3 matrix
    rays = np.array([np.array(f["transform_matrix"])[:3, 2] for f in frames])
    # Normalize
    rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)

    # 2. Compute all pairwise dot products (NxN matrix). Minimum dot product = maximum angle
    dot_products = rays @ rays.T

    # Get the indices (idx1, idx2) of the minimum dot product
    idx1, idx2 = np.unravel_index(np.argmin(dot_products), dot_products.shape)

    # Randomize order
    if random.random() > 0.5:
        idx1, idx2 = idx2, idx1

    # 3. Sample targets from frames other than idx1 and idx2
    # (Since these two are the most separated, other frames lie between them)
    available_targets = [i for i in range(num_frames) if i != idx1 and i != idx2]

    if len(available_targets) >= supervise_views:
        targets = random.sample(available_targets, k=supervise_views)
    else:
        targets = random.choices(available_targets, k=supervise_views)

    # Convert numpy dtypes to plain int before returning
    return [int(idx1), int(idx2)] + targets


def main(args):
    """Process all scenes and write the evaluation index JSON.

    Args:
        args: Parsed command-line arguments controlling context view count
            and angle thresholds.
    """
    project_root = Path(__file__).resolve().parent.parent
    # Set up input/output file paths
    input_txt_path = str(project_root / "assets" / "test_MVImgNet2.txt")
    if args.num_views == 2:
        output_json_path = str(project_root / "assets" / "evaluation_index_MVIN2.json")
    else:
        output_json_path = str(
            project_root / "assets" / f"evaluation_index_MVIN2_context{args.num_views}.json"
        )
    supervise_views = 3
    context_views = args.num_views

    # Dictionary to store results
    split_data = {}

    # 1. Read scene paths from the text file
    with open(input_txt_path, "r") as f:
        scene_paths = sorted(line.strip() for line in f if line.strip())

    print(f"Processing {len(scene_paths)} scenes in total...")

    # 2. Process each scene
    for scene_path in tqdm(scene_paths):
        meta_json_path = os.path.join(scene_path, "scene_meta.json")

        # Check whether scene_meta.json exists
        if not os.path.exists(meta_json_path):
            print(f"Warning: {meta_json_path} not found. Skipping.")
            continue

        try:
            # Load metadata and run sampling
            with open(meta_json_path, "r") as f:
                scene_meta = json.load(f)

            if args.num_views == 2:
                sampled_indices = sample_nvs_indices(
                    scene_meta,
                    min_angle=args.min_angle,
                    max_angle=args.max_angle,
                    supervise_views=supervise_views,
                )
            else:
                sampled_indices = sample_and_split_trajectory(
                    scene_meta,
                    total_views=context_views * 2,
                    supervise_view=supervise_views,
                    max_obj_angle=30.0,
                    max_pano_angle=30.0,
                )

            assert (
                sampled_indices is not None
                and len(sampled_indices) == context_views + supervise_views
            ), (
                f"Number of sampled indices does not match the expected count: {len(sampled_indices)} != {context_views + supervise_views}"
            )
            sampled_indices_dict = {
                "context": sampled_indices[:context_views],
                "target": sampled_indices[context_views:],
            }

            for i in sampled_indices_dict["context"] + sampled_indices_dict["target"]:
                assert os.path.exists(
                    os.path.join(scene_path, scene_meta["frames"][i]["file_path"])
                ), f"File not found: {scene_meta['frames'][i]['file_path']}"

            # Store result keyed by scene path
            split_data[scene_path] = sampled_indices_dict

        except Exception as e:  # noqa: BLE001 - skip scene on any error
            print(f"Error: problem occurred while processing {scene_path} - {e}")

    # 3. Write results to a JSON file
    with open(output_json_path, "w") as f:
        # Use indent=4 for human-readable formatting
        json.dump(split_data, f, indent=4)

    print(f"\nProcessing complete! Results saved to {output_json_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process dataset.")

    args = parser.parse_args()
    args.min_angle = 30.0
    args.max_angle = 60.0

    for num_views in [2, 4, 6, 8, 10, 12]:
        args.num_views = num_views
        main(args)
