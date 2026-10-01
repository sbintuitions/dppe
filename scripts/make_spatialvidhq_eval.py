"""Generate an evaluation index JSON for the SpatialVidHQ dataset.

Reads a list of scene paths, samples context and target view indices for each
scene, and writes the resulting evaluation index to a JSON file.
"""

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

from nvs.spatialvidhq_dataset import sample_nvs_indices


def main(args):
    """Process all scenes and write the evaluation index JSON.

    Args:
        args: Parsed command-line arguments with ``suffix``.
    """
    project_root = Path(__file__).resolve().parent.parent
    # Handle the suffix for output file naming
    suffix = args.suffix
    input_txt_path = str(project_root / "assets" / "test_spatialvidhq.txt")
    output_json_path = str(project_root / "assets" / f"evaluation_index_spatialvidhq{suffix}.json")
    supervise_views = 3

    print(f"Input list: {input_txt_path}")
    print(f"Output file: {output_json_path}")

    # Dictionary to store results
    split_data = {}

    # 1. Read scene paths from the text file
    if not os.path.exists(input_txt_path):
        raise FileNotFoundError(f"{input_txt_path} not found.")

    with open(input_txt_path, "r") as f:
        scene_paths = sorted(line.strip() for line in f if line.strip())

    print(f"Processing {len(scene_paths)} scenes in total...")

    # 2. Process each scene
    for scene_path in tqdm(scene_paths):
        # Use scene_meta.json for sampling
        meta_json_path = os.path.join(scene_path, "scene_meta.json")

        if not os.path.exists(meta_json_path):
            print(f"Warning: {meta_json_path} not found. Skipping.")
            continue

        try:
            with open(meta_json_path, "r") as f:
                scene_meta = json.load(f)

            sampled_indices = sample_nvs_indices(scene_meta, supervise_views=supervise_views)
            assert sampled_indices is not None
            sampled_indices_dict = {"context": sampled_indices[:2], "target": sampled_indices[2:]}

            split_data[scene_path] = sampled_indices_dict

        except Exception as e:  # noqa: BLE001 - skip scene on any error
            print(f"Error: problem occurred while processing {scene_path} - {e}")

    # 3. Write results to a JSON file
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(split_data, f, indent=4)

    print(f"\nProcessing complete! Succeeded: {len(split_data)}/{len(scene_paths)}")
    print(f"Results saved to {output_json_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process evaluation index JSON for SpatialVidHQ")
    parser.add_argument(
        "--suffix", type=str, default="", help="Suffix for the output json file, e.g., '_ctx2'"
    )
    args = parser.parse_args()

    main(args)
