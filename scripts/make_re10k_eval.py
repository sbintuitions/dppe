"""Generate a filtered evaluation index JSON for the RealEstate10K (RE10K) dataset.

Reads an original evaluation index JSON, remaps scene paths using the provided
data root, validates each entry by attempting to load it via EvalDataset, and
writes out a cleaned JSON containing only successfully loadable scenes.
"""

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

from nvs.dataset import EvalDataset


def main(args):
    """Load, filter, and save the RE10K evaluation index.

    Args:
        args: Parsed command-line arguments with ``suffix`` and ``data_root``.
    """
    project_root = Path(__file__).resolve().parent.parent
    # Build file paths dynamically using the suffix
    suffix = args.suffix
    original_json_path = str(project_root / "assets" / f"org_evaluation_index_re10k{suffix}.json")
    temp_json_name = f"temp_evaluation_index_re10k{suffix}.json"
    temp_json_path = str(project_root / "assets" / temp_json_name)
    new_json_path = str(project_root / "assets" / f"evaluation_index_re10k{suffix}.json")
    new_path_template = os.path.join(args.data_root, "test_{}")

    print(f"Target file: {original_json_path}")
    if not os.path.exists(original_json_path):
        raise FileNotFoundError(f"{original_json_path} not found.")

    # 1. Load the original JSON
    with open(original_json_path, "r") as f:
        index_info = json.load(f)

    all_index_info = {}
    for scene, info in index_info.items():
        if info is None or not isinstance(info, dict):
            continue

        new_key = new_path_template.format(scene)

        # Use the context / target entries from the original JSON as-is
        if "context" not in info or "target" not in info:
            continue

        all_index_info[new_key] = info

    mode_str = "extraction"
    print(
        f"Index {mode_str} data: {len(all_index_info)} entries (original data: {len(index_info)} entries)"
    )

    # 2. Save all entries to a temporary file
    # Create the assets directory if it does not exist
    os.makedirs(str(project_root / "assets"), exist_ok=True)
    with open(temp_json_path, "w") as f:
        json.dump(all_index_info, f, indent=4)

    # 3. Instantiate EvalDataset with the temporary file
    print("Starting load test via the dataset class...")
    dataset = EvalDataset(
        folder="assets",
        test_index_fp=temp_json_name,
        patch_size=256,
        zoom_factor=1.0,
        input_views=2,
        supervise_views=3,
    )

    valid_index_info = {}

    # 4. Call __getitem__ for each entry, keeping only those that load successfully
    for i in tqdm(range(len(dataset))):
        data_dir = str(dataset.data_dirs[i], encoding="utf-8")
        try:
            _ = dataset[i]
            valid_index_info[data_dir] = all_index_info[data_dir]
        except Exception as e:  # noqa: BLE001 - skip entries that fail to load
            print(f"[Skip] {data_dir}: {e}")

    # 5. Save the filtered, clean data as the final JSON
    with open(new_json_path, "w") as f:
        json.dump(valid_index_info, f, indent=4)

    # 6. Clean up the temporary file
    if os.path.exists(temp_json_path):
        os.remove(temp_json_path)

    print(
        f"Filtering complete: successfully loaded {len(valid_index_info)}/{len(all_index_info)} entries"
    )
    print(f"Results saved to {new_json_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter and process evaluation index JSON for RE10K"
    )
    parser.add_argument(
        "--suffix", type=str, default="", help="Suffix for the json file, e.g., '_ctx2'"
    )
    parser.add_argument(
        "--data_root", type=str, required=True, help="Root path to the RealEstate10K test data"
    )
    args = parser.parse_args()

    main(args)
