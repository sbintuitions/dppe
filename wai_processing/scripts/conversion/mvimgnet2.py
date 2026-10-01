# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
import shutil
from pathlib import Path

import numpy as np
from argconf import argconf_parse
from natsort import natsorted
from tqdm import tqdm
from wai_processing.core import store_data
from wai_processing.utils.globals import WAI_PROC_CONFIG_PATH

# Import COLMAP binary reading utilities
from wai_processing.utils.read_write_model import read_cameras_binary, read_images_binary
from wai_processing.utils.wrapper import convert_scenes_wrapper

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
logger = logging.getLogger(__name__)


def qvec2rotmat(qvec):
    """Convert COLMAP quaternion to 3x3 rotation matrix."""
    qvec = qvec / np.linalg.norm(qvec)
    r, i, j, k = qvec
    return np.array(
        [
            [1 - 2 * (j**2 + k**2), 2 * (i * j - k * r), 2 * (i * k + j * r)],
            [2 * (i * j + k * r), 1 - 2 * (i**2 + k**2), 2 * (j * k - i * r)],
            [2 * (i * k - j * r), 2 * (j * k + i * r), 1 - 2 * (i**2 + j**2)],
        ]
    )


def convert_scene(cfg, scene_name):
    """
    Process an MVImgNet2 scene into the WAI format using COLMAP binary files.
    """
    original_scene_path = scene_name.replace("_", "/")
    scene_root = Path(cfg.original_root)
    scene_dir = scene_root / original_scene_path
    target_scene_root = Path(cfg.root) / scene_name

    # Prepare directories
    images_dir = target_scene_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Path to binary sparse files
    cameras_bin = scene_dir / "sparse/0/cameras.bin"
    images_bin = scene_dir / "sparse/0/images.bin"

    # Read binary files
    colmap_cameras = read_cameras_binary(str(cameras_bin))
    colmap_images = read_images_binary(str(images_bin))

    # Create a mapping for fast lookup by filename
    # Using the filename (e.g., "030.jpg") as key
    images_by_name = {img.name: img for img in colmap_images.values()}

    image_files = natsorted(os.listdir(scene_dir / "images"))
    wai_frames = []

    if not colmap_cameras:
        logger.error(f"No cameras found in {cameras_bin}")
        return

    cam_id = list(colmap_cameras.keys())[0]
    cam = colmap_cameras[cam_id]

    print(f"Found {len(image_files)} images and {len(images_by_name)} pose entries.")

    for image_name in tqdm(image_files):
        # Match using the filename
        img_data = images_by_name.get(image_name)

        if img_data is None:
            # Check if it exists with path prefix (e.g., "images/030.jpg")
            img_data = next(
                (img for img in colmap_images.values() if img.name.endswith(image_name)), None
            )

        if img_data is None:
            logger.warning(f"Could not find pose for image: {image_name}")
            continue

        # Convert pose: W2C (from COLMAP) to C2W
        R = qvec2rotmat(img_data.qvec)
        t = img_data.tvec.reshape([3, 1])
        w2c = np.concatenate([np.concatenate([R, t], 1), [[0, 0, 0, 1]]], 0)
        c2w = np.linalg.inv(w2c)

        # Copy image
        src_path = scene_dir / "images" / image_name
        dst_path = images_dir / image_name
        if src_path.exists():
            shutil.copy(src_path, dst_path)
        else:
            logger.error(f"Source image not found: {src_path}")
            continue

        # Prepare frame metadata
        wai_frame = {
            "frame_name": os.path.splitext(image_name)[0],
            "image": str(f"images/{image_name}"),
            "file_path": str(f"images/{image_name}"),
            "transform_matrix": c2w.tolist(),
            "h": int(cam.height),
            "w": int(cam.width),
        }

        # Handle intrinsic parameters based on model
        # COLMAP params: PINHOLE [fx, fy, cx, cy]
        # SIMPLE_RADIAL [fx, fy, cx, cy, k]
        if cam.model == "PINHOLE":
            wai_frame.update(
                {
                    "fl_x": cam.params[0],
                    "fl_y": cam.params[1],
                    "cx": cam.params[2],
                    "cy": cam.params[3],
                }
            )
        elif cam.model == "SIMPLE_RADIAL":
            wai_frame.update(
                {
                    "fl_x": cam.params[0],
                    "fl_y": cam.params[0],
                    "cx": cam.params[1],
                    "cy": cam.params[2],
                    "k": cam.params[3],
                }
            )

        wai_frames.append(wai_frame)

    # Final scene metadata
    scene_meta = {
        "scene_name": scene_name,
        "dataset_name": cfg.dataset_name,
        "version": cfg.version,
        "shared_intrinsics": True,
        "camera_model": cam.model,
        "camera_convention": "opencv",
        "frames": wai_frames,
    }
    store_data(target_scene_root / "scene_meta.json", scene_meta, "scene_meta")


def get_original_scene_names(cfg):
    root = Path(cfg.original_root)
    seq_list = []
    for scene in os.listdir(root):
        if scene != "tar_files":
            scene_path = root / scene
            if scene_path.is_dir():
                for seq in os.listdir(scene_path):
                    seq_list.append(f"{scene}_{seq}")
    return seq_list


def get_single_scene_list(cfg, scene_name=None):
    return scene_name


if __name__ == "__main__":
    import cv2

    cv2.setNumThreads(0)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    from functools import partial

    cfg = argconf_parse(WAI_PROC_CONFIG_PATH / "conversion/mvimgnet2.yaml")
    target_root_dir = Path(cfg.root)
    target_root_dir.mkdir(parents=True, exist_ok=True)

    # 1. Get all scenes
    all_scene_names = get_original_scene_names(cfg)
    logger.info(f"Total scenes to process: {len(all_scene_names)}")

    # 2. Setting parallel
    max_workers = 16
    chunks = [all_scene_names[i::max_workers] for i in range(max_workers)]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                convert_scenes_wrapper,
                convert_scene,
                cfg,
                get_original_scene_names_func=partial(get_single_scene_list, scene_name=chunk),
            )
            for chunk in chunks
        ]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Parallel Processing"):
            try:
                # Display result
                _.result()
            except Exception as e:
                logger.error(f"Worker generated an exception: {e}")

    logger.info("All parallel tasks submitted and completed.")
