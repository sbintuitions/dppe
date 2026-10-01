"""MVIN2 dataset for novel view synthesis with angle-based view sampling.

Provides angle-aware context/target view selection and camera pose normalization
for multi-view image network training and evaluation.
"""

import json
import os
import random
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import Dataset

from dppe.torch import _invert_SE3
from nvs.idx_sampling import sample_and_split_trajectory


def get_camera_ray(transform_matrix):
    """Extract the camera viewing direction (Z-axis) from a camera-to-world matrix."""
    c2w = np.array(transform_matrix)
    ray = c2w[:3, 2]
    return ray / np.linalg.norm(ray)


def calc_angle(ray_a, ray_b):
    """Compute the angle (in degrees) between two unit vectors."""
    cos_angle = np.clip(np.dot(ray_a, ray_b), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def sample_nvs_indices(scene_meta, min_angle=30.0, max_angle=60.0, supervise_views=1):
    """Sample NVS context and target view indices based on angular constraints."""
    frames = scene_meta["frames"]
    num_frames = len(frames)

    if num_frames < 3:
        return None

    # 1. Determine idx1 and precompute rays/angles for all frames
    idx1 = random.randint(0, num_frames - 1)
    ray1 = get_camera_ray(frames[idx1]["transform_matrix"])

    frame_rays = []
    angles_from_idx1 = []

    for frame in frames:
        ray = get_camera_ray(frame["transform_matrix"])
        frame_rays.append(ray)
        angles_from_idx1.append(calc_angle(ray1, ray))

    # 2. Get idx2 candidates (those within the specified angle range)
    valid_idx2_candidates = [
        i
        for i, angle in enumerate(angles_from_idx1)
        if i != idx1 and min_angle <= angle <= max_angle
    ]

    if valid_idx2_candidates:
        idx2 = random.choice(valid_idx2_candidates)
    else:
        idx2 = idx1

    # 3. Get target candidates (rugby-ball-shaped angular constraint)
    target_candidates = []

    if idx1 != idx2:
        ray2 = frame_rays[idx2]
        angle_1_2 = angles_from_idx1[idx2]  # Angle between idx1 and idx2

        for i in range(num_frames):
            if i == idx1 or i == idx2:
                continue

            angle_1_i = angles_from_idx1[i]
            angle_2_i = calc_angle(ray2, frame_rays[i])

            # Constraint: angle from both references must be smaller than the baseline angle (angle_1_2)
            if angle_1_i < angle_1_2 and angle_2_i < angle_1_2:
                target_candidates.append(i)

    # 4. Determine target views
    if target_candidates:
        if len(target_candidates) >= supervise_views:
            # Enough candidates: sample without replacement
            target_indices = random.sample(target_candidates, k=supervise_views)
        else:
            # Not enough candidates: sample with replacement
            target_indices = random.choices(target_candidates, k=supervise_views)
    else:
        # No valid targets found (e.g., too few frames)
        target_indices = [idx1] * supervise_views

    return [idx1] + [idx2] + target_indices


def _normalize_poses_identity_unit_distance(
    in_c2ws: torch.Tensor,
    ref0_idx: int,
    ref1_idx: int,
):
    """
    Normalize the poses such that the ref0 camera is the identity
    and the ref1 camera is unit distance to the ref0 camera.
    """

    ref0_c2w = in_c2ws[ref0_idx]
    c2ws = torch.einsum("ij,njk->nik", _invert_SE3(ref0_c2w), in_c2ws)

    ref0_c2w = c2ws[ref0_idx]
    ref1_c2w = c2ws[ref1_idx]
    dist = torch.linalg.norm(ref1_c2w[:3, 3] - ref0_c2w[:3, 3])
    if dist > 1e-4:  # numerically stable
        c2ws[:, :3, 3] /= dist

    return c2ws


def resize_crop_with_subpixel_accuracy(
    image: np.ndarray, K: np.ndarray, patch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Resize and crop the image to have the smallest side equal to `patch_size`,
    while maintaining sub-pixel accuracy using a single warpAffine transformation.

    Args:
        image (np.ndarray): Input image.
        K (np.ndarray): Camera intrinsic matrix.
        patch_size (int): Target size of the smaller dimension.

    Returns:
        tuple[np.ndarray, np.ndarray]: Resized and cropped image, updated intrinsic matrix.
    """
    h, w = image.shape[:2]
    scale = patch_size / min(h, w)

    # Compute the affine transformation matrix combining scaling and cropping
    new_w, new_h = w * scale, h * scale
    crop_x = (new_w - patch_size) / 2
    crop_y = (new_h - patch_size) / 2

    M = np.array([[scale, 0, -crop_x], [0, scale, -crop_y]], dtype=np.float32)

    # Apply affine transformation with sub-pixel accuracy
    is_downsampling = min(h, w) > patch_size
    interpolation = cv2.INTER_AREA if is_downsampling else cv2.INTER_CUBIC
    cropped_resized_image = cv2.warpAffine(image, M, (patch_size, patch_size), flags=interpolation)

    # Update intrinsic matrix K
    K_scaled = K.copy()
    K_scaled[:2, :] *= scale
    K_scaled[0, 2] -= crop_x
    K_scaled[1, 2] -= crop_y

    return cropped_resized_image, K_scaled


def center_zoom_in_with_subpixel_accuracy(
    image: np.ndarray, K: np.ndarray, scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Zoom into the center of the image while maintaining sub-pixel accuracy using warpAffine.

    Args:
        image (np.ndarray): Input image.
        K (np.ndarray): Camera intrinsic matrix.
        scale (float): Zoom-in factor.

    Returns:
        tuple[np.ndarray, np.ndarray]: Zoomed image, updated intrinsic matrix.
    """
    if scale == 1.0:
        return image, K

    h, w = image.shape[:2]
    center_x, center_y = w / 2, h / 2

    # Compute the affine transformation matrix for zooming in at the center
    M = np.array(
        [[scale, 0, (1 - scale) * center_x], [0, scale, (1 - scale) * center_y]],
        dtype=np.float32,
    )

    # Apply affine transformation
    zoomed_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_AREA)

    # Update intrinsic matrix K.
    K_zoomed = K.copy()
    K_zoomed[:2, :] *= scale
    K_zoomed[0, 2] += (1 - scale) * center_x
    K_zoomed[1, 2] += (1 - scale) * center_y

    return zoomed_image, K_zoomed


def load_and_maybe_update_meta_info(json_path: str) -> tuple[bool, dict]:
    """Load the meta information from the `transforms.json` file.

    If the image paths (e.g., `images/xxx.jpg` ) stored in the `transforms.json` file
    are not found, try with `images_{1, 2, 4, 8}/xxx.jpg` instead, and update the
    camera intrinsics accordingly.
    """
    if not os.path.exists(json_path):
        return False, {}
    with open(json_path, "r") as f:
        meta_info = json.load(f)

    # Check if the image paths are valid
    frames = meta_info["frames"]
    if len(frames) == 0:
        return False, {}

    # Use the jpg version if it exists (for DL3DV)
    if "-jpeg" in json_path:
        for frame in frames:
            frame["file_path"] = os.path.splitext(frame["file_path"])[0] + ".jpeg"

    # Check if the image paths are valid
    maybe_relative_path_to_img = frames[0]["file_path"]
    if maybe_relative_path_to_img.startswith("/"):
        _start_path = os.path.abspath(os.path.dirname(json_path))
        for frame in frames:
            frame["file_path"] = os.path.relpath(frame["file_path"], start=_start_path)
        relative_path_to_img = frames[0]["file_path"]
    else:
        relative_path_to_img = maybe_relative_path_to_img

    abs_path_to_img = os.path.join(os.path.dirname(json_path), relative_path_to_img)
    if not os.path.exists(abs_path_to_img):
        # Try with images_{1, 2, 4, 8}/xxx.jpg
        assert "images/" in relative_path_to_img, (
            f"Invalid image path in the meta info file: {relative_path_to_img}"
        )
        factor = None
        for _factor in [1, 2, 4, 8]:
            _relative_path_to_img = relative_path_to_img.replace("images/", f"images_{_factor}/")
            _abs_path_to_img = os.path.join(os.path.dirname(json_path), _relative_path_to_img)
            if os.path.exists(_abs_path_to_img):
                factor = _factor
                break

        if factor is None:
            # No valid image path found
            return False, meta_info

        # Found a valid image path, update the meta info
        for frame in frames:
            frame["file_path"] = frame["file_path"].replace("images/", f"images_{factor}/")

        w, h = meta_info["w"], meta_info["h"]
        assert w % factor == 0 and h % factor == 0, (
            f"Invalid factor: {factor} with w={w} and h={h}"
        )

        for key in ["fl_x", "fl_y", "cx", "cy"]:
            meta_info[key] /= factor
        for key in ["w", "h"]:
            meta_info[key] //= factor

    return True, meta_info


def load_frames_from_meta_info(
    data_dir: str,
    meta_info: dict[str, Any],
    frame_ids: list[int],
    patch_size: int = 256,
    zoom_factor: float = 1.0,
    random_zoom: bool = False,
    camera_pose_only: bool = False,
) -> dict[str, Any] | np.ndarray:
    """Load images and camera parameters for the specified frame indices."""

    # Load the camera intrinsic
    frames = meta_info["frames"]
    K_raw = np.array(
        [
            [frames[0]["fl_x"], 0, frames[0]["cx"]],
            [0, frames[0]["fl_y"], frames[0]["cy"]],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    # Shortcut for loading only camera poses
    if camera_pose_only:
        c2ws = []
        for frame_id in frame_ids:
            frame = frames[frame_id]
            c2w = np.array(frame["transform_matrix"], dtype=np.float32)  # @ blender2opencv
            c2ws.append(c2w)
        return np.stack(c2ws)

    # Load the images
    images, Ks, c2ws, abs_image_paths = [], [], [], []

    for index, frame_id in enumerate(frame_ids):
        frame = frames[frame_id]

        rel_image_path = frame["file_path"]
        abs_image_path = os.path.join(data_dir, rel_image_path)
        image = imageio.imread(abs_image_path)[..., :3]

        per_image_zoom_factor = np.random.uniform(1.0, zoom_factor) if random_zoom else zoom_factor

        image, K = center_zoom_in_with_subpixel_accuracy(
            image,
            K_raw,
            per_image_zoom_factor,
        )
        image, K = resize_crop_with_subpixel_accuracy(image, K, patch_size)

        c2w = np.array(frame["transform_matrix"], dtype=np.float32)  # @ blender2opencv

        images.append(image)
        Ks.append(K)
        c2ws.append(c2w)
        abs_image_paths.append(abs_image_path)

    return {
        "image": np.stack(images),
        "K": np.stack(Ks),
        "camtoworld": np.stack(c2ws),
        "image_path": abs_image_paths,
    }


class TrainDataset(Dataset):
    """Training dataset with angle-based view sampling for NVS."""

    def __init__(
        self,
        data_dirs: list[str],
        patch_size: int = 256,
        zoom_factor: float = 1.0,  # 1.0 means disabled
        random_zoom: bool = False,  # only useful when zoom_factor is > 1.0
        input_views: int = 2,
        supervise_views: int = 6,
        verbose: bool = False,
    ):
        super().__init__()
        # No list/dict in the dataset, which would cause "memory leak"
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-715050814
        self.data_dirs = np.array(data_dirs).astype(np.bytes_)
        self.patch_size = patch_size
        self.zoom_factor = zoom_factor
        self.random_zoom = random_zoom
        self.input_views = input_views
        self.supervise_views = supervise_views
        self.verbose = verbose
        if self.verbose:
            print(f"[TrainDataset] Initialized with {len(self.data_dirs)} scenes.")

    def __len__(self):
        """Return the number of scenes."""
        return len(self.data_dirs)

    def __getitem__(self, _: Any) -> dict[str, Any]:
        # Choose a random scene
        data_dir = str(np.random.choice(self.data_dirs), encoding="utf-8")
        valid, meta_info = load_and_maybe_update_meta_info(
            os.path.join(data_dir, "scene_meta.json")
        )
        if not valid:
            if self.verbose:
                print(f"⚠️ [Skip] Invalid scene: {data_dir}")
            return self.__getitem__(None)

        # --- Multi view num training ---
        if self.input_views > 2:
            total_views_needed = self.input_views * 2

            if len(meta_info["frames"]) < total_views_needed:
                if self.verbose:
                    print(
                        f"⚠️ [Skip] Not enough frames ({len(meta_info['frames'])} < {total_views_needed}): {data_dir}"
                    )
                return self.__getitem__(None)

            frame_ids = sample_and_split_trajectory(
                scene_meta=meta_info,
                total_views=total_views_needed,
                supervise_view=self.supervise_views,
            )

            if not frame_ids or len(frame_ids) < (self.input_views + self.supervise_views):
                return self.__getitem__(None)
        # --- Original random sampling for 2 views input ---
        else:
            # Select views: first 2 indices are context, rest are target
            frame_ids = sample_nvs_indices(meta_info)
            if frame_ids is None:
                return self.__getitem__(None)

        try:
            # Load frames
            loaded = load_frames_from_meta_info(
                data_dir,
                meta_info,
                frame_ids,
                patch_size=self.patch_size,
                zoom_factor=self.zoom_factor,
                random_zoom=self.random_zoom,
            )
        except Exception as e:  # noqa: BLE001 - skip scene on any load error
            print(f"Error in {data_dir}: {e}. frame_ids: {frame_ids}")
            return self.__getitem__(None)

        # Preprocess poses
        camtoworld = torch.from_numpy(loaded["camtoworld"]).float()  # (N, 4, 4)
        # Clean camera matrix
        camtoworld[..., 3, :3] = 0.0
        camtoworld[..., 3, 3] = 1.0

        # Norm of each col of rotation matrix should be 1.
        rot_mat = camtoworld[..., :3, :3]  # (N, 3, 3)
        col_norms = torch.linalg.norm(rot_mat, dim=1)  # (N, 3)
        norm_errors = torch.abs(col_norms - 1.0)

        if (norm_errors > 0.001).any():
            print(f"⚠️ [Skip] Invalid rotation matrix in {data_dir}.")
            return self.__getitem__(None)

        camtoworld = _normalize_poses_identity_unit_distance(
            camtoworld, ref0_idx=0, ref1_idx=self.input_views - 1
        )
        K = torch.from_numpy(loaded["K"]).float()
        image = torch.from_numpy(loaded["image"]).float()
        image_path = loaded["image_path"]

        return {
            "camtoworld": camtoworld,
            "K": K,
            "image": image,
            "image_path": image_path,
        }


class EvalDataset(Dataset):
    """Evaluation dataset that loads pre-defined scene/view indices from a JSON file."""

    def __init__(
        self,
        file: str,
        patch_size: int = 256,
        zoom_factor: float = 1.0,  # 1.0 means disabled
        verbose: bool = False,
        first_n: int | None = None,
        rank: int | None = None,
        world_size: int | None = None,
        input_views: int = 2,
        supervise_views: int = 3,
    ):
        super().__init__()
        self.data_file = file
        self.patch_size = patch_size
        self.zoom_factor = zoom_factor
        self.input_views = input_views
        self.supervise_views = supervise_views

        assert os.path.exists(self.data_file), f"Index file not found: {self.data_file}"
        with open(self.data_file, "r") as f:
            scenes = json.load(f)
        if verbose:
            print(f"[EvalDataset] Found {len(scenes)} scenes in the index file. ")
        # Convert scene path keys to a sorted list and use it as the base for slicing
        scene_keys = sorted(scenes.keys())

        if first_n is not None:
            scene_keys = scene_keys[:first_n]

        if rank is not None and world_size is not None:
            scene_keys = scene_keys[rank::world_size]

        self.data_dirs = np.array(scene_keys).astype(np.bytes_)
        self.contexts = np.array([scenes[k]["context"] for k in scene_keys])
        self.targets = np.array([scenes[k]["target"] for k in scene_keys])

    def __len__(self):
        return len(self.data_dirs)

    def __getitem__(self, scene_id: int) -> dict[str, Any]:
        data_dir = str(self.data_dirs[scene_id], encoding="utf-8")
        valid, meta_info = load_and_maybe_update_meta_info(
            os.path.join(data_dir, "scene_meta.json")
        )
        assert valid, f"Invalid scene: {data_dir}"

        # Load context and target views together
        context_view_ids = self.contexts[scene_id][: self.input_views]
        target_view_ids = self.targets[scene_id][: self.supervise_views]
        frame_ids = np.concatenate([context_view_ids, target_view_ids])

        try:
            loaded = load_frames_from_meta_info(
                data_dir,
                meta_info,
                frame_ids,
                patch_size=self.patch_size,
                zoom_factor=self.zoom_factor,
            )
        except Exception as e:
            print(f"Error in {data_dir}: {e}. frame_ids: {frame_ids}")
            raise

        # Preprocess poses
        camtoworld = torch.from_numpy(loaded["camtoworld"]).float()
        camtoworld = _normalize_poses_identity_unit_distance(
            camtoworld, ref0_idx=0, ref1_idx=self.input_views - 1
        )
        K = torch.from_numpy(loaded["K"]).float()
        image = torch.from_numpy(loaded["image"]).float()
        image_path = loaded["image_path"]

        return {
            "camtoworld": camtoworld,
            "K": K,
            "image": image,
            "image_path": image_path,
            "scene": scene_id,
        }
