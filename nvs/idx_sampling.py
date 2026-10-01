"""Index sampling utilities for multi-view trajectory-based view selection.

Provides penalty-based view sampling, 1D trajectory ordering, and
even/odd splitting for input/target view assignment.
"""

import random

import numpy as np


def sample_and_split_trajectory(
    scene_meta, total_views, supervise_view, max_obj_angle=30.0, max_pano_angle=30.0
):
    """Sample views from frames, reorder them into a 1D trajectory, and split into
    input (even-indexed) and target (odd-indexed) views for NVS.
    """
    frames = scene_meta["frames"]
    num_frames = len(frames)

    # --- Internal helper functions ---
    def get_camera_poses(matrices):
        """Extract camera centers and normalized viewing directions from transform matrices."""
        c2w = np.array(matrices)
        centers = c2w[:, :3, 3]
        rays = c2w[:, :3, 2]
        norms = np.linalg.norm(rays, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return centers, rays / norms

    def calc_angles(ray_ref, rays_array):
        """Compute angles (in degrees) between a reference ray and an array of rays."""
        dot_products = np.sum(rays_array * ray_ref, axis=1)
        cos_angles = np.clip(dot_products, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angles))

    # Retrieve pose information
    matrices = [f["transform_matrix"] for f in frames]
    C, R = get_camera_poses(matrices)

    # --- 1. Penalty-based sampling ---
    if num_frames <= total_views:
        selected_indices = list(range(num_frames))
    else:
        idx0 = random.randint(0, num_frames - 1)
        selected_indices = [idx0]
        last_idx = idx0
        main_scores = np.zeros(num_frames)

        for _ in range(1, total_views):
            c0, r0 = C[last_idx], R[last_idx]

            # Distance score
            diff_c = C - c0
            distances = np.linalg.norm(diff_c, axis=1)
            max_dist = np.max(distances)
            dist_scores = np.zeros(num_frames)

            if max_dist < 1e-8:
                dist_scores.fill(4.0)
            else:
                dist_scores[distances > (max_dist * 3.0 / 4.0)] = 1.0
                dist_scores[
                    (distances > (max_dist * 2.0 / 4.0)) & (distances <= (max_dist * 3.0 / 4.0))
                ] = 2.0
                dist_scores[
                    (distances > (max_dist * 1.0 / 4.0)) & (distances <= (max_dist * 2.0 / 4.0))
                ] = 3.0
                dist_scores[distances <= (max_dist * 1.0 / 4.0)] = 4.0

            # Angle score
            diff_r = R - r0
            norm_r = np.linalg.norm(diff_r, axis=1) + 1e-8
            norm_c = distances + 1e-8
            dot_products = np.sum((diff_r / norm_r[:, None]) * (diff_c / norm_c[:, None]), axis=1)
            max_angles = np.where(dot_products < 0, max_obj_angle, max_pano_angle)

            angles = calc_angles(r0, R)
            angle_scores = np.zeros(num_frames)

            angle_scores[angles <= (max_angles / 4.0)] = 4.0
            angle_scores[(angles > (max_angles / 4.0)) & (angles <= (max_angles / 2.0))] = 3.0
            angle_scores[(angles > (max_angles / 2.0)) & (angles <= (max_angles * 3.0 / 4.0))] = (
                2.0
            )
            angle_scores[(angles > (max_angles * 3.0 / 4.0)) & (angles <= max_angles)] = 1.0

            step_scores = angle_scores * dist_scores
            main_scores += step_scores

            valid_mask = main_scores > 0.0
            valid_mask[selected_indices] = False

            if not np.any(valid_mask):
                remaining = [i for i in range(num_frames) if i not in selected_indices]
                next_idx = random.choice(remaining)
            else:
                valid_indices = np.where(valid_mask)[0]
                valid_scores = main_scores[valid_indices]
                min_score = np.min(valid_scores)
                min_score_indices = valid_indices[valid_scores == min_score]
                next_idx = random.choice(min_score_indices)

            selected_indices.append(int(next_idx))
            last_idx = int(next_idx)

    # --- 2. Reorder into a 1D trajectory (Trajectory Building) ---
    if not selected_indices:
        return [], []

    trajectory = [selected_indices[0]]
    for i in range(1, len(selected_indices)):
        curr_idx = selected_indices[i]
        curr_center = C[curr_idx]

        traj_centers = C[trajectory]
        distances = np.linalg.norm(traj_centers - curr_center, axis=1)

        closest_pos = np.argmin(distances)
        closest_center = traj_centers[closest_pos]

        if len(trajectory) == 1:
            trajectory.append(curr_idx)
            continue

        if closest_pos == 0:
            right_center = C[trajectory[1]]
            if np.linalg.norm(curr_center - right_center) < np.linalg.norm(
                closest_center - right_center
            ):
                trajectory.insert(1, curr_idx)
            else:
                trajectory.insert(0, curr_idx)

        elif closest_pos == len(trajectory) - 1:
            left_center = C[trajectory[-2]]
            if np.linalg.norm(curr_center - left_center) < np.linalg.norm(
                closest_center - left_center
            ):
                trajectory.insert(closest_pos, curr_idx)
            else:
                trajectory.append(curr_idx)

        else:
            left_idx, right_idx = trajectory[closest_pos - 1], trajectory[closest_pos + 1]
            if np.linalg.norm(curr_center - C[left_idx]) < np.linalg.norm(
                curr_center - C[right_idx]
            ):
                trajectory.insert(closest_pos, curr_idx)
            else:
                trajectory.insert(closest_pos + 1, curr_idx)

    # --- 3. Split into even/odd indices and shuffle ---
    input_views = trajectory[0::2]  # Even indices (input views)
    target_candidates = trajectory[1::2]  # Odd indices (target candidates)

    random.shuffle(target_candidates)
    target_views = target_candidates[:supervise_view]

    return input_views + target_views
