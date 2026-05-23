# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Foot clearance via downward raycasts to terrain (sole lowest point vs local ground height)."""

from __future__ import annotations

import numpy as np
import omni
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils.math import quat_apply
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh
from pxr import UsdGeom

from .rtv6_constants import (
    FOOT_RAY_CAST_LIFT,
    FOOT_RAY_MAX_DIST,
    FOOT_SOLE_Z_OFFSET_B,
    FOOTPRINT_HALF_LENGTH,
    FOOTPRINT_HALF_WIDTH,
    FOOTPRINT_RAY_RESOLUTION,
)

def _load_terrain_warp_mesh(mesh_prim_path: str, device: torch.device) -> wp.Mesh:
    """Load the terrain collision mesh for warp raycasting (same logic as :class:`~isaaclab.sensors.RayCaster`)."""
    mesh_prim = sim_utils.get_first_matching_child_prim(mesh_prim_path, lambda prim: prim.GetTypeName() == "Plane")
    if mesh_prim is None:
        mesh_prim = sim_utils.get_first_matching_child_prim(mesh_prim_path, lambda prim: prim.GetTypeName() == "Mesh")
        if mesh_prim is None or not mesh_prim.IsValid():
            raise RuntimeError(f"Invalid terrain mesh prim path: {mesh_prim_path}")
        mesh_prim = UsdGeom.Mesh(mesh_prim)
        points = np.asarray(mesh_prim.GetPointsAttr().Get())
        transform_matrix = np.array(omni.usd.get_world_transform_matrix(mesh_prim)).T
        points = np.matmul(points, transform_matrix[:3, :3].T)
        points += transform_matrix[:3, 3]
        indices = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get())
        return convert_to_warp_mesh(points, indices, device=device)
    mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
    return convert_to_warp_mesh(mesh.vertices, mesh.faces, device=device)


def _make_footprint_offsets(device: torch.device) -> torch.Tensor:
    """Ray start offsets in the foot link frame (xy grid at sole height)."""
    x = torch.arange(
        -FOOTPRINT_HALF_LENGTH,
        FOOTPRINT_HALF_LENGTH + 1.0e-9,
        FOOTPRINT_RAY_RESOLUTION,
        device=device,
    )
    y = torch.arange(
        -FOOTPRINT_HALF_WIDTH,
        FOOTPRINT_HALF_WIDTH + 1.0e-9,
        FOOTPRINT_RAY_RESOLUTION,
        device=device,
    )
    grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
    offsets = torch.zeros(grid_x.numel(), 3, device=device)
    offsets[:, 0] = grid_x.flatten()
    offsets[:, 1] = grid_y.flatten()
    offsets[:, 2] = FOOT_SOLE_Z_OFFSET_B
    return offsets


class RTv6FeetClearanceHelper:
    """Raycast terrain under each foot sole patch (world-down rays)."""

    def __init__(self, env, terrain_mesh_prim_path: str, device: torch.device) -> None:
        self.device = device
        self._terrain_mesh = _load_terrain_warp_mesh(terrain_mesh_prim_path, device)
        self._footprint_offsets_b = _make_footprint_offsets(device)
        self._num_rays_per_foot = self._footprint_offsets_b.shape[0]
        self._ray_lift = torch.tensor([0.0, 0.0, FOOT_RAY_CAST_LIFT], device=device)
        self._ray_dir = torch.tensor([0.0, 0.0, -1.0], device=device)

    def sole_clearance(
        self, asset: Articulation, foot_body_ids: list[int] | slice | torch.Tensor
    ) -> torch.Tensor:
        """Lowest sole point minus highest terrain hit under each foot. Shape ``(num_envs, num_feet)``."""
        link_pos = asset.data.body_link_pos_w[:, foot_body_ids]
        link_quat = asset.data.body_link_quat_w[:, foot_body_ids]
        num_envs, num_feet, _ = link_pos.shape
        k = self._num_rays_per_foot

        offsets = self._footprint_offsets_b.view(1, 1, k, 3).expand(num_envs, num_feet, -1, -1)
        quat = link_quat.unsqueeze(2).expand(-1, -1, k, -1).reshape(-1, 4)
        offset_flat = offsets.reshape(-1, 3)
        sole_pts = link_pos.unsqueeze(2) + quat_apply(quat, offset_flat).view(num_envs, num_feet, k, 3)

        sole_z_min = sole_pts[..., 2].amin(dim=2)
        ray_starts = sole_pts.reshape(-1, 3) + self._ray_lift
        ray_dirs = self._ray_dir.unsqueeze(0).expand(ray_starts.shape[0], -1)

        hits, _, _, _ = raycast_mesh(
            ray_starts,
            ray_dirs,
            mesh=self._terrain_mesh,
            max_dist=FOOT_RAY_MAX_DIST,
        )
        hit_z = hits[:, 2].view(num_envs, num_feet, k)
        valid = torch.isfinite(hit_z)
        terrain_z = torch.where(valid, hit_z, torch.full_like(hit_z, float("-inf"))).amax(dim=2)
        no_hit = ~valid.any(dim=2)
        clearance = sole_z_min - terrain_z
        return torch.where(no_hit, torch.zeros_like(clearance), torch.clamp(clearance, min=0.0))


def feet_clearance_capped(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    target_height: float = 0.05,
    min_air_time: float = 0.02,
) -> torch.Tensor:
    """Swing-phase clearance: sole-to-terrain distance (raycast), capped at ``target_height``."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    in_air = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids] > min_air_time
    clearance = env._feet_clearance_helper.sole_clearance(asset, asset_cfg.body_ids)
    scaled = torch.clamp(clearance / target_height, min=0.0, max=1.0)
    return (scaled * in_air.float()).sum(dim=1)
