"""Scene-feature widths and infinite top-plane geometry for simulated worlds."""

import numpy as np
import torch

SIMPLE_ROLLING_SCENE_FEATURE_SIZE = 10
OPPOSING_RAMPS_SCENE_FEATURE_SIZE = 19
SURFACE_FEATURE_SCHEMA = "top_planes_v1"


def pack_surface_features(
    box_centers: np.ndarray,
    rotation_matrices: np.ndarray,
    box_thicknesses: np.ndarray,
    ball_radii: np.ndarray,
) -> np.ndarray:
    """Build flat top-plane records from batched box poses.

    B is environment count and S is surface count. Each box rotation maps its
    local axes into world coordinates; its local Z axis is the outward top-face
    normal. Output records are ordered by surface, with each center immediately
    followed by its normal, and the ball radius stored last.

    @param box_centers: Float array shaped (B, S, 3), containing box-center XYZ
        positions in world coordinates.
    @param rotation_matrices: Float array shaped (B, S, 3, 3), containing box
        rotations whose columns are the local axes in world coordinates.
    @param box_thicknesses: Float array shaped (S,), containing box sizes along
        each local Z axis.
    @param ball_radii: Float array shaped (B, 1), containing one radius per
        environment.

    @return: Float array shaped (B, 6*S + 1), containing consecutive
        [top_center_xyz, normal_xyz] records followed by ball radius.
    """
    raise NotImplementedError

# These need to be packed and unpacked because they are part of the training 
# data / data loader -- easier to keep those as tensors
def unpack_surface_features(
    surface_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack flat surface records without assuming a fixed surface count.

    @param surface_features: Float tensor shaped (B, 6*S + 1), containing
        consecutive [top_center_xyz, normal_xyz] records and ball radius.

    @return: Surface centers shaped (B, S, 3), outward unit normals shaped
        (B, S, 3), and ball radii shaped (B, 1).
    """
    raise NotImplementedError


def find_closest_surface(
    positions: torch.Tensor,
    surface_centers: torch.Tensor,
    surface_normals: torch.Tensor,
) -> torch.Tensor:
    """Find the infinite plane nearest each ball center (in batch).

    B is batch size and S is surface count. Distance is the absolute signed
    distance from a ball center to a plane. Exact ties select the first surface
    in the supplied order.

    @param positions: Float tensor shaped (B, 3), containing ball-center XYZ
        positions in world coordinates.
    @param surface_centers: Float tensor shaped (B, S, 3), containing one point
        on each top plane.
    @param surface_normals: Float tensor shaped (B, S, 3), containing outward
        unit normals in world coordinates.

    @return: Long tensor shaped (B,), containing selected surface indices.
    """
    raise NotImplementedError


def compute_surface_gap(
    positions: torch.Tensor,
    surface_centers: torch.Tensor,
    surface_normals: torch.Tensor,
    ball_radii: torch.Tensor,
) -> torch.Tensor:
    """Compute signed ball-surface gaps for already-selected planes.

    The gap is dot(position - surface_center, surface_normal) - ball_radius.
    Positive values indicate clearance on the outward side, zero indicates
    tangency, and negative values indicate penetration into the plane's solid
    half-space.

    @param positions: Float tensor shaped (B, 3), containing ball-center XYZ
        positions in world coordinates.
    @param surface_centers: Float tensor shaped (B, 3), containing the selected
        point on each plane.
    @param surface_normals: Float tensor shaped (B, 3), containing selected
        outward unit normals in world coordinates.
    @param ball_radii: Float tensor shaped (B, 1), containing ball radii.

    @return: Float tensor shaped (B, 1), containing signed surface gaps.
    """
    raise NotImplementedError
