import unittest

import numpy as np
import torch

from sim.scene_features import (
    compute_surface_gap,
    find_closest_surface,
    pack_surface_features,
    unpack_surface_features,
)
from world_models.latent_state import LatentStateModel, ObservationEncoder


class SurfaceFeaturesTest(unittest.TestCase):
    def test_pack_uses_rotated_top_face_and_half_thickness(self):
        box_centers = np.asarray(
            [[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]],
            dtype=np.float32,
        )
        rotation_matrices = np.asarray(
            [[
                np.eye(3),
                [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            ]],
            dtype=np.float32,
        )

        packed = pack_surface_features(
            box_centers,
            rotation_matrices,
            np.asarray([2.0, 4.0], dtype=np.float32),
            np.asarray([[0.5]], dtype=np.float32),
        )

        np.testing.assert_array_equal(
            packed,
            np.asarray(
                [[
                    1.0, 2.0, 4.0, 0.0, 0.0, 1.0,
                    10.0, 22.0, 30.0, 0.0, 1.0, 0.0,
                    0.5,
                ]],
                dtype=np.float32,
            ),
        )

    def test_pack_and_unpack_preserve_surface_order_and_batch_rows(self):
        box_centers = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
        rotation_matrices = np.repeat(
            np.expand_dims(np.eye(3, dtype=np.float32), (0, 1)),
            6,
            axis=0,
        ).reshape(2, 3, 3, 3)
        thicknesses = np.asarray([0.2, 0.4, 0.6], dtype=np.float32)
        radii = np.asarray([[0.1], [0.3]], dtype=np.float32)

        packed = pack_surface_features(
            box_centers,
            rotation_matrices,
            thicknesses,
            radii,
        )
        surface_centers, surface_normals, unpacked_radii = unpack_surface_features(
            torch.from_numpy(packed)
        )

        expected_centers = box_centers.copy()
        expected_centers[:, :, 2] += thicknesses / 2.0
        torch.testing.assert_close(
            surface_centers,
            torch.from_numpy(expected_centers),
        )
        torch.testing.assert_close(
            surface_normals,
            torch.tensor([[[0.0, 0.0, 1.0]] * 3] * 2),
        )
        torch.testing.assert_close(unpacked_radii, torch.from_numpy(radii))

    def test_closest_surface_uses_absolute_plane_distance_per_batch_row(self):
        positions = torch.tensor(
            [[2.0, 0.25, 1.0], [-0.1, 2.0, 3.0], [1.0, 1.0, 5.0]]
        )
        surface_centers = torch.zeros((3, 3, 3))
        surface_normals = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]] * 3
        )

        surface_indices = find_closest_surface(
            positions,
            surface_centers,
            surface_normals,
        )

        torch.testing.assert_close(surface_indices, torch.tensor([1, 0, 0]))

    def test_closest_surface_ignores_offsets_within_each_infinite_plane(self):
        positions = torch.tensor([[0.0, 0.0, 0.4]])
        surface_centers = torch.tensor(
            [[[100.0, -50.0, 0.0], [0.0, 0.0, 2.0]]]
        )
        surface_normals = torch.tensor(
            [[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]]
        )

        torch.testing.assert_close(
            find_closest_surface(positions, surface_centers, surface_normals),
            torch.tensor([0]),
        )

    def test_surface_gap_handles_clearance_tangency_and_penetration(self):
        diagonal_normal = torch.tensor([2.0**-0.5, 0.0, 2.0**-0.5])
        surface_center = torch.tensor(
            [[10.0, -4.0, 1.0], [3.0, 2.0, -1.0], [0.0, 5.0, 0.0]]
        )
        surface_normal = torch.stack(
            (
                torch.tensor([0.0, 0.0, 1.0]),
                diagonal_normal,
                torch.tensor([0.0, 1.0, 0.0]),
            )
        )
        positions = torch.stack(
            (
                torch.tensor([12.0, 8.0, 1.25]),
                surface_center[1] + 0.2 * diagonal_normal + torch.tensor([5.0, 0.0, -5.0]),
                torch.tensor([0.0, 5.19, 0.0]),
            )
        )

        gaps = compute_surface_gap(
            positions,
            surface_center,
            surface_normal,
            torch.full((3, 1), 0.2),
        )

        torch.testing.assert_close(
            gaps,
            torch.tensor([[0.05], [0.0], [-0.01]]),
            atol=1e-6,
            rtol=0.0,
        )

    def test_encoder_assembles_position_selected_context_and_radius(self):
        encoder = ObservationEncoder()
        position = torch.tensor([[1.0, 2.0, 0.25]])
        surface_features = torch.tensor(
            [[
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                0.0, 0.0, 4.0, 0.0, 0.0, 1.0,
                0.2,
            ]]
        )

        encoded = encoder(position, surface_features)

        torch.testing.assert_close(
            encoded,
            torch.tensor([[1.0, 2.0, 0.25, 0.05, 0.0, 0.0, 1.0, 0.2]]),
        )

    def test_latent_state_updater_includes_encoded_observation_difference(self):
        model = LatentStateModel(latent_size=5, hidden_size=4)

        self.assertEqual(ObservationEncoder.scene_feature_size, 8)
        self.assertEqual(model.state_updater[0].in_features, 5 + 2 * 8)

    def test_unselected_farther_surface_does_not_change_encoder_input(self):
        encoder = ObservationEncoder()
        position = torch.tensor([[0.0, 0.0, 0.3]])
        nearest_surface = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        radius = [0.2]
        two_surfaces = torch.tensor(
            [[*nearest_surface, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, *radius]]
        )
        three_surfaces = torch.tensor(
            [[
                *nearest_surface,
                0.0, 0.0, 2.0, 0.0, 0.0, 1.0,
                0.0, 0.0, -3.0, 0.0, 0.0, 1.0,
                *radius,
            ]]
        )

        torch.testing.assert_close(
            encoder(position, two_surfaces),
            encoder(position, three_surfaces),
        )


if __name__ == "__main__":
    unittest.main()
